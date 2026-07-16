"""Compute attribute direction .npz files for each encoder from an attribute CSV.

For every requested encoder (dinov2 / clip / diffae) this script:
  1. extracts representations for all images listed in the CSV (batched),
  2. computes mean(positive) - mean(negative) per attribute column,
  3. saves attributes/{encoder}.npz with one key per attribute.

Per-image representations are cached as .npy under --rep_cache so re-runs
(e.g. with new attributes) are fast.

Example:
    python tools/compute_attributes.py \
        --image_dir /path/to/celebahq \
        --attr_file annotations/list_attr_celebahq.csv \
        --encoders dinov2 clip \
        --max_images 5000

    # DiffAE needs its semantic encoder checkpoint in checkpoints/
    python tools/compute_attributes.py --image_dir ... --encoders diffae
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app import config  # noqa: E402
from app.models import _dino_transform, _diffae_transform  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CsvImageDataset(Dataset):
    def __init__(self, image_dir, filenames, transform=None):
        self.image_dir = Path(image_dir)
        self.filenames = filenames
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        image = Image.open(self.image_dir / name).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, name


def collate_pil(batch):
    images, names = zip(*batch)
    return list(images), list(names)


def build_batch_encoder(encoder_key):
    """Return (encode_fn(batch)->tensor[B,D], transform, collate_fn)."""
    if encoder_key == "dinov2":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        model.eval().to(device)

        @torch.no_grad()
        def encode(images):
            return model(images.to(device)).float().cpu()

        return encode, _dino_transform, None

    if encoder_key == "clip":
        from transformers import CLIPModel, CLIPProcessor

        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.eval().to(device)

        @torch.no_grad()
        def encode(images):  # images: list of PIL
            inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
            vision_outputs = model.vision_model(**inputs)
            return vision_outputs.last_hidden_state[:, 0, :].float().cpu()  # CLS token

        return encode, None, collate_pil

    if encoder_key == "diffae":
        info = config.resolve_models().get("diffae", {})
        enc_path = info.get("encoder_checkpoint")
        if not enc_path:
            raise RuntimeError("DiffAE semantic encoder checkpoint not found in "
                               f"{config.CHECKPOINT_DIR} (filename must contain 'encoder').")
        model = torch.load(enc_path, map_location=device, weights_only=False)
        model.eval().to(device)

        @torch.no_grad()
        def encode(images):
            return model(images.to(device)).flatten(1).float().cpu()

        return encode, _diffae_transform, None

    raise ValueError(f"Unknown encoder: {encoder_key}")


def extract_representations(encoder_key, image_dir, filenames, cache_dir, batch_size):
    """Return {filename: np.ndarray[D]}, computing and caching missing ones."""
    cache_dir = Path(cache_dir) / encoder_key
    cache_dir.mkdir(parents=True, exist_ok=True)

    reps = {}
    missing = []
    for name in filenames:
        npy = cache_dir / (os.path.splitext(name)[0] + ".npy")
        if npy.is_file():
            reps[name] = np.load(npy)
        else:
            missing.append(name)

    print(f"[{encoder_key}] {len(reps)} cached, {len(missing)} to compute")
    if missing:
        encode, transform, collate_fn = build_batch_encoder(encoder_key)
        dataset = CsvImageDataset(image_dir, missing, transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=2, collate_fn=collate_fn)
        done = 0
        for images, names in loader:
            features = encode(images)
            for feat, name in zip(features, names):
                arr = feat.numpy().astype(np.float32)
                np.save(cache_dir / (os.path.splitext(name)[0] + ".npy"), arr)
                reps[name] = arr
            done += len(names)
            print(f"[{encoder_key}] extracted {done}/{len(missing)}", end="\r")
        print()
    return reps


def compute_attribute_vectors(attr_df, reps, attributes):
    """Return {attribute: mean(pos)-mean(neg)} using available representations."""
    available = [f for f in attr_df.index if f in reps]
    if not available:
        raise RuntimeError("No overlap between CSV entries and extracted representations.")
    stacked = np.stack([reps[f] for f in available], axis=0)
    labels = attr_df.loc[available]

    vectors = {}
    for attr in attributes:
        pos = labels[attr].values == 1
        neg = ~pos
        if pos.sum() == 0 or neg.sum() == 0:
            print(f"  ! Skipping '{attr}': one class is empty")
            continue
        vectors[attr] = (stacked[pos].mean(axis=0) - stacked[neg].mean(axis=0)).astype(np.float32)
        print(f"  {attr}: {int(pos.sum())} positive / {int(neg.sum())} negative")
    return vectors


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image_dir", required=True, help="Folder with the images referenced by the CSV")
    parser.add_argument("--attr_file", default=str(REPO_ROOT / "annotations/list_attr_celebahq.csv"),
                        help="CelebA-style attribute CSV (filename column + one column per attribute)")
    parser.add_argument("--encoders", nargs="+", default=["dinov2", "clip", "diffae"],
                        choices=["dinov2", "clip", "diffae"], help="Encoders to process")
    parser.add_argument("--attributes", nargs="*", default=None,
                        help="Attribute columns to export (default: all columns in the CSV)")
    parser.add_argument("--output", default=str(config.ATTRIBUTE_DIR), help="Output directory for .npz files")
    parser.add_argument("--rep_cache", default=str(REPO_ROOT / "cache/reps"),
                        help="Directory to cache per-image representations")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_images", type=int, default=None,
                        help="Optionally limit to the first N images of the CSV")
    args = parser.parse_args()

    if not os.path.isdir(args.image_dir):
        raise SystemExit(f"Image directory not found: {args.image_dir}")

    attr_df = pd.read_csv(args.attr_file, sep=",", header=0, index_col=0)
    attributes = args.attributes or list(attr_df.columns)
    unknown = [a for a in attributes if a not in attr_df.columns]
    if unknown:
        raise SystemExit(f"Attributes not in CSV: {unknown}\nAvailable: {list(attr_df.columns)}")

    filenames = [f for f in attr_df.index if (Path(args.image_dir) / f).is_file()]
    print(f"{len(filenames)}/{len(attr_df)} CSV images found in {args.image_dir}")
    if not filenames:
        raise SystemExit("No CSV-listed images found in the image directory.")
    if args.max_images:
        filenames = filenames[:args.max_images]
        print(f"Limiting to first {len(filenames)} images")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for encoder_key in args.encoders:
        print(f"\n=== {encoder_key} ===")
        try:
            reps = extract_representations(encoder_key, args.image_dir, filenames,
                                           args.rep_cache, args.batch_size)
        except RuntimeError as e:
            print(f"Skipping {encoder_key}: {e}")
            continue
        vectors = compute_attribute_vectors(attr_df, reps, attributes)
        out_file = out_dir / f"{encoder_key}.npz"
        np.savez(out_file, **vectors)
        print(f"Saved {len(vectors)} attribute vectors -> {out_file}")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
