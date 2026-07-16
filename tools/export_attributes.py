"""Export attribute direction vectors into the attributes/ layout used by the app.

Computes mean-difference vectors from a directory of per-image .npy
representations + a CelebA-style attribute CSV, and stores them in a single
{model_key}.npz that the app can load.

Example:
    python tools/export_attributes.py \
        --rep_path /path/to/rep_dinov2 \
        --attr_file /path/to/list_attr_celeba.csv \
        --model_key dinov2 \
        --attributes Blond_Hair Smiling Eyeglasses
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


def mean_attribute_difference(directory_path, attr_file, attribute_name):
    """Mean(positive class) - mean(negative class) representation vector."""
    attr_df = pd.read_csv(attr_file, sep=",", header=0, index_col=0)
    if attribute_name not in attr_df.columns:
        raise ValueError(f"Attribute '{attribute_name}' not found in attribute file.")

    selected = attr_df[attr_df[attribute_name] == 1].index.tolist()
    remaining = attr_df[attr_df[attribute_name] != 1].index.tolist()

    def to_npy(files):
        return [os.path.splitext(f)[0] + ".npy" for f in files]

    def load(files):
        arrays = []
        for f in to_npy(files):
            p = os.path.join(directory_path, f)
            if os.path.exists(p):
                arrays.append(np.load(p))
        return arrays

    selected_arrays = load(selected)
    remaining_arrays = load(remaining)
    if not selected_arrays:
        raise ValueError(f"No valid .npy files found for attribute '{attribute_name}' (positive class).")
    if not remaining_arrays:
        raise ValueError("No valid .npy files found for remaining data (negative class).")

    mean_selected = np.mean(np.stack(selected_arrays, axis=0), axis=0)
    mean_remaining = np.mean(np.stack(remaining_arrays, axis=0), axis=0)
    return (mean_selected - mean_remaining).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep_path", required=True, help="Directory of per-image .npy representations")
    parser.add_argument("--attr_file", required=True, help="CelebA-style attribute CSV")
    parser.add_argument("--model_key", required=True, choices=["dinov2", "clip", "diffae"])
    parser.add_argument("--attributes", nargs="+", required=True, help="Attribute column names")
    parser.add_argument("--output", default=str(REPO_ROOT / "attributes"), help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    vectors = {}
    for name in args.attributes:
        print(f"Computing direction vector for '{name}' ...")
        vectors[name] = mean_attribute_difference(args.rep_path, args.attr_file, name)

    out_file = out_dir / f"{args.model_key}.npz"
    np.savez(out_file, **vectors)
    print(f"Saved {len(vectors)} attribute vectors to {out_file}")


if __name__ == "__main__":
    main()
