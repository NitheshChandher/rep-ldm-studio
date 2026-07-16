"""Configuration and discovery of model checkpoints / attribute vectors."""
import os
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent

CHECKPOINT_DIR = Path(os.environ.get("REPLDM_CHECKPOINT_DIR", REPO_ROOT / "checkpoints"))
ATTRIBUTE_DIR = Path(os.environ.get("REPLDM_ATTRIBUTE_DIR", REPO_ROOT / "attributes"))
OUTPUT_DIR = Path(os.environ.get("REPLDM_OUTPUT_DIR", REPO_ROOT / "outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SD_MODEL_ID = "CompVis/stable-diffusion-v1-4"

CKPT_EXTS = (".pt", ".pth", ".ckpt")

# key -> spec. `keywords` are matched (case-insensitive) against checkpoint
# filenames inside CHECKPOINT_DIR to locate the UNet for each model.
MODEL_SPECS = {
    "dinov2": {
        "display": "DINOv2-LDM",
        "keywords": ("dino",),
        "exclude": ("encoder",),
    },
    "clip": {
        "display": "CLIP-LDM",
        "keywords": ("clip",),
        "exclude": ("encoder",),
    },
    "diffae": {
        "display": "DiffAE-LDM",
        "keywords": ("diffae", "semantic"),
        "exclude": ("encoder",),
        # DiffAE additionally needs its semantic encoder checkpoint
        "encoder_keywords": ("encoder",),
    },
}


def _find_checkpoint(keywords, exclude=()):
    if not CHECKPOINT_DIR.is_dir():
        return None
    for f in sorted(CHECKPOINT_DIR.iterdir()):
        if f.suffix.lower() not in CKPT_EXTS:
            continue
        name = f.name.lower()
        if any(k in name for k in keywords) and not any(e in name for e in exclude):
            return f
    return None


def resolve_models():
    """Return {key: {display, checkpoint, encoder_checkpoint, available}}."""
    out = {}
    for key, spec in MODEL_SPECS.items():
        ckpt = _find_checkpoint(spec["keywords"], spec.get("exclude", ()))
        enc_ckpt = None
        if "encoder_keywords" in spec:
            enc_ckpt = _find_checkpoint(spec["encoder_keywords"])
        available = ckpt is not None and ("encoder_keywords" not in spec or enc_ckpt is not None)
        out[key] = {
            "key": key,
            "display": spec["display"],
            "checkpoint": str(ckpt) if ckpt else None,
            "encoder_checkpoint": str(enc_ckpt) if enc_ckpt else None,
            "available": available,
        }
    return out


def load_attribute_vectors(model_key):
    """Load attribute direction vectors for a model.

    Supported layouts inside ATTRIBUTE_DIR:
      1. {model_key}.npz            -> keys are attribute names
      2. {model_key}/ *.npy         -> one file per attribute (filename = attribute name)
      3. attribute.py               -> module exposing ATTRIBUTES = {model_key: {name: vector}}
    Returns {attribute_name: np.ndarray}.
    """
    result = {}

    npz_path = ATTRIBUTE_DIR / f"{model_key}.npz"
    if npz_path.is_file():
        with np.load(npz_path) as data:
            for name in data.files:
                result[name] = np.asarray(data[name], dtype=np.float32)
        return result

    sub_dir = ATTRIBUTE_DIR / model_key
    if sub_dir.is_dir():
        for f in sorted(sub_dir.glob("*.npy")):
            result[f.stem] = np.load(f).astype(np.float32)
        if result:
            return result

    attr_py = ATTRIBUTE_DIR / "attribute.py"
    if attr_py.is_file():
        import importlib.util

        spec = importlib.util.spec_from_file_location("user_attributes", attr_py)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        attrs = getattr(module, "ATTRIBUTES", None)
        if isinstance(attrs, dict) and model_key in attrs:
            for name, vec in attrs[model_key].items():
                result[name] = np.asarray(vec, dtype=np.float32)
    return result
