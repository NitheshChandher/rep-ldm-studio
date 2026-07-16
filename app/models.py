"""Lazy loading / caching of representation encoders, UNets and the SD VAE."""
import threading

import torch
import torchvision.transforms as transforms
from diffusers import AutoencoderKL

from app import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_lock = threading.Lock()
_encoders = {}          # encoder cache: key -> callable(pil) -> tensor [D]
_unet_cache = {"path": None, "unet": None}
_vae = None

# Same transform as extract_rep.py for DINOv2
_dino_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Same transform as extract_rep.py for DIFFAE
_diffae_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


def _build_encoder(model_key, model_info):
    if model_key == "dinov2":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        model.eval().to(device)

        def encode(pil):
            x = _dino_transform(pil).unsqueeze(0).to(device)
            with torch.no_grad():
                return model(x)[0].float().cpu()

        return encode

    if model_key == "clip":
        from transformers import CLIPModel, CLIPProcessor

        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.eval().to(device)

        def encode(pil):
            inputs = processor(images=[pil], return_tensors="pt").to(device)
            with torch.no_grad():
                vision_outputs = model.vision_model(**inputs)
            return vision_outputs.last_hidden_state[0, 0, :].float().cpu()  # CLS token

        return encode

    if model_key == "diffae":
        enc_path = model_info.get("encoder_checkpoint")
        if not enc_path:
            raise RuntimeError("DiffAE semantic encoder checkpoint not found in checkpoints/ "
                               "(expected a file with 'encoder' in its name).")
        model = torch.load(enc_path, map_location=device, weights_only=False)
        model.eval().to(device)

        def encode(pil):
            x = _diffae_transform(pil).unsqueeze(0).to(device)
            with torch.no_grad():
                return model(x)[0].flatten().float().cpu()

        return encode

    raise ValueError(f"Unknown model key: {model_key}")


def extract_representation(model_key, model_info, pil_image):
    """Return the representation vector for a PIL image as tensor [D]."""
    with _lock:
        if model_key not in _encoders:
            _encoders[model_key] = _build_encoder(model_key, model_info)
        encoder = _encoders[model_key]
    return encoder(pil_image.convert("RGB"))


def get_unet(checkpoint_path):
    """Load (and cache one) UNet. Switching models frees the previous one."""
    with _lock:
        if _unet_cache["path"] != str(checkpoint_path):
            if _unet_cache["unet"] is not None:
                _unet_cache["unet"] = None
                torch.cuda.empty_cache()
            unet = torch.load(checkpoint_path, map_location=device, weights_only=False).to(device)
            unet.eval()
            _unet_cache["path"] = str(checkpoint_path)
            _unet_cache["unet"] = unet
        return _unet_cache["unet"]


def get_vae():
    global _vae
    with _lock:
        if _vae is None:
            _vae = AutoencoderKL.from_pretrained(
                config.SD_MODEL_ID, subfolder="vae", revision=None
            ).to(device)
            _vae.eval()
        return _vae
