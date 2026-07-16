"""Core generation pipelines adapted from interpolate.py and attribute_manipulation.py."""
import numpy as np
import torch
import torchvision.transforms as transforms
from diffusers import DDIMScheduler
from PIL import Image
from torch import autocast, inference_mode

from app.inversion_utils import inversion_forward_process, inversion_reverse_process
from app import config, models
from app.models import device


def _cos(a, b):
    a = torch.nn.functional.normalize(a.reshape(-1), dim=0)
    b = torch.nn.functional.normalize(b.reshape(-1), dim=0)
    return (a * b).sum()


def slerp(x0, x1, alpha):
    """Spherical linear interpolation (Eq. 67, DDIM paper)."""
    theta = torch.arccos(_cos(x0, x1))
    x_shape = x0.shape
    x_interp = (torch.sin((1 - alpha) * theta) * x0.flatten()
                + torch.sin(alpha * theta) * x1.flatten()) / torch.sin(theta)
    return x_interp.view(*x_shape)


def _seed_everything(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def _preprocess(pil, res):
    transform = transforms.Compose([
        transforms.Resize(res),
        transforms.CenterCrop(res),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    return transform(pil.convert("RGB"))


def _autocast_ctx():
    return autocast("cuda") if torch.cuda.is_available() else torch.no_grad()


def _decode_latents(vae, w0, chunk=3):
    """Decode latents to PIL images in chunks to limit VRAM usage."""
    images = []
    with _autocast_ctx(), inference_mode():
        for i in range(0, w0.shape[0], chunk):
            x_dec = vae.decode(1 / 0.18215 * w0[i:i + chunk]).sample
            imgs = (x_dec / 2 + 0.5).clamp(0, 1)
            imgs = imgs.detach().cpu().float().permute(0, 2, 3, 1).numpy()
            imgs = (imgs * 255).round().astype("uint8")
            images.extend(Image.fromarray(im) for im in imgs)
    return images


def run_interpolation(model_info, img1, img2, *, sampling="linear-spherical",
                      num_diffusion_steps=100, skip=36, cfg_src=1.0, cfg_tar=1.0,
                      eta=1.0, num_frames=9, progress=lambda p, m: None):
    """Interpolate between two uploaded PIL images. Returns list of PIL images."""
    _seed_everything()
    model_key = model_info["key"]

    progress(0.02, "Extracting representations")
    z0 = models.extract_representation(model_key, model_info, img1).to(device)
    z1 = models.extract_representation(model_key, model_info, img2).to(device)
    z0 = z0.view(1, 1, -1)
    z1 = z1.view(1, 1, -1)
    z_pair = torch.cat([z0, z1], dim=0)

    progress(0.08, "Loading model")
    unet = models.get_unet(model_info["checkpoint"])
    res = unet.sample_size * 8  # derive image resolution from the UNet latent size

    x0 = _preprocess(img1, res).unsqueeze(0).to(device)
    x1 = _preprocess(img2, res).unsqueeze(0).to(device)
    inp = torch.cat([x0, x1], dim=0)

    scheduler = DDIMScheduler.from_config(config.SD_MODEL_ID, subfolder="scheduler")
    scheduler.set_timesteps(num_diffusion_steps)
    vae = models.get_vae()

    progress(0.12, "Encoding images")
    with _autocast_ctx(), inference_mode():
        w0 = (vae.encode(inp).latent_dist.mode() * 0.18215).float()

        progress(0.15, "Inverting images (forward process)")
        wt, zs, wts = inversion_forward_process(
            unet, scheduler, w0, etas=eta, encoder_hidden_states=z_pair,
            cfg_scale=cfg_src, prog_bar=True, num_inference_steps=num_diffusion_steps)

    torch.cuda.empty_cache()
    progress(0.55, "Interpolating")

    K = num_frames
    alpha = np.linspace(0, 1, K)
    z_interp = torch.zeros([K, 1, z_pair.shape[2]], device=device)
    zs_interp = torch.zeros([zs.shape[0], K, zs.shape[2], zs.shape[3], zs.shape[4]], device=device)
    wts_interp = torch.zeros([K, wts.shape[2], wts.shape[3], wts.shape[4]], device=device)
    t_start = num_diffusion_steps - skip

    if sampling == "linear-spherical":
        for k in range(K):
            z_interp[k] = (1 - alpha[k]) * z0 + alpha[k] * z1
            for t in range(zs.shape[0]):
                zs_interp[t, k] = slerp(zs[t, 0], zs[t, 1], alpha[k])
            wts_interp[k] = slerp(wts[t_start, 0], wts[t_start, 1], alpha[k])
    elif sampling == "linear":
        for k in range(K):
            z_interp[k] = (1 - alpha[k]) * z0 + alpha[k] * z1
            zs_interp[:, k] = zs[:, 0]
            wts_interp[k] = wts[t_start, 0]
    elif sampling == "spherical":
        for k in range(K):
            z_interp[k] = z0
            for t in range(zs.shape[0]):
                zs_interp[t, k] = slerp(zs[t, 0], zs[t, 1], alpha[k])
            wts_interp[k] = slerp(wts[t_start, 0], wts[t_start, 1], alpha[k])
    else:
        raise ValueError(f"Invalid sampling method: {sampling}")

    progress(0.6, "Generating interpolated images (reverse process)")
    with _autocast_ctx(), inference_mode():
        w0, _ = inversion_reverse_process(
            unet, scheduler, xT=wts_interp, etas=eta, encoder_hidden_states=z_interp,
            cfg_scales=[cfg_tar], prog_bar=True, zs=zs_interp[:t_start])

    torch.cuda.empty_cache()
    progress(0.9, "Decoding images")
    images = _decode_latents(vae, w0)
    progress(0.98, "Saving results")
    return images


def run_attribute_edit(model_info, img, attribute_name, *, lamda=1.0,
                       num_diffusion_steps=100, skip=36, cfg_src=1.0, cfg_tar=1.0,
                       eta=1.0, progress=lambda p, m: None):
    """Edit an attribute of an uploaded PIL image. Returns [edited PIL image]."""
    _seed_everything()
    model_key = model_info["key"]

    attributes = config.load_attribute_vectors(model_key)
    if attribute_name not in attributes:
        raise ValueError(f"Attribute '{attribute_name}' not available for model '{model_key}'. "
                         f"Available: {sorted(attributes)}")
    attribute_emb = torch.tensor(attributes[attribute_name], dtype=torch.float32,
                                 device=device).view(1, 1, -1)

    progress(0.02, "Extracting representation")
    z0 = models.extract_representation(model_key, model_info, img).to(device).view(1, 1, -1)

    progress(0.08, "Loading model")
    unet = models.get_unet(model_info["checkpoint"])
    res = unet.sample_size * 8

    x0 = _preprocess(img, res).unsqueeze(0).to(device)

    scheduler = DDIMScheduler.from_config(config.SD_MODEL_ID, subfolder="scheduler")
    scheduler.set_timesteps(num_diffusion_steps)
    vae = models.get_vae()

    progress(0.12, "Encoding image")
    with _autocast_ctx(), inference_mode():
        w0 = (vae.encode(x0).latent_dist.mode() * 0.18215).float()

        progress(0.15, "Inverting image (forward process)")
        wt, zs, wts = inversion_forward_process(
            unet, scheduler, w0, etas=eta, encoder_hidden_states=z0,
            cfg_scale=cfg_src, prog_bar=True, num_inference_steps=num_diffusion_steps)

        torch.cuda.empty_cache()

        progress(0.6, "Applying attribute and generating (reverse process)")
        z_edit = z0 + lamda * attribute_emb
        w0, _ = inversion_reverse_process(
            unet, scheduler, xT=wts[num_diffusion_steps - skip], etas=eta,
            encoder_hidden_states=z_edit, cfg_scales=[cfg_tar], prog_bar=True,
            zs=zs[:(num_diffusion_steps - skip)])

    torch.cuda.empty_cache()
    progress(0.9, "Decoding image")
    images = _decode_latents(vae, w0)
    progress(0.98, "Saving results")
    return images
