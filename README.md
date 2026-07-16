# Rep-LDM Studio

Web interface for **representation-conditioned diffusion models** (DINOv2-LDM, CLIP-LDM, DiffAE-LDM) with two features:

1. **Interpolation** — upload two images and generate a sequence of images interpolated in representation space (with optional spherical interpolation of noise maps).
2. **Feature Editing** — upload an image, pick an attribute (e.g. *Blond_Hair*, *Smiling*) and a strength λ, and generate the edited image.

Representations for uploaded images are extracted on the fly (DINOv2 / CLIP / DiffAE semantic encoder). All generated images can be downloaded from the UI.

Companion app for the [rep-ldm](https://github.com/nithesh/rep-ldm) research codebase (*Evaluating Representation Conditioned Diffusion Models*).

## Setup

```bash
pip install -r requirements.txt
```

Requires a CUDA GPU for practical generation speed. The Stable Diffusion v1-4 VAE/scheduler and the DINOv2/CLIP encoders are downloaded automatically on first use.

## Expected folder layout

```
rep-ldm-studio/
├── checkpoints/                 # UNet checkpoints (auto-discovered by filename)
│   ├── dinov2-ldm.pt            # name must contain "dino"
│   ├── clip-ldm.pt              # name must contain "clip"
│   ├── diffae-ldm.pt            # name must contain "diffae" (or "semantic")
│   └── diffae-encoder.pt        # DiffAE semantic encoder — name must contain "encoder"
├── attributes/                  # attribute direction vectors, one of:
│   ├── dinov2.npz               #   a) {model_key}.npz — keys = attribute names
│   ├── clip.npz
│   ├── diffae.npz
│   ├── dinov2/Blond_Hair.npy    #   b) {model_key}/<Attribute>.npy per attribute
│   └── attribute.py             #   c) module with ATTRIBUTES = {model_key: {name: vector}}
└── app/
```

Override locations with environment variables: `REPLDM_CHECKPOINT_DIR`, `REPLDM_ATTRIBUTE_DIR`, `REPLDM_OUTPUT_DIR`.

To compute `.npz` attribute files from per-image representations + a CelebA-style CSV:

```bash
python tools/export_attributes.py \
    --rep_path /path/to/rep_dinov2 \
    --attr_file /path/to/list_attr_celeba.csv \
    --model_key dinov2 \
    --attributes Blond_Hair Smiling Eyeglasses
```

## Running

```bash
./run.sh                         # uses $REPLDM_PYTHON or ~/anaconda3/envs/di/bin/python
# or
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000

## Architecture

- `app/main.py` — FastAPI endpoints (`/api/models`, `/api/interpolate`, `/api/edit`, `/api/job/{id}`, `/api/download/{id}/{file}`)
- `app/pipeline.py` — DDPM-inversion based interpolation & attribute editing
- `app/models.py` — lazy loading/caching of encoders, UNets, and the SD VAE
- `app/inversion_utils.py` — edit-friendly DDPM inversion (forward + reverse)
- `app/jobs.py` — single-GPU background job queue with progress reporting
- `app/static/` — vanilla HTML/JS/CSS frontend
- `diffae/` — semantic encoder definition (needed to unpickle DiffAE encoder checkpoints)

## Notes

- Image resolution is derived automatically from each UNet's `sample_size` (× 8).
- Generation runs on a single background worker; the UI polls job progress.
- Outputs are stored under `outputs/<job_id>/` and served at `/api/download/<job_id>/<file>`.
