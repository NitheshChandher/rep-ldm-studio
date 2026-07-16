"""FastAPI backend for the Rep-LDM web app.

Run from the repository root:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
import io
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from app import config
from app.jobs import JobManager

app = FastAPI(title="Rep-LDM Studio")
job_manager = JobManager()

STATIC_DIR = config.APP_DIR / "static"


def _read_upload(upload: UploadFile) -> Image.Image:
    data = upload.file.read()
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, f"Could not read image '{upload.filename}'")


def _get_model_or_400(model_key: str):
    info = config.resolve_models().get(model_key)
    if info is None:
        raise HTTPException(400, f"Unknown model '{model_key}'")
    if not info["available"]:
        raise HTTPException(400, f"Checkpoint for '{info['display']}' not found in "
                                 f"{config.CHECKPOINT_DIR}")
    return info


def _save_images(job, images, labels, inputs=None):
    job_dir = config.OUTPUT_DIR / job.id
    job_dir.mkdir(parents=True, exist_ok=True)
    for pil, label in zip(images, labels):
        fname = f"{label}.png"
        pil.save(job_dir / fname)
        job.results.append({"file": fname, "label": label})
    for name, pil in (inputs or []):
        fname = f"input_{name}.png"
        pil.save(job_dir / fname)
        job.inputs.append(fname)


@app.get("/api/models")
def list_models():
    result = []
    for key, info in config.resolve_models().items():
        attrs = sorted(config.load_attribute_vectors(key)) if info["available"] else []
        result.append({
            "key": key,
            "display": info["display"],
            "available": info["available"],
            "attributes": attrs,
        })
    return {"models": result,
            "checkpoint_dir": str(config.CHECKPOINT_DIR),
            "attribute_dir": str(config.ATTRIBUTE_DIR)}


@app.post("/api/interpolate")
def interpolate(model: str = Form(...),
                sampling: str = Form("linear-spherical"),
                num_diffusion_steps: int = Form(100),
                skip: int = Form(36),
                cfg_src: float = Form(1.0),
                cfg_tar: float = Form(1.0),
                num_frames: int = Form(9),
                image1: UploadFile = File(...),
                image2: UploadFile = File(...)):
    if sampling not in ("linear", "spherical", "linear-spherical"):
        raise HTTPException(400, f"Invalid sampling method '{sampling}'")
    if not (2 <= num_frames <= 15):
        raise HTTPException(400, "num_frames must be between 2 and 15")
    if not (0 < skip < num_diffusion_steps):
        raise HTTPException(400, "skip must be between 1 and num_diffusion_steps-1")

    info = _get_model_or_400(model)
    img1, img2 = _read_upload(image1), _read_upload(image2)

    def work(job):
        from app import pipeline
        images = pipeline.run_interpolation(
            info, img1, img2, sampling=sampling,
            num_diffusion_steps=num_diffusion_steps, skip=skip,
            cfg_src=cfg_src, cfg_tar=cfg_tar, num_frames=num_frames,
            progress=job.set_progress)
        labels = [f"alpha_{i / (len(images) - 1):.2f}" for i in range(len(images))]
        _save_images(job, images, labels, inputs=[("1", img1), ("2", img2)])

    job = job_manager.submit("interpolate", work)
    return {"job_id": job.id}


@app.post("/api/edit")
def edit(model: str = Form(...),
         attribute: str = Form(...),
         lamda: float = Form(1.0),
         num_diffusion_steps: int = Form(100),
         skip: int = Form(36),
         cfg_src: float = Form(1.0),
         cfg_tar: float = Form(1.0),
         image: UploadFile = File(...)):
    if not (0 < skip < num_diffusion_steps):
        raise HTTPException(400, "skip must be between 1 and num_diffusion_steps-1")

    info = _get_model_or_400(model)
    attrs = config.load_attribute_vectors(model)
    if attribute not in attrs:
        raise HTTPException(400, f"Attribute '{attribute}' not found for model '{model}'. "
                                 f"Available: {sorted(attrs)}")
    img = _read_upload(image)

    def work(job):
        from app import pipeline
        images = pipeline.run_attribute_edit(
            info, img, attribute, lamda=lamda,
            num_diffusion_steps=num_diffusion_steps, skip=skip,
            cfg_src=cfg_src, cfg_tar=cfg_tar, progress=job.set_progress)
        _save_images(job, images, [f"{attribute}_lambda_{lamda:g}"], inputs=[("1", img)])

    job = job_manager.submit("edit", work)
    return {"job_id": job.id}


@app.get("/api/job/{job_id}")
def job_status(job_id: str):
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job.to_dict()


@app.get("/api/download/{job_id}/{filename}")
def download(job_id: str, filename: str):
    path = (config.OUTPUT_DIR / job_id / filename).resolve()
    if not path.is_file() or config.OUTPUT_DIR.resolve() not in path.parents:
        raise HTTPException(404, "File not found")
    return FileResponse(path, media_type="image/png", filename=filename)


app.mount("/outputs", StaticFiles(directory=str(config.OUTPUT_DIR)), name="outputs")
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
