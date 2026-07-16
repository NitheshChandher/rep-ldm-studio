"""FFHQ-style face alignment for uploaded images.

Adapted from the align_face scripts (NVlabs/ffhq-dataset method, via
https://lzhbrian.me), updated for Pillow >= 10 and to work on in-memory
PIL images. The dlib 68-landmark shape predictor is downloaded
automatically on first use.
"""
import bz2
import threading
import urllib.request
from pathlib import Path

import numpy as np
import PIL.Image
import scipy.ndimage

from app import config

PREDICTOR_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
CACHE_DIR = Path(config.REPO_ROOT) / "cache"

_lock = threading.Lock()
_predictor = None
_detector = None


def _get_predictor():
    """Lazily download and load the dlib shape predictor."""
    global _predictor, _detector
    import dlib

    with _lock:
        if _predictor is None:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            dat_path = CACHE_DIR / "shape_predictor_68_face_landmarks.dat"
            if not dat_path.is_file():
                bz2_path = dat_path.with_suffix(".dat.bz2")
                print(f"Downloading dlib shape predictor to {dat_path} ...")
                urllib.request.urlretrieve(PREDICTOR_URL, bz2_path)
                dat_path.write_bytes(bz2.decompress(bz2_path.read_bytes()))
                bz2_path.unlink()
            _predictor = dlib.shape_predictor(str(dat_path))
            _detector = dlib.get_frontal_face_detector()
    return _predictor, _detector


def _get_landmark(img_rgb):
    """Return the 68-point landmark array for the largest detected face, or None."""
    predictor, detector = _get_predictor()
    arr = np.asarray(img_rgb)
    dets = detector(arr, 1)
    if not dets:
        return None
    # Use the largest face
    det = max(dets, key=lambda d: d.width() * d.height())
    shape = predictor(arr, det)
    return np.array([[p.x, p.y] for p in shape.parts()])


def align_face(img, output_size=1024, transform_size=4096, enable_padding=True):
    """Align a face in a PIL image using the FFHQ method.

    Returns the aligned PIL image, or None if no face was detected.
    """
    img = img.convert("RGB")
    lm = _get_landmark(img)
    if lm is None:
        return None

    lm_eye_left = lm[36:42]
    lm_eye_right = lm[42:48]
    lm_mouth_outer = lm[48:60]

    # Auxiliary vectors
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_avg = (lm_mouth_outer[0] + lm_mouth_outer[6]) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Oriented crop rectangle
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # Shrink
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)),
                 int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.LANCZOS)
        quad /= shrink
        qsize /= shrink

    # Crop
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
            int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0),
            min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
           int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0),
           max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        arr = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), "reflect")
        h, w, _ = arr.shape
        yy, xx, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(xx) / pad[0], np.float32(w - 1 - xx) / pad[2]),
                          1.0 - np.minimum(np.float32(yy) / pad[1], np.float32(h - 1 - yy) / pad[3]))
        blur = qsize * 0.02
        arr += (scipy.ndimage.gaussian_filter(arr, [blur, blur, 0]) - arr) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        arr += (np.median(arr, axis=(0, 1)) - arr) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(arr), 0, 255)), "RGB")
        quad += pad[:2]

    # Transform
    img = img.transform((transform_size, transform_size), PIL.Image.Transform.QUAD,
                        (quad + 0.5).flatten(), PIL.Image.Resampling.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.LANCZOS)

    return img


def try_align(img, output_size=1024):
    """Align a face if one is detected; otherwise return the original image."""
    try:
        aligned = align_face(img, output_size=output_size)
    except Exception as e:
        print(f"Face alignment failed ({e}); using original image.")
        return img, False
    if aligned is None:
        print("No face detected; using original image.")
        return img, False
    return aligned, True
