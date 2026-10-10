"""FastAPI backend for the AGE AI frontend.

Wraps the trained ensemble (same logic as predict.py) behind the two HTTP
endpoints the frontend calls:

    POST /upload  -> stores the image, returns {"imageUrl": "/uploads/..."}
    POST /predict -> runs face detection + alignment + ensemble inference,
                      returns {"age", "min_age", "max_age", "confidence"}

Run with:
    uvicorn backend.main:app --reload --port 5000
(from the age_prediction_pipeline directory, with the venv active)
"""
import io
import os
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_ROOT / "src"))

from align_faces import align_image  # noqa: E402
from data import build_transforms  # noqa: E402
from face_detector import detect_landmarks  # noqa: E402
from model import DualHeadAgeModel  

CHECKPOINTS = [
    PIPELINE_ROOT / "runs" / "resnet50" / "best_model.pt",
    PIPELINE_ROOT / "runs" / "efficientnet_b0" / "best_model.pt",
    PIPELINE_ROOT / "runs" / "mobilenet_v2" / "best_model.pt",
]
# Weight on each model's regression head within its own blend; the classification
# head gets (1 - this). Matches predict.py's default (regression scored lower MAE).
REGRESSION_WEIGHT = 0.7

UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB, matches the frontend's stated limit
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png"}

ALLOWED_ORIGINS = os.environ.get(
    "FRONTEND_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
).split(",")

UPLOAD_DIR.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_ensemble: list[dict] = []


def load_ensemble() -> list[dict]:
    loaded = []
    for path in CHECKPOINTS:
        if not path.exists():
            print(f"Warning: checkpoint missing, skipping: {path}")
            continue
        ckpt = torch.load(path, map_location=device)
        model = DualHeadAgeModel(
            num_classes=ckpt["num_classes"], backbone=ckpt["backbone"], pretrained=False
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        loaded.append(
            {
                "backbone": ckpt["backbone"],
                "model": model,
                "min_age": ckpt["min_age"],
                "img_size": ckpt["img_size"],
            }
        )
    return loaded


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ensemble.extend(load_ensemble())
    if not _ensemble:
        print("Warning: no model checkpoints loaded — /predict will fail until runs/*/best_model.pt exist.")
    yield
    _ensemble.clear()


app = FastAPI(title="AGE AI backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


def _load_validated_image(file: UploadFile, raw: bytes) -> Image.Image:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(400, "Only JPG and PNG images are supported.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(400, "Image exceeds the 5 MB size limit.")
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not read the uploaded file as an image.")


@app.post("/upload")
async def upload(image: UploadFile = File(...)):
    raw = await image.read()
    _load_validated_image(image, raw)  # validate before writing anything to disk

    ext = Path(image.filename or "").suffix.lower()
    if ext not in (".jpg", ".jpeg", ".png"):
        ext = ".jpg"
    name = f"{uuid.uuid4().hex}{ext}"
    (UPLOAD_DIR / name).write_bytes(raw)

    return {"imageUrl": f"/uploads/{name}"}


@app.post("/predict")
async def predict(image: UploadFile = File(...), image_url: str = Form(default="")):
    if not _ensemble:
        raise HTTPException(503, "No trained model checkpoints are loaded on the server.")

    raw = await image.read()
    img = _load_validated_image(image, raw)

    landmarks = detect_landmarks(img)
    if landmarks is None:
        raise HTTPException(422, "No face detected in the image. Try a clearer, front-facing photo.")

    aligned_cache: dict[int, torch.Tensor] = {}
    per_model = []

    for entry in _ensemble:
        size = entry["img_size"]
        if size not in aligned_cache:
            aligned = align_image(img, landmarks, size=size)
            transform = build_transforms(train=False, img_size=size)
            aligned_cache[size] = transform(aligned).unsqueeze(0).to(device)
        x = aligned_cache[size]

        with torch.no_grad():
            class_logits, reg_out = entry["model"](x)
            probs = F.softmax(class_logits, dim=1)[0]

        top1_idx = int(torch.argmax(probs))
        class_pred_age = top1_idx + entry["min_age"]
        class_confidence = float(probs[top1_idx])
        reg_pred_age = float(reg_out.item())
        final_pred_age = REGRESSION_WEIGHT * reg_pred_age + (1 - REGRESSION_WEIGHT) * class_pred_age

        per_model.append({"final_pred_age": final_pred_age, "class_confidence": class_confidence})

    ensemble_age = sum(m["final_pred_age"] for m in per_model) / len(per_model)
    spread = max(m["final_pred_age"] for m in per_model) - min(m["final_pred_age"] for m in per_model)
    avg_confidence = sum(m["class_confidence"] for m in per_model) / len(per_model)
    half_range = max(spread / 2, 3)

    return {
        "age": round(ensemble_age),
        "min_age": max(0, round(ensemble_age - half_range)),
        "max_age": round(ensemble_age + half_range),
        "confidence": round(avg_confidence * 100, 1),
    }


@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": len(_ensemble)}
