"""
FastAPI backend — PlantGuard AI
  Development : uvicorn main:app --reload --port 8000  (from api/)
  Production  : uvicorn api.main:app --host 0.0.0.0 --port 7860  (from project root)
"""

import csv
import io
import sys
import os

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from inference import load_all_models, predict_image

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR  = os.path.join(BASE_DIR, "results")
REACT_DIST   = os.path.join(BASE_DIR, "webapp", "dist")

MODELS_META = [
    {
        "id": "efficientnet",
        "name": "EfficientNet v2-S",
        "shortName": "EfficientNet",
        "accuracy": "94.86%",
        "size": "78 MB",
        "description": "Google's compound-scaled mobile CNN optimised for accuracy-per-FLOP. Highest accuracy among all models.",
    },
    {
        "id": "convnext",
        "name": "ConvNeXt-Tiny",
        "shortName": "ConvNeXt",
        "accuracy": "89.50%",
        "size": "106 MB",
        "description": "Pure-CNN architecture redesigned with transformer best-practices (depthwise convs, LayerNorm, GELU).",
    },
    {
        "id": "vit_moe_v3",
        "name": "Vision Transformer + MoE",
        "shortName": "ViT + MoE",
        "accuracy": "91.125%",
        "size": "37 MB",
        "description": "Custom ViT with Mixture-of-Experts routing, learnable positional embeddings, 4 stacked transformer blocks.",
    },
    {
        "id": "yolo",
        "name": "YOLOv11 Nano",
        "shortName": "YOLO v11",
        "accuracy": "86.50%",
        "size": "~3 MB",
        "description": "Ultralytics YOLOv11 nano classification head, fastest inference of the four.",
    },
]

_models: dict = {}


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _read_standard_csv(path: str) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                rows.append({
                    "epoch":      int(float(row["epoch"])),
                    "train_acc":  round(float(row["train_acc"]) * 100, 2),
                    "train_loss": round(float(row["train_loss"]), 4),
                    "val_acc":    round(float(row["val_acc"]) * 100, 2),
                    "val_loss":   round(float(row["val_loss"]), 4),
                    "val_prec":   round(float(row["val_prec"]) * 100, 2),
                    "val_rec":    round(float(row["val_rec"]) * 100, 2),
                    "val_f1":     round(float(row["val_f1"]) * 100, 2),
                })
            except (ValueError, KeyError):
                continue
    return rows


def _read_yolo_csv(path: str) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                rows.append({
                    "epoch":      int(float(row["epoch"].strip())),
                    "train_loss": round(float(row["train/loss"].strip()), 4),
                    "val_acc":    round(float(row["metrics/accuracy_top1"].strip()) * 100, 2),
                    "val_loss":   round(float(row["val/loss"].strip()), 4),
                })
            except (ValueError, KeyError):
                continue
    return rows


def _best_row(rows: list[dict]) -> dict:
    return max(rows, key=lambda r: r["val_acc"]) if rows else {}


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _models

    # Download model weights from HF Hub if running in production
    try:
        from download_models import ensure_models
        ensure_models()
    except Exception as e:
        print(f"[startup] model download skipped: {e}")

    _models = load_all_models()
    yield
    _models.clear()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="PlantGuard AI", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── API routes (must be defined BEFORE the static-file catch-all) ─────────────

@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": list(_models.keys())}


@app.get("/models")
async def get_models():
    available = [m for m in MODELS_META if m["id"] in _models]
    return {"models": available}


@app.get("/metrics")
async def get_metrics():
    CSV_MAP = {
        "vit_moe_v3":   (os.path.join(RESULTS_DIR, "vit_moe_v3.1_training_metrics.csv"),       _read_standard_csv),
        "efficientnet": (os.path.join(RESULTS_DIR, "efficientnet_v2_comparable_metrics.csv"),   _read_standard_csv),
        "convnext":     (os.path.join(RESULTS_DIR, "convnext_v2_training_metricsfor20epoch.csv"), _read_standard_csv),
        "yolo":         (os.path.join(RESULTS_DIR, "runs/classify/yolo_metrics/yolo_v2_scratch/results.csv"), _read_yolo_csv),
    }

    curves: dict = {}
    for model_id, (path, reader) in CSV_MAP.items():
        try:
            curves[model_id] = reader(path)
        except Exception as e:
            print(f"[metrics] {model_id}: {e}")
            curves[model_id] = []

    short_names = {m["id"]: m["shortName"] for m in MODELS_META}
    summary = []
    for model_id, rows in curves.items():
        if not rows:
            continue
        best = _best_row(rows)
        summary.append({
            "model_id":   model_id,
            "name":       short_names.get(model_id, model_id),
            "best_epoch": best.get("epoch"),
            "best_acc":   best.get("val_acc"),
            "best_prec":  best.get("val_prec"),
            "best_rec":   best.get("val_rec"),
            "best_f1":    best.get("val_f1"),
            "best_loss":  best.get("val_loss"),
        })

    return {"summary": summary, "curves": curves}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_id: str = Form(default="efficientnet"),
):
    if model_id not in _models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_id}' not loaded. Available: {list(_models.keys())}",
        )

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Cannot parse uploaded file as an image.")

    result = predict_image(_models[model_id], image, model_id)
    result["model_id"] = model_id
    return JSONResponse(result)


# ── Serve React build (catch-all — must be LAST) ──────────────────────────────
if os.path.exists(REACT_DIST):
    app.mount("/", StaticFiles(directory=REACT_DIST, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
