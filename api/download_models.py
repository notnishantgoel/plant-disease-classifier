"""
Download model weights from Hugging Face Hub at container startup.
All weights live in a single public model repo.

Repo layout expected on HF Hub:
  best_vit_moe_v3_model.pth
  best_efficientnet_v2_comparable.pth
  best_convnext_v2_model_for20epoch.pth
  yolo_best.pt
"""

import os
from huggingface_hub import hf_hub_download

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
YOLO_DIR   = os.path.join(BASE_DIR, "results/runs/classify/yolo_metrics/yolo_v2_scratch/weights")

# ── UPDATE THIS to your HF model repo  e.g. "notnishantgoel/plant-disease-models"
HF_REPO_ID = "notnishantgoel/plant-disease-models"

PYTORCH_MODELS = [
    "best_vit_moe_v3_model.pth",
    "best_efficientnet_v2_comparable.pth",
    "best_convnext_v2_model_for20epoch.pth",
]


def ensure_models() -> None:
    """Download any missing model weights from HF Hub."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(YOLO_DIR,   exist_ok=True)

    for filename in PYTORCH_MODELS:
        dest = os.path.join(MODELS_DIR, filename)
        if os.path.exists(dest):
            print(f"[models] already present: {filename}")
            continue
        print(f"[models] downloading {filename} …")
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            local_dir=MODELS_DIR,
            local_dir_use_symlinks=False,
        )
        print(f"[models] done: {filename}")

    # YOLO weights are expected at the original inference path
    yolo_dest = os.path.join(YOLO_DIR, "best.pt")
    if os.path.exists(yolo_dest):
        print("[models] already present: yolo_best.pt")
        return
    print("[models] downloading yolo_best.pt …")
    hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="yolo_best.pt",
        local_dir=YOLO_DIR,
        local_dir_use_symlinks=False,
    )
    # rename to the filename inference.py expects
    os.rename(
        os.path.join(YOLO_DIR, "yolo_best.pt"),
        yolo_dest,
    )
    print("[models] done: yolo_best.pt → best.pt")


if __name__ == "__main__":
    ensure_models()
