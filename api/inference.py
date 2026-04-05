"""
Model loading and inference for all 4 architectures.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from vit_moe_arch import ViT_MoE_v3

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
YOLO_WEIGHTS = os.path.join(
    BASE_DIR, "results/runs/classify/yolo_metrics/yolo_v2_scratch/weights/best.pt"
)

CLASS_NAMES = ["Blight_fungus", "Mosiac_Virus", "Spider_Mite", "Thrip_pest"]
NUM_CLASSES = len(CLASS_NAMES)

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_state(model, path):
    device = get_device()
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_vit_moe_v3():
    model = ViT_MoE_v3(num_classes=NUM_CLASSES)
    return _load_state(model, os.path.join(MODELS_DIR, "best_vit_moe_v3_model.pth"))


def load_vit_moe_v3_ema():
    model = ViT_MoE_v3(num_classes=NUM_CLASSES)
    return _load_state(model, os.path.join(MODELS_DIR, "best_vit_moe_v3_ema.pth"))


def load_efficientnet():
    model = models.efficientnet_v2_s(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    return _load_state(model, os.path.join(MODELS_DIR, "best_efficientnet_v2_comparable.pth"))


def load_convnext():
    model = models.convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, NUM_CLASSES)
    return _load_state(model, os.path.join(MODELS_DIR, "best_convnext_v2_model_for20epoch.pth"))


def load_yolo():
    from ultralytics import YOLO
    return YOLO(YOLO_WEIGHTS)


def load_all_models():
    device = get_device()
    print(f"[inference] Loading models on: {device}")
    loaded = {}

    for key, loader in [
        ("vit_moe_v3", load_vit_moe_v3),
        ("efficientnet", load_efficientnet),
        ("convnext", load_convnext),
        ("yolo", load_yolo),
    ]:
        try:
            loaded[key] = loader()
            print(f"  OK  {key}")
        except Exception as e:
            print(f"  FAIL {key}: {e}")

    return loaded


def predict_image(model, image: Image.Image, model_id: str) -> dict:
    if model_id == "yolo":
        results = model.predict(image, imgsz=224, verbose=False)
        probs = results[0].probs
        names = results[0].names  # {0: 'Blight_fungus', ...} — sorted alphabetically by YOLO

        conf_list = probs.data.cpu().tolist()
        class_probs = {}
        for i, prob in enumerate(conf_list):
            name = names.get(i, CLASS_NAMES[i]) if names else CLASS_NAMES[i]
            class_probs[name] = round(prob * 100, 2)

        top_class = max(class_probs, key=class_probs.get)
        return {
            "predicted_class": top_class,
            "confidence": class_probs[top_class],
            "all_probabilities": class_probs,
        }

    device = get_device()
    img_tensor = INFERENCE_TRANSFORM(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)[0]

    class_probs = {CLASS_NAMES[i]: round(probs[i].item() * 100, 2) for i in range(NUM_CLASSES)}
    predicted_idx = int(probs.argmax())
    predicted_class = CLASS_NAMES[predicted_idx]

    return {
        "predicted_class": predicted_class,
        "confidence": class_probs[predicted_class],
        "all_probabilities": class_probs,
    }
