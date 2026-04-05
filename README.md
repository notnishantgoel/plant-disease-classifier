---
title: PlantGuard AI
emoji: 🌿
colorFrom: green
colorTo: emerald
sdk: docker
app_port: 7860
pinned: false
---

# 🌿 PlantGuard AI — Plant Disease Classifier

AI-powered plant disease classification across **4 deep learning models**, trained from scratch on 4,000 balanced images.

## Disease Classes
| Class | Description |
|-------|-------------|
| 🍂 Blight (Fungal) | Rapid tissue death, brown/black lesions |
| 🟡 Mosaic Virus | Mottled yellow-green leaf patterns |
| 🕷️ Spider Mite | Yellowing, stippling, fine webbing |
| 🪲 Thrip Pest | Silvery streaks, leaf deformation |

## Models
| Model | Val Accuracy | Size |
|-------|-------------|------|
| EfficientNet v2-S | **94.86%** | 78 MB |
| ViT + MoE v3.1 | 91.125% | 37 MB |
| ConvNeXt-Tiny | 89.50% | 106 MB |
| YOLOv11 Nano | 86.50% | ~3 MB |

All models trained **from scratch** — no pre-trained weights.

## Features
- 📁 Drag-drop or browse image upload
- 📷 Camera capture (mobile)
- 🔀 Switch between 4 models instantly
- 📊 Comparative analysis dashboard (accuracy, precision, recall, F1, loss curves)
- 💊 Treatment recommendations per disease

## Stack
- **Backend**: FastAPI + PyTorch + Ultralytics
- **Frontend**: React + Vite + Recharts
- **Android**: React Native (Expo)

## Local Development

```bash
# Backend
cd api && pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend (separate terminal)
cd webapp && npm install && npm run dev
```

> **Note**: Model weights (`models/`) and dataset (`data/`) are not included in this repo due to size.  
> They are downloaded automatically at startup from Hugging Face Hub.
