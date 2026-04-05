from ultralytics import YOLO
import torch
import os

def train_yolo_from_scratch(data_dir, epochs=10):
    # Determine device for Mac
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Training YOLOv11 on device: {device}")

    # 1. Initialize YOLOv11-Nano Architecture
    # By passing the .yaml file instead of a .pt file, we force it to build
    # the architecture completely from scratch without any pre-trained weights.
    print("Initializing YOLOv11 (Nano) strictly from scratch...")
    model = YOLO('yolo11n-cls.yaml') 

    # 2. Train the Model
    # The ultralytics library handles the entire loop, logging, and CSV generation natively.
    print(f"\nStarting YOLO training for {epochs} epochs...")
    results = model.train(
        data=data_dir,
        epochs=epochs,
        imgsz=224,               # Matching our ViT and ConvNeXt size
        device=device,           # Using Apple Silicon
        pretrained=False,        # Guaranteeing a blind, fair fight
        project='../results/runs/classify/yolo_metrics',  # Folder where results will be saved
        name='yolo_v2_scratch'   # Subfolder name
    )
    
    print("\n" + "="*50)
    print("YOLO TRAINING COMPLETE!")
    print(f"Your model weights and the training CSV spreadsheet have been automatically")
    print(f"saved by Ultralytics into the following directory:")
    print(f" -> ../results/runs/classify/yolo_metrics/yolo_v2_scratch/")
    print("="*50)

if __name__ == "__main__":
    # YOLO's dataloader natively understands the standard 'train', 'val', 'test' folder layout
    # that we built with your v2 dataset script.
    PROCESSED_DATA_DIR = os.path.abspath("../data/processed/v2_balanced_dataset")
    
    train_yolo_from_scratch(PROCESSED_DATA_DIR, epochs=10)
