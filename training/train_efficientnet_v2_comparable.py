"""
EfficientNetV2-S — Comparable Training Setup
=============================================
This script uses the EXACT same training recipe as ViT+MoE v3.1
so the comparison is purely about architecture:

  Identical to v3.1:
    - Training from scratch (no pretrained weights)
    - Same augmentation (RandomResizedCrop, ColorJitter, rotation)
    - Same CutMix/MixUp (25% probability)
    - Same cosine LR scheduler with 3-epoch warmup
    - Same label smoothing (0.05)
    - Same optimizer (AdamW, lr=3e-4, weight_decay=0.01)
    - Same gradient clipping (1.0)
    - Same EMA (decay=0.998)
    - Same 30 epochs
    - Same batch size (32)

  Only difference: model architecture (EfficientNetV2-S vs ViT+MoE)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import efficientnet_v2_s
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import time
import math
import copy
import numpy as np


# =============================================================
# 1. Data Loading — SAME augmentation as ViT+MoE v3.1
# =============================================================
def get_dataloaders(data_dir, batch_size=32, num_workers=2):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), train_transform),
        'val': datasets.ImageFolder(os.path.join(data_dir, 'val'), val_transform),
    }

    dataloaders = {
        'train': DataLoader(
            image_datasets['train'], batch_size=batch_size,
            shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True
        ),
        'val': DataLoader(
            image_datasets['val'], batch_size=batch_size,
            shuffle=False, num_workers=num_workers, pin_memory=True
        ),
    }

    num_classes = len(image_datasets['train'].classes)
    class_names = image_datasets['train'].classes
    print(f"Classes ({num_classes}): {class_names}")
    print(f"Train: {len(image_datasets['train'])} | Val: {len(image_datasets['val'])}")
    return dataloaders, num_classes


# =============================================================
# 2. CutMix / MixUp — SAME as v3.1
# =============================================================
def cutmix_mixup(images, labels, num_classes, cutmix_prob=0.5, mixup_alpha=0.2):
    batch_size = images.size(0)
    one_hot = F.one_hot(labels, num_classes).float()

    if np.random.rand() < cutmix_prob:
        lam = np.random.beta(1.0, 1.0)
        rand_index = torch.randperm(batch_size, device=images.device)
        target_a = one_hot
        target_b = one_hot[rand_index]

        _, _, H, W = images.shape
        cut_ratio = math.sqrt(1.0 - lam)
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)
        cy = np.random.randint(H)
        cx = np.random.randint(W)
        y1 = max(0, cy - cut_h // 2)
        y2 = min(H, cy + cut_h // 2)
        x1 = max(0, cx - cut_w // 2)
        x2 = min(W, cx + cut_w // 2)

        images[:, :, y1:y2, x1:x2] = images[rand_index, :, y1:y2, x1:x2]
        lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)
        mixed_labels = lam * target_a + (1 - lam) * target_b
    else:
        lam = np.random.beta(mixup_alpha, mixup_alpha) if mixup_alpha > 0 else 1.0
        rand_index = torch.randperm(batch_size, device=images.device)
        images = lam * images + (1 - lam) * images[rand_index]
        mixed_labels = lam * one_hot + (1 - lam) * one_hot[rand_index]

    return images, mixed_labels


# =============================================================
# 3. EMA — SAME as v3.1
# =============================================================
class ModelEMA:
    def __init__(self, model, decay=0.998):
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        self.decay = decay
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, model_p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.ema_model.state_dict()


# =============================================================
# 4. LR Scheduler — SAME as v3.1
# =============================================================
def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return 0.1 + 0.9 * epoch / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return max(min_lr / optimizer.defaults['lr'],
                   0.5 * (1 + math.cos(math.pi * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =============================================================
# 5. Training Loop — SAME structure as v3.1
# =============================================================
def train_model(data_dir, epochs=30, batch_size=32, lr=3e-4, weight_decay=0.01,
                warmup_epochs=3, label_smoothing=0.05, use_cutmix_mixup=True,
                use_ema=True, grad_clip=1.0):

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Training on: {device}")

    # Data
    dataloaders, num_classes = get_dataloaders(data_dir, batch_size=batch_size)

    # Model — from scratch, no pretrained weights
    print("Initializing EfficientNetV2-S from scratch (weights=None)...")
    model = efficientnet_v2_s(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Loss — same label smoothing as v3.1
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Optimizer — same settings as v3.1
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))

    # LR Scheduler — same cosine with warmup as v3.1
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_epochs, epochs)

    # EMA — same decay as v3.1
    ema = ModelEMA(model) if use_ema else None

    history = []
    best_acc = 0.0
    best_ema_acc = 0.0

    print(f"\n{'='*60}")
    print(f"Training EfficientNetV2-S for {epochs} epochs")
    print(f"  LR: {lr} | Weight Decay: {weight_decay}")
    print(f"  Warmup: {warmup_epochs} epochs | Label Smoothing: {label_smoothing}")
    print(f"  CutMix/MixUp: {use_cutmix_mixup} | EMA: {use_ema}")
    print(f"  Gradient Clipping: {grad_clip}")
    print(f"  ** Same recipe as ViT+MoE v3.1 for fair comparison **")
    print(f"{'='*60}\n")

    for epoch in range(epochs):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{epochs} (lr: {current_lr:.6f})")
        epoch_results = {'epoch': epoch + 1, 'lr': current_lr}

        # ---- TRAINING PHASE ----
        model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []

        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)

            if use_cutmix_mixup and np.random.rand() < 0.25:
                inputs, mixed_labels = cutmix_mixup(inputs, labels, num_classes)
                optimizer.zero_grad()
                outputs = model(inputs)
                log_probs = F.log_softmax(outputs, dim=-1)
                loss = -(mixed_labels * log_probs).sum(dim=-1).mean()
            else:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            if ema is not None:
                ema.update(model)

            running_loss += loss.item() * inputs.size(0)
            preds = torch.max(outputs, 1)[1]
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_loss = running_loss / len(dataloaders['train'].dataset)
        train_acc = accuracy_score(all_labels, all_preds)
        train_prec, train_rec, train_f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )
        epoch_results.update({
            'train_loss': train_loss, 'train_acc': train_acc,
            'train_prec': train_prec, 'train_rec': train_rec, 'train_f1': train_f1,
        })
        print(f"  Train | Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | "
              f"Prec: {train_prec:.4f} | Rec: {train_rec:.4f} | F1: {train_f1:.4f}")

        # ---- VALIDATION PHASE ----
        for eval_name, eval_model in [('val', model)] + ([('val_ema', ema.ema_model)] if ema else []):
            eval_model.eval()
            running_loss = 0.0
            all_preds, all_labels = [], []

            with torch.no_grad():
                for inputs, labels in dataloaders['val']:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = eval_model(inputs)
                    loss = criterion(outputs, labels)
                    running_loss += loss.item() * inputs.size(0)
                    all_preds.extend(torch.max(outputs, 1)[1].cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            val_loss = running_loss / len(dataloaders['val'].dataset)
            val_acc = accuracy_score(all_labels, all_preds)
            val_prec, val_rec, val_f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='macro', zero_division=0
            )

            prefix = eval_name
            epoch_results.update({
                f'{prefix}_loss': val_loss, f'{prefix}_acc': val_acc,
                f'{prefix}_prec': val_prec, f'{prefix}_rec': val_rec, f'{prefix}_f1': val_f1,
            })
            label = "Val" if eval_name == 'val' else "EMA"
            print(f"  {label:>5} | Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | "
                  f"Prec: {val_prec:.4f} | Rec: {val_rec:.4f} | F1: {val_f1:.4f}")

            if eval_name == 'val' and val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), '../models/best_efficientnet_v2_comparable.pth')
                print(f"    -> Best model saved! (val_acc: {best_acc:.4f})")

            if eval_name == 'val_ema' and val_acc > best_ema_acc:
                best_ema_acc = val_acc
                torch.save(ema.state_dict(), '../models/best_efficientnet_v2_comparable_ema.pth')
                print(f"    -> Best EMA model saved! (ema_acc: {best_ema_acc:.4f})")

        scheduler.step()

        epoch_duration = time.time() - epoch_start
        mins, secs = int(epoch_duration // 60), int(epoch_duration % 60)
        print(f"  [*] Epoch {epoch+1} completed in {mins}m {secs}s")
        epoch_results['epoch_time_seconds'] = epoch_duration
        history.append(epoch_results)

    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"  Best Val Accuracy:     {best_acc:.4f}")
    if ema:
        print(f"  Best EMA Val Accuracy: {best_ema_acc:.4f}")
    print(f"{'='*60}")

    df = pd.DataFrame(history)
    csv_path = '../results/efficientnet_v2_comparable_metrics.csv'
    df.to_csv(csv_path, index=False)
    print(f"Metrics saved to '{csv_path}'")


# =============================================================
# 6. Entry Point — SAME hyperparams as v3.1
# =============================================================
if __name__ == "__main__":
    PROCESSED_DATA_DIR = "../data/processed/v2_balanced_dataset"

    train_model(
        PROCESSED_DATA_DIR,
        epochs=30,
        batch_size=32,
        lr=3e-4,
        weight_decay=0.01,
        warmup_epochs=3,
        label_smoothing=0.05,
        use_cutmix_mixup=True,
        use_ema=True,
        grad_clip=1.0,
    )
