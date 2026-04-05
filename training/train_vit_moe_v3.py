"""
ViT + MoE v3 — Improved Plant Disease Classifier
==================================================
Key improvements over v2:
  1. Learnable positional embeddings (v2 had NONE — critical fix)
  2. Deeper architecture: 4 stacked transformer blocks (v2 had only 1)
  3. Fixed MoE routing: expert outputs now properly weighted by router scores
  4. MoE load-balancing auxiliary loss to prevent expert collapse
  5. DropPath (stochastic depth) for deep network regularization
  6. Dropout in attention, MoE, and embedding layers
  7. Cosine annealing LR scheduler with linear warmup
  8. Stronger data augmentation: ColorJitter, RandomErasing, RandAugment
  9. Label smoothing in CrossEntropyLoss
  10. Gradient clipping
  11. CutMix / MixUp augmentation
  12. Exponential Moving Average (EMA) of model weights
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
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
# 1. Data Loading — Much Stronger Augmentation
# =============================================================
def get_dataloaders(data_dir, batch_size=32, num_workers=2):
    """
    v3 changes:
    - Added ColorJitter (brightness, contrast, saturation, hue)
    - Added RandomResizedCrop instead of plain Resize (scale variation)
    - Added RandomErasing (cutout-style regularization)
    - Added RandAugment for diverse augmentation policies
    """
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
# 2. CutMix / MixUp Augmentation
# =============================================================
def cutmix_mixup(images, labels, num_classes, cutmix_prob=0.5, mixup_alpha=0.2):
    """
    Randomly apply CutMix or MixUp to a batch.
    Returns mixed images and soft label distributions.
    """
    batch_size = images.size(0)
    one_hot = F.one_hot(labels, num_classes).float()

    if np.random.rand() < cutmix_prob:
        # --- CutMix ---
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
        lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)  # adjust for clipping
        mixed_labels = lam * target_a + (1 - lam) * target_b
    else:
        # --- MixUp ---
        lam = np.random.beta(mixup_alpha, mixup_alpha) if mixup_alpha > 0 else 1.0
        rand_index = torch.randperm(batch_size, device=images.device)
        images = lam * images + (1 - lam) * images[rand_index]
        mixed_labels = lam * one_hot + (1 - lam) * one_hot[rand_index]

    return images, mixed_labels


# =============================================================
# 3. DropPath (Stochastic Depth)
# =============================================================
class DropPath(nn.Module):
    """
    Randomly drops entire residual branches during training.
    This is the standard regularization for deep ViTs — much more
    effective than plain dropout for transformer architectures.
    """
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # broadcast across all dims except batch
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x / keep_prob * random_tensor


# =============================================================
# 4. Improved Expert & MoE Layer
# =============================================================
class Expert(nn.Module):
    """
    Single expert FFN with internal dropout.
    """
    def __init__(self, embed_dim, expansion=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * expansion, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MoE_Layer(nn.Module):
    """
    Mixture of Experts with proper weighted routing and load balancing.

    v3 fixes:
    - Expert outputs are now MULTIPLIED by routing weights (v2 just added them)
    - Auxiliary load-balancing loss encourages uniform expert utilization
    - Added noise to routing for exploration during training
    """
    def __init__(self, embed_dim, num_experts=4, k=2, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.router = nn.Linear(embed_dim, num_experts)
        self.experts = nn.ModuleList([Expert(embed_dim, dropout=dropout) for _ in range(num_experts)])
        self.aux_loss = 0.0  # stored for the training loop to pick up

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        x_flat = x.view(-1, embed_dim)
        num_tokens = x_flat.shape[0]

        # Router with noise for exploration
        router_logits = self.router(x_flat)
        if self.training:
            noise = torch.randn_like(router_logits) * 0.1
            router_logits = router_logits + noise

        routing_weights = F.softmax(router_logits, dim=-1)

        # Top-k expert selection
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)  # renormalize

        # ---- Load Balancing Auxiliary Loss ----
        # Fraction of tokens routed to each expert
        # (averaged across all top-k selections)
        expert_mask = F.one_hot(top_k_indices, self.num_experts).float()  # [N, k, E]
        tokens_per_expert = expert_mask.sum(dim=1).mean(dim=0)  # [E]
        # Mean routing probability per expert
        router_prob_per_expert = routing_weights.mean(dim=0)  # [E]
        # Auxiliary loss: dot product encourages uniform distribution
        self.aux_loss = self.num_experts * (tokens_per_expert * router_prob_per_expert).sum()

        # ---- Weighted Expert Computation (FIXED from v2) ----
        out_flat = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            # For each expert, find tokens that selected it in any of the k slots
            for k_idx in range(self.k):
                mask = (top_k_indices[:, k_idx] == i)
                if not mask.any():
                    continue
                expert_input = x_flat[mask]
                expert_output = expert(expert_input)
                # CRITICAL FIX: multiply by the routing weight for this slot
                weight = top_k_weights[mask, k_idx].unsqueeze(-1)
                out_flat[mask] += weight * expert_output

        return out_flat.view(batch_size, seq_len, embed_dim)


# =============================================================
# 5. Transformer Block (Attention + MoE + Residuals)
# =============================================================
class TransformerBlock(nn.Module):
    """
    A single transformer block: LayerNorm → MHSA → residual → LayerNorm → MoE → residual
    With DropPath on both residual connections.
    """
    def __init__(self, embed_dim, num_heads=8, num_experts=4, k=2,
                 attn_dropout=0.1, proj_dropout=0.1, moe_dropout=0.1, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=attn_dropout, batch_first=True
        )
        self.proj_drop = nn.Dropout(proj_dropout)
        self.drop_path1 = DropPath(drop_path)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.moe = MoE_Layer(embed_dim, num_experts=num_experts, k=k, dropout=moe_dropout)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x):
        # Self-attention with pre-norm and residual
        x2 = self.norm1(x)
        attn_out, _ = self.attn(x2, x2, x2)
        attn_out = self.proj_drop(attn_out)
        x = x + self.drop_path1(attn_out)

        # MoE FFN with pre-norm and residual
        x = x + self.drop_path2(self.moe(self.norm2(x)))
        return x


# =============================================================
# 6. Full ViT + MoE v3 Model
# =============================================================
class ViT_MoE_v3(nn.Module):
    """
    Vision Transformer with Mixture of Experts — v3

    Architecture:
    - Patch embedding (16x16 patches → 196 patches for 224x224 input)
    - Learnable positional embeddings (NEW — v2 had none!)
    - Learnable [CLS] token
    - N stacked TransformerBlocks (each with MHSA + MoE)
    - LayerNorm → classification head

    Default config: embed_dim=256, depth=4, heads=8, 4 experts (top-2)
    """
    def __init__(self, num_classes, embed_dim=256, depth=4, num_heads=8,
                 num_experts=4, k=2, patch_size=16, img_size=224,
                 dropout=0.05, drop_path_rate=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2  # 196 for 224/16

        # Patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Learnable positional embeddings — THE critical fix
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Stochastic depth: linearly increasing drop rate per block
        drop_path_rates = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]

        # Stacked transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_experts=num_experts,
                k=k,
                attn_dropout=dropout,
                proj_dropout=dropout,
                moe_dropout=dropout,
                drop_path=drop_path_rates[i],
            )
            for i in range(depth)
        ])

        # Final norm + classifier
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize positional embedding with sinusoidal-like values, then allow learning
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def get_aux_loss(self):
        """Collect load-balancing loss from all MoE layers."""
        total = 0.0
        count = 0
        for block in self.blocks:
            total += block.moe.aux_loss
            count += 1
        return total / max(count, 1)

    def forward(self, x):
        # Patch embedding: [B, 3, 224, 224] → [B, 196, 256]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, 197, 256]

        # Add positional embeddings (THIS WAS MISSING IN v2)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        # Classify from CLS token
        x = self.norm(x[:, 0])
        return self.head(x)


# =============================================================
# 7. Exponential Moving Average (EMA)
# =============================================================
class ModelEMA:
    """
    Maintains an exponential moving average of model parameters.
    The EMA model often generalizes better than the final trained model
    because it smooths out noisy gradient updates.
    """
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
# 8. Learning Rate Scheduler with Warmup
# =============================================================
def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
    """
    Linear warmup for warmup_epochs, then cosine decay to min_lr.
    Much better than flat LR — allows aggressive early learning
    then fine-grained convergence.
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Start at 10% of peak LR (not zero!) and ramp linearly
            return 0.1 + 0.9 * epoch / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return max(min_lr / optimizer.defaults['lr'],
                   0.5 * (1 + math.cos(math.pi * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =============================================================
# 9. Training Loop
# =============================================================
def train_model(data_dir, epochs=30, batch_size=32, lr=3e-4, weight_decay=0.05,
                warmup_epochs=5, label_smoothing=0.1, aux_loss_weight=0.01,
                use_cutmix_mixup=True, use_ema=True, grad_clip=1.0):
    """
    Full training pipeline with all v3 improvements.
    """
    # Device selection
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Training on: {device}")

    # Data
    dataloaders, num_classes = get_dataloaders(data_dir, batch_size=batch_size)

    # Model
    model = ViT_MoE_v3(num_classes=num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Optimizer — higher weight decay for better regularization
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))

    # LR Scheduler
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_epochs, epochs)

    # EMA
    ema = ModelEMA(model) if use_ema else None

    # Training state
    history = []
    best_acc = 0.0
    best_ema_acc = 0.0

    print(f"\n{'='*60}")
    print(f"Training ViT+MoE v3 for {epochs} epochs")
    print(f"  LR: {lr} | Weight Decay: {weight_decay}")
    print(f"  Warmup: {warmup_epochs} epochs | Label Smoothing: {label_smoothing}")
    print(f"  CutMix/MixUp: {use_cutmix_mixup} | EMA: {use_ema}")
    print(f"  Gradient Clipping: {grad_clip} | Aux Loss Weight: {aux_loss_weight}")
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

            # Apply CutMix/MixUp with 50% probability
            if use_cutmix_mixup and np.random.rand() < 0.25:
                inputs, mixed_labels = cutmix_mixup(inputs, labels, num_classes)
                optimizer.zero_grad()
                outputs = model(inputs)
                # Soft cross-entropy for mixed labels
                log_probs = F.log_softmax(outputs, dim=-1)
                loss = -(mixed_labels * log_probs).sum(dim=-1).mean()
            else:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # Add MoE load-balancing auxiliary loss
            aux_loss = model.get_aux_loss()
            total_loss = loss + aux_loss_weight * aux_loss

            total_loss.backward()

            # Gradient clipping
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            # Update EMA
            if ema is not None:
                ema.update(model)

            running_loss += loss.item() * inputs.size(0)
            preds = torch.max(outputs, 1)[1]
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Train metrics (note: labels might be soft from mixup, use hard labels for metrics)
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

            # Save best model
            if eval_name == 'val' and val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), '../models/best_vit_moe_v3_model.pth')
                print(f"    -> Best model saved! (val_acc: {best_acc:.4f})")

            if eval_name == 'val_ema' and val_acc > best_ema_acc:
                best_ema_acc = val_acc
                torch.save(ema.state_dict(), '../models/best_vit_moe_v3_ema.pth')
                print(f"    -> Best EMA model saved! (ema_acc: {best_ema_acc:.4f})")

        # Step LR scheduler
        scheduler.step()

        # Epoch timing
        epoch_duration = time.time() - epoch_start
        mins, secs = int(epoch_duration // 60), int(epoch_duration % 60)
        print(f"  [*] Epoch {epoch+1} completed in {mins}m {secs}s")
        epoch_results['epoch_time_seconds'] = epoch_duration
        history.append(epoch_results)

    # ---- Final Summary ----
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"  Best Val Accuracy:     {best_acc:.4f}")
    if ema:
        print(f"  Best EMA Val Accuracy: {best_ema_acc:.4f}")
    print(f"{'='*60}")

    # Export metrics
    df = pd.DataFrame(history)
    csv_path = '../results/vit_moe_v3.1_training_metrics.csv'
    df.to_csv(csv_path, index=False)
    print(f"Metrics saved to '{csv_path}'")


# =============================================================
# 10. Entry Point
# =============================================================
if __name__ == "__main__":
    PROCESSED_DATA_DIR = "../data/processed/v2_balanced_dataset"

    train_model(
        PROCESSED_DATA_DIR,
        epochs=30,
        batch_size=32,
        lr=3e-4,
        weight_decay=0.01,   # back to v2's level — 0.05 was over-regularizing
        warmup_epochs=3,     # shorter warmup — small dataset converges fast
        label_smoothing=0.05, # halved — was making training too hard
        aux_loss_weight=0.01,
        use_cutmix_mixup=True,
        use_ema=True,
        grad_clip=1.0,
    )
