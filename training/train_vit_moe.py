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

# ---------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------
def get_dataloaders(data_dir, batch_size=32):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train']),
        'val': datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val'])
    }

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=2),
        'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=2)
    }

    num_classes = len(image_datasets['train'].classes)
    return dataloaders, num_classes

# ---------------------------------------------------------
# 2. Simplified ViT + MoE Architecture
# ---------------------------------------------------------
class Expert(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
    def forward(self, x):
        return self.net(x)

class MoE_Layer(nn.Module):
    def __init__(self, embed_dim, num_experts=4, k=2):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.router = nn.Linear(embed_dim, num_experts)
        self.experts = nn.ModuleList([Expert(embed_dim) for _ in range(num_experts)])

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        x_flat = x.view(-1, embed_dim)
        router_logits = self.router(x_flat)
        routing_weights = F.softmax(router_logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        out_flat = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            expert_mask = (top_k_indices == i).any(dim=-1)
            if not expert_mask.any(): continue
            out_flat[expert_mask] += expert(x_flat[expert_mask])
            
        return out_flat.view(batch_size, seq_len, embed_dim)

class SimpleViT_MoE(nn.Module):
    def __init__(self, num_classes, embed_dim=256, num_experts=4):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.moe = MoE_Layer(embed_dim, num_experts=num_experts)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x2 = self.norm1(x)
        attn_out, _ = self.attn(x2, x2, x2)
        x = x + attn_out
        x = x + self.moe(self.norm2(x))
        return self.head(x[:, 0])

# ---------------------------------------------------------
# 3. Training Loop & Analysis
# ---------------------------------------------------------
def train_model(data_dir, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Training on device: {device}")

    dataloaders, num_classes = get_dataloaders(data_dir)
    model = SimpleViT_MoE(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    history = []
    best_acc = 0.0

    print(f"Starting Training for {epochs} Epochs...")
    for epoch in range(epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{epochs}")
        epoch_results = {'epoch': epoch + 1}
        
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss, all_preds, all_labels = 0.0, [], []
            
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward(), optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                all_preds.extend(torch.max(outputs, 1)[1].cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            acc = accuracy_score(all_labels, all_preds)
            prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
            
            epoch_results.update({
                f'{phase}_loss': epoch_loss, f'{phase}_acc': acc,
                f'{phase}_prec': prec, f'{phase}_rec': rec, f'{phase}_f1': f1
            })
            
            print(f"{phase.capitalize()} | Loss: {epoch_loss:.4f} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")

            # Save the best model
            if phase == 'val' and acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), '../models/best_vit_moe_v2_model.pthfor20epoch')
                print(" -> Best model saved to 'best_vit_moe_v2_model.pthfor20epoch'!")

        # Calculate and record epoch duration
        epoch_duration = time.time() - epoch_start_time
        mins = int(epoch_duration // 60)
        secs = int(epoch_duration % 60)
        print(f"[*] Epoch {epoch+1} completed in {mins}m {secs}s")
        epoch_results['epoch_time_seconds'] = epoch_duration

        history.append(epoch_results)

    # Export to Spreadsheet
    df = pd.DataFrame(history)
    df.to_csv('../results/vit_moe_v2_training_metrics_for20epoch.csv', index=False)
    print("\nMetrics saved to 'vit_moe_v2_training_metrics.csv'")

if __name__ == "__main__":
    # Pointing to the NEW dataset folder
    PROCESSED_DATA_DIR = "../data/processed/v2_balanced_dataset"
    
    # Train ViT + MoE for 10 epochs
    train_model(PROCESSED_DATA_DIR, epochs=20)