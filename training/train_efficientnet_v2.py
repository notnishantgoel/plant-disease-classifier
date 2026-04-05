import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import efficientnet_v2_s
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import time

# ---------------------------------------------------------
# 1. Data Loading (Using your new V2 Balanced Dataset)
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
# 2. Training Loop & Analysis
# ---------------------------------------------------------
def train_model(data_dir, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Training EfficientNetV2 on device: {device}")

    dataloaders, num_classes = get_dataloaders(data_dir)
    print(f"Found {num_classes} classes in the dataset.")
    
    # Initialize EfficientNetV2-S (Google's SOTA Mobile CNN) from scratch
    # weights=None ensures it is "blind", just like your ViT
    print("Initializing EfficientNetV2-S model from scratch (no pre-trained weights)...")
    model = efficientnet_v2_s(weights=None)
    
    # Modify the classification head
    # EfficientNet's classifier is a Sequential block where index 1 is the Linear layer
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    model = model.to(device)
    
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
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                all_preds.extend(torch.max(outputs, 1)[1].cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            # Calculate Metrics
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
                torch.save(model.state_dict(), '../models/best_efficientnet_v2_model.pth')
                print(" -> Best model saved to 'best_efficientnet_v2_model.pth'!")

        # Record epoch duration
        epoch_duration = time.time() - epoch_start_time
        mins = int(epoch_duration // 60)
        secs = int(epoch_duration % 60)
        print(f"[*] Epoch {epoch+1} completed in {mins}m {secs}s")
        epoch_results['epoch_time_seconds'] = epoch_duration

        history.append(epoch_results)

    # Export to Spreadsheet
    df = pd.DataFrame(history)
    df.to_csv('../results/efficientnet_v2_training_metrics.csv', index=False)
    print("\nMetrics saved to 'efficientnet_v2_training_metrics.csv'")

if __name__ == "__main__":
    PROCESSED_DATA_DIR = "../data/processed/v2_balanced_dataset"
    train_model(PROCESSED_DATA_DIR, epochs=10)