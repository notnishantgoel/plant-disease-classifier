import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os

# ---------------------------------------------------------
# 1. Rebuild the ViT + MoE Architecture
# (Must exactly match the training script structure)
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
# 2. Interactive Prediction Loop
# ---------------------------------------------------------
def interactive_prediction_loop(model_path, dataset_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    # 1. Get the class names
    train_dir = os.path.join(dataset_dir, 'train')
    try:
        # PyTorch ImageFolder sorts classes alphabetically
        class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    except Exception as e:
        print(f"Error reading dataset directory to get class names: {e}")
        return

    num_classes = len(class_names)
    
    # 2. Initialize Model & Load Weights ONCE
    print("\nLoading ViT+MoE model...")
    model = SimpleViT_MoE(num_classes=num_classes)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Weights loaded successfully!\n")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return
        
    model = model.to(device)
    model.eval() # Set to evaluation mode

    # 3. Define exactly the same transforms used for validation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("="*50)
    print("ViT+MoE INTERACTIVE PREDICTOR")
    print("Type 'q' or 'quit' to exit.")
    print("="*50)

    # 4. Interactive Loop
    while True:
        image_path = input("\nEnter image path (or 'q' to quit): ").strip()
        
        # Handle drag-and-drop quotes (Mac/Windows terminals often add quotes)
        if image_path.startswith(("'", '"')) and image_path.endswith(("'", '"')):
            image_path = image_path[1:-1]
            
        if image_path.lower() in ['q', 'quit', 'exit']:
            print("Exiting interactive predictor. Goodbye!")
            break
            
        if not os.path.exists(image_path):
            print(f"Error: Could not find file at '{image_path}'")
            continue

        # Load and process the image
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
        except Exception as e:
            print(f"Error loading image: {e}")
            continue

        # Run Inference
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, 0)
            
            predicted_class = class_names[predicted_idx.item()]
            confidence_pct = confidence.item() * 100

        # Print Results
        print("\n" + "-"*40)
        print(f"Predicted Class : {predicted_class}")
        print(f"Confidence      : {confidence_pct:.2f}%")
        print("-" * 40)
        
        print("Other possibilities:")
        for i, prob in enumerate(probabilities):
            if i != predicted_idx.item():
                print(f"  - {class_names[i]}: {prob.item() * 100:.2f}%")
        print("-" * 40)

if __name__ == "__main__":
    # Path to your saved weights
    MODEL_WEIGHTS = "../models/best_vit_moe_v2_model.pth"

    # Path to your dataset (so the script knows the names of the classes)
    DATASET_DIR = "../data/processed/v2_balanced_dataset"
    
    # Start the interactive loop
    interactive_prediction_loop(MODEL_WEIGHTS, DATASET_DIR)