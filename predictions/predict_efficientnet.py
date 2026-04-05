import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet_v2_s
from PIL import Image
import os

# ---------------------------------------------------------
# Interactive Prediction Loop for EfficientNetV2
# ---------------------------------------------------------
def interactive_prediction_loop(model_path, dataset_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    # 1. Get the class names dynamically from the dataset folder
    train_dir = os.path.join(dataset_dir, 'train')
    try:
        class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    except Exception as e:
        print(f"Error reading dataset directory to get class names: {e}")
        return

    num_classes = len(class_names)
    
    # 2. Initialize EfficientNetV2 & Load Weights ONCE
    print("\nLoading EfficientNetV2-S model...")
    
    # Initialize "blind" just like we did in training
    model = efficientnet_v2_s(weights=None)
    
    # Modify the final classification head to match your dataset classes
    # EfficientNet uses index 1 for its final Linear layer
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Weights loaded successfully!\n")
    except Exception as e:
        print(f"Error loading weights. Make sure '{model_path}' exists.")
        print(f"Details: {e}")
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
    print("EfficientNetV2 INTERACTIVE PREDICTOR")
    print("Type 'q' or 'quit' to exit.")
    print("="*50)

    # 4. Interactive Loop
    while True:
        try:
            image_path = input("\nEnter image path (or 'q' to quit): ").strip()
        except KeyboardInterrupt:
            print("\nExiting interactive predictor. Goodbye!")
            break
            
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
    # Path to your saved EfficientNetV2 weights
    MODEL_WEIGHTS = "../models/best_efficientnet_v2_model.pth"

    # Path to your dataset (so the script knows the names of the classes)
    DATASET_DIR = "../data/processed/v2_balanced_dataset"
    
    # Start the interactive loop
    interactive_prediction_loop(MODEL_WEIGHTS, DATASET_DIR)