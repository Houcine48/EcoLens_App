import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import os
import argparse

# --- CONFIGURATION ---
MODEL_PATH = "models/best_model.pth"
CLASSES = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']

def load_trained_model():
    print(f"🔄 Chargement du modèle depuis : {MODEL_PATH}")
    try:
        # Reconstruction de l'architecture
        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(num_ftrs, len(CLASSES)))
        
        # Chargement des poids (CPU safe)
        state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except FileNotFoundError:
        print("❌ Erreur : Modèle introuvable.")
        return None

def predict_single_image(image_path, model):
    if not os.path.exists(image_path):
        print(f"❌ Image introuvable : {image_path}")
        return

    # Transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0) # Batch de 1
    
    # Inférence
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        score, idx = torch.max(probs, 1)
        
    label = CLASSES[idx.item()]
    conf = score.item() * 100
    
    print(f"\n🖼️ Image : {os.path.basename(image_path)}")
    print(f"🤖 Prédiction : {label}")
    print(f"📊 Confiance : {conf:.2f}%")
    return label

if __name__ == "__main__":
    # On peut lancer ce script en ligne de commande : python models/test.py --image "mon_image.jpg"
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Chemin de l'image à tester")
    args = parser.parse_args()
    
    model = load_trained_model()
    
    if model:
        if args.image:
            predict_single_image(args.image, model)
        else:
            print("ℹ️ Astuce : Lancez avec --image 'chemin.jpg' pour tester.")