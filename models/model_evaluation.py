import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# --- CONFIGURATION ---
DATA_DIR = r"G:\Mon Drive\Projet pi\Dataset\Garbage classification" # Adaptez si besoin
MODEL_PATH = "models/best_model.pth"
REPORT_OUTPUT = "models/evaluation_report.png"
CLASSES = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']

def evaluate_model():
    print("🚀 Démarrage de l'évaluation complète...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Chargement Modèle
    model = models.resnet50(pretrained=False)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, len(CLASSES)))
    
    if not os.path.exists(MODEL_PATH):
        print("❌ Modèle non trouvé.")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    
    # 2. Chargement Données
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # On utilise tout le dataset pour l'évaluation finale (ou un dossier 'Test' séparé si vous avez)
    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    all_preds = []
    all_labels = []
    
    print(f"⏳ Analyse de {len(dataset)} images... (Cela peut prendre du temps sur CPU)")
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # 3. Génération des Rapports
    print("\n📝 Rapport de Classification :")
    report = classification_report(all_labels, all_preds, target_names=CLASSES)
    print(report)
    
    # 4. Matrice de Confusion (Sauvegarde Image)
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title("Matrice de Confusion (Performance Modèle)")
    plt.ylabel('Réalité')
    plt.xlabel('Prédiction')
    
    plt.savefig(REPORT_OUTPUT)
    print(f"✅ Graphique sauvegardé sous : {REPORT_OUTPUT}")
    
    # Sauvegarde des métriques brutes pour le Dashboard (CSV)
    df_cm = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
    df_cm.to_csv("models/confusion_matrix.csv")

if __name__ == "__main__":
    evaluate_model()