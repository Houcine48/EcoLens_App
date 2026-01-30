import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import os
import time
import copy

# --- CONFIGURATION ---
# Si vous lancez depuis VS Code, on remonte d'un cran pour trouver data/
# Adaptez ce chemin vers votre Drive G: si besoin
DATA_DIR = r"G:\Mon Drive\Projet pi\Dataset\Garbage classification" 
SAVE_PATH = "models/best_model.pth"

BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 10
CLASSES = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']

def train_model():
    print("🚀 Démarrage du script de ré-entraînement (Retraining)...")
    
    # 1. Configuration du Matériel (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Utilisation du processeur : {device}")

    # 2. Préparation des Données
    # Transformations (Data Augmentation pour la robustesse)
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])

    if not os.path.exists(DATA_DIR):
        print(f"❌ Erreur : Dossier de données introuvable : {DATA_DIR}")
        return

    full_dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms)
    
    # Split 80% Train / 20% Val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"📊 Données chargées : {len(train_dataset)} entrainement, {len(val_dataset)} validation.")

    # 3. Architecture du Modèle (ResNet50)
    print("🧠 Chargement de l'architecture ResNet50...")
    model = models.resnet50(pretrained=True)
    
    # On remplace la dernière couche pour nos 6 classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_ftrs, len(CLASSES))
    )
    
    model = model.to(device)

    # 4. Configuration de l'apprentissage
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    
    # 5. Boucle d'entraînement
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    start_time = time.time()

    for epoch in range(EPOCHS):
        print(f'\nÉpoque {epoch+1}/{EPOCHS}')
        print('-' * 10)

        # Chaque époque a une phase d'entraînement et de validation
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # Itération sur les données
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + Optimize (seulement en train)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Sauvegarde du meilleur modèle (Deep Copy)
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), SAVE_PATH)
                print("💾 Modèle sauvegardé (Nouveau record !)")

    time_elapsed = time.time() - start_time
    print(f'\n✅ Entraînement terminé en {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'🏆 Meilleure précision validation : {best_acc:.4f}')

    # Recharger les meilleurs poids
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    # Protection pour Windows (Multiprocessing)
    train_model()