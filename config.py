import os
import platform

# --- DÉTECTION AUTOMATIQUE DE L'ENVIRONNEMENT ---

def get_dataset_path():
    system = platform.system()
    
    # Cas 1 : Google Colab (Linux)
    if os.path.exists('/content/drive'):
        return '/content/drive/MyDrive/Projet pi/Dataset/Garbage classification'
    
    # Cas 2 : Windows (Votre PC avec Drive G:)
    elif system == 'Windows':
        # Essayons plusieurs lettres possibles pour le Drive
        candidates = [
            r"G:\Mon Drive\Projet pi\Dataset\Garbage classification",
            r"G:\My Drive\Projet pi\Dataset\Garbage classification",
            r"H:\Mon Drive\Projet pi\Dataset\Garbage classification"
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
                
    # Cas 3 : Mode dégradé (Dossier local data/)
    return "./data/raw"

# --- CONSTANTES GLOBALES ---
DATASET_PATH = get_dataset_path()
MODELS_PATH = r"G:\Mon Drive\Projet pi\Models" if platform.system() == 'Windows' else "/content/drive/MyDrive/Projet pi/Models"
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Test rapide quand on lance ce fichier
if __name__ == "__main__":
    print(f"🌍 Environnement détecté : {platform.system()}")
    print(f"📂 Chemin Dataset : {DATASET_PATH}")
    
    if os.path.exists(DATASET_PATH):
        print("✅ Connexion réussie !")
        print(f"   Classes trouvées : {os.listdir(DATASET_PATH)}")
    else:
        print("❌ Connexion échouée. Vérifiez le lecteur G: ou le nom du dossier.")