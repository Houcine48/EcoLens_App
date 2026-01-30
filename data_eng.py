import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging # Pour des logs pro

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, element_at, udf, when
from pyspark.sql.types import StringType, FloatType, IntegerType, StructType, StructField
from config import DATASET_PATH, CLASSES

# --- CONFIGURATION LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 1. CONFIGURATION SPARK ---
def init_spark():
    logger.info("? Demarrage de la Session Spark...")
    try:
        return SparkSession.builder \
            .appName("EcoLens_ETL_Advanced") \
            .config("spark.driver.memory", "4g") \
            .getOrCreate()
    except Exception as e:
        logger.error(f"?? CRASH SPARK : Impossible de d�marrer. {e}")
        return None

# --- 2. UDF AVANC�E (Le Cerveau du Nettoyage) ---
# Retourne une structure complexe avec toutes les infos
def advanced_check(path):
    # Nettoyage du chemin (Compatibilit� Windows/Linux)
    clean_path = path.replace("file:", "").replace("///", "/") if os.name != 'nt' else path.replace("file:///", "")
    
    # 1. Extraction Extension
    try:
        ext = clean_path.split('.')[-1].lower()
    except:
        ext = "unknown"

    # V�rification Extension
    if ext not in ['jpg', 'jpeg', 'png', 'bmp']:
        return "Extension Invalide", 0.0, 0, 0, ext

    try:
        # 2. Lecture Image
        img = cv2.imread(clean_path)
        if img is None:
            return "Fichier Corrompu", 0.0, 0, 0, ext
        
        # 3. V�rification Dimensions (Nouveau !)
        h, w = img.shape[:2]
        if h < 50 or w < 50: # Si l'image fait moins de 50px, l'IA ne verra rien
            return "Trop Petit", 0.0, w, h, ext
            
        # 4. V�rification Flou
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        status = "Flou" if blur < 5 else "OK"
        
        return status, float(blur), w, h, ext

    except Exception as e:
        # Si OpenCV plante sur une image bizarre, on ne crashe pas le script
        return f"Erreur Lecture: {str(e)}", 0.0, 0, 0, ext

# D�finition de la structure de retour pour Spark
audit_schema = StructType([
    StructField("status", StringType(), False),
    StructField("blur_score", FloatType(), False),
    StructField("width", IntegerType(), False),
    StructField("height", IntegerType(), False),
    StructField("extension", StringType(), False)
])

audit_udf = udf(advanced_check, audit_schema)

# --- 3. PIPELINE PRINCIPAL ---
def run_etl_pipeline():
    spark = init_spark()
    if not spark: return None
    
    logger.info(f"?? Ingestion depuis : {DATASET_PATH}")
    
    try:
        # Lecture (On lit TOUT, m�me les fichiers texte, pour les filtrer proprement apr�s)
        raw_df = spark.read.format("binaryFile") \
            .option("pathGlobFilter", "*") \
            .option("recursiveFileLookup", "true") \
            .load(DATASET_PATH)
            
        logger.info(f"?? Fichiers d�tect�s : {raw_df.count()}")

        # Transformation
        df_processed = raw_df.withColumn("label", element_at(split(col("path"), "/"), -2))
        
        # Filtrage Labels
        df_filtered = df_processed.filter(col("label").isin(CLASSES))
        
        # Application de l'Audit (Calcul distribu�)
        logger.info("? Audit Qualit� approfondi en cours...")
        
        df_audit = df_filtered.withColumn("audit", audit_udf(col("path"))) \
            .select(
                col("path"),
                col("label"),
                col("audit.status").alias("status"),
                col("audit.blur_score").alias("blur_score"),
                col("audit.width").alias("width"),
                col("audit.height").alias("height"),
                col("audit.extension").alias("file_type")
            )
        
        # Conversion et Sauvegarde
        local_df = df_audit.toPandas()
        
        # Rapport d'erreurs
        error_count = local_df[local_df['status'] != 'OK'].shape[0]
        if error_count > 0:
            logger.warning(f"?? {error_count} images probl�matiques d�tect�es !")
            print(local_df[local_df['status'] != 'OK']['status'].value_counts())
        else:
            logger.info("? Aucune anomalie d�tect�e.")

        local_df.to_csv("dataset_full_report.csv", index=False)
        logger.info("?? Rapport complet sauvegard� : dataset_full_report.csv")
        
        spark.stop()
        return local_df

    except Exception as e:
        logger.critical(f"?? �CHEC CRITIQUE DU PIPELINE : {e}")
        return None

# --- 4. VISUALISATION ---
def visualize(df):
    if df is None or df.empty: return
    
    logger.info("?? G�n�ration des dashboards...")
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    
    # Graph 1 : Distribution
    sns.countplot(y=df['label'], palette='viridis', order=df['label'].value_counts().index, ax=ax[0])
    ax[0].set_title("Distribution par Classe")
    
    # Graph 2 : Qualit� (Nouveau)
    sns.countplot(x=df['status'], palette='coolwarm', ax=ax[1])
    ax[1].set_title("Qualit� des Donn�es (Anomalies)")
    
    plt.tight_layout()
    plt.show()

# --- MAIN ---
if __name__ == "__main__":
    # Petit hack pour importer config si lanc� directement
    try:
        from config import DATASET_PATH, CLASSES
    except ImportError:
        logger.warning("Fichier config.py non trouv�, utilisation des valeurs par d�faut.")
        # Valeurs de secours
        DATASET_PATH = r"G:\Mon Drive\Projet pi\Dataset\Garbage classification"
        CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

    df_final = run_etl_pipeline()
    visualize(df_final)