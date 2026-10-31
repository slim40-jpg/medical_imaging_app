import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = "medsam-secret-key"
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
    RESULT_FOLDER = os.path.join(BASE_DIR, "static", "results")
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif"}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024

    # Identifiant du modèle sur Hugging Face
    MEDSAM_MODEL_ID = "flaviagiammarino/medsam-vit-base"
    
    # Taille d'entrée attendue par MedSAM (le processor s'en occupe, mais on garde la trace)
    INPUT_SIZE = 1024