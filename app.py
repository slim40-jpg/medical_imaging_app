import os
import uuid
import cv2
import numpy as np
import torch
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from transformers import SamModel, SamProcessor

# --- Imports de tes méthodes conventionnelles ---
# Assure-toi d'avoir copié le dossier "modules" (avec conventional_seg.py et preprocessing.py) 
# de ton premier projet dans ce dossier "medsam_app".
from modules.preprocessing import apply_clahe
from modules.conventional_seg import otsu_threshold, watershed_segmentation, overlay_contours

from config import Config

app = Flask(__name__)
app.config.from_object(Config)

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["RESULT_FOLDER"], exist_ok=True)

print("Chargement de MedSAM...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SamModel.from_pretrained(app.config["MEDSAM_MODEL_ID"]).to(device)
processor = SamProcessor.from_pretrained(app.config["MEDSAM_MODEL_ID"])
model.eval()
print(f"MedSAM chargé sur {device}.")

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("image")
    if not file or file.filename == "":
        flash("Aucune image sélectionnée.", "warning")
        return redirect(url_for("index"))
    
    uid = uuid.uuid4().hex[:10]
    ext = file.filename.rsplit(".", 1)[1].lower()
    filename = f"src_{uid}.{ext}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    return render_template("draw.html", image_url=url_for('static', filename='uploads/' + filename), filename=filename)

@app.route("/process", methods=["POST"])
def process():
    data = request.get_json()
    filename = data["filename"]
    box = [float(v) for v in data["box"]]

    image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    raw_image = Image.open(image_path).convert("RGB")
    original_cv = cv2.imread(image_path)
    
        # ==========================================
    # 1. MÉTHODES CONVENTIONNELLES (OpenCV)
    # ==========================================
    try:
        # Prétraitement
        gray = apply_clahe(original_cv)
        
        # Seuillage Otsu
        otsu_mask = otsu_threshold(gray)
        otsu_vis = overlay_contours(gray, otsu_mask, color=(255, 0, 0))
        
        # Watershed (Ligne de partage des eaux)
        watershed_vis = watershed_segmentation(gray)
        
    except Exception as e:
        print("!!! ERREUR DANS LES METHODES CONVENTIONNELLES :", e)
        # En cas d'erreur, on met l'image originale pour ne pas planter
        otsu_vis = original_cv
        watershed_vis = original_cv

    # Sauvegarde des résultats conventionnels
    result_uid = uuid.uuid4().hex[:10]
    otsu_name = f"otsu_{result_uid}.png"
    cv2.imwrite(os.path.join(app.config["RESULT_FOLDER"], otsu_name), otsu_vis)
    
    ws_name = f"ws_{result_uid}.png"
    cv2.imwrite(os.path.join(app.config["RESULT_FOLDER"], ws_name), watershed_vis)

    # ==========================================
    # 2. DEEP LEARNING (MedSAM)
    # ==========================================
    inputs = processor(raw_image, input_boxes=[[box]], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)

    probs = processor.image_processor.post_process_masks(
        outputs.pred_masks.sigmoid().cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
        binarize=False
    )
    
    mask_array = probs[0].squeeze().numpy()
    if mask_array.ndim == 3:
        mask_array = mask_array[0]
    
    binary_mask = (mask_array > 0.5).astype(np.uint8) * 255
    
    # Sauvegarde du masque MedSAM
    medsam_mask_name = f"medsam_mask_{result_uid}.png"
    cv2.imwrite(os.path.join(app.config["RESULT_FOLDER"], medsam_mask_name), binary_mask)

    # Superposition MedSAM (Vert)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    medsam_overlay = original_cv.copy()
    cv2.drawContours(medsam_overlay, contours, -1, (0, 255, 0), 2)
    medsam_overlay_name = f"medsam_overlay_{result_uid}.png"
    cv2.imwrite(os.path.join(app.config["RESULT_FOLDER"], medsam_overlay_name), medsam_overlay)

    return jsonify({
        "original": url_for('static', filename='uploads/' + filename),
        "otsu": url_for('static', filename='results/' + otsu_name),
        "watershed": url_for('static', filename='results/' + ws_name),
        "medsam_mask": url_for('static', filename='results/' + medsam_mask_name),
        "medsam_overlay": url_for('static', filename='results/' + medsam_overlay_name)
    })

@app.route("/result")
def result():
    return render_template("results.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)