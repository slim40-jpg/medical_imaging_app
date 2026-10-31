# modules/preprocessing.py
"""
Module de prétraitement d'images médicales :
- Amélioration du contraste (CLAHE)
- Réduction du bruit (médian, gaussien, bilatéral)
- Filtrage (moyenneur, gaussien, médian)
"""
import cv2
import numpy as np


def to_gray(image: np.ndarray) -> np.ndarray:
    """Convertit une image en niveaux de gris si nécessaire."""
    if image is None:
        raise ValueError("Image invalide (None).")
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if len(image.shape) == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return image


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0,
                tile_size=(8, 8)) -> np.ndarray:
    """
    Égalisation adaptative d'histogramme avec limitation de contraste (CLAHE).
    Recommandée pour les images médicales à faible contraste.
    """
    gray = to_gray(image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return clahe.apply(gray)


def denoise(image: np.ndarray, method: str = "median",
            kernel_size: int = 3) -> np.ndarray:
    """
    Réduction du bruit :
    - 'median'   : filtre médian (bruit impulsionnel / poivre & sel)
    - 'gaussian' : filtre gaussien (bruit gaussien)
    - 'bilateral': filtre bilatéral (préserve les contours)
    - 'none'     : aucun débruitage
    """
    gray = to_gray(image)
    if kernel_size % 2 == 0:
        kernel_size += 1  # OpenCV exige des tailles impaires

    if method == "median":
        return cv2.medianBlur(gray, kernel_size)
    if method == "gaussian":
        return cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    if method == "bilateral":
        return cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    return gray


def apply_filter(image: np.ndarray, kind: str = "gaussian",
                 kernel_size: int = 5) -> np.ndarray:
    """
    Filtrage générique :
    - 'mean'     : filtre moyenneur
    - 'gaussian' : filtre gaussien
    - 'median'   : filtre médian
    """
    gray = to_gray(image)
    if kernel_size % 2 == 0:
        kernel_size += 1

    if kind == "mean":
        return cv2.blur(gray, (kernel_size, kernel_size))
    if kind == "gaussian":
        return cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    if kind == "median":
        return cv2.medianBlur(gray, kernel_size)
    return gray


def full_pipeline(image: np.ndarray,
                  use_clahe: bool = True,
                  denoise_method: str = "median",
                  kernel_size: int = 3) -> np.ndarray:
    """Chaîne complète : contraste puis débruitage."""
    out = to_gray(image)
    if use_clahe:
        out = apply_clahe(out)
    out = denoise(out, method=denoise_method, kernel_size=kernel_size)
    return out