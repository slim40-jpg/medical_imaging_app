# modules/conventional_seg.py
import cv2
import numpy as np
from skimage.segmentation import active_contour
from skimage.filters import gaussian

from modules.preprocessing import to_gray

# ---------------------------------------------------------------
# 1. Seuillage d'Otsu (Amélioré)
# ---------------------------------------------------------------
def otsu_threshold(image):
    """
    Sur une IRM, Otsu global échoue car le fond noir domine l'histogramme.
    Correction : on ignore les pixels du fond (les plus sombres) 
    avant de calculer le seuil.
    """
    gray = to_gray(image)
    
    # 1. On ignore les 10% de pixels les plus sombres (le fond de l'IRM)
    # pour que l'histogramme se concentre sur les tissus.
    pixels = gray.flatten()
    pixels = pixels[pixels > np.percentile(pixels, 10)]
    if len(pixels) == 0: return gray
    
    # 2. Calcul d'Otsu manuel sur les pixels restants
    hist, _ = np.histogram(pixels, bins=256, range=(0, 256))
    hist = hist.astype(float)
    
    # Formule mathématique d'Otsu
    total = pixels.size
    sum_total = np.dot(np.arange(256), hist)
    sumB, wB, max_var, best_thresh = 0.0, 0.0, 0.0, 0
    
    for t in range(256):
        wB += hist[t]
        if wB == 0: continue
        wF = total - wB
        if wF == 0: break
        
        sumB += t * hist[t]
        mB = sumB / wB
        mF = (sum_total - sumB) / wF
        var_between = wB * wF * (mB - mF) ** 2
        
        if var_between > max_var:
            max_var = var_between
            best_thresh = t
            
    # 3. Application du seuil trouvé
    _, binary = cv2.threshold(gray, best_thresh, 255, cv2.THRESH_BINARY)
    return binary

# ---------------------------------------------------------------
# 2. Seuillage adaptatif (Corrigé)
# ---------------------------------------------------------------
def adaptive_threshold(image, block_size=51, C=10):
    """
    Correction : block_size grand (51) pour moyenner sur une zone plus large,
    et C élevé (10) pour ignorer les micro-variations de texture (bruit).
    """
    gray = to_gray(image)
    if block_size % 2 == 0:
        block_size += 1
    # On applique un flou préalable pour réduire le bruit poivre et sel
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    return cv2.adaptiveThreshold(blurred, 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY,
                                 block_size, C)

# ---------------------------------------------------------------
# 3. Watershed (Amélioré avec filtrage des petites régions)
# ---------------------------------------------------------------
def watershed_segmentation(image):
    gray = to_gray(image)
    orig_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # 1. Binarisation Otsu
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. Nettoyage morphologique (Ouvre les contours et ferme les trous)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # 3. Distance transform
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    if dist.max() == 0: return orig_bgr
    
    _, sure_fg = cv2.threshold(dist, 0.3 * dist.max(), 255, 0) # 0.3 au lieu de 0.5
    sure_fg = np.uint8(sure_fg)

    # 4. Marquage
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # 5. Watershed
    markers = cv2.watershed(orig_bgr, markers)
    
    # 6. NOUVEAU : Supprimer les régions trop petites (bruit)
    result = orig_bgr.copy()
    result[markers == -1] = [0, 0, 255] # Frontières en rouge
    
    # On filtre les régions internes < 1000 pixels
    for label in np.unique(markers):
        if label <= 1: continue # 0 = unknown, 1 = bg, -1 = borders
        if np.sum(markers == label) < 1000:
            result[markers == label] = [0, 0, 0] # On noircit le bruit
            
    return result

# ---------------------------------------------------------------
# 4. Snake (Correction majeure des paramètres)
# ---------------------------------------------------------------
def active_contour_segmentation(image, center=None, radius=None):
    gray = to_gray(image)
    h, w = gray.shape

    # On initialise le cercle au centre, rayon plus petit (1/6 de la dimension)
    if center is None:
        center = (h // 2, w // 2)
    if radius is None:
        radius = min(h, w) // 6

    s = np.linspace(0, 2 * np.pi, 400)
    init_r = center[0] + radius * np.sin(s)
    init_c = center[1] + radius * np.cos(s)
    init = np.array([init_r, init_c]).T

    gray_float = gray.astype(np.float64) / 255.0
    # On lisse un peu plus pour que le snake ne s'accroche pas au bruit
    smoothed = gaussian(gray_float, sigma=4, preserve_range=False)

    # CORRECTION ICI : beta très bas (0.1) pour que le contour puisse se déformer
    # et alpha bas (0.01) pour qu'il puisse se contracter sur l'os.
    snake = active_contour(
        smoothed, init,
        alpha=0.01, beta=0.1, gamma=0.001,
        max_num_iter=5000,
        convergence=0.1
    )

    result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    pts = snake.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(result, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    return result

def overlay_contours(image, mask, color=(0, 255, 0)):
    gray = to_gray(image)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if mask is None: return base
    mask_gray = to_gray(mask) if len(mask.shape) == 3 else mask
    contours, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(base, contours, -1, color, 2)
    return base