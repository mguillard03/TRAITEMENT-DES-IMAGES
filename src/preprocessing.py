import cv2
import numpy as np

def apply_clahe_rgb(img_rgb):
    """
    Applique une égalisation d'histogramme adaptative (CLAHE) sur le canal L (luminance)
    pour rehausser le contraste sans modifier les couleurs.
    """
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # clipLimit : seuil pour le contraste, tileGridSize : taille des pavés d'égalisation
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_equalized = clahe.apply(l)
    
    lab_equalized = cv2.merge((l_equalized, a, b))
    return cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2RGB)

def preprocess_image(img_rgb, median_kernel=3):
    """
    Pipeline complet de prétraitement :
    1. Égalisation de contraste (CLAHE)
    2. Réduction du bruit (Filtre Médian)
    """
    # 1. Rehaussement du contraste
    img_equalized = apply_clahe_rgb(img_rgb)
    
    # 2. Réduction du bruit
    # Préserve les bords (arbres/routes) tout en supprimant le grain
    img_denoised = cv2.medianBlur(img_equalized, median_kernel)
    
    return img_denoised