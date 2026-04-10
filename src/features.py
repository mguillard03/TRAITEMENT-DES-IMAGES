import cv2
import numpy as np

def compute_ratio_vert(img_rgb):
    """Calcule le ratio G / (R+G+B)"""
    img_f = img_rgb.astype(np.float32)
    R, G, B = cv2.split(img_f)
    ratio = G / (R + G + B + 1e-6)
    return ratio

def get_all_representations(img_rgb):
    """
    Retourne les différentes vues (Point 3 du sujet)
    Utilisé pour la visualisation et la comparaison.
    """
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(img_hsv)
    
    ratio_v = compute_ratio_vert(img_rgb)
    
    return {
        "RGB": img_rgb,
        "Hue": H,
        "Saturation": S,
        "Value": V,
        "RatioVert": ratio_v
    }

def compute_local_stats(data_gray, window_size=5):
    """
    Calcule la moyenne et la variance locale sur une fenêtre donnée.
    Point 4 du sujet.
    """
    kernel = np.ones((window_size, window_size), np.float32) / (window_size**2)
    
    # Moyenne locale E[X]
    mu = cv2.filter2D(data_gray, -1, kernel)
    
    # Variance locale E[X²] - (E[X])²
    mu2 = cv2.filter2D(data_gray**2, -1, kernel)
    var = mu2 - mu**2
    
    # On s'assure qu'il n'y a pas de valeurs négatives dues aux arrondis
    var = np.maximum(var, 0)
    
    return mu, var

def extract_full_features(img_rgb):
    """
    Combine les représentations de pixels (Point 3) 
    et les features locales (Point 4) pour le K-means.
    """
    # 1. Couleurs (Point 3)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue = img_hsv[:,:,0] / 180.0       # Normalisé 0-1
    sat = img_hsv[:,:,1] / 255.0       # Normalisé 0-1
    ratio_v = compute_ratio_vert(img_rgb)
    
    # 2. Features locales sur le ratio vert (Point 4)
    mean_loc, var_loc = compute_local_stats(ratio_v, window_size=5)
    
    # 3. Assemblage du vecteur de caractéristiques (H, W, N)
    # On empile les plans pour que chaque pixel soit un vecteur de taille 5
    features_stack = np.stack([
        hue, 
        sat, 
        ratio_v, 
        mean_loc, 
        var_loc
    ], axis=-1)
    
    return features_stack