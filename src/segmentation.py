import numpy as np
from sklearn.cluster import KMeans
import cv2

def apply_kmeans(features_stack, k=3):
    """
    Applique l'algorithme K-means sur l'ensemble des pixels.
    Point 6 du sujet.
    """
    h, w, c = features_stack.shape
    # Mise à plat des données : chaque pixel devient une ligne, chaque feature une colonne
    data = features_stack.reshape(-1, c)
    
    # n_init=10 permet de relancer l'algo avec différentes positions de départ pour éviter les minima locaux
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(data)
    
    # On reforme l'image segmentée
    return labels.reshape(h, w)

def identify_forest_cluster(segmented_img, ratio_vert_img):
    """
    Identifie quel cluster (0, 1 ou 2) correspond à la forêt 
    en cherchant celui qui a le ratio vert moyen le plus élevé.
    """
    k = len(np.unique(segmented_img))
    means = []
    for i in range(k):
        mean_val = ratio_vert_img[segmented_img == i].mean()
        means.append(mean_val)
    
    # L'index du cluster avec le ratio vert max
    return np.argmax(means)

def clean_mask(mask, kernel_size=3):
    """
    Applique des opérations morphologiques pour nettoyer le masque binaire.
    Point 8 du sujet.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # 1. Ouverture : Supprime le bruit extérieur (petits points blancs isolés)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # 2. Fermeture : Bouche les trous à l'intérieur (petits points noirs dans la forêt)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    
    return closing