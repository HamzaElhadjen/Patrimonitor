# Fichier: sprint3_registration.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

def align_images(imgA_path, imgB_path):
    print("[INFO] Chargement des images temporelles...")
    imgA = cv2.imread(imgA_path)
    imgB = cv2.imread(imgB_path)
    
    if imgA is None or imgB is None:
        print("[ERREUR] Impossible de trouver image_A.jpg ou image_B.jpg.")
        return

    # 1. Conversion en niveaux de gris pour l'algorithme SIFT
    grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

    print("[INFO] Extraction des points de repère (SIFT)...")
    sift = cv2.SIFT_create()
    kpsA, desA = sift.detectAndCompute(grayA, None)
    kpsB, desB = sift.detectAndCompute(grayB, None)

    print("[INFO] Calcul des correspondances (FLANN Matcher)...")
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desB, desA, k=2)

    # 2. Filtre rigoureux de Lowe pour éliminer les mauvais points
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    print(f"[INFO] {len(good_matches)} points solides trouvés sur la maçonnerie.")

    if len(good_matches) > 10:
        # 3. Extraction des coordonnées géométriques
        ptsA = np.float32([kpsA[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        ptsB = np.float32([kpsB[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        print("[INFO] Calcul de l'Homographie et correction de l'angle de vue...")
        # 4. Calcul mathématique de la matrice de déformation
        H, _ = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, 5.0)

        h, w = imgA.shape[:2]
        # 5. Redressement de l'Image B sur l'Image A
        warped_imgB = cv2.warpPerspective(imgB, H, (w, h))

        # 6. Création d'une image de contrôle : 50% Image A + 50% Image B redressée
        blended = cv2.addWeighted(imgA, 0.5, warped_imgB, 0.5, 0)

        # Affichage visuel des résultats
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1), plt.title("1. Image A (Janvier)"), plt.imshow(cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB)), plt.axis('off')
        plt.subplot(1, 3, 2), plt.title("2. Image B (Février - Déformée)"), plt.imshow(cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB)), plt.axis('off')
        plt.subplot(1, 3, 3), plt.title("3. Résultat : B redressée sur A (Fusion 50/50)"), plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)), plt.axis('off')
        plt.tight_layout()
        plt.show()

    else:
        print("[ERREUR] Pas assez de correspondances pour recaler l'image.")

if __name__ == "__main__":
    align_images("image_A.jpg", "image_B.jpg")