import cv2
import numpy as np

# On prend une image de ton dataset pour faire l'Image A (Janvier)
input_image_path = r"C:\Users\hamza\OneDrive\Bureau\italy project\crack_segmentation_dataset\test\images\CFD_001.jpg"

img = cv2.imread(input_image_path)
h, w = img.shape[:2]

# Sauvegarde de l'Image A (La référence)
cv2.imwrite("image_A.jpg", img)

# On simule un mauvais angle de caméra pour l'Image B (Février)
pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
pts2 = np.float32([[15, 25], [w-30, 5], [5, h-15], [w-45, h-35]]) # Déformation mathématique

matrice_perspective = cv2.getPerspectiveTransform(pts1, pts2)
img_b = cv2.warpPerspective(img, matrice_perspective, (w, h), borderMode=cv2.BORDER_REPLICATE)

# Sauvegarde de l'Image B (L'image déformée)
cv2.imwrite("image_B.jpg", img_b)
print("Succès : image_A.jpg et image_B.jpg ont été créées avec succès !")