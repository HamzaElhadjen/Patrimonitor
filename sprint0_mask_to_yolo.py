import os
import cv2
import numpy as np
import tqdm # Si erreur d'import, lance: pip install tqdm

def mask_to_yolo_polygons(mask_path, image_width, image_height, class_id=0):
    """
    Lit une image de masque binaire (Noir&Blanc) et extrait les polygones
    au format YOLOv8-Seg (coordonnées normalisées [0-1]).
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return ""

    # Binarisation stricte pour nettoyer le masque
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # OpenCV 4.x: findContours retourne (contours, hierarchy)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    yolo_lines = []
    for contour in contours:
        # Ignorer les bruits trop petits (moins de 5 points)
        if contour.shape[0] < 5:
            continue

        # Approximation pour réduire le nombre de points (lississage)
        epsilon = 0.001 * cv2.arcLength(contour, True) 
        approx_contour = cv2.approxPolyDP(contour, epsilon, True)
        
        # Ignorer si l'approximation a trop réduit le contour
        if approx_contour.shape[0] < 4:
            continue

        # Aplatir et normaliser les coordonnées [0, 1]
        polygon_points = approx_contour.flatten().astype(np.float32)
        polygon_points[::2] /= image_width # X normalisé
        polygon_points[1::2] /= image_height # Y normalisé
        
        # Format YOLOv8-Seg: class_id x1 y1 x2 y2 ...
        line = f"{class_id} " + " ".join([f"{p:.6f}" for p in polygon_points])
        yolo_lines.append(line)

    return "\n".join(yolo_lines)

def convert_dataset(masks_dir, images_dir, output_labels_dir):
    """
    Parcourt le dataset, convertit chaque masque et génère le dossier labels/
    """
    os.makedirs(output_labels_dir, exist_ok=True)
    
    mask_files = [f for f in os.listdir(masks_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    success_count = 0
    print(f"[INFO] Début de la conversion de {len(mask_files)} masques vers : {output_labels_dir}")

    for mask_name in tqdm.tqdm(mask_files):
        mask_path = os.path.join(masks_dir, mask_name)
        img_name = mask_name 
        img_path = os.path.join(images_dir, img_name)

        # Nous avons besoin des dimensions de l'image originale
        img = cv2.imread(img_path)
        if img is None:
            # Si l'image n'est pas trouvée, utiliser les dimensions du masque
            img = cv2.imread(mask_path)
            if img is None: continue
            h, w = img.shape[:2]
        else:
            h, w = img.shape[:2]

        # Conversion
        yolo_annotations = mask_to_yolo_polygons(mask_path, w, h)
        
        if yolo_annotations:
            # Sauvegarde du fichier label TXT
            label_name = os.path.splitext(mask_name)[0] + ".txt"
            label_path = os.path.join(output_labels_dir, label_name)
            with open(label_path, 'w') as f:
                f.write(yolo_annotations)
            success_count += 1

    print(f"[FIN] Terminé. {success_count}/{len(mask_files)} labels générés avec succès.\n")

if __name__ == "__main__":
    # Chemin absolu récupéré depuis ton terminal
    DATASET_ROOT = r"C:\Users\hamza\OneDrive\Bureau\italy project\crack_segmentation_dataset"
    
    # 1. Traitement du dossier TRAIN
    TRAIN_MASKS = os.path.join(DATASET_ROOT, "train", "masks")
    TRAIN_IMAGES = os.path.join(DATASET_ROOT, "train", "images")
    TRAIN_LABELS = os.path.join(DATASET_ROOT, "train", "labels")
    
    # 2. Traitement du dossier TEST
    TEST_MASKS = os.path.join(DATASET_ROOT, "test", "masks")
    TEST_IMAGES = os.path.join(DATASET_ROOT, "test", "images")
    TEST_LABELS = os.path.join(DATASET_ROOT, "test", "labels")

    print("[INFO] Lancement de la conversion Technique #3")
    
    # Vérification de la structure avant de lancer
    if os.path.exists(TRAIN_MASKS) and os.path.exists(TEST_MASKS):
        print("--- Phase 1: Conversion du dossier TRAIN ---")
        convert_dataset(TRAIN_MASKS, TRAIN_IMAGES, TRAIN_LABELS)
        
        print("--- Phase 2: Conversion du dossier TEST ---")
        convert_dataset(TEST_MASKS, TEST_IMAGES, TEST_LABELS)
    else:
        # Fallback si le dataset n'est pas partitionné
        ROOT_MASKS = os.path.join(DATASET_ROOT, "masks")
        ROOT_IMAGES = os.path.join(DATASET_ROOT, "images")
        ROOT_LABELS = os.path.join(DATASET_ROOT, "labels")
        print("\n[INFO] Sous-dossiers train/test non trouvés. Conversion du dossier racine...")
        convert_dataset(ROOT_MASKS, ROOT_IMAGES, ROOT_LABELS)