"""Briques du mode fantôme pour guider une re-photographie."""
from __future__ import annotations
import cv2
import numpy as np


def ghost_overlay(reference, live, alpha=0.5):
    """Superpose la référence recalée à l'image courante; nécessite des dimensions identiques."""
    if reference.shape != live.shape:
        live = cv2.resize(live, (reference.shape[1], reference.shape[0]))
    return cv2.addWeighted(reference, float(alpha), live, 1.0-float(alpha), 0)


def estimate_alignment(reference, live):
    """Estime une homographie ORB et renvoie les métriques; ne fournit pas une pose 6-DoF."""
    orb = cv2.ORB_create(nfeatures=1500)
    k1, d1 = orb.detectAndCompute(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), None)
    k2, d2 = orb.detectAndCompute(cv2.cvtColor(live, cv2.COLOR_BGR2GRAY), None)
    if d1 is None or d2 is None:
        raise ValueError("Descripteurs ORB insuffisants.")
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d2, d1, k=2)
    good = [m for pair in matches if len(pair) == 2 for m, n in [pair] if m.distance < 0.75*n.distance]
    if len(good) < 8:
        raise ValueError("Pas assez de correspondances pour estimer l'alignement.")
    src = np.float32([k2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if H is None:
        raise ValueError("Homographie non calculable.")
    n = int(inliers.sum())
    return {"homography": H, "matches": len(good), "inliers": n,
            "inlier_ratio": n/len(good),
            "pose_6dof_available": False}
