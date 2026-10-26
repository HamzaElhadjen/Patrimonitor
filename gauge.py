"""Jauge virtuelle expérimentale : fonctions géométriques et déplacement local.
Les sorties sont des estimations image, pas une mesure certifiée sans calibration.
"""
from __future__ import annotations
import cv2
import numpy as np


def local_tangent_normal(skeleton: np.ndarray, x: int, y: int, radius: int = 8):
    """Estime tangente et normale par PCA des pixels du squelette dans un voisinage."""
    sk = skeleton > 0
    y0, y1 = max(0, y-radius), min(sk.shape[0], y+radius+1)
    x0, x1 = max(0, x-radius), min(sk.shape[1], x+radius+1)
    yy, xx = np.where(sk[y0:y1, x0:x1])
    pts = np.column_stack((xx+x0, yy+y0)).astype(np.float64)
    if len(pts) < 3:
        raise ValueError("Squelette insuffisant autour du point sélectionné.")
    pts -= pts.mean(axis=0)
    cov = np.cov(pts.T)
    vals, vecs = np.linalg.eigh(cov)
    tangent = vecs[:, np.argmax(vals)]
    tangent /= np.linalg.norm(tangent) + 1e-12
    normal = np.array([-tangent[1], tangent[0]])
    return tangent, normal


def local_ecc_displacement(ref_gray: np.ndarray, target_gray: np.ndarray,
                           point_xy: tuple[int, int], half_size: int = 40):
    """Estime une translation locale sous-pixel par ECC sur une fenêtre."""
    x, y = map(int, point_xy)
    h, w = ref_gray.shape[:2]
    x0, x1 = max(0, x-half_size), min(w, x+half_size+1)
    y0, y1 = max(0, y-half_size), min(h, y+half_size+1)
    a = ref_gray[y0:y1, x0:x1]
    b = target_gray[y0:y1, x0:x1]
    if a.shape != b.shape or min(a.shape) < 16:
        raise ValueError("Fenêtre locale trop petite ou images incompatibles.")
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
    try:
        score, warp = cv2.findTransformECC(a.astype(np.float32)/255.0,
                                           b.astype(np.float32)/255.0,
                                           warp, cv2.MOTION_TRANSLATION,
                                           criteria, None, 1)
    except cv2.error as exc:
        raise ValueError(f"ECC local non convergent : {exc}") from exc
    return np.array([warp[0, 2], warp[1, 2]], dtype=float), float(score)


def project_relative_motion(displacement_a, displacement_b, tangent, normal,
                             mm_per_px=None):
    """Projette le déplacement relatif sur normale (ouverture) et tangente (glissement)."""
    rel = np.asarray(displacement_b, float) - np.asarray(displacement_a, float)
    scale = float(mm_per_px) if mm_per_px else 1.0
    unit = "mm" if mm_per_px else "px"
    return {
        "opening": float(np.dot(rel, normal) * scale),
        "slip": float(np.dot(rel, tangent) * scale),
        "unit": unit,
        "relative_vector": rel.tolist(),
        "calibrated": mm_per_px is not None,
    }
