"""Outils d'incertitude expérimentaux. Les probabilités ne sont pas calibrées par défaut."""
from __future__ import annotations
import cv2
import numpy as np


def bootstrap_homography(src_pts, dst_pts, iterations=100, seed=42,
                         ransac_threshold=5.0):
    """Bootstrap des correspondances et retourne les homographies valides."""
    src = np.asarray(src_pts, np.float32).reshape(-1, 2)
    dst = np.asarray(dst_pts, np.float32).reshape(-1, 2)
    if len(src) < 4 or len(src) != len(dst):
        raise ValueError("Il faut au moins quatre paires de points correspondants.")
    rng = np.random.default_rng(seed)
    mats = []
    for _ in range(int(iterations)):
        idx = rng.integers(0, len(src), len(src))
        if len(np.unique(idx)) < 4:
            continue
        H, _ = cv2.findHomography(src[idx], dst[idx], cv2.RANSAC, ransac_threshold)
        if H is not None and np.isfinite(H).all():
            mats.append(H / (H[2, 2] if abs(H[2, 2]) > 1e-12 else 1.0))
    if len(mats) < max(5, iterations // 10):
        raise ValueError("Bootstrap instable : trop peu d'homographies valides.")
    return np.stack(mats)


def local_tolerance_map(shape, rmse_px, base_px=1.0, max_px=20.0):
    """Proxy prudent : tolérance uniforme liée au RMSE global, non une incertitude locale."""
    tol = float(np.clip(base_px + 2.0 * max(0.0, rmse_px), base_px, max_px))
    return np.full(shape[:2], tol, dtype=np.float32)


def compare_masks_with_tolerance(mask_a, mask_b, tolerance_map):
    """Carte de changement géométrique simple et score heuristique, non probabilité calibrée."""
    a = (mask_a > 0).astype(np.uint8) * 255
    b = (mask_b > 0).astype(np.uint8) * 255
    tol = float(np.nanmedian(tolerance_map))
    k = max(1, int(np.ceil(tol)))
    dilated = cv2.dilate(a, np.ones((2*k+1, 2*k+1), np.uint8))
    new = cv2.bitwise_and(b, cv2.bitwise_not(dilated))
    # Score uniquement heuristique : ne pas le présenter comme probabilité.
    score = float((new > 0).sum() / max(1, (b > 0).sum()))
    return new, {"tolerance_proxy_px": tol, "new_pixel_fraction_heuristic": score,
                 "statistical_significance_established": False}
