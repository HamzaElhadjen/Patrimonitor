"""Extraction descriptive d'un graphe de fissures depuis un masque binaire."""
from __future__ import annotations
import cv2
import numpy as np
from skimage.morphology import skeletonize


def extract_crack_graph(mask):
    """Retourne nœuds et statistiques de segments approximatifs sur squelette."""
    binary = (mask > 0).astype(np.uint8)
    if not binary.any():
        return {"nodes": [], "edges": [], "summary": {"length_px": 0, "branches": 0}}
    sk = skeletonize(binary > 0).astype(np.uint8)
    # Comptage des voisins 8-connexes.
    neigh = cv2.filter2D(sk, cv2.CV_16S, np.ones((3, 3), np.uint8))
    degree = neigh - sk
    ys, xs = np.where((sk > 0) & ((degree == 1) | (degree >= 3)))
    nodes = [{"id": i, "x": int(x), "y": int(y),
              "type": "extremite" if degree[y, x] == 1 else "jonction",
              "degree": int(degree[y, x])} for i, (x, y) in enumerate(zip(xs, ys))]
    # Les arêtes exactes nécessitent un traçage topologique; ici, métriques globales explicites.
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    widths = 2.0 * dist[sk > 0]
    length = int(sk.sum())
    edges = []
    if length:
        edges.append({
            "id": 0, "length_px_approx": length,
            "mean_width_px": float(widths.mean()) if len(widths) else 0.0,
            "max_width_px": float(widths.max()) if len(widths) else 0.0,
            "orientation_deg_approx": None,
            "note": "Agrégat global, pas encore un appariement segment-à-segment."
        })
    return {"nodes": nodes, "edges": edges,
            "summary": {"length_px": length,
                        "endpoints": sum(n["type"] == "extremite" for n in nodes),
                        "junctions": sum(n["type"] == "jonction" for n in nodes),
                        "branches": sum(n["type"] == "jonction" for n in nodes)}}
