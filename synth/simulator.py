"""Générateur déterministe de fissures synthétiques sur textures simples."""
from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path


def generate_crack_image(width=640, height=480, seed=42, growth=0.0,
                         texture="stone", jpeg_quality=95):
    """Génère une image BGR, masque vérité terrain et paramètres de simulation."""
    rng = np.random.default_rng(seed)
    base = {"stone": (155, 157, 160), "brick": (125, 105, 95),
            "mortar": (175, 175, 168), "plaster": (185, 184, 178)}.get(texture, (160, 160, 160))
    noise = rng.normal(0, 8, (height, width, 1))
    img = np.clip(np.array(base, np.float32)[None, None, :] + noise, 0, 255).astype(np.uint8)
    # Texture de joints / grain rudimentaire.
    for _ in range(45):
        x, y = int(rng.integers(width)), int(rng.integers(height))
        cv2.circle(img, (x, y), int(rng.integers(1, 5)), (base[0]-8, base[1]-8, base[2]-8), -1)
    mask = np.zeros((height, width), np.uint8)
    x, y = int(width*0.15), int(height*0.2)
    pts = [(x, y)]
    steps = int(height*0.55)
    for i in range(steps):
        x += int(rng.choice([-2, -1, 0, 1, 2]))
        y += 1
        x = int(np.clip(x, 5, width-6))
        if y >= height-5:
            break
        pts.append((x, y))
        if rng.random() < 0.025:
            bx = int(np.clip(x + rng.integers(-30, 31), 5, width-6))
            by = int(np.clip(y + rng.integers(8, 30), 5, height-6))
            cv2.line(mask, (x, y), (bx, by), 255, int(rng.integers(1, 3)))
    thickness = max(1, int(1 + growth))
    if len(pts) > 1:
        cv2.polylines(mask, [np.asarray(pts, np.int32)], False, 255, thickness=thickness)
    # Rendu de la fissure sombre, avec bords irréguliers.
    crack = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    img[crack > 0] = (35, 35, 38)
    if rng.random() < 0.5:
        img = cv2.GaussianBlur(img, (3, 3), float(rng.uniform(0.1, 1.0)))
    if rng.random() < 0.5:
        img = np.clip(img.astype(np.int16) + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    encode = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    ok, data = cv2.imencode(".jpg", img, encode)
    if ok:
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img, mask, {"seed": seed, "growth_parameter": growth, "texture": texture,
                       "width": width, "height": height, "jpeg_quality": jpeg_quality,
                       "synthetic_only": True}


def generate_dataset(output_dir, count=20, seed=42):
    """Exporte images, masques et annotations JSON reproductibles."""
    import json
    out = Path(output_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    metadata = []
    textures = ["stone", "brick", "mortar", "plaster"]
    for i in range(int(count)):
        params = {"seed": int(rng.integers(0, 2**31-1)),
                  "growth": float(rng.uniform(0, 3)),
                  "texture": textures[i % len(textures)]}
        img, mask, meta = generate_crack_image(seed=params["seed"], growth=params["growth"],
                                               texture=params["texture"])
        cv2.imwrite(str(out / "images" / f"{i:05d}.jpg"), img)
        cv2.imwrite(str(out / "masks" / f"{i:05d}.png"), mask)
        metadata.append(meta)
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
