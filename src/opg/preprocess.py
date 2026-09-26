"""Stage 1: image preprocessing (plan tasks 1.8 and 2.1, owner: Shubh).

Keep every step optional and switchable, so the ablation in task 4.4 can
measure what each one contributes.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_gray(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def clahe(img: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation (used by the Six-class dataset)."""
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile)).apply(img)


def crop_black_border(img: np.ndarray, threshold: int = 10) -> tuple[np.ndarray, tuple[int, int]]:
    """Remove uniform dark borders. Returns the crop and its (x, y) offset in the original."""
    mask = img > threshold
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return img, (0, 0)
    x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    return img[y1:y2, x1:x2], (int(x1), int(y1))


def resize_long_side(img: np.ndarray, long_side: int = 1024) -> tuple[np.ndarray, float]:
    """Resize keeping the aspect ratio. Returns the image and the scale factor applied."""
    h, w = img.shape[:2]
    scale = long_side / max(h, w)
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA), scale


def preprocess(path: str | Path, use_clahe: bool = True, crop: bool = True) -> dict:
    """Full stage-1 pipeline. Keeps the offset so boxes can be mapped back to the original."""
    img = load_gray(path)
    offset = (0, 0)
    if crop:
        img, offset = crop_black_border(img)
    if use_clahe:
        img = clahe(img)
    return {"image": img, "offset": offset}
