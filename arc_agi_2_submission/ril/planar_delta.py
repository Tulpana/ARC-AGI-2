# ril/planar_delta.py
from __future__ import annotations

import os
from typing import Dict, List, Tuple

from . import np_compat as npc

USE_DELTA = os.getenv("ARC_PLANAR_DELTA", "1") not in ("0", "false", "False")


def _compute_with_numpy(A, B, max_shift):
    np = npc.NP  # type: ignore
    if np is None:
        npc.require_numpy("planar_delta")
        np = npc.NP  # type: ignore
    A = np.asarray(A)
    B = np.asarray(B)
    H, W = A.shape
    translations = []
    best = (0, 0, 0.0)
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            # overlapping window
            y0 = max(0, dy)
            y1 = min(H, H + dy)
            x0 = max(0, dx)
            x1 = min(W, W + dx)
            if y0 >= y1 or x0 >= x1:
                continue
            As = A[y0:y1, x0:x1]
            Bs = B[y0 - dy : y1 - dy, x0 - dx : x1 - dx]
            iou = (As == Bs).mean()
            translations.append((dx, dy, float(iou)))
            if iou > best[2]:
                best = (dx, dy, float(iou))

    emergent_mask = (B != A) & (A == 0)
    # crude directional growth scores
    gy = (emergent_mask.any(axis=1)).astype(int)
    gx = (emergent_mask.any(axis=0)).astype(int)
    growth_scores = {
        "down": float(gy[-1]) if gy.size else 0.0,
        "up": float(gy[0]) if gy.size else 0.0,
        "right": float(gx[-1]) if gx.size else 0.0,
        "left": float(gx[0]) if gx.size else 0.0,
    }
    growth_hint = max(growth_scores, key=growth_scores.get)
    return {
        "best": best,
        "translations": translations,
        "emergent_mask": emergent_mask.astype(int),
        "growth_scores": growth_scores,
        "growth_hint": growth_hint,
    }


def _compute_pure_python(A, B, max_shift):
    # A,B are list-of-lists ints
    H = len(A)
    W = len(A[0]) if A else 0

    def eq(a, b):
        return 1 if a == b else 0

    translations, best = [], (0, 0, 0.0)
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            y0 = 0 if dy >= 0 else -dy
            x0 = 0 if dx >= 0 else -dx
            y1 = H - dy if dy >= 0 else H
            x1 = W - dx if dx >= 0 else W
            num = den = 0
            for y in range(y0, y1):
                for x in range(x0, x1):
                    den += 1
                    if A[y][x] == B[y + dy][x + dx]:
                        num += 1
            if den == 0:
                continue
            iou = num / den
            translations.append((dx, dy, iou))
            if iou > best[2]:
                best = (dx, dy, iou)
    # emergent_mask
    emergent = [
        [1 if (B[y][x] != A[y][x] and A[y][x] == 0) else 0 for x in range(W)]
        for y in range(H)
    ]
    top = sum(emergent[0]) if H else 0
    bot = sum(emergent[-1]) if H else 0
    left = sum(row[0] for row in emergent) if W else 0
    right = sum(row[-1] for row in emergent) if W else 0
    scores = {
        "down": float(bot > 0),
        "up": float(top > 0),
        "right": float(right > 0),
        "left": float(left > 0),
    }
    hint = max(scores, key=scores.get)
    return {
        "best": best,
        "translations": translations,
        "emergent_mask": emergent,
        "growth_scores": scores,
        "growth_hint": hint,
    }


def compute_planar_delta(A, B, max_shift: int = 6):
    """
    Returns dict with keys: best, translations, emergent_mask, growth_scores, growth_hint.
    Works offline; prefers NumPy if present; pure-Python otherwise.
    """

    if not USE_DELTA:
        return {
            "best": (0, 0, 0.0),
            "translations": [],
            "emergent_mask": [],
            "growth_scores": {"up": 0, "down": 0, "left": 0, "right": 0},
            "growth_hint": "none",
        }
    if npc.NP is not None:
        return _compute_with_numpy(A, B, max_shift)
    # Pure-Python fallback (slower but fine for small grids / smoke tests)
    return _compute_pure_python(A, B, max_shift)


__all__ = ["compute_planar_delta"]
