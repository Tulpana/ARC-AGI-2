"""Palette rescue utilities for ARC solver gating.

These helpers are intentionally lightweight so they can be imported both from
unit tests and from the production solver without pulling additional
dependencies. They operate on NumPy arrays to make palette manipulation
predictable and easy to reason about.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

try:  # pragma: no cover - allow import when NumPy is missing
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - fallback for non-NumPy runtimes
    np = None  # type: ignore[assignment]

ArrayLike = Any


def palette_rescue(
    cands: Sequence[ArrayLike],
    gate_ctx: Mapping[str, object],
    *,
    k_floor: int = 2,
    keep_ratio: float = 0.6,
) -> list[ArrayLike]:
    """Ensure the candidate pool preserves at least ``k_floor`` colours."""

    if np is None:
        raise RuntimeError("palette_rescue requires NumPy to be available")

    if k_floor <= 1:
        return [np.array(c, copy=True) for c in cands]

    test_colors = [
        int(c)
        for c in gate_ctx.get("test_input_colors", [])  # type: ignore[arg-type]
        if int(c) != 0
    ]
    pool = [_ensure_uint8(c) for c in cands if c is not None]

    if not pool:
        base = gate_ctx.get("baseline_pred")
        if base is None:
            return []
        base_arr = _ensure_uint8(base)
        seeded = _seed_palette_variant(base_arr, test_colors)
        return [seeded]

    pool_colors = set().union(*(colors_of(z) for z in pool))
    if len(pool_colors) >= k_floor:
        return [arr.copy() for arr in pool]

    seeded: list[ArrayLike] = []
    for z in pool:
        need = [c for c in test_colors if c not in colors_of(z)]
        if not need:
            seeded.append(z.copy())
            continue
        seeded.append(_blend_in_colors(z, need, keep_ratio=keep_ratio))
    return seeded


def colors_of(z: ArrayLike) -> set[int]:
    """Return the non-zero colour set of a grid."""

    if np is None:
        raise RuntimeError("palette_rescue requires NumPy to be available")
    return {int(v) for v in np.unique(z) if int(v) != 0}


def _ensure_uint8(grid):
    if np is None:
        raise RuntimeError("palette_rescue requires NumPy to be available")
    arr = np.asarray(grid, dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError("palette_rescue expects 2D grids")
    return arr.copy()


def _seed_palette_variant(base: ArrayLike, colors: Sequence[int]) -> ArrayLike:
    if np is None:
        raise RuntimeError("palette_rescue requires NumPy to be available")
    out = base.copy()
    if out.size == 0 or not colors:
        return out

    ys, xs = np.nonzero(out)
    if ys.size == 0:
        y0, x0 = 0, 0
    else:
        y0, x0 = int(ys[0]), int(xs[0])

    h, w = out.shape
    for idx, color in enumerate(colors):
        y = max(0, min(h - 1, y0 + (idx % 3)))
        x = max(0, min(w - 1, x0 + (idx // 3)))
        y1 = min(h, y + 2)
        x1 = min(w, x + 2)
        out[y:y1, x:x1] = np.uint8(color)
    return out


def _blend_in_colors(z: ArrayLike, need: Sequence[int], *, keep_ratio: float) -> ArrayLike:
    if np is None:
        raise RuntimeError("palette_rescue requires NumPy to be available")
    out = z.copy()
    background_mask = out == 0

    if not np.any(background_mask):
        counts = Counter(int(v) for v in out.flatten() if int(v) not in need)
        if counts:
            replace_colour = max(counts.items(), key=lambda kv: kv[1])[0]
            background_mask = out == replace_colour
        else:
            background_mask = np.ones_like(out, dtype=bool)

    coords = np.argwhere(background_mask)
    if coords.size == 0:
        return out

    coords_list = [tuple(map(int, pt)) for pt in coords]
    base_limit = int(len(coords_list) * max(0.0, 1.0 - keep_ratio))
    limit = max(len(need), base_limit)
    if limit <= 0 or not need:
        return out

    # Never modify more cells than we actually have available.
    limit = min(limit, len(coords_list))

    placed = 0
    colors_placed: set[int] = set()
    coord_idx = 0

    # First pass: guarantee that each colour gets at least one dedicated cell
    for color in need:
        if placed >= limit or coord_idx >= len(coords_list):
            break
        y, x = coords_list[coord_idx]
        out[y, x] = np.uint8(color)
        placed += 1
        coord_idx += 1
        colors_placed.add(color)

    # Second pass: if we still have quota to meet the keep_ratio requirement,
    # continue filling remaining background cells using a round-robin schedule.
    if placed < limit and coord_idx < len(coords_list) and need:
        while placed < limit and coord_idx < len(coords_list):
            color = need[placed % len(need)]
            y, x = coords_list[coord_idx]
            out[y, x] = np.uint8(color)
            colors_placed.add(color)
            placed += 1
            coord_idx += 1

    # If we ran out of unique cells before giving each colour at least one spot,
    # cycle through the available coordinates again so that no colour is omitted.
    if len(colors_placed) < len(need):
        remaining = [c for c in need if c not in colors_placed]
        for idx, color in enumerate(remaining):
            y, x = coords_list[idx % len(coords_list)]
            out[y, x] = np.uint8(color)
            colors_placed.add(color)
            if len(colors_placed) == len(need):
                break
    return out


__all__ = ["palette_rescue"]
