"""Minimal NumPy compatibility shim for ARC RIL operations."""
from __future__ import annotations

import os
from collections import deque
from typing import List, Sequence, Tuple

try:  # optional dependency
    import numpy as _np_mod  # type: ignore
except Exception:  # pragma: no cover - import guard
    _np_mod = None  # type: ignore
    _HAVE_NUMPY = False
else:
    if getattr(_np_mod, "__arc_numpy_stub__", False):  # pragma: no cover - shim detection
        _np_mod = None  # type: ignore[assignment]
        _HAVE_NUMPY = False
    else:
        _HAVE_NUMPY = True

USE_NUMPY = _HAVE_NUMPY and os.environ.get("RIL_NO_NUMPY", "0") != "1"
if USE_NUMPY:
    _np = _np_mod  # type: ignore
else:
    _np = None  # type: ignore

HAVE_NUMPY = _HAVE_NUMPY

Grid = List[List[int]]
Mask = List[List[bool]]


def has_numpy() -> bool:
    return USE_NUMPY


# historical alias used by downstream code
HAS_NUMPY = has_numpy()


# ---------- grid primitives (both backends) ----------
def asgrid(x: Sequence[Sequence[int]]) -> Grid:
    """Ensure grid is list[list[int]]. Accepts list-of-lists or numpy array."""
    if _np is not None and hasattr(x, "tolist"):
        return [[int(v) for v in row] for row in x.tolist()]
    return [[int(v) for v in row] for row in x]


def shape(g: Sequence[Sequence[int]]) -> Tuple[int, int]:
    grid = asgrid(g)
    return (len(grid), len(grid[0]) if grid else 0)


def clone(g: Sequence[Sequence[int]], fill: int = 0) -> Grid:
    h, w = shape(g)
    return [[fill for _ in range(w)] for _ in range(h)]


def eq(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]]) -> bool:
    return asgrid(a) == asgrid(b)


def rotate90(g: Sequence[Sequence[int]], k: int = 1) -> Grid:
    grid = asgrid(g)
    if not grid:
        return []
    k %= 4
    for _ in range(k):
        grid = [list(row) for row in zip(*grid[::-1])]
    return grid


def flip(g: Sequence[Sequence[int]], axis: str = "h") -> Grid:
    grid = asgrid(g)
    if axis in ("h", "horizontal"):
        return [row[::-1] for row in grid]
    return grid[::-1]


def pad(
    g: Sequence[Sequence[int]],
    pad_top: int = 0,
    pad_bottom: int = 0,
    pad_left: int = 0,
    pad_right: int = 0,
    fill: int = 0,
) -> Grid:
    grid = asgrid(g)
    h, w = shape(grid)
    core = [[fill] * pad_left + row + [fill] * pad_right for row in grid]
    fullw = w + pad_left + pad_right
    top_rows = [[fill] * fullw for _ in range(pad_top)]
    bottom_rows = [[fill] * fullw for _ in range(pad_bottom)]
    return top_rows + core + bottom_rows


def bbox_of_color(g: Sequence[Sequence[int]], color: int) -> Tuple[int, int, int, int] | None:
    grid = asgrid(g)
    h, w = shape(grid)
    ys = [y for y in range(h) for x in range(w) if grid[y][x] == color]
    xs = [x for y in range(h) for x in range(w) if grid[y][x] == color]
    if not xs:
        return None
    return (min(ys), min(xs), max(ys), max(xs))


def unique_colors(g: Sequence[Sequence[int]]) -> List[int]:
    grid = asgrid(g)
    return sorted({c for row in grid for c in row})


def count_nonzero(g: Sequence[Sequence[int]]) -> int:
    grid = asgrid(g)
    return sum(1 for row in grid for c in row if c != 0)


# 4-neighborhood connected components (color-aware)
def components(g: Sequence[Sequence[int]], ignore_color: int = 0):
    grid = asgrid(g)
    h, w = shape(grid)
    seen = [[False] * w for _ in range(h)]
    comps = []
    for y in range(h):
        for x in range(w):
            col = grid[y][x]
            if col == ignore_color or seen[y][x]:
                continue
            q: deque[Tuple[int, int]] = deque([(y, x)])
            seen[y][x] = True
            pixels = []
            while q:
                cy, cx = q.popleft()
                pixels.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny][nx] and grid[ny][nx] == col:
                        seen[ny][nx] = True
                        q.append((ny, nx))
            comps.append({"color": col, "pixels": pixels})
    return comps


# Backend bridge (used by callers that *prefer* numpy when present)
def to_numpy(g: Sequence[Sequence[int]]):  # pragma: no cover - exercised in NumPy envs
    if _np is None:
        raise RuntimeError("NumPy unavailable")
    return _np.array(asgrid(g), dtype=int)


# Backwards-compat helpers for legacy imports
NP = _np


def require_numpy(feature_name: str) -> None:
    if _np is None:
        raise RuntimeError(f"{feature_name} requires NumPy but it is not available in this runtime.")


__all__ = [
    "USE_NUMPY",
    "HAVE_NUMPY",
    "NP",
    "has_numpy",
    "HAS_NUMPY",
    "asgrid",
    "shape",
    "clone",
    "eq",
    "rotate90",
    "flip",
    "pad",
    "bbox_of_color",
    "unique_colors",
    "count_nonzero",
    "components",
    "to_numpy",
    "require_numpy",
]
