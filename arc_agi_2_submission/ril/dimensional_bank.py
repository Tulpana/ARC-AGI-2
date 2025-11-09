"""Dimensional hypothesis seeding utilities.

This module mirrors ``arc-agi-2-entry-2/ril/dimensional_bank.py`` so that the
Kaggle submission package has access to the same block-wise downsampling
helpers used during development.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

from ril.np_compat import NP

np = NP  # type: ignore

GridLike = Sequence[Sequence[int]]
Array = "np.ndarray"


def _to_array(grid: GridLike) -> Array:
    """Best-effort conversion of ``grid`` into a 2-D numpy array."""

    try:
        arr = np.asarray(grid)
    except Exception:  # pragma: no cover - defensive guard
        arr = np.empty((0, 0), dtype=int)
    if arr.ndim != 2:
        arr = np.reshape(arr, (arr.shape[0], -1)) if arr.size else np.empty((0, 0), dtype=int)
    return arr


def block_mode_downsample(grid: GridLike, block_h: int, block_w: int) -> Array:
    """Reduce ``grid`` using majority pooling with ``block_h``×``block_w`` tiles."""

    if block_h <= 0 or block_w <= 0:
        raise ValueError("block sizes must be positive integers")

    arr = _to_array(grid)
    if arr.size == 0:
        return np.empty((0, 0), dtype=arr.dtype)

    H, W = arr.shape
    if H % block_h != 0 or W % block_w != 0:
        raise ValueError("grid dimensions must be divisible by block size")

    gh, gw = H // block_h, W // block_w
    out = np.empty((gh, gw), dtype=arr.dtype)
    for y in range(gh):
        row_start = y * block_h
        row_end = row_start + block_h
        for x in range(gw):
            col_start = x * block_w
            col_end = col_start + block_w
            block = arr[row_start:row_end, col_start:col_end]
            vals, counts = np.unique(block, return_counts=True)
            out[y, x] = vals[int(np.argmax(counts))]
    return out


def dimensional_hypotheses(
    grid: GridLike,
    target_sizes: Iterable[Tuple[int, int]] = ((9, 3), (3, 9), (4, 4), (4, 5), (3, 7)),
) -> List[Array]:
    """Return block-mode downsampled grids for shapes compatible with ``grid``."""

    arr = _to_array(grid)
    if arr.size == 0:
        return []

    H, W = arr.shape
    cands: List[Array] = []
    for h, w in target_sizes:
        if h <= 0 or w <= 0:
            continue
        if H % h == 0 and W % w == 0:
            try:
                cand = block_mode_downsample(arr, H // h, W // w)
            except ValueError:
                continue
            cands.append(cand)
    return cands


__all__ = ["block_mode_downsample", "dimensional_hypotheses"]
