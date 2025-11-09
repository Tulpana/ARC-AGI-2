"""Delta alignment utilities for ARC grids."""

from __future__ import annotations

from ril.utils.safe_np import np


def shift(x: np.ndarray, dy: int, dx: int, fill: int = 0) -> np.ndarray:
    """Shift ``x`` by the given offsets, filling uncovered cells with ``fill``."""
    array = np.asarray(x)
    if array.ndim != 2:
        raise ValueError("shift expects a 2D array")
    height, width = array.shape
    out = np.full_like(array, fill)

    if height == 0 or width == 0:
        return out

    y_src_start = max(0, -dy)
    y_dst_start = max(0, dy)
    x_src_start = max(0, -dx)
    x_dst_start = max(0, dx)
    overlap_h = height - abs(dy)
    overlap_w = width - abs(dx)

    if overlap_h > 0 and overlap_w > 0:
        out[y_dst_start : y_dst_start + overlap_h, x_dst_start : x_dst_start + overlap_w] = (
            array[y_src_start : y_src_start + overlap_h, x_src_start : x_src_start + overlap_w]
        )

    return out


def align_and_delta(
    x: np.ndarray,
    y: np.ndarray,
    *,
    microshift: bool = True,
    max_shift: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Align ``x`` to ``y`` and compute delta masks.

    Returns a tuple containing the aligned ``x`` followed by masks representing
    preserved pixels, additions, deletions, and recolorings.
    """

    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    if x_arr.ndim != 2 or y_arr.ndim != 2:
        raise ValueError("align_and_delta expects 2D arrays")

    target_h, target_w = y_arr.shape
    aligned = np.zeros((target_h, target_w), dtype=y_arr.dtype)

    src_h, src_w = x_arr.shape
    copy_h = min(src_h, target_h)
    copy_w = min(src_w, target_w)
    if copy_h > 0 and copy_w > 0:
        aligned[:copy_h, :copy_w] = x_arr[:copy_h, :copy_w]

    if microshift and max_shift > 0:
        best_shift = (0, 0)
        best_iou = -1.0
        y_nonzero = y_arr != 0
        aligned_nonzero = aligned != 0
        union = np.logical_or(aligned_nonzero, y_nonzero)
        if union.any():
            best_iou = np.logical_and(aligned_nonzero, y_nonzero).sum() / union.sum()
        for dy in range(-max_shift, max_shift + 1):
            for dx in range(-max_shift, max_shift + 1):
                if dy == 0 and dx == 0:
                    continue
                shifted = shift(aligned, dy, dx)
                shifted_nonzero = shifted != 0
                inter = np.logical_and(shifted_nonzero, y_nonzero).sum()
                uni = np.logical_or(shifted_nonzero, y_nonzero).sum()
                iou = inter / uni if uni else 1.0
                if iou > best_iou:
                    best_iou = iou
                    best_shift = (dy, dx)
        if best_shift != (0, 0):
            aligned = shift(aligned, *best_shift)

    preserved = (aligned == y_arr) & (y_arr != 0)
    additions = (aligned == 0) & (y_arr != 0)
    deletions = (aligned != 0) & (y_arr == 0)
    recolors = (aligned != y_arr) & (aligned != 0) & (y_arr != 0)

    return aligned, preserved, additions, deletions, recolors
