"""Connectivity utilities."""

from __future__ import annotations

from typing import List, Tuple

from ril.utils.safe_np import np


BBox = Tuple[int, int, int, int]
ComponentInfo = Tuple[int, BBox, int]


def cc4(mask: np.ndarray) -> tuple[np.ndarray, List[ComponentInfo]]:
    """Label 4-connected components in ``mask``.

    Returns a tuple of the label matrix and a list of component metadata
    (component id, bounding box, size).
    """

    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != 2:
        raise ValueError("cc4 expects a 2D mask")

    height, width = arr.shape
    labels = np.zeros((height, width), dtype=np.int32)
    seen = np.zeros((height, width), dtype=bool)
    components: List[ComponentInfo] = []

    ys, xs = np.nonzero(arr)
    cid = 0

    for y0, x0 in zip(ys, xs):
        if seen[y0, x0]:
            continue
        cid += 1
        stack = [(y0, x0)]
        seen[y0, x0] = True
        labels[y0, x0] = cid

        y_min = y_max = y0
        x_min = x_max = x0
        size = 0

        while stack:
            y, x = stack.pop()
            size += 1
            if y < y_min:
                y_min = y
            if y > y_max:
                y_max = y
            if x < x_min:
                x_min = x
            if x > x_max:
                x_max = x

            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width and not seen[ny, nx] and arr[ny, nx]:
                    seen[ny, nx] = True
                    labels[ny, nx] = cid
                    stack.append((ny, nx))

        components.append((cid, (y_min, x_min, y_max, x_max), size))

    return labels, components
