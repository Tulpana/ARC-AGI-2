"""Simple motif crop adapter for ARC candidate generation."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

Grid = Sequence[Sequence[int]]
GridLike = Iterable[Iterable[int]]


def _bbox_of_nonzero(grid: Grid) -> Tuple[int, int, int, int]:
    """Return the bounding box of non-zero cells as (y0, x0, y1, x1)."""

    if not grid:
        return (0, 0, -1, -1)
    height = len(grid)
    width = len(grid[0]) if grid[0] else 0
    min_y, min_x = height, width
    max_y, max_x = -1, -1
    for y, row in enumerate(grid):
        for x, value in enumerate(row):
            if value != 0:
                if y < min_y:
                    min_y = y
                if x < min_x:
                    min_x = x
                if y > max_y:
                    max_y = y
                if x > max_x:
                    max_x = x
    if max_y < min_y or max_x < min_x:
        return (0, 0, -1, -1)
    return (min_y, min_x, max_y, max_x)


def _crop(grid: Grid, y0: int, x0: int, y1: int, x1: int) -> List[List[int]]:
    """Return a sub-grid defined by the inclusive bounding box."""

    return [list(row[x0 : x1 + 1]) for row in grid[y0 : y1 + 1]]


def generate_motif_crop(
    training_pairs: Iterable[Tuple[GridLike, GridLike]],
    test_grid: GridLike,
) -> List[List[List[int]]]:
    """Generate a cropped candidate using the tightest training bbox."""

    boxes: List[Tuple[int, int, int, int]] = []
    for pair in training_pairs:
        if not isinstance(pair, tuple) or len(pair) != 2:
            continue
        _, output_grid = pair
        if not isinstance(output_grid, Sequence):
            continue
        bbox = _bbox_of_nonzero(output_grid)  # type: ignore[arg-type]
        if bbox[2] >= bbox[0] and bbox[3] >= bbox[1]:
            boxes.append(bbox)
    if not boxes:
        return []

    y0 = min(b[0] for b in boxes)
    x0 = min(b[1] for b in boxes)
    y1 = min(b[2] for b in boxes)
    x1 = min(b[3] for b in boxes)

    if y1 < y0 or x1 < x0:
        return []

    if not isinstance(test_grid, Sequence) or not test_grid:
        return []
    height = len(test_grid)
    width = len(test_grid[0]) if test_grid[0] else 0
    if height == 0 or width == 0:
        return []

    y0 = max(0, min(y0, height - 1))
    x0 = max(0, min(x0, width - 1))
    y1 = max(y0, min(y1, height - 1))
    x1 = max(x0, min(x1, width - 1))

    return [_crop(test_grid, y0, x0, y1, x1)]  # type: ignore[arg-type]
