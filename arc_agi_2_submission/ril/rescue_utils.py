"""Lightweight rescue helpers shared by gating heuristics."""

from __future__ import annotations

from typing import List, Tuple

Grid = List[List[int]]


def get_bounding_box(grid: Grid) -> Tuple[int, int, int, int]:
    """Return (min_r, min_c, max_r, max_c) for non-zero cells in ``grid``."""
    if not grid or not grid[0]:
        return (0, 0, 0, 0)
    height = len(grid)
    width = len(grid[0])
    min_r, min_c = height, width
    max_r, max_c = -1, -1
    for r in range(height):
        row = grid[r]
        for c in range(width):
            if row[c] != 0:
                if r < min_r:
                    min_r = r
                if c < min_c:
                    min_c = c
                if r > max_r:
                    max_r = r
                if c > max_c:
                    max_c = c
    if max_r < 0 or max_c < 0:
        return (0, 0, 0, 0)
    return (min_r, min_c, max_r, max_c)
