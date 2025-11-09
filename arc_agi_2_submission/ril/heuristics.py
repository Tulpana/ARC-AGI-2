"""Stdlib-safe heuristic fallbacks for the ARC RIL router."""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence

Grid = List[List[int]]
Example = Dict[str, object]


def _copy_grid(inp: Grid) -> Grid:
    return [row[:] for row in inp]


def identity(inp: Grid) -> Grid:
    """Return the input grid unchanged."""
    return _copy_grid(inp)


def copy_most_common_color(inp: Grid) -> Grid:
    """Fill the entire canvas with the most frequent color from the input grid."""
    freq: Counter[int] = Counter()
    for row in inp:
        freq.update(int(cell) for cell in row)
    if not freq:
        return _copy_grid(inp)
    color, _ = max(freq.items(), key=lambda kv: kv[1])
    h = len(inp)
    w = len(inp[0]) if h else 0
    return [[int(color) for _ in range(w)] for _ in range(h)]


def tile_small_training_output(train_examples: Sequence[Example], test_inp: Grid):
    """Tile a training output when it fits evenly into the test input."""
    if not test_inp or not isinstance(test_inp, list):
        return None
    Ht = len(test_inp)
    Wt = len(test_inp[0]) if Ht else 0
    for ex in train_examples:
        if not isinstance(ex, dict):
            continue
        out = ex.get("output")
        if not isinstance(out, list) or not out:
            continue
        h = len(out)
        w = len(out[0]) if h else 0
        if h == 0 or w == 0:
            continue
        if Ht % h != 0 or Wt % w != 0:
            continue
        tiled: Grid = []
        for r in range(Ht):
            row = []
            for c in range(Wt):
                row.append(int(out[r % h][c % w]))
            tiled.append(row)
        return tiled
    return None


def heuristic_candidates(train_examples: Sequence[Example], test_input: Grid) -> List[Grid]:
    """Return heuristic grids when the external solver has no answer."""
    cands: List[Grid] = []
    tiled = tile_small_training_output(train_examples, test_input)
    if tiled is not None:
        cands.append(tiled)
    if isinstance(test_input, list):
        cands.append(identity(test_input))
        cands.append(copy_most_common_color(test_input))
    return cands


__all__ = [
    "identity",
    "copy_most_common_color",
    "tile_small_training_output",
    "heuristic_candidates",
]
