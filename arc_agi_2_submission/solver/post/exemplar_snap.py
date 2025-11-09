"""Exemplar-guided post-processing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple
import math

from ril.utils.safe_np import np


Grid = Sequence[Sequence[int]]
NormalizedCoord = Tuple[float, float]


@dataclass(frozen=True)
class Component:
    color: int
    pixels: Tuple[Tuple[int, int], ...]

    @property
    def size(self) -> int:
        return len(self.pixels)

    def centroid(self, height: int, width: int) -> NormalizedCoord:
        if not self.pixels:
            return (0.0, 0.0)
        area = len(self.pixels)
        row_mean = sum(p[0] for p in self.pixels) / area
        col_mean = sum(p[1] for p in self.pixels) / area
        denom_r = float(max(height, 1))
        denom_c = float(max(width, 1))
        return (
            float((row_mean + 0.5) / denom_r),
            float((col_mean + 0.5) / denom_c),
        )


def _neighbors4(height: int, width: int, r: int, c: int) -> Iterable[Tuple[int, int]]:
    if r > 0:
        yield (r - 1, c)
    if r + 1 < height:
        yield (r + 1, c)
    if c > 0:
        yield (r, c - 1)
    if c + 1 < width:
        yield (r, c + 1)


def _components(grid: Grid) -> List[Component]:
    arr = np.asarray(grid)
    if arr.ndim != 2 or arr.size == 0:
        return []
    height, width = arr.shape
    seen = np.zeros((height, width), dtype=bool)
    components: List[Component] = []
    for r in range(height):
        for c in range(width):
            if seen[r, c]:
                continue
            color = int(arr[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            pixels: List[Tuple[int, int]] = []
            while stack:
                y, x = stack.pop()
                pixels.append((y, x))
                for ny, nx in _neighbors4(height, width, y, x):
                    if seen[ny, nx]:
                        continue
                    if int(arr[ny, nx]) != color:
                        continue
                    seen[ny, nx] = True
                    stack.append((ny, nx))
            components.append(Component(color=color, pixels=tuple(pixels)))
    return components


def collect_exemplar_signatures(outputs: Iterable[Grid]) -> Dict[int, List[NormalizedCoord]]:
    """Return a mapping colour -> normalised pixel coordinates from exemplars."""

    exemplar_map: Dict[int, List[NormalizedCoord]] = {}
    for grid in outputs:
        arr = np.asarray(grid)
        if arr.ndim != 2 or arr.size == 0:
            continue
        height, width = arr.shape
        if height <= 0 or width <= 0:
            continue
        denom_r = float(height)
        denom_c = float(width)
        for r in range(height):
            row = arr[r]
            for c in range(width):
                try:
                    color = int(row[c])
                except Exception:
                    continue
                coord = (
                    float((r + 0.5) / denom_r),
                    float((c + 0.5) / denom_c),
                )
                exemplar_map.setdefault(color, []).append(coord)
    return exemplar_map


def _nearest_label_from_exemplars(
    coord: NormalizedCoord,
    exemplars: Mapping[int, Sequence[NormalizedCoord]],
    current_color: int,
    *,
    max_dist_sq: float = 0.05,
) -> int:
    if not exemplars:
        return current_color

    best_color = current_color
    best_dist = math.inf
    current_dist = math.inf

    for color, coords in exemplars.items():
        if not coords:
            continue
        colour_best = math.inf
        for ey, ex in coords:
            dy = coord[0] - float(ey)
            dx = coord[1] - float(ex)
            dist = dy * dy + dx * dx
            if dist < colour_best:
                colour_best = dist
        if colour_best >= best_dist:
            if color == current_color and colour_best < current_dist:
                current_dist = colour_best
            continue
        if color == current_color:
            current_dist = colour_best
            best_color = current_color
            best_dist = colour_best
        else:
            best_color = color
            best_dist = colour_best

    if best_color == current_color:
        return current_color

    if best_dist >= max_dist_sq:
        return current_color

    if current_dist < math.inf and best_dist >= current_dist - 1e-9:
        return current_color

    return best_color


def snap_singletons(
    grid: Grid,
    exemplars: Mapping[int, Sequence[NormalizedCoord]] | None,
    *,
    max_dist_sq: float = 0.05,
) -> List[List[int]]:
    """Relabel isolated single-pixel components using exemplar guidance."""

    arr = np.asarray(grid)
    if arr.ndim != 2 or arr.size == 0:
        if hasattr(grid, "tolist"):
            return arr.tolist()  # type: ignore[return-value]
        return [list(row) for row in grid]  # type: ignore[return-value]

    height, width = arr.shape
    out = arr.astype(int).tolist()
    if not exemplars:
        return out

    components = _components(arr)
    for comp in components:
        if comp.size != 1:
            continue
        y, x = comp.pixels[0]
        coord = (
            float((y + 0.5) / float(max(height, 1))),
            float((x + 0.5) / float(max(width, 1))),
        )
        new_color = _nearest_label_from_exemplars(
            coord,
            exemplars,
            int(arr[y, x]),
            max_dist_sq=max_dist_sq,
        )
        if new_color != int(arr[y, x]):
            out[y][x] = int(new_color)
    return out


__all__ = [
    "collect_exemplar_signatures",
    "snap_singletons",
]
