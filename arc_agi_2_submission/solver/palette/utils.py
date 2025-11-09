"""Palette utility helpers for finisher post-processing."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence, Tuple

from ril.utils.safe_np import np

GridLike = Sequence[Sequence[int]]


def palette_hist(grid: GridLike) -> dict[int, float]:
    """Return the normalised colour histogram for ``grid``."""

    counts: Counter[int] = Counter()
    for row in grid:
        counts.update(row)
    total = sum(counts.values()) or 1
    return {colour: counts[colour] / total for colour in counts}


def palette_set(grid: GridLike) -> set[int]:
    """Return the set of colours present in ``grid``."""

    return {value for row in grid for value in row}


def palette_completeness(candidate: GridLike, allowed: Iterable[int]) -> float:
    """Return the ratio of allowed colours present in ``candidate``."""

    allowed_set = set(allowed)
    if not allowed_set:
        return 1.0
    present = palette_set(candidate)
    return len(present & allowed_set) / len(allowed_set)


def extra_colour_stats(candidate: GridLike, allowed: Iterable[int]) -> Tuple[float, int, int]:
    """Return extra-colour rate together with raw cell counts."""

    allowed_set = set(allowed)
    total_cells = 0
    extra_cells = 0

    for row in candidate:
        if not isinstance(row, Sequence):
            continue
        row_len = len(row)
        total_cells += row_len
        for value in row:
            try:
                colour = int(value)
            except Exception:
                colour = value
            if allowed_set and colour not in allowed_set:
                extra_cells += 1

    rate = float(extra_cells) / float(total_cells or 1)
    return rate, extra_cells, total_cells


def extra_colour_rate(candidate: GridLike, allowed: Iterable[int]) -> float:
    """Return the fraction of colours in ``candidate`` that are not allowed."""

    rate, _extra, _total = extra_colour_stats(candidate, allowed)
    return rate


def enforce_palette_completion(
    candidate: list[list[int]],
    birth_palette: Sequence[int],
    paint_colour: int | None = None,
) -> list[list[int]]:
    """Ensure that every colour in ``birth_palette`` appears in ``candidate``.

    Missing colours are filled greedily by repainting the longest contiguous run
    of ``0``s in each row.  If no background run exists, fall back to repainting
    the first cell whose colour is not part of ``birth_palette``.  When
    ``paint_colour`` is provided we repaint with that colour; otherwise we use
    the missing colour itself.
    """

    height = len(candidate)
    width = max((len(row) for row in candidate if isinstance(row, Sequence)), default=0)
    if height == 0 or width == 0:
        return candidate

    present = palette_set(candidate)
    missing = [colour for colour in birth_palette if colour not in present]
    if not missing:
        return candidate

    for target in missing:
        best_length = 0
        best_coords: tuple[int, int, int] | None = None

        for row_index, row in enumerate(candidate):
            if not isinstance(row, Sequence):
                continue
            row_len = len(row)
            column = 0
            while column < row_len:
                if row[column] == 0:
                    start = column
                    while column < row_len and row[column] == 0:
                        column += 1
                    run_length = column - start
                    if run_length > best_length:
                        best_length = run_length
                        best_coords = (row_index, start, column)
                else:
                    column += 1

        fill_colour = target if paint_colour is None else paint_colour
        if best_coords is None:
            replaced = False
            for row_index, row in enumerate(candidate):
                if not isinstance(row, Sequence):
                    continue
                for column, value in enumerate(row):
                    if value not in birth_palette:
                        candidate[row_index][column] = fill_colour
                        replaced = True
                        break
                if replaced:
                    break
        else:
            row_index, start, end = best_coords
            row = candidate[row_index]
            row_len = len(row)
            for column in range(start, min(end, row_len)):
                row[column] = fill_colour

    return candidate


def prune_extra_colours(
    candidate: list[list[int]],
    allowed: Iterable[int],
) -> list[list[int]]:
    """Replace colours not in ``allowed`` with a fallback colour."""

    allowed_set = set(allowed)
    if not allowed_set:
        return candidate

    arr = np.asarray(candidate)
    if arr.size == 0:
        return candidate

    mask_extra = ~np.isin(arr, list(allowed_set))
    if not mask_extra.any():
        return candidate

    fallback = _mode_colour_of_local_neighbourhood(arr, mask_extra, allowed_set)

    work = arr.copy()
    work[mask_extra] = fallback

    replaced = work.tolist()
    for row_index, row in enumerate(replaced):
        for column_index, value in enumerate(row):
            candidate[row_index][column_index] = int(value)

    return candidate


def _mode_colour_of_local_neighbourhood(
    grid: np.ndarray,
    mask: np.ndarray,
    allowed: set[int],
) -> int:
    """Return the dominant allowed colour surrounding ``mask`` cells."""

    if grid.size == 0 or not mask.any():
        return next(iter(allowed), 0)

    counts: Counter[int] = Counter()
    height, width = grid.shape

    for r, c in np.argwhere(mask):
        r0 = max(0, int(r) - 1)
        r1 = min(height, int(r) + 2)
        c0 = max(0, int(c) - 1)
        c1 = min(width, int(c) + 2)

        neighbourhood = grid[r0:r1, c0:c1]
        neighbourhood_mask = mask[r0:r1, c0:c1]

        for value, is_extra in zip(neighbourhood.flat, neighbourhood_mask.flat):
            colour = int(value)
            if not is_extra and colour in allowed:
                counts[colour] += 1

    if counts:
        return max(counts.items(), key=lambda item: (item[1], -item[0]))[0]

    allowed_counts: Counter[int] = Counter(
        int(value) for value in grid.flat if int(value) in allowed
    )
    if allowed_counts:
        return max(allowed_counts.items(), key=lambda item: (item[1], -item[0]))[0]

    return next(iter(allowed), 0)
