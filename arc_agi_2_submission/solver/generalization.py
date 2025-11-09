"""Dimension-agnostic rule extraction and application helpers for ARC puzzles.

The utilities in this module implement the "ARC Generalization Directives"
provided in the engineering memo.  They deliberately avoid hard-coded grid
sizes, colour ids or coordinates.  Instead the functions learn local
transformations from the training examples and apply them proportionally to the
geometry of the test input.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field
import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from ril.utils.safe_np import np

from core.components import cc4

from .palette import utils as palette_utils

from .post.shape_snap import snap_lonely_cells
from .post.exemplar_snap import snap_singletons


GATE_MIN = 0.55
GATE_MAX = 0.92
MONOCOLOR_GATE_DELTA = 0.05
TINY_GRID_BASE_THRESHOLD = 0.70

DEFAULT_SHAPE_WEIGHT = 0.7
DEFAULT_PALETTE_WEIGHT = 0.3
MONOCOLOR_PALETTE_BONUS = 0.1
MONOCOLOR_PALETTE_CAP = 0.5
TINY_GRID_WEIGHTS = (0.4, 0.6)


try:
    Grid = np.ndarray  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover - safe_np fallback
    Grid = Any



@dataclass
class Example:
    """Container bundling a training pair."""

    grid_in: Grid
    grid_out: Grid


@dataclass
class LocalRule:
    """Local transformation expressed in relative space."""

    color_hist: Dict[int, float]
    delta: List[Tuple[float, float, int]]
    bbox_fraction: Tuple[float, float]
    centroid_fraction: Tuple[float, float]
    mask_pattern: np.ndarray
    birth_colors: set[int] = field(default_factory=set)
    death_colors: set[int] = field(default_factory=set)


@dataclass
class RuleSet:
    """Bundle holding the learned rules and palette guards."""

    rules: List[LocalRule]
    allowed_palette: set[int]
    birth_palette: set[int]
    death_palette: set[int]
    base_threshold: float = 0.85
    is_monocolor_target: bool = False
    is_tiny_grid: bool = False

    def palette_score(self, grid: Grid) -> float:
        """Return the palette completeness score for ``grid``."""

        present = set(np.unique(np.asarray(grid)))
        if not self.allowed_palette:
            return 1.0
        return len(present & self.allowed_palette) / len(self.allowed_palette)


def extract_local_rules(
    examples: Sequence[Tuple[Grid, Grid]] | Sequence[Example],
    *,
    test_input: Grid | None = None,
    base_threshold: float = 0.85,
) -> RuleSet:
    """Extract local delta rules from the provided examples.

    Parameters
    ----------
    examples:
        Training pairs.  They can be raw ``(input, output)`` tuples or ``Example``
        instances.
    test_input:
        Optional test grid.  Its palette contributes to ``allowed_palette`` so
        that inference never emits colours unseen across the task bundle.
    base_threshold:
        Similarity baseline that will later be adapted via
        :func:`entropy_scaled_gating`.
    """

    allowed_palette: set[int] = set()
    death_palette: set[int] = set()
    rules: List[LocalRule] = []
    palette_sizes: List[int] = []
    grid_areas: List[int] = []

    def _ensure_tuple(pair: Tuple[Grid, Grid] | Example) -> Tuple[Grid, Grid]:
        if isinstance(pair, Example):
            return pair.grid_in, pair.grid_out
        return pair

    example_pairs = [_ensure_tuple(pair) for pair in examples]

    if test_input is not None:
        allowed_palette.update(int(c) for c in np.unique(np.asarray(test_input)))

    for grid_in, grid_out in example_pairs:
        arr_in = np.asarray(grid_in)
        arr_out = np.asarray(grid_out)

        if arr_out.size:
            palette_sizes.append(len(np.unique(arr_out)))
            if arr_out.ndim == 2:
                grid_areas.append(int(arr_out.shape[0] * arr_out.shape[1]))

        if arr_in.shape != arr_out.shape:
            raise ValueError("Training grids must share the same shape")

        allowed_palette.update(int(c) for c in np.unique(arr_in))
        allowed_palette.update(int(c) for c in np.unique(arr_out))

        in_palette = set(int(c) for c in np.unique(arr_in))
        out_palette = set(int(c) for c in np.unique(arr_out))
        death_palette.update(in_palette - out_palette)

        diff_mask = arr_in != arr_out
        if not diff_mask.any():
            continue

        h, w = arr_in.shape

        for color in np.unique(arr_out[diff_mask]):
            color = int(color)
            color_mask = diff_mask & (arr_out == color)
            if not color_mask.any():
                continue
            labels, components = cc4(color_mask)
            for cid, bbox, _size in components:
                component_mask = labels == cid
                if not component_mask.any():
                    continue
                rule = _build_rule(
                    arr_in,
                    arr_out,
                    component_mask,
                    bbox,
                    colour=color,
                    grid_shape=(h, w),
                )
                rule.birth_colors.add(color)
                rules.append(rule)

        for color in np.unique(arr_in[diff_mask]):
            color = int(color)
            color_mask = diff_mask & (arr_in == color)
            if not color_mask.any():
                continue
            labels, components = cc4(color_mask)
            for cid, bbox, _size in components:
                component_mask = labels == cid
                if not component_mask.any():
                    continue
                rule = _build_rule(
                    arr_in,
                    arr_out,
                    component_mask,
                    bbox,
                    colour=int(np.median(arr_out[component_mask])),
                    grid_shape=(h, w),
                )
                rule.death_colors.add(color)
                rules.append(rule)

    birth_palette = compute_birth_palette(example_pairs)

    monocolor_family = bool(palette_sizes) and max(palette_sizes) <= 2
    tiny_grid_family = bool(grid_areas) and max(grid_areas) <= 9
    if test_input is not None:
        test_arr = np.asarray(test_input)
        if test_arr.ndim == 2 and test_arr.size:
            tiny_grid_family = tiny_grid_family or (test_arr.shape[0] * test_arr.shape[1] <= 9)

    return RuleSet(
        rules=rules,
        allowed_palette=allowed_palette,
        birth_palette=birth_palette,
        death_palette=death_palette,
        base_threshold=base_threshold,
        is_monocolor_target=monocolor_family,
        is_tiny_grid=tiny_grid_family,
    )


def compute_birth_palette(examples: Sequence[Tuple[Grid, Grid]]) -> set[int]:
    """Return colours that consistently appear as births across examples."""

    if not examples:
        return set()

    births: List[set[int]] = []
    for grid_in, grid_out in examples:
        arr_in = np.asarray(grid_in)
        arr_out = np.asarray(grid_out)
        ci = set(int(c) for c in np.unique(arr_in))
        co = set(int(c) for c in np.unique(arr_out))
        births.append(co - ci)

    freq = Counter(colour for group in births for colour in group)
    if not freq:
        return set()

    required = max(1, math.ceil(len(examples) / 2))
    return {colour for colour, count in freq.items() if count >= required}


def _classify_palette_issue(pc: float | None, er: float | None) -> str:
    """Return a coarse palette health label."""

    try:
        pc_val = float(pc) if pc is not None else None
    except (TypeError, ValueError):
        pc_val = None
    try:
        er_val = float(er) if er is not None else None
    except (TypeError, ValueError):
        er_val = None

    tolerance = 1e-6
    has_missing = pc_val is not None and pc_val < 1.0 - tolerance
    palette_perfect = pc_val is not None and pc_val >= 1.0 - tolerance
    has_extra = er_val is not None and er_val > tolerance

    if palette_perfect and not has_extra:
        return "ok"
    if has_missing and not has_extra:
        return "missing_only"
    if palette_perfect and has_extra:
        return "extra_only"
    if has_missing and has_extra:
        return "missing_and_extra"
    if has_extra:
        return "extra_only"
    if has_missing:
        return "missing_only"
    return "ok"


def apply_rule_set(rule_set: RuleSet, grid: Grid) -> Grid:
    """Apply the learned ``rule_set`` to ``grid``."""

    arr = np.asarray(grid).copy()
    h, w = arr.shape if arr.ndim == 2 else (0, 0)
    if h == 0 or w == 0:
        return arr

    gate_base = float(rule_set.base_threshold)
    if rule_set.is_monocolor_target:
        gate_base = max(0.0, gate_base - MONOCOLOR_GATE_DELTA)
    if rule_set.is_tiny_grid:
        gate_base = min(gate_base, TINY_GRID_BASE_THRESHOLD)

    threshold = entropy_scaled_gating(arr, gate_base)

    for rule in rule_set.rules:
        win_h = max(1, int(round(rule.bbox_fraction[0] * h)))
        win_w = max(1, int(round(rule.bbox_fraction[1] * w)))
        if win_h <= 0 or win_w <= 0 or win_h > h or win_w > w:
            continue

        best_score = 0.0
        best_anchor: Tuple[int, int] | None = None

        for top in range(0, h - win_h + 1):
            for left in range(0, w - win_w + 1):
                patch = arr[top : top + win_h, left : left + win_w]
                score = rule_match_score(rule, patch)

                center_r = (top + win_h / 2.0) / h
                center_c = (left + win_w / 2.0) / w
                distance = np.hypot(
                    center_r - rule.centroid_fraction[0],
                    center_c - rule.centroid_fraction[1],
                )
                proximity = max(0.0, 1.0 - distance)
                score *= (0.5 + 0.5 * proximity)

                if score >= threshold and score > best_score:
                    best_score = score
                    best_anchor = (top, left)

        if best_anchor is None:
            continue

        arr = _apply_delta(
            arr,
            rule,
            anchor=best_anchor,
            window_shape=(win_h, win_w),
            allowed_palette=rule_set.allowed_palette,
            birth_palette=rule_set.birth_palette,
            death_palette=rule_set.death_palette,
        )

    arr = _apply_palette_rescue(
        arr,
        allowed_palette=rule_set.allowed_palette,
        birth_palette=rule_set.birth_palette,
    )
    return arr


def _apply_palette_rescue(
    grid: Grid,
    *,
    allowed_palette: Iterable[int],
    birth_palette: Iterable[int],
    background_colors: Iterable[int] | None = None,
    legend_palette: Iterable[int] | None = None,
) -> Grid:
    """Inject missing birth colours using zone-level recolouring."""

    arr = np.asarray(grid).copy()
    present = set(int(c) for c in np.unique(arr))
    legend = {int(c) for c in legend_palette} if legend_palette else set()
    allowed = set(int(c) for c in allowed_palette)
    if legend:
        allowed.update(legend)
    births = [int(c) for c in birth_palette if int(c) not in present]

    if not births:
        return arr

    # Prefer 0 as background, or use dominant background
    is_all_zero = (arr == 0).sum() == arr.size
    if is_all_zero:
        # Grid is blank (all zeros) - seed colors directly
        return _seed_missing_colors_fallback(arr, births, allowed)

    # Find background, preferring 0 if it exists, otherwise most common color
    if 0 in np.unique(arr):
        background = 0
    else:
        # Try to find a background that's NOT a missing birth color
        unique_vals, counts = np.unique(arr, return_counts=True)
        missing_births_set = set(int(c) for c in births)  # births already filtered to missing

        # Sort by count (descending)
        sorted_pairs = sorted(zip(counts, unique_vals), reverse=True)

        # Prefer most common non-missing-birth color as background
        background = None
        for count, val in sorted_pairs:
            if int(val) not in missing_births_set:
                background = int(val)
                break

        # If all colors are missing births, use fallback (can't find safe background)
        if background is None:
            # This is a monocolor grid with only missing birth colors
            # Use fallback to seed other colors
            return _seed_missing_colors_fallback(arr, births, allowed)

    background_mask = arr == background

    if not background_mask.any():
        # No background cells at all - use fallback seeding
        return _seed_missing_colors_fallback(arr, births, allowed)

    labels, components = cc4(background_mask)
    components = sorted(components, key=lambda item: item[2], reverse=True)
    if not components:
        # No connected components found - use fallback seeding
        return _seed_missing_colors_fallback(arr, births, allowed)

    return _satisfy_birth_colours(
        arr,
        births,
        labels,
        components,
        allowed,
        legend_palette=legend,
    )


def _seed_missing_colors_fallback(
    arr: Grid,
    births: Sequence[int],
    allowed: Set[int],
) -> Grid:
    """Seed missing birth colors in blank or near-blank predictions.

    This handles cases where the prediction is all background or has no
    clear zones to recolor. We seed colors at distributed positions to
    ensure palette coverage WITHOUT destroying existing valid colors.
    """
    if not births or arr.size == 0:
        return arr

    h, w = arr.shape
    result = arr.copy()

    # Get current present colors
    present = set(int(c) for c in np.unique(arr))

    # Find background cells (zeros first, then most common color NOT in births)
    zero_mask = arr == 0
    if zero_mask.any():
        background_mask = zero_mask
    else:
        # Find a suitable background color that's not a missing birth
        # and not already validly placed
        background_color = None
        unique_vals, counts = np.unique(arr, return_counts=True)
        sorted_pairs = sorted(zip(counts, unique_vals), reverse=True)

        # Prefer colors that are NOT in the births list (i.e., already present and valid)
        births_set = set(int(b) for b in births)
        for count, val in sorted_pairs:
            if int(val) not in births_set:
                background_color = int(val)
                break

        if background_color is None:
            # All colors are missing births - can only seed minimally
            # Use smallest count color as background
            background_color = int(sorted_pairs[-1][1]) if sorted_pairs else 0

        background_mask = arr == background_color

    # Get all background positions
    bg_positions = np.argwhere(background_mask)

    if len(bg_positions) == 0:
        # No background cells - grid is full
        # Find positions with colors NOT in allowed palette to replace
        out_of_palette_mask = np.zeros_like(arr, dtype=bool)
        for val in np.unique(arr):
            if int(val) not in allowed:
                out_of_palette_mask |= (arr == val)

        if out_of_palette_mask.any():
            bg_positions = np.argwhere(out_of_palette_mask)
        else:
            # Really can't find anywhere to seed without destroying valid colors
            # Return as-is (preserve existing valid colors)
            return result

    # Distribute missing birth colors across available positions
    num_positions = len(bg_positions)
    for idx, color in enumerate(births):
        if color not in allowed:
            continue

        # Use at least 1 cell per color, more if space allows
        cells_per_color = max(1, min(4, num_positions // max(1, len(births))))

        for cell_idx in range(cells_per_color):
            pos_idx = (idx * cells_per_color + cell_idx) % num_positions
            if pos_idx < len(bg_positions):
                r, c = bg_positions[pos_idx]
                if 0 <= r < h and 0 <= c < w:
                    result[r, c] = color

    return result


BIRTH_ZONE_IOU_THRESHOLD = float(os.getenv("RIL_BIRTH_IOU_THRESHOLD", "0.08"))


def _satisfy_birth_colours(
    arr: Grid,
    births: Sequence[int],
    labels: Grid,
    components: Sequence[Tuple[int, Tuple[slice, slice], int]],
    allowed: Set[int],
    *,
    tau: float = BIRTH_ZONE_IOU_THRESHOLD,
    legend_palette: Set[int] | None = None,
) -> Grid:
    """Attempt to recolour background zones until all ``births`` appear."""

    present = {int(c) for c in np.unique(arr)}
    used_components: set[int] = set()

    for colour in births:
        if colour in present or colour not in allowed:
            continue

        best_gain = 0.0
        best_zone: np.ndarray | None = None
        best_cid: int | None = None

        for cid, _bbox, _size in components:
            if cid in used_components:
                continue

            zone_mask = labels == cid
            if not np.any(zone_mask):
                continue

            gain = _palette_iou_gain(arr, zone_mask, colour)
            if gain > best_gain:
                best_gain = gain
                best_zone = zone_mask
                best_cid = cid

        if best_zone is not None and best_cid is not None and best_gain > tau:
            arr = _recolor_zone(
                arr,
                best_zone,
                colour,
                allowed,
                legend_palette=legend_palette,
            )
            present.add(colour)
            used_components.add(best_cid)

    return arr


def normalized_entropy(grid: Grid) -> float:
    """Return the entropy of ``grid`` normalised to ``[0, 1]``."""

    arr = np.asarray(grid)
    if arr.size == 0:
        return 0.0

    colours, counts = np.unique(arr, return_counts=True)
    if counts.size == 0:
        return 0.0

    probs = counts.astype(float) / float(arr.size)
    entropy = float(-(probs * np.log2(probs + 1e-12)).sum())
    max_entropy = float(np.log2(len(colours))) if colours.size else 0.0
    if max_entropy <= 0.0:
        return 0.0

    ratio = entropy / max_entropy
    return float(max(0.0, min(1.0, ratio)))


def entropy_scaled_gating(grid: Grid, base_thr: float, *, is_birth_candidate: bool = False) -> float:
    """Return a gate threshold scaled by the grid entropy."""

    entropy_fraction = normalized_entropy(grid)
    scaled = float(base_thr) * (1.0 - entropy_fraction)
    scaled = max(GATE_MIN, min(GATE_MAX, scaled))
    if is_birth_candidate:
        scaled *= 0.9
        scaled = max(GATE_MIN, min(GATE_MAX, scaled))
    return float(scaled)


def dimensionless_kernel_size(shape: Tuple[int, int], ratio: float = 0.05) -> int:
    """Return a kernel radius scaled by the smallest grid dimension."""

    h, w = shape
    k = max(1, int(round(min(h, w) * ratio)))
    k = min(3, k)
    return k


def dimensionless_dilate(mask: Grid, ratio: float = 0.05, iterations: int = 1) -> Grid:
    """Dilate ``mask`` using a dimension-independent kernel."""

    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != 2:
        raise ValueError("dimensionless_dilate expects a 2D mask")

    radius = dimensionless_kernel_size(arr.shape, ratio)
    out = arr.copy()
    for _ in range(max(1, iterations)):
        out = _dilate(out, radius)
    return out


def dimensionless_erode(mask: Grid, ratio: float = 0.05, iterations: int = 1) -> Grid:
    """Erode ``mask`` using a dimension-independent kernel."""

    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != 2:
        raise ValueError("dimensionless_erode expects a 2D mask")

    radius = dimensionless_kernel_size(arr.shape, ratio)
    out = arr.copy()
    for _ in range(max(1, iterations)):
        out = _erode(out, radius)
    return out


def dimensionless_ring(mask: Grid, ratio: float = 0.05) -> Grid:
    """Return a ring mask created via dilation minus erosion."""

    arr = np.asarray(mask, dtype=bool)
    dilated = dimensionless_dilate(arr, ratio=ratio, iterations=1)
    eroded = dimensionless_erode(arr, ratio=ratio, iterations=1)
    return np.logical_and(dilated, ~eroded)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_rule(
    arr_in: Grid,
    arr_out: Grid,
    mask: Grid,
    bbox: Tuple[int, int, int, int],
    *,
    colour: int,
    grid_shape: Tuple[int, int],
) -> LocalRule:
    y_min, x_min, y_max, x_max = bbox
    window_in = arr_in[y_min : y_max + 1, x_min : x_max + 1]
    window_out = arr_out[y_min : y_max + 1, x_min : x_max + 1]
    window_mask = np.asarray(mask[y_min : y_max + 1, x_min : x_max + 1], dtype=bool)

    hist = _normalised_histogram(window_in)

    coords = np.argwhere(mask)
    bbox_h = max(1, y_max - y_min + 1)
    bbox_w = max(1, x_max - x_min + 1)
    delta: List[Tuple[float, float, int]] = []
    for r, c in coords:
        rel_r = (r - y_min + 0.5) / bbox_h
        rel_c = (c - x_min + 0.5) / bbox_w
        delta.append((float(rel_r), float(rel_c), int(window_out[r - y_min, c - x_min])))

    centroid_r = (coords[:, 0].mean() if coords.size else y_min) + 0.5
    centroid_c = (coords[:, 1].mean() if coords.size else x_min) + 0.5
    h, w = grid_shape
    centroid_fraction = (centroid_r / h, centroid_c / w)

    bbox_fraction = (bbox_h / h, bbox_w / w)

    return LocalRule(
        color_hist=hist,
        delta=delta,
        bbox_fraction=bbox_fraction,
        centroid_fraction=centroid_fraction,
        mask_pattern=window_mask,
    )


def rule_match_score(rule: LocalRule, patch: Grid) -> float:
    """Return a hybrid histogram/IoU score between ``rule`` and ``patch``."""

    patch_arr = np.asarray(patch)
    hist_score = _histogram_similarity(_normalised_histogram(patch_arr), rule.color_hist)

    rule_mask = np.asarray(rule.mask_pattern, dtype=bool)
    patch_mask = _structure_mask(patch_arr)

    if rule_mask.size == 0:
        iou_score = 1.0 if not patch_mask.any() else 0.0
    else:
        normalised_rule_mask = _resize_mask(rule_mask, patch_mask.shape)
        iou_score = iou(normalised_rule_mask, patch_mask)

    return 0.5 * hist_score + 0.5 * iou_score


def _normalised_histogram(patch: Grid) -> Dict[int, float]:
    colours, counts = np.unique(np.asarray(patch), return_counts=True)
    total = float(counts.sum())
    if total == 0:
        return {}
    return {int(col): float(count) / total for col, count in zip(colours, counts)}


def _histogram_similarity(a: Dict[int, float], b: Dict[int, float]) -> float:
    if not a and not b:
        return 1.0
    keys = set(a) | set(b)
    if not keys:
        return 1.0
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
    norm_a = np.sqrt(sum(v * v for v in a.values()))
    norm_b = np.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot) / float(norm_a * norm_b)


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Return the IoU between two boolean masks of the same shape."""

    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    if a.shape != b.shape:
        return 0.0
    union = np.logical_or(a, b).sum(dtype=float)
    intersection = np.logical_and(a, b).sum(dtype=float)
    if union == 0.0:
        return 1.0
    return float(intersection) / float(union)


def _structure_mask(patch: Grid) -> np.ndarray:
    arr = np.asarray(patch)
    if arr.size == 0:
        return np.zeros_like(arr, dtype=bool)
    colours, counts = np.unique(arr, return_counts=True)
    if counts.size == 0:
        return np.zeros_like(arr, dtype=bool)
    background = colours[int(np.argmax(counts))]
    return np.asarray(arr != background, dtype=bool)


def _resize_mask(mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = shape
    if target_h <= 0 or target_w <= 0:
        return np.zeros((max(target_h, 0), max(target_w, 0)), dtype=bool)

    src = np.asarray(mask, dtype=bool)
    if src.size == 0:
        return np.zeros((target_h, target_w), dtype=bool)

    src_h, src_w = src.shape
    if src_h == target_h and src_w == target_w:
        return src.copy()

    row_idx = np.clip(
        np.floor(np.linspace(0, max(src_h - 1, 0), num=target_h)).astype(int),
        0,
        max(src_h - 1, 0),
    )
    col_idx = np.clip(
        np.floor(np.linspace(0, max(src_w - 1, 0), num=target_w)).astype(int),
        0,
        max(src_w - 1, 0),
    )
    return src[row_idx][:, col_idx]


def _apply_delta(
    grid: Grid,
    rule: LocalRule,
    *,
    anchor: Tuple[int, int],
    window_shape: Tuple[int, int],
    allowed_palette: Iterable[int],
    birth_palette: Iterable[int],
    death_palette: Iterable[int],
) -> Grid:
    arr = np.asarray(grid).copy()
    h, w = arr.shape
    top, left = anchor
    win_h, win_w = window_shape
    allowed = set(int(c) for c in allowed_palette)
    births = set(int(c) for c in birth_palette)
    deaths = set(int(c) for c in death_palette)

    for rel_r, rel_c, colour in rule.delta:
        r = top + int(round(rel_r * max(1, win_h - 1)))
        c = left + int(round(rel_c * max(1, win_w - 1)))
        if not (0 <= r < h and 0 <= c < w):
            continue
        if colour not in allowed and colour not in births and colour not in deaths:
            continue
        arr[r, c] = colour
    return arr


def _dominant_background(
    arr: Grid,
    allowed: Iterable[int],
    background_colors: Iterable[int] | None = None,
) -> int:
    colours, counts = np.unique(arr, return_counts=True)
    if counts.size == 0:
        return 0

    allowed_set = set(int(c) for c in allowed)
    preferred = (
        {int(c) for c in background_colors}
        if background_colors is not None
        else None
    )
    ordered = sorted(zip(counts, colours), reverse=True)

    if preferred:
        for _count, colour in ordered:
            if int(colour) in preferred:
                return int(colour)

    for _count, colour in ordered:
        if not allowed_set or int(colour) in allowed_set:
            return int(colour)

    return int(ordered[0][1]) if ordered else 0


def _zone_iou(arr: Grid, zone_mask: Grid, colour: int) -> float:
    mask = np.asarray(zone_mask, dtype=bool)
    if not mask.any():
        return 0.0

    colour_mask = np.asarray(arr == colour, dtype=bool)
    intersection = np.logical_and(colour_mask, mask).sum(dtype=float)
    union = mask.sum(dtype=float)
    if union <= 0.0:
        return 0.0

    return float(intersection / union)


def _palette_iou_gain(arr: Grid, zone_mask: Grid, colour: int) -> float:
    mask = np.asarray(zone_mask, dtype=bool)
    if not mask.any():
        return 0.0

    before = _zone_iou(arr, mask, colour)
    return max(0.0, 1.0 - before)


def _recolor_zone(
    arr: Grid,
    mask: Grid,
    colour: int,
    allowed: Iterable[int],
    *,
    legend_palette: Set[int] | None = None,
) -> Grid:
    allowed_set = {int(c) for c in allowed}
    target = int(colour)
    if allowed_set and target not in allowed_set:
        return np.asarray(arr)

    mask_arr = np.asarray(mask, dtype=bool)
    if not mask_arr.any():
        return np.asarray(arr)

    protected = {int(c) for c in legend_palette} if legend_palette else set()
    if protected:
        zone_values = {int(value) for value in np.asarray(arr)[mask_arr]}
        if zone_values & protected and target not in protected:
            return np.asarray(arr)

    base = np.asarray(arr).tolist()
    mask_list = mask_arr.tolist()
    height = min(len(base), len(mask_list))
    for r in range(height):
        row = base[r]
        mask_row = mask_list[r]
        width = min(len(row), len(mask_row))
        for c in range(width):
            if mask_row[c]:
                row[c] = target
    return np.asarray(base)


def _dilate(mask: Grid, radius: int) -> Grid:
    arr = np.asarray(mask, dtype=bool)
    h, w = arr.shape
    out = np.zeros_like(arr)
    for y in range(h):
        y0 = max(0, y - radius)
        y1 = min(h, y + radius + 1)
        for x in range(w):
            x0 = max(0, x - radius)
            x1 = min(w, x + radius + 1)
            if np.any(arr[y0:y1, x0:x1]):
                out[y, x] = True
    return out


def _erode(mask: Grid, radius: int) -> Grid:
    arr = np.asarray(mask, dtype=bool)
    h, w = arr.shape
    out = np.zeros_like(arr)
    for y in range(h):
        y0 = max(0, y - radius)
        y1 = min(h, y + radius + 1)
        for x in range(w):
            x0 = max(0, x - radius)
            x1 = min(w, x + radius + 1)
            if np.all(arr[y0:y1, x0:x1]):
                out[y, x] = True
    return out


def _grid_palette_set(grid: Grid | Sequence[Sequence[int]]) -> Set[int]:
    arr = np.asarray(grid)
    if arr.size == 0:
        return set()
    return {int(c) for c in np.unique(arr)}


def _normalise_grid_sequence(grids: Any) -> List[np.ndarray]:
    """Return a list of 2D numpy arrays extracted from ``grids``."""

    if grids is None:
        return []

    arr = np.asarray(grids)
    if arr.ndim == 2:
        return [arr]
    if arr.ndim == 3:
        return [np.asarray(arr[idx]) for idx in range(arr.shape[0])]

    try:
        sequence = list(grids)  # type: ignore[arg-type]
    except TypeError:
        return [arr] if arr.ndim >= 2 else []

    normalised: List[np.ndarray] = []
    for item in sequence:
        item_arr = np.asarray(item)
        if item_arr.ndim >= 2:
            normalised.append(item_arr)
    return normalised


def _is_monocolor_family(grids: List[np.ndarray]) -> bool:
    palette_sizes = [len(np.unique(grid)) for grid in grids if grid.size]
    if not palette_sizes:
        return False
    return max(palette_sizes) <= 2


def _is_tiny_grid_family(grids: List[np.ndarray], fallback: Any = None) -> bool:
    areas = [int(grid.shape[0] * grid.shape[1]) for grid in grids if grid.ndim == 2]
    if not areas and fallback is not None:
        for candidate in _normalise_grid_sequence(fallback):
            if candidate.ndim == 2:
                areas.append(int(candidate.shape[0] * candidate.shape[1]))
    if not areas:
        return False
    return max(areas) <= 9


def _score_weights(train_out: Any, test_in: Any) -> tuple[float, float, bool, bool]:
    """Return shape/palette weights together with family flags."""

    train_out_grids = _normalise_grid_sequence(train_out)
    is_monocolor = _is_monocolor_family(train_out_grids)
    is_tiny = _is_tiny_grid_family(train_out_grids, fallback=test_in)

    shape_w, palette_w = DEFAULT_SHAPE_WEIGHT, DEFAULT_PALETTE_WEIGHT
    if is_tiny:
        shape_w, palette_w = TINY_GRID_WEIGHTS

    if is_monocolor:
        target_palette = min(
            MONOCOLOR_PALETTE_CAP, palette_w + MONOCOLOR_PALETTE_BONUS
        )
        if target_palette > palette_w:
            palette_w = target_palette

    shape_w = max(0.0, 1.0 - palette_w)
    total = shape_w + palette_w
    if total > 0:
        shape_w /= total
        palette_w /= total

    return shape_w, palette_w, is_monocolor, is_tiny


def compute_allowed_and_birth_palettes(
    train_in: Grid | Sequence[Sequence[int]],
    train_out: Grid | Sequence[Sequence[int]],
    test_in: Grid | Sequence[Sequence[int]],
) -> tuple[Set[int], Set[int]]:
    train_in_palette = _grid_palette_set(train_in)
    train_out_palette = _grid_palette_set(train_out)
    test_palette = _grid_palette_set(test_in)
    allowed = train_in_palette | train_out_palette | test_palette
    births = train_out_palette - train_in_palette
    return allowed, births


def _coerce_palette_values(payload: Any) -> Set[int]:
    """Normalise ``payload`` into a flat set of colours."""

    colours: Set[int] = set()
    if payload is None:
        return colours

    if isinstance(payload, Mapping):
        for key in ("sequence", "allowed", "palette", "colors", "colours"):
            if key in payload:
                colours.update(_coerce_palette_values(payload.get(key)))
        return colours

    if isinstance(payload, (list, tuple, set)):
        for item in payload:
            colours.update(_coerce_palette_values(item))
        return colours

    try:
        colours.add(int(payload))
    except Exception:
        pass
    return colours


def palette_completeness(
    candidate: Grid | Sequence[Sequence[int]],
    allowed: Set[int],
    births: Set[int] | None = None,
) -> float:
    present = _grid_palette_set(candidate)
    allowed_score = len(present & allowed) / len(allowed) if allowed else 1.0
    if births:
        birth_score = len(present & births) / len(births) if births else 1.0
        return 0.5 * allowed_score + 0.5 * birth_score
    return allowed_score


def _env_flag(name: str, default: bool = False) -> bool:
    """Return True when the environment variable ``name`` is truthy."""

    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    if raw in {"", "0", "false", "no"}:
        return False
    return True


def _mode_colour(grid: Sequence[Sequence[int]]) -> int:
    """Return the most common colour in ``grid`` (ties resolved by lower colour)."""

    counts: Counter[int] = Counter()
    for row in grid:
        if isinstance(row, Sequence):
            for value in row:
                try:
                    counts[int(value)] += 1
                except Exception:
                    continue
    if not counts:
        return 0
    max_count = max(counts.values())
    return min((colour for colour, count in counts.items() if count == max_count), default=0)


def _grid_dimensions(grid: Sequence[Sequence[int]]) -> tuple[int, int]:
    height = len(grid)
    width = 0
    for row in grid:
        if isinstance(row, Sequence):
            width = max(width, len(row))
    return height, width


def _iter_colour_coords(
    grid: Sequence[Sequence[int]],
    *,
    colour: int,
    ignore_matching_input: Sequence[Sequence[int]] | None = None,
) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for r, row in enumerate(grid):
        if not isinstance(row, Sequence):
            continue
        for c, value in enumerate(row):
            try:
                val = int(value)
            except Exception:
                continue
            if val != colour:
                continue
            if (
                ignore_matching_input
                and r < len(ignore_matching_input)
                and isinstance(ignore_matching_input[r], Sequence)
                and c < len(ignore_matching_input[r])
                and ignore_matching_input[r][c] == val
            ):
                continue
            coords.append((r, c))
    return coords


def _infer_pattern_box(
    examples: Sequence[Mapping[str, object]],
    colour: int,
) -> tuple[float, float, float, float] | None:
    """Return an averaged bounding box for ``colour`` across the examples."""

    total_weight = 0
    accum = [0.0, 0.0, 0.0, 0.0]

    for example in examples:
        grid_out = example.get("output")
        grid_in = example.get("input")
        if not isinstance(grid_out, Sequence):
            continue
        coords = _iter_colour_coords(grid_out, colour=colour, ignore_matching_input=grid_in if isinstance(grid_in, Sequence) else None)
        if not coords:
            continue
        height, width = _grid_dimensions(grid_out)
        if height == 0 or width == 0:
            continue
        min_r = min(r for r, _ in coords)
        max_r = max(r for r, _ in coords)
        min_c = min(c for _, c in coords)
        max_c = max(c for _, c in coords)
        norm_r = max(height - 1, 1)
        norm_c = max(width - 1, 1)
        weight = len(coords)
        accum[0] += (min_r / norm_r) * weight
        accum[1] += (max_r / norm_r) * weight
        accum[2] += (min_c / norm_c) * weight
        accum[3] += (max_c / norm_c) * weight
        total_weight += weight

    if total_weight == 0:
        return None

    return tuple(value / total_weight for value in accum)  # type: ignore[return-value]


def _project_box_to_grid(
    box: tuple[float, float, float, float],
    *,
    height: int,
    width: int,
) -> set[tuple[int, int]]:
    if height <= 0 or width <= 0:
        return set()
    min_r, max_r, min_c, max_c = box
    norm_r = max(height - 1, 1)
    norm_c = max(width - 1, 1)
    start_r = max(0, min(height - 1, int(round(min_r * norm_r))))
    end_r = max(start_r, min(height - 1, int(round(max_r * norm_r))))
    start_c = max(0, min(width - 1, int(round(min_c * norm_c))))
    end_c = max(start_c, min(width - 1, int(round(max_c * norm_c))))
    coords: set[tuple[int, int]] = set()
    for r in range(start_r, end_r + 1):
        for c in range(start_c, end_c + 1):
            coords.add((r, c))
    return coords


def _infer_frequency_positions(
    examples: Sequence[Mapping[str, object]],
    colour: int,
    *,
    height: int,
    width: int,
) -> set[tuple[int, int]]:
    if height <= 0 or width <= 0:
        return set()

    counts: Counter[tuple[int, int]] = Counter()
    total_examples = 0
    for example in examples:
        grid_out = example.get("output")
        grid_in = example.get("input")
        if not isinstance(grid_out, Sequence):
            continue
        coords = _iter_colour_coords(grid_out, colour=colour, ignore_matching_input=grid_in if isinstance(grid_in, Sequence) else None)
        if not coords:
            continue
        total_examples += 1
        g_height, g_width = _grid_dimensions(grid_out)
        if g_height == 0 or g_width == 0:
            continue
        norm_r = max(g_height - 1, 1)
        norm_c = max(g_width - 1, 1)
        seen: set[tuple[int, int]] = set()
        for r, c in coords:
            rel_r = int(round((r / norm_r) * max(height - 1, 0))) if height > 1 else 0
            rel_c = int(round((c / norm_c) * max(width - 1, 0))) if width > 1 else 0
            rel = (max(0, min(height - 1, rel_r)), max(0, min(width - 1, rel_c)))
            seen.add(rel)
        for rel in seen:
            counts[rel] += 1

    if not counts:
        return set()

    threshold = max(1, (total_examples + 1) // 2)
    return {coord for coord, count in counts.items() if count >= threshold}


def _apply_alternative_fill(
    candidate: list[list[int]],
    *,
    allowed_palette: set[int],
    base_grid: list[list[int]],
    context: Dict[str, Any],
) -> tuple[list[list[int]] | None, Dict[str, Any]]:
    examples_obj = context.get("train_examples")
    test_input = context.get("test_input")
    if not isinstance(examples_obj, Sequence) or not isinstance(test_input, Sequence):
        return None, {}

    missing = [colour for colour in sorted(allowed_palette) if colour not in palette_utils.palette_set(candidate)]
    if not missing:
        return None, {}

    fallback = [row[:] for row in candidate]
    diagnostics: Dict[str, Any] = {"applied": []}
    test_height, test_width = _grid_dimensions(test_input)
    background = _mode_colour(base_grid)

    for colour in missing:
        box = _infer_pattern_box(examples_obj, colour)
        coords: set[tuple[int, int]] = set()
        strategy = ""
        if box is not None:
            coords = _project_box_to_grid(box, height=test_height, width=test_width)
            strategy = "pattern"
        freq_coords = _infer_frequency_positions(
            examples_obj,
            colour,
            height=test_height,
            width=test_width,
        )
        if freq_coords:
            if coords:
                coords |= freq_coords
                strategy = f"{strategy}+ml" if strategy else "ml"
            else:
                coords = freq_coords
                strategy = "ml"

        # If no geometry-based coords found and grid is mostly blank, seed minimally
        if not coords:
            # For blank grids, ensure at least minimal presence
            present_colors = palette_utils.palette_set(fallback)
            if len(present_colors) <= 1:  # Blank or monocolor grid
                # Seed at corner to ensure color appears
                h, w = len(fallback), len(fallback[0]) if fallback else 0
                if h > 0 and w > 0:
                    coords = {(0, 0)}
                    strategy = "minimal_seed"
            else:
                continue

        applied = 0
        for r, c in coords:
            if r >= len(fallback) or c >= len(fallback[r]):
                continue
            current = fallback[r][c]
            if current in allowed_palette and current != background:
                continue
            fallback[r][c] = int(colour)
            applied += 1
        if applied:
            diagnostics["applied"].append({"colour": int(colour), "cells": applied, "strategy": strategy or "ml"})

    if diagnostics["applied"]:
        return fallback, diagnostics
    return None, {}


def finalize_candidate(
    candidate: Grid | Sequence[Sequence[int]],
    context: Dict[str, Any] | None,
) -> list[list[int]]:
    """Apply palette completion and pruning safeguards to ``candidate``.

    Parameters
    ----------
    candidate:
        Predicted grid, typically a numpy array or nested sequence of ints.
    context:
        Optional mapping that carries ``allowed_palette`` and ``birth_palette``
        entries.  The function records diagnostics under ``_metrics`` inside
        ``context`` when provided.
    """

    base_grid: list[list[int]]
    if hasattr(candidate, "tolist"):
        base_grid = candidate.tolist()  # type: ignore[assignment]
    else:
        base_grid = [list(row) for row in candidate]  # type: ignore[assignment]

    work_grid = [row[:] for row in base_grid]

    context = context or {}
    allowed_raw = context.get("allowed_palette")
    birth_raw = context.get("birth_palette")
    allowed_palette = {int(c) for c in allowed_raw} if allowed_raw else set()
    if birth_raw:
        birth_palette = sorted({int(c) for c in birth_raw})
    else:
        birth_palette = []

    legend_palette = set()
    legend_palette.update(_coerce_palette_values(context.get("legend_palette")))
    legend_palette.update(_coerce_palette_values(context.get("legend_colors")))
    legend_palette.update(_coerce_palette_values(context.get("legend")))
    if legend_palette:
        allowed_palette.update(legend_palette)
        context.setdefault("_legend_palette", sorted(legend_palette))

    metrics = context.setdefault("_metrics", {})

    def _as_float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _cell_count(grid: Sequence[Sequence[int]]) -> int:
        total = 0
        for row in grid:
            if isinstance(row, Sequence):
                total += len(row)
        return total

    present_before = palette_utils.palette_set(base_grid)
    missing_before = [colour for colour in birth_palette if colour not in present_before]
    extra_rate_before = 0.0
    extra_cells_before = 0
    total_cells_before = 0
    if allowed_palette:
        rate, extra_cells_before, total_cells_before = palette_utils.extra_colour_stats(
            base_grid, allowed_palette
        )
        extra_rate_before = rate

    if allowed_palette:
        metrics["palette_completeness_before"] = palette_utils.palette_completeness(
            base_grid, allowed_palette
        )
        metrics["extra_colour_rate_before"] = extra_rate_before
        metrics["extra_colour_cells_before"] = extra_cells_before
        metrics["total_cells_before"] = total_cells_before
    else:
        metrics.setdefault("palette_completeness_before", 1.0)
        metrics.setdefault("extra_colour_rate_before", 0.0)
        metrics.setdefault("extra_colour_cells_before", 0)
        metrics.setdefault("total_cells_before", _cell_count(base_grid))

    metrics["palette_issue_before"] = _classify_palette_issue(
        _as_float(metrics.get("palette_completeness_before")), extra_rate_before
    )

    if birth_palette:
        if missing_before:
            context["_missing_birth_colours"] = missing_before
        metrics["missing_birth_colours_before"] = len(missing_before)
        max_seeds_raw = context.get("palette_seed_limit")
        try:
            max_seeds = int(max_seeds_raw)
        except (TypeError, ValueError):
            max_seeds = 3
        max_seeds = max(0, min(max_seeds, 3))
        if missing_before and max_seeds > 0:
            seed_pool = [colour for colour in missing_before if colour not in legend_palette]
            if seed_pool:
                seeds = seed_pool[:max_seeds]
                work_grid = palette_utils.enforce_palette_completion(work_grid, seeds)
                metrics["births_enforced"] = len(seeds)
                metrics["palette_seed_limit"] = max_seeds
            elif legend_palette:
                metrics.setdefault("births_enforced", 0)
                context["_legend_protected_births"] = missing_before
    else:
        metrics.setdefault("missing_birth_colours_before", 0)

    disable_extras_filter = _env_flag("ARC_DISABLE_PALETTE_EXTRAS_FILTER", False)
    alternative_fill_enabled = _env_flag("ARC_ENABLE_ALTERNATIVE_FILL", True)

    if allowed_palette:
        if not disable_extras_filter:
            work_grid = palette_utils.prune_extra_colours(work_grid, allowed_palette)
            metrics["extras_filter_applied"] = True
        else:
            metrics["extras_filter_applied"] = False
        metrics["palette_completeness"] = palette_utils.palette_completeness(
            work_grid, allowed_palette
        )
        rate_after, extra_after, total_after = palette_utils.extra_colour_stats(
            work_grid, allowed_palette
        )
        metrics["extra_colour_rate"] = rate_after
        metrics["extra_colour_cells"] = extra_after
        metrics["total_cells"] = total_after
        if disable_extras_filter:
            metrics.setdefault("extras_filter_disabled", True)
    else:
        metrics.setdefault("palette_completeness", 1.0)
        metrics.setdefault("extra_colour_rate", 0.0)
        metrics.setdefault("extra_colour_cells", 0)
        metrics.setdefault("total_cells", _cell_count(work_grid))

    metrics["palette_issue"] = _classify_palette_issue(
        _as_float(metrics.get("palette_completeness")), _as_float(metrics.get("extra_colour_rate"))
    )
    context["palette_issue"] = metrics["palette_issue"]

    if alternative_fill_enabled and allowed_palette:
        alt_grid, diag = _apply_alternative_fill(
            work_grid,
            allowed_palette=allowed_palette,
            base_grid=base_grid,
            context=context,
        )
        if alt_grid is not None:
            new_comp = palette_utils.palette_completeness(alt_grid, allowed_palette)
            new_rate, new_extra, new_total = palette_utils.extra_colour_stats(alt_grid, allowed_palette)
            old_comp = _as_float(metrics.get("palette_completeness")) or 0.0
            old_rate = _as_float(metrics.get("extra_colour_rate")) or 1.0
            improved = (new_comp > old_comp + 1e-6) or (new_rate < old_rate - 1e-6)
            if improved:
                work_grid = alt_grid
                metrics["palette_completeness"] = new_comp
                metrics["extra_colour_rate"] = new_rate
                metrics["extra_colour_cells"] = new_extra
                metrics["total_cells"] = new_total
                metrics["palette_issue"] = _classify_palette_issue(new_comp, new_rate)
                context["palette_issue"] = metrics["palette_issue"]
                if diag:
                    context["_alternative_fill"] = diag
                    metrics["alternative_fill_applied"] = True
            elif diag:
                metrics.setdefault("alternative_fill_attempted", True)

    work_grid = snap_lonely_cells(work_grid)
    exemplar_signatures = context.get("exemplar_signatures")
    if exemplar_signatures:
        work_grid = snap_singletons(work_grid, exemplar_signatures)
    return work_grid


def validate_candidate(
    candidate: Grid | Sequence[Sequence[int]],
    gold: Grid | Sequence[Sequence[int]],
    train_in: Grid | Sequence[Sequence[int]],
    train_out: Grid | Sequence[Sequence[int]],
    test_in: Grid | Sequence[Sequence[int]],
) -> Dict[str, float]:
    cand_arr = np.asarray(candidate)
    gold_arr = np.asarray(gold)
    if cand_arr.shape != gold_arr.shape:
        raise ValueError("candidate and gold must share the same shape")

    allowed, births = compute_allowed_and_birth_palettes(train_in, train_out, test_in)
    gold_palette = _grid_palette_set(gold_arr)

    ious: list[float] = []
    for colour in gold_palette:
        pred_mask = np.asarray(cand_arr == colour, dtype=bool)
        gold_mask = np.asarray(gold_arr == colour, dtype=bool)
        ious.append(iou(pred_mask, gold_mask))

    mean_iou = sum(ious) / len(ious) if ious else 1.0
    birth_coverage = len(_grid_palette_set(cand_arr) & births) / len(births) if births else 1.0
    pal_comp = palette_completeness(cand_arr, allowed, births)

    shape_w, palette_w, is_monocolor, is_tiny = _score_weights(train_out, test_in)
    total = shape_w * mean_iou + palette_w * pal_comp

    metrics: Dict[str, Any] = {
        "iou": mean_iou,
        "palette_completeness": pal_comp,
        "birth_palette_coverage": birth_coverage,
        "score": total,
        "weights": {"shape": shape_w, "palette": palette_w},
    }
    if is_monocolor:
        metrics["monocolor_family"] = True
    if is_tiny:
        metrics["tiny_grid_family"] = True
    return metrics


def adaptive_palette_rescue(
    train_in: Grid | Sequence[Sequence[int]],
    train_out: Grid | Sequence[Sequence[int]],
    test_in: Grid | Sequence[Sequence[int]],
    candidate: Grid | Sequence[Sequence[int]],
    background_colors: Iterable[int] | None = None,
    legend_palette: Iterable[int] | None = None,
) -> list[list[int]]:
    allowed, births = compute_allowed_and_birth_palettes(train_in, train_out, test_in)
    rescued = _apply_palette_rescue(
        candidate,
        allowed_palette=allowed,
        birth_palette=births,
        background_colors=background_colors,
        legend_palette=legend_palette,
    )
    return np.asarray(rescued).tolist()


def stageA_threshold(base_thr: float, cand_meta: Dict) -> float:
    axis = str(cand_meta.get("axis", ""))
    family = str(cand_meta.get("family", ""))
    vertical_families = {
        "Ring",
        "PathV",
        "RailV",
        "LegendCol",
        "symmetry_complete_vertical",
        "palette_restore@vertical",
    }
    threshold = float(base_thr)
    if axis.startswith("V") and family in vertical_families:
        vertical_cap = float(os.getenv("RIL_VERTICAL_CAP", "0.83"))
        threshold = min(threshold, vertical_cap)

    if bool(cand_meta.get("monocolor_family")) or family == "MonocolorTarget":
        threshold = max(GATE_MIN, threshold - MONOCOLOR_GATE_DELTA)

    if bool(cand_meta.get("tiny_grid_family")) or family == "TinyGrid":
        threshold = max(GATE_MIN, min(threshold, TINY_GRID_BASE_THRESHOLD))

    return threshold

__all__ = [
    "Example",
    "LocalRule",
    "RuleSet",
    "adaptive_palette_rescue",
    "apply_rule_set",
    "compute_allowed_and_birth_palettes",
    "compute_birth_palette",
    "dimensionless_dilate",
    "dimensionless_erode",
    "dimensionless_kernel_size",
    "dimensionless_ring",
    "entropy_scaled_gating",
    "normalized_entropy",
    "extract_local_rules",
    "finalize_candidate",
    "iou",
    "palette_completeness",
    "rule_match_score",
    "stageA_threshold",
    "validate_candidate",
]
