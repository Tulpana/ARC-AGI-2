"""Pattern-based candidate generators for the RIL solver (stdlib-friendly)."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .np_compat import asgrid, components, shape

Grid = List[List[int]]
Example = Dict[str, Grid]
Mask = List[List[bool]]


# ----------------------------------------------------------------------------
# Helper primitives


def _ensure_grid(grid: Sequence[Sequence[int]]) -> Grid:
    return asgrid(grid)


def _copy_grid(grid: Grid) -> Grid:
    return [row[:] for row in grid]


def _zero_grid(h: int, w: int, fill: int = 0) -> Grid:
    return [[fill for _ in range(w)] for _ in range(h)]


def _zero_mask(h: int, w: int, fill: bool = False) -> Mask:
    return [[fill for _ in range(w)] for _ in range(h)]


def _mode_color(grid: Grid | Sequence[int]) -> int:
    if isinstance(grid, list) and grid and isinstance(grid[0], list):
        seq: Iterable[int] = (cell for row in grid for cell in row)
    else:
        seq = grid  # type: ignore[assignment]
    cnt = Counter(int(v) for v in seq)
    return int(cnt.most_common(1)[0][0]) if cnt else 0


def _mask_from_pixels(h: int, w: int, pixels: List[Tuple[int, int]]) -> Mask:
    mask = _zero_mask(h, w)
    for y, x in pixels:
        mask[y][x] = True
    return mask


def _mask_any(mask: Mask) -> bool:
    return any(any(row) for row in mask)


def _mask_sum(mask: Mask) -> int:
    return sum(1 for row in mask for val in row if val)


def _mask_coords(mask: Mask) -> List[Tuple[int, int]]:
    coords: List[Tuple[int, int]] = []
    for y, row in enumerate(mask):
        for x, val in enumerate(row):
            if val:
                coords.append((y, x))
    return coords


def _mask_or(dst: Mask, src: Mask) -> None:
    for y, row in enumerate(src):
        for x, val in enumerate(row):
            if val:
                dst[y][x] = True


def _outline_mask(mask: Mask) -> Mask:
    h = len(mask)
    w = len(mask[0]) if h else 0
    result = _zero_mask(h, w)
    for y in range(h):
        for x in range(w):
            if not mask[y][x]:
                continue
            keep = False
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if ny < 0 or ny >= h or nx < 0 or nx >= w or not mask[ny][nx]:
                    keep = True
                    break
            if keep:
                result[y][x] = True
    return result


def _union_bounding_box(mask: Mask) -> Optional[Tuple[int, int, int, int]]:
    coords = _mask_coords(mask)
    if not coords:
        return None
    ys = [y for y, _ in coords]
    xs = [x for _, x in coords]
    return min(ys), max(ys), min(xs), max(xs)


def _extract_border_colors(grid: Grid) -> Counter:
    h, w = shape(grid)
    if h == 0 or w == 0:
        return Counter()
    colors: List[int] = []
    colors.extend(int(x) for x in grid[0])
    if h > 1:
        colors.extend(int(x) for x in grid[h - 1])
    if h > 2:
        colors.extend(int(grid[r][0]) for r in range(1, h - 1))
        if w > 1:
            colors.extend(int(grid[r][w - 1]) for r in range(1, h - 1))
    return Counter(colors)


def _component_masks(arr: Grid, bg_color: int) -> List[Dict[str, object]]:
    h, w = shape(arr)
    comps = []
    for comp in components(arr, ignore_color=bg_color):
        pixels = comp.get("pixels", [])  # type: ignore[assignment]
        coords = [(int(y), int(x)) for y, x in pixels]
        mask = _mask_from_pixels(h, w, coords)
        color_counts = Counter(int(arr[y][x]) for y, x in coords)
        mode_color = int(color_counts.most_common(1)[0][0]) if color_counts else 0
        comps.append({
            "mask": mask,
            "coords": coords,
            "size": len(coords),
            "mode_color": mode_color,
        })
    return comps


def _training_color_priors(train_examples: Sequence[Example]) -> Dict[str, Optional[int]]:
    border_counter: Counter = Counter()
    bridge_counter: Counter = Counter()
    swap_counter: Counter = Counter()

    for ex in train_examples:
        inp = _ensure_grid(ex.get("input", []))
        out = _ensure_grid(ex.get("output", []))
        h_in, w_in = shape(inp)
        h_out, w_out = shape(out)
        if h_out == 0 or w_out == 0:
            continue
        border_counter.update(_extract_border_colors(out))
        if (h_in, w_in) == (h_out, w_out) and h_in * w_in:
            diff_coords = [
                (r, c)
                for r in range(h_out)
                for c in range(w_out)
                if inp[r][c] != out[r][c]
            ]
            for r, c in diff_coords:
                bridge_counter[int(out[r][c])] += 1
            bg_in = _mode_color(inp)
            for r in range(h_out):
                for c in range(w_out):
                    if inp[r][c] == bg_in and out[r][c] != bg_in:
                        swap_counter[int(out[r][c])] += 1

    def _dominant(counter: Counter) -> Optional[int]:
        return int(counter.most_common(1)[0][0]) if counter else None

    return {
        "border_color": _dominant(border_counter),
        "bridge_color": _dominant(bridge_counter),
        "swap_color": _dominant(swap_counter),
    }


def _component_centroid(comp: Dict[str, object]) -> Tuple[float, float]:
    coords: List[Tuple[int, int]] = comp["coords"]  # type: ignore[index]
    if not coords:
        return 0.0, 0.0
    ys = [y for y, _ in coords]
    xs = [x for _, x in coords]
    return float(sum(ys)) / len(ys), float(sum(xs)) / len(xs)


def _outline_neighbors(arr: Grid, mask: Mask) -> Counter:
    h, w = shape(arr)
    counts: Counter = Counter()
    for y, x in _mask_coords(mask):
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not mask[ny][nx]:
                counts[int(arr[ny][nx])] += 1
    return counts


def _manhattan_path(start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
    y0, x0 = start
    y1, x1 = end
    path: List[Tuple[int, int]] = []
    y, x = y0, x0
    while y != y1:
        y += 1 if y1 > y else -1
        path.append((y, x))
    while x != x1:
        x += 1 if x1 > x else -1
        path.append((y, x))
    return path


def _closest_pair_indices(comps: List[Dict[str, object]], topk: int = 3) -> List[Tuple[int, int]]:
    pairs: List[Tuple[float, Tuple[int, int]]] = []
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            ci = _component_centroid(comps[i])
            cj = _component_centroid(comps[j])
            dist = abs(ci[0] - cj[0]) + abs(ci[1] - cj[1])
            pairs.append((dist, (i, j)))
    pairs.sort(key=lambda x: x[0])
    return [pair for _, pair in pairs[:topk]]


def _bridge_between(comps: List[Dict[str, object]], idx_a: int, idx_b: int) -> List[Tuple[int, int]]:
    mask_a = comps[idx_a]["mask"]  # type: ignore[index]
    mask_b = comps[idx_b]["mask"]  # type: ignore[index]
    coords_a = _mask_coords(mask_a)
    coords_b = _mask_coords(mask_b)
    best_dist = math.inf
    best_pair: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
    for ya, xa in coords_a:
        for yb, xb in coords_b:
            dist = abs(ya - yb) + abs(xa - xb)
            if dist < best_dist:
                best_dist = dist
                best_pair = ((ya, xa), (yb, xb))
    if best_pair is None:
        return []
    return _manhattan_path(best_pair[0], best_pair[1])


def _gap_fill_cells(comps: List[Dict[str, object]], union_mask: Mask) -> List[Tuple[int, int]]:
    h = len(union_mask)
    w = len(union_mask[0]) if h else 0
    ownership = [[-1 for _ in range(w)] for _ in range(h)]
    for idx, comp in enumerate(comps):
        mask = comp["mask"]  # type: ignore[index]
        for y, x in _mask_coords(mask):
            ownership[y][x] = idx
    additions: List[Tuple[int, int]] = []
    for y in range(h):
        for x in range(w):
            if union_mask[y][x]:
                continue
            neighbors = set()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    owner = ownership[ny][nx]
                    if owner >= 0:
                        neighbors.add(owner)
            if len(neighbors) >= 2:
                additions.append((y, x))
    return additions


def _values_for_mask(grid: Grid, mask: Mask) -> List[int]:
    values: List[int] = []
    for y, x in _mask_coords(mask):
        values.append(int(grid[y][x]))
    return values


def _apply_mask_from_grid(base_color: int, mask: Mask, src: Grid) -> Grid:
    h, w = shape(src)
    candidate = _zero_grid(h, w, base_color)
    for y, x in _mask_coords(mask):
        candidate[y][x] = int(src[y][x])
    return candidate


def _apply_mask_constant(base_color: int, mask: Mask, value: int, h: int, w: int) -> Grid:
    candidate = _zero_grid(h, w, base_color)
    for y, x in _mask_coords(mask):
        candidate[y][x] = int(value)
    return candidate


def _candidate_grid(candidate: Any) -> Grid:
    if isinstance(candidate, dict) and "grid" in candidate:
        return _ensure_grid(candidate.get("grid", []))
    if isinstance(candidate, tuple) and len(candidate) == 2:
        grid_like, _meta = candidate
        return _ensure_grid(grid_like)
    return _ensure_grid(candidate)


def _dedup_append(results: List[Any], candidate: Any) -> None:
    grid = _candidate_grid(candidate)
    if not grid:
        return
    for existing in results:
        if _candidate_grid(existing) == grid:
            return
    if isinstance(candidate, dict):
        candidate = {**candidate}
        candidate["grid"] = grid
        results.append(candidate)
    elif isinstance(candidate, tuple) and len(candidate) == 2:
        grid_like, meta = candidate
        results.append((_ensure_grid(grid_like), dict(meta)))
    else:
        results.append(grid)


def _wrap_candidate(grid: Grid, *, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"grid": grid}
    if meta:
        payload["meta"] = dict(meta)
    return payload


# ----------------------------------------------------------------------------
# Context construction and heuristics


def _training_diff_map(train_examples: Sequence[Example]) -> Optional[Dict[str, object]]:
    pattern_counts: Dict[Tuple[int, int], Counter] = {}
    size_counter: Counter = Counter()

    for ex in train_examples:
        inp = _ensure_grid(ex.get("input", []))
        out = _ensure_grid(ex.get("output", []))
        h_in, w_in = shape(inp)
        h_out, w_out = shape(out)
        if (h_in, w_in) != (h_out, w_out) or h_out == 0 or w_out == 0:
            continue

        diff_coords = [
            (r, c)
            for r in range(h_out)
            for c in range(w_out)
            if inp[r][c] != out[r][c]
        ]
        if not diff_coords or len(diff_coords) > 48:
            continue

        ys = [r for r, _ in diff_coords]
        xs = [c for _, c in diff_coords]
        r0, r1 = min(ys), max(ys)
        c0, c1 = min(xs), max(xs)
        size = (r1 - r0 + 1, c1 - c0 + 1)
        size_counter[size] += 1

        for r, c in diff_coords:
            key = (r - r0, c - c0)
            color = int(out[r][c])
            pattern_counts.setdefault(key, Counter()).update([color])

    if not pattern_counts:
        return None

    entries = {
        key: counter.most_common(1)[0][0]
        for key, counter in pattern_counts.items()
    }
    dominant_size = None
    if size_counter:
        dominant_size = max(size_counter.items(), key=lambda kv: kv[1])[0]

    return {
        "size": dominant_size,
        "entries": entries,
    }


def _training_notch_offsets(train_examples: Sequence[Example]) -> Optional[List[Tuple[int, int, int]]]:
    offsets: Dict[Tuple[int, int], Counter] = defaultdict(Counter)

    for ex in train_examples:
        inp = _ensure_grid(ex.get("input", []))
        out = _ensure_grid(ex.get("output", []))
        h_in, w_in = shape(inp)
        h_out, w_out = shape(out)
        if (h_in, w_in) != (h_out, w_out) or h_out == 0 or w_out == 0:
            continue

        diff_coords = [
            (r, c)
            for r in range(h_out)
            for c in range(w_out)
            if inp[r][c] != out[r][c]
        ]
        if not diff_coords or len(diff_coords) > 24:
            continue

        diff_colors = Counter(int(out[r][c]) for r, c in diff_coords)
        if not diff_colors:
            continue
        minority_color, _ = diff_colors.most_common()[-1]

        centre_r = (h_out - 1) / 2.0
        centre_c = (w_out - 1) / 2.0
        for r, c in diff_coords:
            color = int(out[r][c])
            if color != minority_color:
                continue
            dy = int(round(r - centre_r))
            dx = int(round(c - centre_c))
            offsets[(dy, dx)].update([color])

    if not offsets:
        return None

    results: List[Tuple[int, int, int]] = []
    for (dy, dx), counter in offsets.items():
        color = counter.most_common(1)[0][0]
        results.append((int(dy), int(dx), int(color)))
    return results


def build_pattern_context(train_examples: Sequence[Example], test_input: Grid) -> Dict[str, object]:
    arr = _ensure_grid(test_input)
    bg_color = _mode_color(arr)
    priors = _training_color_priors(train_examples)
    diff_map = _training_diff_map(train_examples)
    notch_offsets = _training_notch_offsets(train_examples)
    hints = {
        "outline": _train_has_outline_hint(train_examples),
        "union": _train_has_union_hint(train_examples),
        "chasm": _train_has_chasm_hint(train_examples),
        "color_transition": bool(diff_map),
        "diff_offsets": bool(diff_map and diff_map.get("entries")),
        "notch": bool(notch_offsets),
        "cross": bool(diff_map and diff_map.get("size") == (2, 2)),
    }
    return {
        "bg_color": bg_color,
        "priors": priors,
        "hints": hints,
        "diff_map": diff_map,
        "notch_offsets": notch_offsets,
    }


def _train_has_outline_hint(train_examples: Sequence[Example]) -> bool:
    for ex in train_examples:
        inp = _ensure_grid(ex.get("input", []))
        out = _ensure_grid(ex.get("output", []))
        h_in, w_in = shape(inp)
        h_out, w_out = shape(out)
        if (h_in, w_in) != (h_out, w_out) or h_in == 0 or w_in == 0:
            continue
        bg = _mode_color(out)
        comps = _component_masks(inp, _mode_color(inp))
        if not comps:
            continue
        union_mask = _zero_mask(h_in, w_in)
        for comp in comps:
            _mask_or(union_mask, comp["mask"])  # type: ignore[arg-type]
        outline = _outline_mask(union_mask)
        candidate = _apply_mask_from_grid(bg, outline, out)
        if candidate == out:
            return True
    return False


def _train_has_union_hint(train_examples: Sequence[Example]) -> bool:
    for ex in train_examples:
        inp = _ensure_grid(ex.get("input", []))
        out = _ensure_grid(ex.get("output", []))
        h_in, w_in = shape(inp)
        h_out, w_out = shape(out)
        if h_in == 0 or w_in == 0 or h_out == 0 or w_out == 0:
            continue
        comps_in = _component_masks(inp, _mode_color(inp))
        comps_out = _component_masks(out, _mode_color(out))
        if not comps_in or not comps_out:
            continue
        num_in = len(comps_in)
        num_out = len(comps_out)
        area_in = sum(_mask_sum(comp["mask"]) for comp in comps_in)  # type: ignore[arg-type]
        area_out = sum(_mask_sum(comp["mask"]) for comp in comps_out)  # type: ignore[arg-type]
        if num_out < num_in and area_out >= area_in:
            return True
    return False


def _train_has_chasm_hint(train_examples: Sequence[Example]) -> bool:
    for ex in train_examples:
        inp = _ensure_grid(ex.get("input", []))
        out = _ensure_grid(ex.get("output", []))
        h_in, w_in = shape(inp)
        h_out, w_out = shape(out)
        if (h_in, w_in) != (h_out, w_out) or h_in == 0 or w_in == 0:
            continue
        comps = _component_masks(inp, _mode_color(inp))
        if not comps:
            continue
        union_mask = _zero_mask(h_in, w_in)
        for comp in comps:
            _mask_or(union_mask, comp["mask"])  # type: ignore[arg-type]
        holes = _find_holes(union_mask)
        if not holes:
            continue
        bg_out = _mode_color(out)
        for hole in holes:
            candidate = _apply_mask_from_grid(bg_out, hole, out)
            if candidate == out:
                return True
    return False


def heuristic_pattern_priors(
    train_examples: Sequence[Example], cand: Dict[str, object], meta: Optional[Dict[str, object]]
) -> float:
    if not meta:
        return 0.0
    hints = meta.get("hints", {}) if isinstance(meta, dict) else {}
    typ = cand.get("type", "")
    pattern_name = cand.get("pattern_name")
    pattern_variant = cand.get("pattern_variant") or cand.get("meta", {}).get("pattern_variant")
    bonus = 0.0
    if typ == "pattern_outline" and hints.get("outline"):
        bonus += 0.08
    elif typ == "pattern_union" and hints.get("union"):
        bonus += 0.08
    elif typ == "pattern_chasm" and hints.get("chasm"):
        bonus += 0.08

    if typ == "pattern_union" and hints.get("color_transition") and pattern_variant in {
        "diff_outline",
        "diff_replay",
    }:
        bonus += 0.04

    if pattern_name == "bar_clone" and isinstance(cand.get("bar_width"), int):
        bonus += min(0.02 * max(int(cand.get("bar_width", 0)), 0), 0.06)

    if pattern_name == "cross_flip" and hints.get("union"):
        bonus += 0.03

    if pattern_name == "notch" and hints.get("notch"):
        bonus += 0.03

    return min(max(bonus, 0.0), 0.12)


# ----------------------------------------------------------------------------
# Pattern generators


def generate_outline_patterns(
    train_examples: Sequence[Example], test_input: Grid, params: Dict[str, object]
) -> List[Any]:
    arr = _ensure_grid(test_input)
    h, w = shape(arr)
    if h == 0 or w == 0:
        return []
    bg_color = int(params.get("bg_color", _mode_color(arr)))
    priors = params.get("priors", {}) if isinstance(params, dict) else {}
    border_color = priors.get("border_color") if isinstance(priors, dict) else None

    comps = _component_masks(arr, bg_color)
    if not comps:
        return []

    union_mask = _zero_mask(h, w)
    for comp in comps:
        _mask_or(union_mask, comp["mask"])  # type: ignore[arg-type]
    union_outline = _outline_mask(union_mask)
    union_values = _values_for_mask(arr, union_mask)
    union_mode_color = _mode_color(union_values) if union_values else bg_color

    results: List[Any] = []

    for comp in comps:
        mask = comp["mask"]  # type: ignore[index]
        outline = _outline_mask(mask)
        if not _mask_any(outline):
            continue
        candidate_orig = _apply_mask_from_grid(bg_color, outline, arr)
        _dedup_append(results, candidate_orig)
        if border_color is not None:
            candidate_norm = _apply_mask_constant(bg_color, outline, int(border_color), h, w)
            _dedup_append(results, candidate_norm)

    if _mask_any(union_outline):
        cand_union = _apply_mask_constant(bg_color, union_outline, union_mode_color, h, w)
        _dedup_append(results, cand_union)
        if border_color is not None:
            cand_union_prior = _apply_mask_constant(bg_color, union_outline, int(border_color), h, w)
            _dedup_append(results, cand_union_prior)

        bbox = _union_bounding_box(union_mask)
        if bbox is not None:
            r0, r1, c0, c1 = bbox
            rect_mask = _zero_mask(h, w)
            for y in range(r0, r1 + 1):
                rect_mask[y][c0] = True
                rect_mask[y][c1] = True
            for x in range(c0, c1 + 1):
                rect_mask[r0][x] = True
                rect_mask[r1][x] = True
            rect_color = (
                int(border_color)
                if border_color is not None
                else union_mode_color
            )
            cand_rect = _apply_mask_constant(bg_color, rect_mask, rect_color, h, w)
            _dedup_append(
                results,
                _wrap_candidate(
                    cand_rect,
                    meta={"pattern_variant": "bbox_outline", "rect_color": rect_color},
                ),
            )

    return results


def generate_union_patterns(
    train_examples: Sequence[Example], test_input: Grid, params: Dict[str, object]
) -> List[Any]:
    arr = _ensure_grid(test_input)
    h, w = shape(arr)
    if h == 0 or w == 0:
        return []
    bg_color = int(params.get("bg_color", _mode_color(arr)))
    priors = params.get("priors", {}) if isinstance(params, dict) else {}
    bridge_color = priors.get("bridge_color") if isinstance(priors, dict) else None
    hints = params.get("hints", {}) if isinstance(params, dict) else {}
    diff_map = params.get("diff_map") if isinstance(params, dict) else None

    comps = _component_masks(arr, bg_color)
    if len(comps) < 2:
        return []

    union_mask = _zero_mask(h, w)
    for comp in comps:
        _mask_or(union_mask, comp["mask"])  # type: ignore[arg-type]

    union_values = _values_for_mask(arr, union_mask)
    fill_color = (
        int(bridge_color)
        if bridge_color is not None
        else _mode_color(union_values) if union_values else bg_color
    )

    union_outline = _outline_mask(union_mask)

    results: List[Any] = []

    cand_union = _apply_mask_constant(bg_color, union_mask, fill_color, h, w)
    _dedup_append(results, cand_union)

    cand_union_orig = _apply_mask_from_grid(bg_color, union_mask, arr)
    _dedup_append(results, cand_union_orig)

    if hints.get("color_transition") and _mask_any(union_outline):
        transition_grid = _copy_grid(arr)
        transition_color = int(bridge_color) if bridge_color is not None else fill_color
        for y, x in _mask_coords(union_outline):
            transition_grid[y][x] = transition_color
        _dedup_append(
            results,
            _wrap_candidate(
                transition_grid,
                meta={
                    "pattern_variant": "diff_outline",
                    "fill_color": transition_color,
                },
            ),
        )

    if diff_map and diff_map.get("entries") and _mask_any(union_mask):
        bbox = _union_bounding_box(union_mask)
        if bbox is not None:
            r0, _r1, c0, _c1 = bbox
            replay_grid = _copy_grid(arr)
            for (dy, dx), color in diff_map.get("entries", {}).items():
                ty = r0 + int(dy)
                tx = c0 + int(dx)
                if 0 <= ty < h and 0 <= tx < w:
                    replay_grid[ty][tx] = int(color)
            _dedup_append(
                results,
                _wrap_candidate(
                    replay_grid,
                    meta={
                        "pattern_variant": "diff_replay",
                        "fill_color": fill_color,
                    },
                ),
            )

    for i, j in _closest_pair_indices(comps, topk=min(3, len(comps) - 1)):
        mask_i = comps[i]["mask"]  # type: ignore[index]
        mask_j = comps[j]["mask"]  # type: ignore[index]
        pair_union = _zero_mask(h, w)
        _mask_or(pair_union, mask_i)
        _mask_or(pair_union, mask_j)
        cand_pair = _apply_mask_from_grid(bg_color, pair_union, arr)
        _dedup_append(results, cand_pair)

        bridge_path = _bridge_between(comps, i, j)
        if bridge_path:
            cand_bridge = _copy_grid(cand_pair)
            bridge_val = (
                int(bridge_color)
                if bridge_color is not None
                else int(comps[i]["mode_color"])  # type: ignore[index]
            )
            for y, x in bridge_path:
                if 0 <= y < h and 0 <= x < w:
                    cand_bridge[y][x] = bridge_val
            _dedup_append(results, cand_bridge)

    gap_cells = _gap_fill_cells(comps, union_mask)
    if gap_cells:
        cand_gap = _copy_grid(arr)
        gap_color = int(bridge_color) if bridge_color is not None else fill_color
        for y, x in gap_cells:
            cand_gap[y][x] = gap_color
        _dedup_append(results, cand_gap)

    return results


def _find_holes(union_mask: Mask) -> List[Mask]:
    bbox = _union_bounding_box(union_mask)
    if bbox is None:
        return []
    r0, r1, c0, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    complement = _zero_mask(h, w)
    for y in range(h):
        for x in range(w):
            complement[y][x] = not union_mask[r0 + y][c0 + x]
    visited = _zero_mask(h, w)
    holes: List[Mask] = []
    for y in range(h):
        for x in range(w):
            if visited[y][x] or not complement[y][x]:
                continue
            stack = [(y, x)]
            visited[y][x] = True
            coords: List[Tuple[int, int]] = []
            touches_border = False
            while stack:
                cy, cx = stack.pop()
                coords.append((cy, cx))
                if cy in (0, h - 1) or cx in (0, w - 1):
                    touches_border = True
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and not visited[ny][nx] and complement[ny][nx]:
                        visited[ny][nx] = True
                        stack.append((ny, nx))
            if touches_border:
                continue
            mask = _zero_mask(len(union_mask), len(union_mask[0]) if union_mask else 0)
            for cy, cx in coords:
                mask[r0 + cy][c0 + cx] = True
            holes.append(mask)
    return holes


def generate_chasm_patterns(
    train_examples: Sequence[Example], test_input: Grid, params: Dict[str, object]
) -> List[Grid]:
    arr = _ensure_grid(test_input)
    h, w = shape(arr)
    if h == 0 or w == 0:
        return []
    bg_color = int(params.get("bg_color", _mode_color(arr)))
    priors = params.get("priors", {}) if isinstance(params, dict) else {}
    swap_color = priors.get("swap_color") if isinstance(priors, dict) else None

    comps = _component_masks(arr, bg_color)
    if not comps:
        return []

    union_mask = _zero_mask(h, w)
    for comp in comps:
        _mask_or(union_mask, comp["mask"])  # type: ignore[arg-type]

    holes = _find_holes(union_mask)
    if not holes:
        for comp in comps:
            coords = comp["coords"]  # type: ignore[index]
            if coords and all(0 < y < h - 1 and 0 < x < w - 1 for y, x in coords):
                holes.append(comp["mask"])  # type: ignore[index]
    if not holes:
        return []

    results: List[Grid] = []

    for hole in holes:
        ring_counts = _outline_neighbors(arr, hole)
        ring_color = ring_counts.most_common(1)[0][0] if ring_counts else bg_color
        hole_values = _values_for_mask(arr, hole)
        hole_color = hole_values[0] if hole_values else None

        candidate_ring = _apply_mask_constant(bg_color, hole, int(ring_color), h, w)
        _dedup_append(results, candidate_ring)

        if swap_color is not None:
            candidate_swap = _apply_mask_constant(bg_color, hole, int(swap_color), h, w)
            _dedup_append(results, candidate_swap)

        if hole_color is not None and hole_color != bg_color:
            candidate_hole = _apply_mask_constant(bg_color, hole, int(hole_color), h, w)
            _dedup_append(results, candidate_hole)

    largest = max(holes, key=_mask_sum)
    if largest:
        ring_counts = _outline_neighbors(arr, largest)
        ring_color = ring_counts.most_common(1)[0][0] if ring_counts else bg_color
        cand_largest = _apply_mask_constant(bg_color, largest, int(ring_color), h, w)
        _dedup_append(results, cand_largest)
        if swap_color is not None:
            cand_largest_swap = _apply_mask_constant(bg_color, largest, int(swap_color), h, w)
            _dedup_append(results, cand_largest_swap)

    if swap_color is not None:
        filled = _copy_grid(arr)
        for hole in holes:
            for y, x in _mask_coords(hole):
                filled[y][x] = int(swap_color)
        _dedup_append(results, filled)

    return results


__all__ = [
    "build_pattern_context",
    "generate_outline_patterns",
    "generate_union_patterns",
    "generate_chasm_patterns",
    "heuristic_pattern_priors",
]
