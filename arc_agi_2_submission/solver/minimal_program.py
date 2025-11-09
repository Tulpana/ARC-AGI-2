"""Minimal synthesis loop targeting high-impact ARC patterns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Literal, Sequence, Tuple

import logging

import numpy as np

from core.components import cc4
from core.delta_core import align_and_delta
from finishers.ring_fill import fill_interior
from finishers.translate_object import apply_translation, best_translation
from gating.tau import beam_for_tau, tau_readiness
from ops.recolor_map import apply_recolor, learn_recolor
from ops.lgpc import learn_lgpc

logger = logging.getLogger(__name__)

Grid = np.ndarray
Demo = Tuple[Grid, Grid]
Program = Callable[[Grid], Grid]


@dataclass
class Anchor:
    mask: np.ndarray
    color: int
    kind: Literal[
        "stripeH",
        "stripeV",
        "path",
        "block",
        "line",
        "gridband",
        "none",
    ]
    axis: Literal["H", "V", "none"]
    skeleton: np.ndarray


@dataclass
class Part:
    mask: np.ndarray
    color: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    family: Literal[
        "blob",
        "thin",
        "glyph",
        "tri",
        "L",
        "ringish",
        "unknown",
    ]


@dataclass
class Behavior:
    verb: Literal["attach", "merge", "translate", "copy", "grow_rail", "fill_ring", "none"]
    target: Literal["anchor_edge", "anchor_skeleton", "border", "pairwise", "legend_indexed"]
    geom_mode: Literal[
        "morph_dilate",
        "snap_bbox_edge",
        "snap_bbox_corner",
        "rail_grow",
        "ring_grow",
    ]
    recolor_mode: Literal["none", "contact_band_to(anchor)", "all_to(anchor)", "legend_map"]
    scope: Literal["touching_only", "within_k", "whole_component"]
    axis_hint: Literal["H", "V", "none"]
    contact_band: int


def _extract_components(grid: Grid) -> list[dict[str, object]]:
    components: list[dict[str, object]] = []
    palette = [int(value) for value in np.unique(grid) if int(value) != 0]
    for color in palette:
        labels, infos = cc4(grid == color)
        for cid, bbox, size in infos:
            mask = labels == cid
            components.append(
                {
                    "mask": mask,
                    "bbox": bbox,
                    "size": int(size),
                    "color": int(color),
                }
            )
    return components


def _component_centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return (0.0, 0.0)
    return (float(ys.mean()), float(xs.mean()))


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return (0, 0, 0, 0)
    return (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()))


def _compute_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.logical_and(a, b).sum())
    if inter == 0:
        return 0.0
    union = float(np.logical_or(a, b).sum())
    return inter / union if union else 0.0


def _classify_anchor_kind(mask: np.ndarray) -> tuple[str, str]:
    y0, x0, y1, x1 = _bbox_from_mask(mask)
    height = max(1, y1 - y0 + 1)
    width = max(1, x1 - x0 + 1)
    area = int(mask.sum())
    bbox_area = height * width
    slenderness = width / height if height else 1.0

    if area == 0:
        return ("none", "none")

    if slenderness >= 3.0 and area / bbox_area > 0.6:
        return ("stripeH", "H")
    if slenderness <= 1 / 3 and area / bbox_area > 0.6:
        return ("stripeV", "V")
    if area <= max(height, width):
        if height >= width:
            return ("line", "V")
        return ("line", "H")
    if area > 0.8 * bbox_area:
        return ("block", "none")
    if slenderness >= 2.0:
        return ("path", "H")
    if slenderness <= 0.5:
        return ("path", "V")
    return ("gridband", "none")


def _anchor_skeleton(mask: np.ndarray, kind: str) -> np.ndarray:
    skeleton = np.zeros_like(mask, dtype=bool)
    if not mask.any():
        return skeleton
    ys, xs = np.nonzero(mask)
    if kind == "stripeH":
        row = int(round(float(ys.mean())))
        skeleton[row, xs.min() : xs.max() + 1] = True
    elif kind == "stripeV":
        col = int(round(float(xs.mean())))
        skeleton[ys.min() : ys.max() + 1, col] = True
    elif kind == "path" or kind == "line":
        for y, x in zip(ys, xs):
            skeleton[y, x] = True
    else:
        skeleton = mask.copy()
    return skeleton


def detect_anchors(examples: Sequence[Demo]) -> list[Anchor]:
    if not examples:
        return []
    first_in, _ = examples[0]
    components = _extract_components(first_in)
    anchors: list[Anchor] = []
    for comp in components:
        mask = np.asarray(comp["mask"], dtype=bool)
        color = int(comp["color"])
        iou_scores: list[float] = []
        for other_in, _ in examples[1:]:
            best = 0.0
            for other_comp in _extract_components(other_in):
                if int(other_comp["color"]) != color:
                    continue
                other_mask = np.asarray(other_comp["mask"], dtype=bool)
                score = _compute_iou(mask, other_mask)
                if score > best:
                    best = score
            if best:
                iou_scores.append(best)
        if iou_scores and min(iou_scores) < 0.9:
            continue
        overall = min(iou_scores) if iou_scores else 1.0
        kind, axis = _classify_anchor_kind(mask)
        skeleton = _anchor_skeleton(mask, kind)
        anchors.append(Anchor(mask=mask, color=color, kind=kind, axis=axis, skeleton=skeleton))
        logger.info("[ANCHOR] kind=%s axis=%s color=%d iou=%.3f", kind, axis, color, overall)
    return anchors


def _component_signature(mask: np.ndarray) -> tuple[int, int, int]:
    y0, x0, y1, x1 = _bbox_from_mask(mask)
    height = max(1, y1 - y0 + 1)
    width = max(1, x1 - x0 + 1)
    area = int(mask.sum())
    return (area, height, width)


def _match_component(template: Anchor, components: list[dict[str, object]]) -> np.ndarray | None:
    target_color = template.color
    t_area, t_height, t_width = _component_signature(template.mask)
    best_mask: np.ndarray | None = None
    best_score = -1.0
    for comp in components:
        if int(comp["color"]) != target_color:
            continue
        mask = np.asarray(comp["mask"], dtype=bool)
        c_area, c_height, c_width = _component_signature(mask)
        if c_area == 0:
            continue
        area_ratio = min(t_area, c_area) / max(t_area, c_area)
        height_ratio = min(t_height, c_height) / max(t_height, c_height)
        width_ratio = min(t_width, c_width) / max(t_width, c_width)
        score = (area_ratio + height_ratio + width_ratio) / 3.0
        if score > best_score:
            best_score = score
            best_mask = mask
    return best_mask


def match_anchors_in_grid(grid: Grid, templates: Sequence[Anchor]) -> list[Anchor]:
    components = _extract_components(grid)
    anchors: list[Anchor] = []
    for template in templates:
        mask = _match_component(template, components)
        if mask is None:
            continue
        kind = template.kind
        axis = template.axis
        if kind == "none":
            kind, axis = _classify_anchor_kind(mask)
        skeleton = _anchor_skeleton(mask, kind)
        anchors.append(Anchor(mask=mask, color=template.color, kind=kind, axis=axis, skeleton=skeleton))
    return anchors


def _classify_part_family(mask: np.ndarray) -> str:
    area, height, width = _component_signature(mask)
    if area == 0:
        return "unknown"
    if min(height, width) == 1 and max(height, width) >= 3:
        return "thin"
    if area <= 3:
        return "glyph"
    if area >= height * width * 0.9:
        return "blob"
    if height == width and area < height * width:
        return "ringish"
    return "unknown"


def detect_parts(grid: Grid, anchors: Sequence[Anchor]) -> list[Part]:
    if not anchors:
        anchors_mask = np.zeros_like(grid, dtype=bool)
    else:
        anchors_mask = np.zeros_like(grid, dtype=bool)
        for anchor in anchors:
            anchors_mask |= np.asarray(anchor.mask, dtype=bool)
    working = np.array(grid, copy=True)
    working[anchors_mask] = 0
    parts: list[Part] = []
    for comp in _extract_components(working):
        mask = np.asarray(comp["mask"], dtype=bool)
        bbox = _bbox_from_mask(mask)
        centroid = _component_centroid(mask)
        family = _classify_part_family(mask)
        parts.append(
            Part(
                mask=mask,
                color=int(comp["color"]),
                bbox=bbox,
                centroid=centroid,
                family=family,
            )
        )
    return parts


def detect_invariants(examples: Sequence[Demo]) -> np.ndarray:
    if not examples:
        return np.zeros((1, 1), dtype=bool)
    invariants = np.ones_like(examples[0][0], dtype=bool)
    for x, y in examples:
        invariants &= np.asarray(x) == np.asarray(y)
    return invariants


def build_protected_mask(anchors: Sequence[Anchor], invariants: np.ndarray) -> np.ndarray:
    protected = np.array(invariants, dtype=bool, copy=True)
    for anchor in anchors:
        protected |= np.asarray(anchor.mask, dtype=bool)
    logger.info("[SCAFFOLD] protected=%d", int(protected.sum()))
    return protected


def prenormalize_examples(examples: Sequence[Demo], anchors: Sequence[Anchor]) -> None:
    axes = [anchor.axis for anchor in anchors if anchor.axis != "none"]
    axis = axes[0] if axes else "none"
    logger.info("[PRENORM] axis=%s ops=%s", axis, "identity")


def _union_masks(items: Iterable[np.ndarray], *, shape: tuple[int, int] | None = None) -> np.ndarray:
    union: np.ndarray | None = None
    for item in items:
        mask = np.asarray(item, dtype=bool)
        if union is None:
            union = np.array(mask, copy=True)
        else:
            union |= mask
    if union is None:
        if shape is None:
            return np.zeros((1, 1), dtype=bool)
        return np.zeros(shape, dtype=bool)
    return union


def _dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    result = np.array(mask, dtype=bool, copy=True)
    for _ in range(iterations):
        expanded = result.copy()
        expanded[:-1, :] |= result[1:, :]
        expanded[1:, :] |= result[:-1, :]
        expanded[:, :-1] |= result[:, 1:]
        expanded[:, 1:] |= result[:, :-1]
        result = expanded
    return result


def _touches(mask_a: np.ndarray, mask_b: np.ndarray) -> bool:
    if not mask_a.any() or not mask_b.any():
        return False
    dilated = _dilate(mask_a, 1)
    return bool(np.logical_and(dilated, mask_b).any())


def morph_attach(piece_mask: np.ndarray, anchor_mask: np.ndarray, protected: np.ndarray, max_iter: int = 6) -> np.ndarray:
    piece = np.array(piece_mask, dtype=bool, copy=True)
    for _ in range(max_iter):
        if _touches(piece, anchor_mask):
            break
        candidate = _dilate(piece, 1)
        candidate[protected] = False
        piece = candidate
    return piece


def _shift_mask(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    shifted = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    ys = ys + dy
    xs = xs + dx
    valid = (ys >= 0) & (ys < height) & (xs >= 0) & (xs < width)
    shifted[ys[valid], xs[valid]] = True
    return shifted


def snap_to_edge(piece_mask: np.ndarray, anchor_mask: np.ndarray, axis: str, protected: np.ndarray) -> np.ndarray:
    best_mask = np.array(piece_mask, dtype=bool, copy=True)
    best_dist = float("inf")
    search = range(-6, 7)
    for dy in search:
        for dx in search:
            if axis == "H" and dy != 0:
                continue
            if axis == "V" and dx != 0:
                continue
            candidate = _shift_mask(piece_mask, dy, dx)
            if np.any(candidate & protected):
                continue
            dist = _distance_to_anchor(candidate, anchor_mask)
            if dist < best_dist:
                best_dist = dist
                best_mask = candidate
    return best_mask


def _distance_to_anchor(piece_mask: np.ndarray, anchor_mask: np.ndarray) -> float:
    if not piece_mask.any() or not anchor_mask.any():
        return float("inf")
    piece_pts = np.argwhere(piece_mask)
    anchor_pts = np.argwhere(anchor_mask)
    min_dist = float("inf")
    for y, x in piece_pts:
        distances = np.abs(anchor_pts[:, 0] - y) + np.abs(anchor_pts[:, 1] - x)
        candidate = float(distances.min())
        if candidate < min_dist:
            min_dist = candidate
    return min_dist


def rail_grow(piece_mask: np.ndarray, skeleton_mask: np.ndarray, axis: str, protected: np.ndarray) -> np.ndarray:
    result = np.array(piece_mask, dtype=bool, copy=True)
    target = np.array(skeleton_mask, dtype=bool, copy=False)
    for _ in range(6):
        contact = _touches(result, target)
        if contact:
            break
        candidate = _dilate(result, 1)
        candidate[protected] = False
        result = candidate
    grown = np.array(result, copy=True)
    grown[target] = True
    grown[protected] = False
    return grown


def recolor_contact(
    grid: Grid,
    piece_mask: np.ndarray,
    anchor_mask: np.ndarray,
    color: int,
    band: int = 1,
) -> int:
    if band <= 0:
        return 0
    band_mask = _dilate(anchor_mask, band) & piece_mask
    count = int(band_mask.sum())
    if count:
        grid[band_mask] = color
    return count


def learn_behaviors(
    examples: Sequence[Demo],
    templates: Sequence[Anchor],
    anchors_per_example: Sequence[Sequence[Anchor]],
    parts_per_example: Sequence[Sequence[Part]],
) -> list[Behavior]:
    behaviors: list[Behavior] = []
    if not templates:
        behaviors.append(
            Behavior(
                verb="none",
                target="border",
                geom_mode="morph_dilate",
                recolor_mode="none",
                scope="whole_component",
                axis_hint="none",
                contact_band=0,
            )
        )
        return behaviors

    attach_votes = 0
    recolor_contact_votes = 0
    anchor_axis = templates[0].axis if templates else "none"
    for (x, y), anchors, parts in zip(examples, anchors_per_example, parts_per_example):
        if not anchors:
            continue
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        anchor_mask = _union_masks((anchor.mask for anchor in anchors), shape=x_arr.shape)
        delta = y_arr != x_arr
        if np.any(delta):
            dilated = _dilate(anchor_mask, 1)
            if np.all(~delta | dilated):
                attach_votes += 1
            colors = np.unique(y_arr[delta])
            if len(colors) == 1 and int(colors[0]) == anchors[0].color:
                recolor_contact_votes += 1
        for part in parts:
            if _touches(part.mask, anchor_mask):
                attach_votes += 1

    if attach_votes:
        recolor_mode = "contact_band_to(anchor)" if recolor_contact_votes else "none"
        behaviors.append(
            Behavior(
                verb="attach",
                target="anchor_edge",
                geom_mode="morph_dilate",
                recolor_mode=recolor_mode,
                scope="touching_only",
                axis_hint=anchor_axis if anchor_axis else "none",
                contact_band=1,
            )
        )

    behaviors.append(
        Behavior(
            verb="none",
            target="border",
            geom_mode="morph_dilate",
            recolor_mode="none",
            scope="whole_component",
            axis_hint="none",
            contact_band=0,
        )
    )
    return behaviors


def _apply_geometry(
    piece: Part,
    behavior: Behavior,
    anchor_mask: np.ndarray,
    anchors: Sequence[Anchor],
    protected: np.ndarray,
) -> tuple[np.ndarray, int, bool]:
    original_mask = np.asarray(piece.mask, dtype=bool)
    new_mask = original_mask
    translated = False
    if behavior.verb == "attach":
        if behavior.geom_mode == "snap_bbox_edge":
            new_mask = snap_to_edge(original_mask, anchor_mask, behavior.axis_hint, protected)
            translated = not np.array_equal(new_mask, original_mask)
        elif behavior.geom_mode == "rail_grow" and anchors:
            new_mask = rail_grow(original_mask, anchors[0].skeleton, behavior.axis_hint, protected)
        else:
            new_mask = morph_attach(original_mask, anchor_mask, protected)
    if np.any(new_mask & protected):
        logger.info("[SAFETY] prune: violates_claim id=%d", 0)
        new_mask = np.array(new_mask, copy=True)
        new_mask[protected] = False
    grown = int(new_mask.sum()) - int(original_mask.sum())
    return new_mask, grown, translated


def apply_behavior(
    grid: Grid,
    behavior: Behavior,
    anchors: Sequence[Anchor],
    parts: Sequence[Part],
    protected: np.ndarray,
) -> Grid:
    if behavior.verb == "none":
        return np.array(grid, copy=True)
    if not anchors:
        return np.array(grid, copy=True)
    result = np.array(grid, copy=True)
    anchor_mask = (
        _union_masks((anchor.mask for anchor in anchors), shape=grid.shape)
        if anchors
        else np.zeros_like(grid, dtype=bool)
    )
    for part in parts:
        new_mask, grown, translated = _apply_geometry(part, behavior, anchor_mask, anchors, protected)
        touching = _touches(new_mask, anchor_mask) if anchors else False
        part_cells = np.asarray(part.mask, dtype=bool)
        result[part_cells & ~new_mask] = 0
        result[new_mask] = part.color
        logger.info(
            "[ATTACH] op=%s color=%d grown=%d translated=%d touching=%s",
            behavior.geom_mode,
            part.color,
            grown,
            1 if translated else 0,
            "yes" if touching else "no",
        )
        if behavior.recolor_mode == "contact_band_to(anchor)" and anchors:
            recolored = recolor_contact(result, new_mask, anchor_mask, anchors[0].color, behavior.contact_band)
            logger.info(
                "[RECOLOR] mode=%s band=%d recolored=%d",
                behavior.recolor_mode,
                behavior.contact_band,
                recolored,
            )
        elif behavior.recolor_mode == "all_to(anchor)" and anchors:
            mask = np.array(new_mask, copy=True)
            mask[protected] = False
            result[mask] = anchors[0].color
            logger.info(
                "[RECOLOR] mode=%s band=%d recolored=%d",
                behavior.recolor_mode,
                behavior.contact_band,
                int(mask.sum()),
            )
    return result


def synthesize_behavior_programs(demos: Sequence[Demo]) -> list[Program]:
    templates = detect_anchors(demos)
    invariants = detect_invariants(demos)
    build_protected_mask(templates, invariants)
    prenormalize_examples(demos, templates)
    anchors_per_example = [match_anchors_in_grid(x, templates) for x, _ in demos]
    parts_per_example = [detect_parts(x, anchors) for (x, _), anchors in zip(demos, anchors_per_example)]
    behaviors = learn_behaviors(demos, templates, anchors_per_example, parts_per_example)

    programs: list[Program] = []
    for rank, behavior in enumerate(behaviors):
        logger.info(
            "[BEHAVIOR] rank=%d verb=%s target=%s geom=%s recolor=%s band=%d axis=%s",
            rank,
            behavior.verb,
            behavior.target,
            behavior.geom_mode,
            behavior.recolor_mode,
            behavior.contact_band,
            behavior.axis_hint,
        )

        def program(grid: Grid, *, _behavior=behavior, _templates=templates) -> Grid:
            anchors = match_anchors_in_grid(grid, _templates)
            if not anchors:
                return np.array(grid, copy=True)
            protected = build_protected_mask(anchors, np.zeros_like(grid, dtype=bool))
            parts = detect_parts(grid, anchors)
            return apply_behavior(grid, _behavior, anchors, parts, protected)

        programs.append(program)

    return programs


def predict_one(demos: Sequence[Demo], test_in: Grid) -> Grid:
    """Predict the output grid for ``test_in`` using demonstrations."""

    demos_np: List[Demo] = [
        (np.array(x, dtype=np.uint8, copy=False), np.array(y, dtype=np.uint8, copy=False))
        for x, y in demos
    ]
    recolor_map = learn_recolor(demos_np)

    _, _, additions, deletions, recolors = align_and_delta(demos_np[0][0], demos_np[0][1])
    _, comps = cc4(demos_np[0][1] != 0)
    tau = tau_readiness(int(additions.sum()), int(deletions.sum()), int(recolors.sum()), len(comps))
    beam = beam_for_tau(tau)

    candidates: List[Program] = []

    lgpc_program = learn_lgpc(demos_np)
    if lgpc_program is not None:
        candidates.append(lgpc_program)

    behavior_programs = synthesize_behavior_programs(demos_np)
    candidates.extend(behavior_programs)

    bt = best_translation(demos_np[0][0], demos_np[0][1], color=None, max_shift=5)
    if bt and bt[3] >= 0.8:
        color, (dy, dx), bbox, _ = bt

        def prog_translate(grid: Grid, *, _color=color, _dy=dy, _dx=dx, _bbox=bbox) -> Grid:
            return apply_translation(grid, _color, _dy, _dx, _bbox)

        candidates.append(prog_translate)

    _, comps_y = cc4(demos_np[0][1] != 0)
    ring_color = dominant_added_color(demos_np[0][0], demos_np[0][1])
    if comps_y and ring_color is not None:
        bbox = comps_y[0][1]

        def prog_ring(grid: Grid, *, _bbox=bbox, _ring=ring_color) -> Grid:
            centre = grid[(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]
            return fill_interior(grid, _bbox, _ring, keep_center=int(centre))

        candidates.append(prog_ring)

    prog_recolor: Program | None = None
    if recolor_map:
        def prog_recolor(grid: Grid, *, _map=recolor_map) -> Grid:
            return apply_recolor(grid, _map)

        candidates.append(prog_recolor)

    pipelines: List[List[Program]] = [[fn] for fn in candidates]
    if prog_recolor is not None:
        for fn in candidates:
            if fn is prog_recolor:
                continue
            pipelines.append([prog_recolor, fn])
            pipelines.append([fn, prog_recolor])

    pipelines = pipelines[:beam]

    best_pipe: Sequence[Program] | None = None
    best_score = -1
    for pipe in pipelines:
        score = 0
        for x_demo, y_demo in demos_np:
            grid = np.array(x_demo, copy=True)
            for fn in pipe:
                grid = fn(grid)
            if np.array_equal(grid, y_demo):
                score += 1
        if score > best_score:
            best_score = score
            best_pipe = pipe

    output = np.array(test_in, dtype=np.uint8, copy=True)
    if best_pipe:
        for fn in best_pipe:
            output = fn(output)
    return output


def dominant_added_color(x: Grid, y: Grid) -> int | None:
    """Return the most frequent colour added between ``x`` and ``y``."""

    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    additions = (x_arr == 0) & (y_arr != 0)
    if not additions.any():
        return None
    values, counts = np.unique(y_arr[additions], return_counts=True)
    candidates = [(int(v), int(counts[i])) for i, v in enumerate(values) if v != 0]
    if not candidates:
        return None
    best, _ = max(candidates, key=lambda item: item[1])
    return best
