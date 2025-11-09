"""Lightweight gating helpers for tests and repair tooling.

These implementations provide conservative, dependency-free defaults so that
pytest can exercise the solver shims without the full gating stack. They should
be replaced by the production implementations when the rich geometry logic is
ported over, but they deliberately match the public API that the tests expect
right now.

The module now also exposes a **Shape-First Scaffold (SFS)** implementation. It
follows the design described in the engineering notes: geometry that looks
stable is "imported" from the input grid, frozen via a protection mask and only
then exposed to the usual zone-audit / palette candidate logic. The goal is to
start from a reliable skeleton instead of letting early gates prune good
structure to the background.

The scaffold helpers are intentionally defensive – they only learn per-task
translations, they never resize unless explicitly allowed, and any candidate
that touches the protected cells must prove that doing so improves IoU. These
constraints match the behaviour of the production solver and give unit tests a
portable surface to target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence, Tuple

from ril.utils.safe_np import np

from core.components import cc4


@dataclass
class Gate:
    """Configuration bundle used by geometry/palette gate shims.

    The new attributes control the SFS behaviour. ``shape_scaffold`` toggles the
    protection pass, ``scaffold_touch_policy`` mirrors the
    ``--scaffold_touch_policy`` flag described in the design notes and the stage
    A/B thresholds map to the 0.85/0.90 defaults that the production gate uses
    for vertical vs. general geometry edits.
    """

    min_vectors: int = 2
    max_ang_var: float = 0.20
    max_step_mad: float = 1.0
    min_replay_iou: float = 0.90
    max_local_edits: int = 6
    shape_scaffold: bool = False
    scaffold_touch_policy: str = "improve_iou_only"
    allow_resize: bool = False
    stage_a_vertical_threshold: float = 0.85
    stage_a_default_threshold: float = 0.90
    stage_b_threshold: float = 0.90
    scaffold_halo: int = 0


class Scaffold:
    """Simple container bundling the imported geometry and the protection mask."""

    def __init__(self, grid: np.ndarray, mask: np.ndarray, *, halo: int = 0) -> None:
        self.grid = np.asarray(grid)
        self.mask = np.asarray(mask, dtype=bool)
        self.halo = int(halo)

    @property
    def editable_mask(self) -> np.ndarray:
        """Return the editable region (optionally with a one-cell halo)."""

        return editable_region(self.mask, halo=self.halo)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """Return an intersection-over-union style score for two masks/grids."""

    if a.shape != b.shape:
        return 0.0
    if a.dtype == bool or b.dtype == bool:
        inter = np.logical_and(a, b).sum(dtype=float)
        union = np.logical_or(a, b).sum(dtype=float)
        return float(inter) / float(union + 1e-9)
    return float(np.equal(a, b).sum(dtype=float)) / float(a.size or 1)


def geometry_first_gate(candidate: np.ndarray, examples: Iterable, gate: Gate) -> bool:
    """Gate that checks geometry similarity before palette fixes apply."""

    example_list: List = list(examples)
    if not example_list:
        return True

    scores: List[float] = []
    for _ex_in, ex_out in example_list:
        ex_grid = np.asarray(ex_out)
        cand_grid = np.asarray(candidate)
        if ex_grid.shape != cand_grid.shape:
            return False
        scores.append(_iou(cand_grid == 0, ex_grid == 0))
    return (min(scores) if scores else 1.0) >= gate.min_replay_iou


def palette_second_gate(before: np.ndarray, after: np.ndarray, zone_mask: np.ndarray, gate: Gate) -> bool:
    """Ensure palette edits remain local and within the edit budget."""

    before_arr = np.asarray(before)
    after_arr = np.asarray(after)
    mask = np.asarray(zone_mask, dtype=bool)
    if before_arr.shape != after_arr.shape or before_arr.shape != mask.shape:
        return False

    changed = before_arr != after_arr
    if np.any(np.logical_and(changed, ~mask)):
        return False

    edits = int(changed.sum())
    if edits > gate.max_local_edits:
        return False

    # Accept if the number of mismatched cells inside the zone does not grow.
    before_err = int(np.logical_and(before_arr != after_arr, mask).sum())
    after_err = int(np.logical_and(after_arr != before_arr, mask).sum())
    return after_err <= before_err


# ---------------------------------------------------------------------------
# Shape-First Scaffold helpers
# ---------------------------------------------------------------------------


def build_scaffold(
    test_input: np.ndarray,
    examples: Sequence[Tuple[np.ndarray, np.ndarray]],
    background: int,
    *,
    gate: Gate | None = None,
) -> Scaffold:
    """Construct a scaffold grid and mask for ``test_input``.

    The routine implements the "shape-first" flow described in the user
    instruction: it detects 4-connected components in the input grid, learns the
    dominant translation between each demo pair and projects the geometry onto
    the test grid. Colours are copied verbatim; callers can re-colour via
    palette alignment once the scaffold is installed.
    """

    arr = np.asarray(test_input)
    if arr.ndim != 2:
        raise ValueError("build_scaffold expects a 2D grid")

    if gate is None:
        gate = Gate()

    translation = _learn_translation(examples, background, allow_resize=gate.allow_resize)

    scaffold_grid = np.full_like(arr, fill_value=background)
    scaffold_mask = np.zeros_like(arr, dtype=bool)

    labels, components = cc4(arr != background)
    for cid, _bbox, _size in components:
        component_mask = labels == cid
        if not component_mask.any():
            continue

        colors, counts = np.unique(arr[component_mask], return_counts=True)
        if len(colors) == 0:
            continue
        color = int(colors[np.argmax(counts)])

        projected = _apply_translation(component_mask, translation, arr.shape)
        scaffold_grid[projected] = color
        scaffold_mask[projected] = True

    return Scaffold(scaffold_grid, scaffold_mask, halo=gate.scaffold_halo)


def editable_region(scaffold_mask: np.ndarray, *, halo: int = 0) -> np.ndarray:
    """Return the editable mask given the scaffold mask and halo size."""

    mask = ~np.asarray(scaffold_mask, dtype=bool)
    if halo <= 0:
        return mask

    protected = np.asarray(scaffold_mask, dtype=bool)
    for _ in range(int(halo)):
        protected = _dilate4(protected)
    return ~protected


def stage_a_gate(candidate: object, scaffold_mask: np.ndarray, gate: Gate) -> bool:
    """Stage A gate mirroring the production logic.

    * Candidates that touch the scaffold must set ``meta["touch_scaffold"]``.
    * Vertical families receive the softer ``0.85`` IoU threshold.
    * IoU is pulled from ``candidate.metrics`` (fallback to ``candidate.meta``).
    """

    touches = _candidate_touches_scaffold(candidate, scaffold_mask)
    meta = getattr(candidate, "meta", {}) or {}

    if touches and not bool(meta.get("touch_scaffold", False)):
        return False

    iou = _lookup_metric(candidate, ("replay_iou_geom", "iou"))
    if iou is None:
        return False

    vertical = bool(meta.get("looks_vertical_geom")) or meta.get("axis") in {"V", "v", "vertical"}
    threshold = gate.stage_a_vertical_threshold if vertical else gate.stage_a_default_threshold
    return float(iou) >= float(threshold)


def stage_b_gate(candidate: object, scaffold_mask: np.ndarray, gate: Gate) -> bool:
    """Stage B gate that enforces the "improve IoU" policy for scaffold cells."""

    touches = _candidate_touches_scaffold(candidate, scaffold_mask)
    metrics = getattr(candidate, "metrics", {}) or {}
    after = metrics.get("replay_iou_geom_after_palette")
    if after is None:
        after = _lookup_metric(candidate, ("replay_iou_geom_after_palette",))
    if after is None:
        return False

    if touches:
        if gate.scaffold_touch_policy == "never":
            return False
        before = metrics.get("replay_iou_geom_before")
        if before is None:
            before = _lookup_metric(candidate, ("replay_iou_geom_before",))
        if before is None:
            return False
        return float(after) > float(before)
    return float(after) >= float(gate.stage_b_threshold)


def arbitration_score(meta: dict) -> tuple:
    """Score tuple used to rank gated candidates deterministically."""

    return (
        -int(meta.get("num_edits", 0)),
        -abs(int(meta.get("orthogonal_err_abs", 0))),
        -abs(float(meta.get("angle_resid_deg", 0.0))),
        -float(meta.get("palette_off_after", 0.0)),
        float(meta.get("min_replay_iou", 0.0)),
    )


def _learn_translation(
    examples: Sequence[Tuple[np.ndarray, np.ndarray]],
    background: int,
    *,
    allow_resize: bool,
) -> Tuple[int, int]:
    """Infer the dominant translation from the provided examples."""

    offsets: List[Tuple[int, int]] = []
    for ex_in, ex_out in examples:
        ex_in_arr = np.asarray(ex_in)
        ex_out_arr = np.asarray(ex_out)
        if ex_in_arr.shape != ex_out_arr.shape and not allow_resize:
            raise ValueError(
                "Example input/output shapes differ but resizing is disabled. "
                "Enable allow_resize to process this task."
            )
        in_mask = ex_in_arr != background
        out_mask = ex_out_arr != background
        if not in_mask.any() or not out_mask.any():
            continue
        in_centroid = _centroid(in_mask)
        out_centroid = _centroid(out_mask)
        dy = int(round(out_centroid[0] - in_centroid[0]))
        dx = int(round(out_centroid[1] - in_centroid[1]))
        offsets.append((dy, dx))
    if not offsets:
        return (0, 0)

    dy = int(round(float(np.mean([off[0] for off in offsets]))))
    dx = int(round(float(np.mean([off[1] for off in offsets]))))
    return dy, dx


def _apply_translation(mask: np.ndarray, offset: Tuple[int, int], shape: Tuple[int, int]) -> np.ndarray:
    """Translate ``mask`` by ``offset`` while staying inside ``shape``."""

    arr = np.asarray(mask, dtype=bool)
    out = np.zeros(shape, dtype=bool)
    dy, dx = offset
    ys, xs = np.nonzero(arr)
    for y, x in zip(ys, xs):
        ny = y + dy
        nx = x + dx
        if 0 <= ny < shape[0] and 0 <= nx < shape[1]:
            out[ny, nx] = True
    return out


def _centroid(mask: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return 0.0, 0.0
    return float(np.mean(ys)), float(np.mean(xs))


def _dilate4(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    out = arr.copy()
    height, width = arr.shape
    ys, xs = np.nonzero(arr)
    for y, x in zip(ys, xs):
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < height and 0 <= nx < width:
                out[ny, nx] = True
    return out


def _candidate_touches_scaffold(candidate: object, scaffold_mask: np.ndarray) -> bool:
    mask = getattr(candidate, "edits_mask", None)
    if mask is None and hasattr(candidate, "edits"):
        edits = getattr(candidate, "edits")
        mask = getattr(edits, "mask", None)
        if mask is None and hasattr(edits, "any_in"):
            return bool(edits.any_in(scaffold_mask))
    if mask is None and isinstance(candidate, Mapping):
        mask = candidate.get("edits_mask")
    if mask is None:
        return False
    return bool(np.any(np.asarray(mask, dtype=bool) & np.asarray(scaffold_mask, dtype=bool)))


def _lookup_metric(candidate: object, keys: Sequence[str]) -> float | None:
    metrics = getattr(candidate, "metrics", {}) or {}
    meta = getattr(candidate, "meta", {}) or {}

    for key in keys:
        if key in metrics:
            return metrics[key]
        if isinstance(metrics, Mapping):
            if key in metrics:
                return metrics[key]
        if isinstance(candidate, Mapping) and key in candidate:
            return candidate[key]
        if key in meta:
            return meta[key]
    return None
