#!/usr/bin/env python3
"""
ARC-AGI-2 Rule Induction Layer (RIL) Solver
5-Stage Pipeline: AE → Candidate → Beam → CSP → Output
- Event-layer injection wired inside solve_arc_task (respects ENABLE_EVENTS, EVENT_VERBOSE)
- Stronger beam scoring (perfect-train-fit boost, shape prior)
- Extra bread-and-butter transforms for early exacts
"""

import os

# ================================ LABEL LEAKAGE GUARD ================================
# This guard ensures the solver never runs in evaluation mode where ground truth
# labels might be accessible during hypothesis generation. Private eval must only
# be used for post-hoc scoring, never for learning or parameter tuning.
if os.getenv("RIL_EVAL_MODE", "0") == "1":
    raise RuntimeError(
        "[LABEL LEAKAGE PREVENTION] Solver must not run under RIL_EVAL_MODE=1. "
        "Ground truth access is forbidden during hypothesis generation. "
        "Set RIL_EVAL_MODE=0 for prediction, or run scoring separately via scripts/private_eval.py"
    )
# ===================================================================================
import random
import json
import hashlib
import math
import statistics
import time
from dataclasses import dataclass, field, asdict
from typing import Iterable, List, Dict, Tuple, Any, Optional, Sequence, Set, Mapping
from pathlib import Path
from collections import Counter, deque


def _now_ms() -> int:
    """Return current monotonic time in milliseconds."""

    return int(time.monotonic() * 1000)

ROUTER_POLICY_DEFAULT = os.environ.get("ARC_ROUTER_POLICY", "hybrid_adapters_first")

from ril.metrics import Metrics, log_pipeline_error

from solver.generalization import finalize_candidate as palette_finalize_candidate
from solver.post.exemplar_snap import collect_exemplar_signatures

from .dimensional_bank import dimensional_hypotheses

from . import rescue_utils

try:
    from .planar_delta import delta
except Exception:  # pragma: no cover - fallback for stdlib smoke
    def delta(a, b):
        H = len(a)
        W = len(a[0]) if H else 0
        total = 0
        for r in range(H):
            ar = a[r]
            br = b[r] if r < len(b) else []
            for c in range(W):
                bc = br[c] if c < len(br) else None
                total += int(ar[c] != bc)
        return total

from ril.np_compat import NP, USE_NUMPY, asgrid

try:  # pragma: no cover - optional dependency in some runtimes
    from gates.palette_rescue import palette_rescue as ensure_palette_rescue
except Exception:  # pragma: no cover - fallback when tests are not available
    ensure_palette_rescue = None  # type: ignore[assignment]

np = NP  # type: ignore

from .palette_pft import PalettePlanStep, compute_palette_feasibility
from .axis_projector import AxisProjector

try:
    from ril.train_to_test_transforms import generate_transform_candidates
except Exception:  # pragma: no cover - optional transforms bundle missing
    generate_transform_candidates = lambda test_grid, train_examples, **kw: []  # type: ignore

try:
    from ril.tiny_grid_specialist import solve_tiny_grid_direct, is_tiny_grid
except Exception:  # pragma: no cover - optional specialist unavailable
    solve_tiny_grid_direct = lambda test, train: []  # type: ignore
    is_tiny_grid = lambda g, threshold=3: False  # type: ignore

try:
    from ril.exact_match_telemetry import ExactMatchTracker
except Exception:  # pragma: no cover - telemetry optional when packaging lean builds
    ExactMatchTracker = None  # type: ignore

DEFAULT_PALETTE_SUPPORT = float(os.getenv("RIL_PALETTE_SUPPORT_MIN", "0.85"))
DEFAULT_CSP_EVIDENCE = float(os.getenv("RIL_CSP_EVIDENCE_MIN", "0.74"))
DEFAULT_PALETTE_FLOOR = int(os.getenv("RIL_PALETTE_FLOOR", "2"))
DEFAULT_PALETTE_RESCUE_KEEP = float(os.getenv("RIL_PALETTE_RESCUE_KEEP_RATIO", "0.6"))
DEFAULT_GATE_SOFT_TOPK = int(os.getenv("RIL_GATE_SOFT_TOPK", "64"))
DEFAULT_MIN_BEAM_AFTER_GATE = int(os.getenv("RIL_MIN_BEAM_AFTER_GATE", "8"))
DEFAULT_SCORING_MIN_AREA = int(os.getenv("RIL_SCORING_MIN_AREA", "10"))

ALLOW_TEST_COLOR_IF = 0.20

from ril.adapters.motif_crop import generate_motif_crop
from ril.pattern_ops import (
    build_pattern_context,
    generate_chasm_patterns,
    generate_outline_patterns,
    generate_union_patterns,
    heuristic_pattern_priors,
)

# ----------------------------- Types -----------------------------------------
Grid = List[List[int]]
Task = Dict[str, Any]


@dataclass
class GateResult:
    """Outcome of an individual gate applied to a candidate."""

    name: str
    accepted: bool
    reason: Optional[str] = None


@dataclass
class Candidate:
    """Internal representation of a solver candidate before normalisation."""

    grid: Grid
    kind: str
    score: float
    source: str = "adapter"
    meta: Dict[str, Any] = field(default_factory=dict)
    param_hash: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    gate_results: List[GateResult] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)

    def as_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "grid": self.grid,
            "confidence": _safe01(self.score),
            "score": _safe01(self.score),
            "type": self.kind or "unknown",
            "source": self.source or "adapter",
            "meta": {**self.meta},
            "op_id": self.kind or "unknown",
        }
        if self.param_hash:
            payload["param_hash"] = self.param_hash
        if self.extra:
            payload.update(self.extra)
        if self.scores:
            payload["scores"] = dict(self.scores)
        payload["meta"].setdefault("src", payload["source"])
        if self.gate_results:
            payload["meta"]["gate_results"] = [
                {k: v for k, v in {"name": gr.name, "accepted": gr.accepted, "reason": gr.reason}.items() if v is not None}
                for gr in self.gate_results
            ]
        return payload

    # Mapping compatibility helpers -------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        payload = self.as_payload()
        if key in payload:
            return payload[key]
        if key in self.extra:
            return self.extra[key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if key == "grid":
            self.grid = value  # type: ignore[assignment]
        elif key in {"confidence", "score"}:
            try:
                self.score = float(value)
            except Exception:
                self.score = 0.0
        elif key in {"type", "op_id"}:
            self.kind = str(value)
        elif key == "source":
            self.source = str(value)
        elif key == "meta" and isinstance(value, dict):
            self.meta = dict(value)
        elif key == "param_hash":
            self.param_hash = str(value) if value is not None else None
        else:
            self.extra[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return key in self.as_payload() or key in self.extra


@dataclass
class GateContext:
    """Aggregate hints derived from the abstract encoder for gating."""

    shape_preserving: bool
    allowed_shapes: set[Tuple[int, int]]
    allow_growth: bool
    allowed_palette: set[int]
    frozen_mask: Optional[List[List[bool]]] = None
    reference_input: Optional[Grid] = None
    frozen_density: float = 0.0
    palette_order: Tuple[int, ...] = field(default_factory=tuple)
    supplemental_palette: Dict[int, float] = field(default_factory=dict)
    core_palette: set[int] = field(default_factory=set)
    test_input_colors: Set[int] = field(default_factory=set)
    emergent_palette: Set[int] = field(default_factory=set)
    vanishing_palette: Set[int] = field(default_factory=set)
    evidence_example_colors: Set[int] = field(default_factory=set)
    evidence_test_colors: Set[int] = field(default_factory=set)
    evidence_weights: Dict[int, float] = field(default_factory=dict)
    dead_colors: Set[int] = field(default_factory=set)
    palette_attr_weight: float = 0.65
    palette_repulse_weight: float = 0.35
    palette_provenance_mode: str = "train_or_test"
    dead_color_policy: str = "ignore"
    palette_floor: int = DEFAULT_PALETTE_FLOOR
    palette_rescue_keep_ratio: float = DEFAULT_PALETTE_RESCUE_KEEP
    gate_soft_topk: int = DEFAULT_GATE_SOFT_TOPK
    palette_soft_floor: int = 0
    palette_soft_ratio: float = 0.0
    min_beam_after_gate: int = DEFAULT_MIN_BEAM_AFTER_GATE
    region_gate_mode: str = "strict"
    palette_gate_mode: str = "hard"
    shape_gate_mode: str = "strict"
    shape_soft_floor: int = 0
    shape_soft_ratio: float = 0.0
    shape_preview_tau_drop: float = 0.0
    train_output_palettes: List[List[int]] = field(default_factory=list)
    recolour_mapping_hint: Dict[int, int] = field(default_factory=dict)
    recolour_only_task: bool = False

# ------------------------- Utility primitives --------------------------------


def _grid_nonzero_area(grid: Grid) -> int:
    return sum(1 for row in grid for cell in row if cell != 0)


def _pool_distinct_colors(candidates: Iterable[Dict[str, Any]]) -> Set[int]:
    colors: Set[int] = set()
    for cand in candidates:
        grid = cand.get("grid")
        if not isinstance(grid, list):
            continue
        try:
            colors.update(_grid_colors(grid))
        except Exception:
            continue
    return colors


def _candidate_prior_conf(candidate: Any) -> float:
    """Return the upstream prior confidence for ``candidate`` if available."""

    scores_obj: Any = getattr(candidate, "scores", None)
    if isinstance(scores_obj, Mapping):
        value = scores_obj.get("prior_conf", 0.0)
    elif isinstance(candidate, dict):
        scores_map = candidate.get("scores")
        value = scores_map.get("prior_conf", 0.0) if isinstance(scores_map, Mapping) else 0.0
    else:
        value = 0.0
    try:
        return float(value)
    except Exception:
        return 0.0


def _route_candidates(
    external: Sequence[Dict[str, Any]],
    adapters: Sequence[Dict[str, Any]],
    cfg: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Return the preferred candidate pool under a simple confidence gate."""

    try:
        min_k = int(cfg.get("router_min_k", 3))
    except Exception:
        min_k = 3
    try:
        min_pri = float(cfg.get("router_min_prior", 0.30))
    except Exception:
        min_pri = 0.30

    ext_ok = len(external) >= min_k and max(
        (_candidate_prior_conf(c) for c in external),
        default=0.0,
    ) >= min_pri

    if not ext_ok:
        return list(external) + list(adapters)
    return list(external)


def _count_components_grid(grid: Grid) -> int:
    if not grid or not grid[0]:
        return 0
    h, w = grid_shape(grid)
    visited = [[False for _ in range(w)] for _ in range(h)]
    total = 0

    def neighbours(r: int, c: int) -> Iterable[Tuple[int, int]]:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if in_bounds(nr, nc, h, w):
                yield nr, nc

    for r in range(h):
        for c in range(w):
            if grid[r][c] == 0 or visited[r][c]:
                continue
            total += 1
            stack = [(r, c)]
            visited[r][c] = True
            while stack:
                cr, cc = stack.pop()
                for nr, nc in neighbours(cr, cc):
                    if visited[nr][nc]:
                        continue
                    if grid[nr][nc] == 0:
                        continue
                    visited[nr][nc] = True
                    stack.append((nr, nc))
    return total


def _average_components(candidates: Iterable[Dict[str, Any]]) -> float:
    total = 0.0
    count = 0
    for cand in candidates:
        grid = cand.get("grid")
        if not isinstance(grid, list):
            continue
        try:
            total += float(_count_components_grid(grid))
            count += 1
        except Exception:
            continue
    return total / count if count else 0.0


def _component_retention(before: Iterable[Dict[str, Any]], after: Iterable[Dict[str, Any]]) -> float:
    before_avg = _average_components(before)
    after_avg = _average_components(after)
    if before_avg <= 0.0:
        return 0.0 if after_avg <= 0.0 else 1.0
    return after_avg / before_avg


def grid_shape(g: Grid) -> Tuple[int, int]:
    return (len(g), len(g[0]) if g else 0)


def _dims(g: Grid) -> Tuple[int, int]:
    return grid_shape(g)


def _allowed_palette(
    train_pairs: Optional[Iterable[Tuple[Grid, Grid]]], test_grid: Grid
) -> Set[int]:
    palette: Set[int] = set()
    if train_pairs:
        for pair in train_pairs:
            if not isinstance(pair, tuple) or len(pair) != 2:
                continue
            inp, out = pair
            if isinstance(inp, list):
                palette.update(int(v) for row in inp for v in row)
            if isinstance(out, list):
                palette.update(int(v) for row in out for v in row)
    if isinstance(test_grid, list):
        palette.update(int(v) for row in test_grid for v in row)
    return palette


def _palette_completion_score(grid: Grid, target_palette: Set[int]) -> float:
    """Return a lightweight score describing palette coverage for ``grid``."""

    if not target_palette or not isinstance(grid, list):
        return 0.0

    try:
        predicted = {int(value) for row in grid for value in row}
    except Exception:
        return 0.0

    if not predicted:
        return 0.0

    overlap = len(predicted & target_palette)
    coverage = overlap / len(target_palette) if target_palette else 0.0

    extras = predicted - target_palette
    extra_ratio = len(extras) / max(1, len(predicted))
    bonus = 0.15 if extra_ratio == 0.0 and coverage >= 1.0 else 0.0
    return 0.6 * coverage + bonus


def _has_only_allowed_colors(grid: Grid, allowed: Set[int]) -> bool:
    for row in grid:
        for value in row:
            if value not in allowed:
                return False
    return True


def _enforce_canvas_size(
    grid: Grid, target_h: int, target_w: int, *, fill: int = 0
) -> Grid:
    h, w = _dims(grid)
    if h == target_h and w == target_w:
        return grid
    output = [[fill for _ in range(target_w)] for __ in range(target_h)]
    for y in range(min(h, target_h)):
        src_row = grid[y]
        out_row = output[y]
        for x in range(min(w, target_w)):
            out_row[x] = src_row[x]
    return output


def in_bounds(r: int, c: int, h: int, w: int) -> bool:
    return 0 <= r < h and 0 <= c < w

def copy_grid(g: Grid) -> Grid:
    return [row[:] for row in g]

def mode_color(g: Grid) -> int:
    cnt = Counter(cell for row in g for cell in row)
    return cnt.most_common(1)[0][0] if cnt else 0

def log_candidate(stage: str, idx: int, grid: Any, context: Mapping[str, Any] | None) -> None:
    """Emit human-readable diagnostics for a generated candidate.

    ``context`` is expected to contain a ``_metrics`` mapping populated by the
    palette finalisation logic.  When present we expose palette quality metrics
    alongside the existing trace logging so that smoke-test transcripts capture
    palette failures without requiring downstream tooling to decode the meta
    payload.
    """

    # existing logging...
    metrics_map: Mapping[str, Any] | None = None
    if context and isinstance(context, Mapping):
        raw_metrics = context.get("_metrics")
        if isinstance(raw_metrics, Mapping):
            metrics_map = raw_metrics

    if metrics_map:
        pal_raw = metrics_map.get("palette_completeness")
        ext_raw = metrics_map.get("extra_colour_rate")

        fmt_parts: list[str] = []
        if isinstance(pal_raw, (int, float)):
            fmt_parts.append(f"palette_completeness={float(pal_raw):.3f}")
        elif pal_raw is not None:
            fmt_parts.append(f"palette_completeness={pal_raw}")

        if isinstance(ext_raw, (int, float)):
            fmt_parts.append(f"extra_colour_rate={float(ext_raw):.3f}")
        elif ext_raw is not None:
            fmt_parts.append(f"extra_colour_rate={ext_raw}")

        if fmt_parts:
            metrics_str = " ".join(fmt_parts)
            print(f"[METRICS] stage={stage} idx={idx} {metrics_str}")


def log_fallback(reason: str, ext_max: float, thr: float, pool: int) -> None:
    print(f"[ROUTER-FALLBACK] reason={reason} ext_max={ext_max:.2f} thr={thr:.2f} pool={pool}")

def _safe01(x) -> float:
    try:
        return 0.0 if x is None else (1.0 if x > 1.0 else (0.0 if x < 0.0 else float(x)))
    except Exception:
        return 0.0


def _bbox_nonzero(grid: Grid) -> Tuple[int, int, int, int]:
    """Return the bounding box of non-zero cells in ``grid``."""

    if not isinstance(grid, list) or not grid or not isinstance(grid[0], list):
        return (0, 0, 0, 0)

    ys: List[int] = []
    xs: List[int] = []
    for y, row in enumerate(grid):
        for x, value in enumerate(row):
            if value != 0:
                ys.append(y)
                xs.append(x)

    if not ys:
        return (0, 0, 0, 0)

    return (min(ys), min(xs), max(ys), max(xs))


def _crop(grid: Grid, y0: int, x0: int, y1: int, x1: int) -> Grid:
    """Crop ``grid`` to the inclusive coordinates provided."""

    return [row[x0 : x1 + 1] for row in grid[y0 : y1 + 1]]


def _motif_crop_candidate(
    training_pairs: Optional[Sequence[Tuple[Grid, Grid]]],
    test_grid: Grid,
) -> Optional[Grid]:
    """Return a deterministic crop candidate derived from training outputs."""

    boxes: List[Tuple[int, int, int, int]] = []
    for pair in training_pairs or []:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        outg = pair[1]
        if not isinstance(outg, list) or not outg or not isinstance(outg[0], list):
            continue
        boxes.append(_bbox_nonzero(outg))

    if not boxes:
        return None

    y0 = min(box[0] for box in boxes)
    x0 = min(box[1] for box in boxes)
    y1 = min(box[2] for box in boxes)
    x1 = min(box[3] for box in boxes)

    if y1 < y0 or x1 < x0:
        return None

    if not isinstance(test_grid, list) or not test_grid or not isinstance(test_grid[0], list):
        return None

    h, w = len(test_grid), len(test_grid[0])
    if y0 < 0 or x0 < 0 or y1 >= h or x1 >= w:
        return None

    return _crop(test_grid, y0, x0, y1, x1)


def is_rectangular(g: Grid) -> bool:
    if not g: return True
    w = len(g[0])
    return all(len(r) == w for r in g)


def border_signature(grid: Optional[Grid]) -> Optional[Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]]:
    """Return a tuple describing the border of ``grid``.

    The signature captures the top row, bottom row, left column and right
    column.  ``None`` is returned for empty or malformed grids.
    """

    if not isinstance(grid, list) or not grid or not isinstance(grid[0], list) or not grid[0]:
        return None
    try:
        top = tuple(int(cell) for cell in grid[0])
        bottom = tuple(int(cell) for cell in grid[-1])
        left = tuple(int(row[0]) for row in grid)
        right = tuple(int(row[-1]) for row in grid)
    except Exception:
        return None
    return (top, bottom, left, right)


def border_palette_counter(grid: Optional[Grid]) -> Counter:
    """Collect a histogram of colors along the border of ``grid``."""

    counts: Counter[int] = Counter()
    if not isinstance(grid, list) or not grid or not isinstance(grid[0], list) or not grid[0]:
        return counts
    h, w = len(grid), len(grid[0])
    for c in range(w):
        counts.update([int(grid[0][c])])
        if h > 1:
            counts.update([int(grid[h - 1][c])])
    for r in range(1, h - 1):
        counts.update([int(grid[r][0])])
        if w > 1:
            counts.update([int(grid[r][w - 1])])
    return counts


def _palette_distribution_match(candidate: Counter, reference: Counter, tolerance: float = 0.12) -> bool:
    """Check if two palette histograms are sufficiently similar.

    The comparison is normalised and allows minor deviations controlled by
    ``tolerance``.  The heuristic is intentionally loose – it is only meant to
    detect when a candidate already mirrors the global colour distribution,
    making further palette/border adjustments unnecessary.
    """

    if not candidate or not reference:
        return False
    cand_total = sum(candidate.values())
    ref_total = sum(reference.values())
    if cand_total <= 0 or ref_total <= 0:
        return False
    if set(candidate.keys()) - set(reference.keys()):
        return False
    for color, ref_count in reference.items():
        expected = ref_count / ref_total
        observed = candidate.get(color, 0) / cand_total
        dominant = max(expected, observed)
        # Ignore tiny mass where statistical noise dominates.
        if dominant < 0.05:
            continue
        if abs(expected - observed) > tolerance:
            return False
    return True

# ----------------------- Connected components --------------------------------
def connected_components(g: Grid, bg: Optional[int] = None) -> List[List[Tuple[int,int]]]:
    """4-neighborhood components of non-bg cells."""
    if not g or not g[0]: return []
    H, W = len(g), len(g[0])
    seen = [[False]*W for _ in range(H)]
    comps = []
    for r in range(H):
        for c in range(W):
            if seen[r][c]: continue
            val = g[r][c]
            if bg is not None and val == bg:
                seen[r][c] = True
                continue
            if bg is None and val == 0:  # default: treat 0 as bg if not specified
                seen[r][c] = True
                continue
            # BFS
            if (bg is None and val != 0) or (bg is not None and val != bg):
                q = deque([(r,c)])
                seen[r][c] = True
                comp = []
                while q:
                    rr, cc = q.popleft()
                    comp.append((rr,cc))
                    for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                        nr, nc = rr+dr, cc+dc
                        if in_bounds(nr,nc,H,W) and not seen[nr][nc]:
                            if (bg is None and g[nr][nc] != 0) or (bg is not None and g[nr][nc] != bg):
                                seen[nr][nc] = True
                                q.append((nr,nc))
                            else:
                                seen[nr][nc] = True  # mark bg as seen so we don't revisit
                comps.append(comp)
            else:
                seen[r][c] = True
    return comps

def crop_to_bbox(g: Grid, cells: List[Tuple[int,int]]) -> Grid:
    if not cells: return [[]]
    rs = [r for r,_ in cells]
    cs = [c for _,c in cells]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    return [row[c0:c1+1] for row in g[r0:r1+1]]

def paste_at(dst: Grid, patch: Grid, top: int, left: int) -> Grid:
    H, W = len(dst), (len(dst[0]) if dst else 0)
    h, w = len(patch), (len(patch[0]) if patch else 0)
    out = copy_grid(dst)
    for i in range(h):
        for j in range(w):
            rr, cc = top+i, left+j
            if in_bounds(rr, cc, H, W):
                out[rr][cc] = patch[i][j]
    return out

def translate_object_to(g: Grid, dest: str = "topleft", bg: Optional[int] = None) -> Grid:
    """Move largest component to a position (topleft|center)."""
    if not g or not g[0]: return g
    H, W = grid_shape(g)
    bgc = bg if bg is not None else mode_color(g)
    comps = connected_components(g, bg=bgc)
    if not comps: return g
    # pick largest comp
    largest = max(comps, key=len)
    obj = crop_to_bbox(g, largest)
    oh, ow = grid_shape(obj)
    out = [[bgc for _ in range(W)] for __ in range(H)]
    if dest == "topleft":
        return paste_at(out, obj, 0, 0)
    # center
    top = max(0, (H - oh)//2)
    left = max(0, (W - ow)//2)
    return paste_at(out, obj, top, left)


def rot90(grid: Grid) -> Grid:
    if not grid or not grid[0]:
        return grid
    H, W = grid_shape(grid)
    return [[grid[H - 1 - r][c] for r in range(H)] for c in range(W)]


def flip_horizontal(grid: Grid) -> Grid:
    if not grid or not grid[0]:
        return grid
    return [row[::-1] for row in grid]


def dihedral_variants(grid: Grid) -> List[Grid]:
    variants: List[Grid] = []
    current = copy_grid(grid)
    for _ in range(4):
        variants.append(current)
        current = rot90(current)
    flipped = flip_horizontal(grid)
    current = flipped
    for _ in range(4):
        variants.append(current)
        current = rot90(current)
    deduped: List[Grid] = []
    seen = set()
    for variant in variants:
        key = tuple(tuple(row) for row in variant)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(variant)
    return deduped or [grid]


def grid_loss_against_train(candidate: Grid, train_examples: List[Dict]) -> float:
    if not candidate or not candidate[0]:
        return 1.0
    best = None
    for example in train_examples or []:
        if not isinstance(example, dict):
            continue
        gold = example.get("output")
        if not gold or not gold[0]:
            continue
        if grid_shape(candidate) != grid_shape(gold):
            continue
        diff = delta(candidate, gold)
        denom = max(1, len(candidate) * len(candidate[0]))
        score = diff / denom
        if best is None or score < best:
            best = score
            if best <= 0.0:
                break
    return 1.0 if best is None else best


def palette_frequencies(train_examples: List[Dict]) -> Counter:
    freq: Counter[int] = Counter()
    for example in train_examples or []:
        if not isinstance(example, dict):
            continue
        out = example.get("output")
        if not out or not out[0]:
            continue
        for row in out:
            for cell in row:
                try:
                    freq.update([int(cell)])
                except Exception:
                    continue
    return freq


def _grid_colors(grid_like: Any) -> set[int]:
    colors: set[int] = set()
    if not isinstance(grid_like, list):
        return colors
    for row in grid_like:
        if not isinstance(row, list):
            continue
        for cell in row:
            try:
                colors.add(int(cell))
            except Exception:
                continue
    return colors


def _collect_example_palette(
    examples: Iterable[Dict[str, Any]], key: str
) -> Tuple[Counter[int], int]:
    """Return per-example palette counts and the number of usable examples."""

    presence: Counter[int] = Counter()
    total = 0
    for example in examples or []:
        if not isinstance(example, dict):
            continue
        grid = example.get(key)
        colors = _grid_colors(grid)
        if not colors:
            continue
        total += 1
        presence.update(colors)
    return presence, total


def _extract_gate_allowed_palette(payload: Optional[Dict[str, Any]]) -> set[int]:
    """Read palette hints attached by the gating stage."""

    allowed: set[int] = set()
    if not isinstance(payload, dict):
        return allowed

    containers = [payload]
    meta = payload.get("meta")
    if isinstance(meta, dict):
        containers.append(meta)

    extra = payload.get("extra") if isinstance(payload, dict) else None
    if isinstance(extra, dict):
        containers.append(extra)

    for container in containers:
        if not isinstance(container, dict):
            continue
        if container is payload:
            keys = ("gate_allowed_palette",)
        else:
            keys = (
                "gate_allowed_palette",
                "allowed_palette",
                "allowed_colors",
                "palette_hint",
                "palette_hints",
                "palette_extras",
            )
        for key in keys:
            if key not in container:
                continue
            values = container.get(key)
            if values is None:
                continue
            if isinstance(values, dict):
                iterable = list(values.keys()) + list(values.values())
            elif isinstance(values, (list, tuple, set)):
                iterable = values
            else:
                iterable = [values]
            for entry in iterable:
                try:
                    allowed.add(int(entry))
                except Exception:
                    continue
    return allowed


def _detect_palette_recolour(
    candidate: Grid,
    reference: Optional[Grid],
    train_palettes: Optional[Iterable[Sequence[int]]] = None,
    mapping_hint: Optional[Mapping[int, int]] = None,
) -> Optional[Dict[int, int]]:
    """Detect whether ``candidate`` is a recolour-only variant of ``reference``."""

    if np is None or reference is None:
        return None
    if not isinstance(candidate, list) or not isinstance(reference, list):
        return None
    try:
        ref_arr = np.asarray(reference, dtype=np.int16)
        cand_arr = np.asarray(candidate, dtype=np.int16)
    except Exception:
        return None
    if ref_arr.shape != cand_arr.shape or ref_arr.size == 0:
        return None

    mapping: Dict[int, int] = {}
    recoloured = False
    try:
        source_colors = np.unique(ref_arr)
    except Exception:
        return None
    for color in source_colors:
        try:
            mask = ref_arr == color
        except Exception:
            return None
        if not np.any(mask):
            continue
        try:
            targets = np.unique(cand_arr[mask])
        except Exception:
            return None
        if len(targets) != 1:
            return None
        target_val = int(targets[0])
        mapping[int(color)] = target_val
        if int(color) != target_val:
            recoloured = True

    if not recoloured:
        return None

    if mapping_hint:
        for src, dst in mapping_hint.items():
            try:
                src_int = int(src)
                dst_int = int(dst)
            except Exception:
                return None
            if src_int in mapping and mapping[src_int] != dst_int:
                return None

    changed_targets = [dst for src, dst in mapping.items() if src != dst]
    if len(set(changed_targets)) != len(changed_targets):
        return None

    try:
        candidate_colors = {int(value) for value in np.unique(cand_arr)}
    except Exception:
        return None
    allowed_targets = set(mapping.values())
    extras = candidate_colors - allowed_targets
    if extras - {0}:
        return None

    try:
        src_hist = np.bincount(ref_arr.ravel(), minlength=10)
        dst_hist = np.bincount(cand_arr.ravel(), minlength=10)
    except Exception:
        return None
    for src_color, dst_color in mapping.items():
        if src_color == dst_color:
            continue
        src_count = int(src_hist[src_color]) if src_color < len(src_hist) else 0
        dst_count = int(dst_hist[dst_color]) if dst_color < len(dst_hist) else 0
        if src_count and dst_count < src_count:
            return None

    if train_palettes:
        train_colors: Set[int] = set()
        for palette in train_palettes:
            if not isinstance(palette, Sequence):
                continue
            for value in palette:
                try:
                    train_colors.add(int(value))
                except Exception:
                    continue
        if train_colors and not candidate_colors.issubset(train_colors | {0}):
            return None

    return mapping


def clamp_to_palette(
    grid: Grid, palette: Counter, *, extras: Optional[Iterable[int]] = None
) -> Grid:
    if not grid or not grid[0]:
        return copy_grid(grid)
    allowed = set(palette.keys()) if palette else set()
    if extras:
        for color in extras:
            try:
                allowed.add(int(color))
            except Exception:
                continue
    if not allowed:
        return copy_grid(grid)
    default = min(allowed)
    clamped: Grid = []
    for row in grid:
        new_row: List[int] = []
        for cell in row:
            value = int(cell) if isinstance(cell, int) else default
            if value not in allowed:
                value = default
            new_row.append(value)
        clamped.append(new_row)
    return clamped


def snap_border_like(train_out: Grid, pred: Grid) -> Grid:
    if not pred or not pred[0] or not train_out or not train_out[0]:
        return pred
    if grid_shape(train_out) != grid_shape(pred):
        return pred
    H, W = grid_shape(pred)
    snapped = copy_grid(pred)
    for c in range(W):
        snapped[0][c] = train_out[0][c]
        snapped[H - 1][c] = train_out[H - 1][c]
    for r in range(H):
        snapped[r][0] = train_out[r][0]
        snapped[r][W - 1] = train_out[r][W - 1]
    return snapped


def recolor_components_by_palette(
    grid: Grid, palette: Counter, *, extras: Optional[Iterable[int]] = None
) -> Grid:
    if not grid or not grid[0]:
        return copy_grid(grid)
    allowed = set(palette.keys()) if palette else set()
    if extras:
        for color in extras:
            try:
                allowed.add(int(color))
            except Exception:
                continue
    if not allowed:
        return copy_grid(grid)
    ranked_items = list(palette.items()) if palette else []
    if not ranked_items and extras:
        ranked_items = [(int(color), 1) for color in extras]
    ranked = sorted(ranked_items, key=lambda kv: (-kv[1], kv[0]))
    if not ranked:
        return copy_grid(grid)
    new_grid = copy_grid(grid)
    comps = connected_components(grid, bg=mode_color(grid))
    for comp in comps:
        if not comp:
            continue
        r0, c0 = comp[0]
        color = grid[r0][c0]
        if color in allowed:
            continue
        size = len(comp)
        best_color = ranked[0][0]
        best_diff = abs(size - ranked[0][1])
        for cand_color, freq in ranked[1:]:
            diff = abs(size - freq)
            if diff < best_diff:
                best_diff = diff
                best_color = cand_color
        for r, c in comp:
            new_grid[r][c] = best_color
    return new_grid


def tight_crop_uncrop(grid: Grid, target_shape: Tuple[int, int]) -> Grid:
    if not grid or not grid[0] or not target_shape:
        return copy_grid(grid)
    th, tw = target_shape
    if th <= 0 or tw <= 0:
        return copy_grid(grid)
    bg = mode_color(grid)
    comps = connected_components(grid, bg=bg)
    if not comps:
        return copy_grid(grid)
    all_cells = [cell for comp in comps for cell in comp]
    cropped = crop_to_bbox(grid, all_cells)
    ch, cw = grid_shape(cropped)
    if ch == th and cw == tw:
        return cropped
    canvas = [[bg for _ in range(tw)] for _ in range(th)]
    top = max(0, (th - ch) // 2)
    left = max(0, (tw - cw) // 2)
    for r in range(min(ch, th)):
        for c in range(min(cw, tw)):
            canvas[top + r][left + c] = cropped[r][c]
    return canvas


def fit_to_shape(grid: Grid, target_shape: Tuple[int, int]) -> Grid:
    if not target_shape:
        return copy_grid(grid)
    th, tw = target_shape
    if th <= 0 or tw <= 0:
        return copy_grid(grid)
    if not grid or not grid[0]:
        return [[0 for _ in range(tw)] for _ in range(th)]
    gh, gw = grid_shape(grid)
    if gh == th and gw == tw:
        return copy_grid(grid)
    bg = mode_color(grid)
    resized = [[bg for _ in range(tw)] for _ in range(th)]
    for r in range(th):
        for c in range(tw):
            source_r = r % gh
            source_c = c % gw
            resized[r][c] = grid[source_r][source_c]
    return resized


def _mask_density(mask: Optional[List[List[bool]]]) -> float:
    if not mask:
        return 0.0
    total = sum(len(row) for row in mask)
    if total <= 0:
        return 0.0
    active = sum(1 for row in mask for cell in row if cell)
    return active / total


def _copy_mask(mask: Optional[List[List[bool]]]) -> Optional[List[List[bool]]]:
    if not mask:
        return None
    return [[bool(cell) for cell in row] for row in mask]


def _erode_mask(mask: Optional[List[List[bool]]], steps: int = 1) -> Optional[List[List[bool]]]:
    if not mask:
        return None
    current = _copy_mask(mask)
    if not current:
        return None
    h = len(current)
    w = len(current[0]) if h else 0
    if h == 0 or w == 0:
        return None

    def _erode_once(values: List[List[bool]]) -> List[List[bool]]:
        out = [[False for _ in range(w)] for _ in range(h)]
        if h <= 2 or w <= 2:
            return out
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                keep = True
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if not values[r + dr][c + dc]:
                            keep = False
                            break
                    if not keep:
                        break
                if keep:
                    out[r][c] = True
        return out

    for _ in range(max(1, int(steps))):
        current = _erode_once(current)
        if not any(any(row) for row in current):
            return None
    return current


def _candidate_palette(candidates: Iterable[Dict[str, Any]]) -> set[int]:
    palette: set[int] = set()
    for cand in candidates:
        grid = cand.get("grid")
        if not isinstance(grid, list):
            continue
        for row in grid:
            if not isinstance(row, list):
                continue
            for cell in row:
                try:
                    palette.add(int(cell))
                except Exception:
                    continue
    return palette


def _rank_background_cells(grid: Grid) -> List[Tuple[int, int]]:
    if not grid or not grid[0]:
        return []
    h, w = grid_shape(grid)
    background: List[Tuple[int, int]] = []
    objects: List[Tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 0:
                background.append((r, c))
            else:
                objects.append((r, c))
    if not background:
        return []
    if not objects:
        centre_r = (h - 1) / 2.0
        centre_c = (w - 1) / 2.0
        background.sort(
            key=lambda rc: abs(rc[0] - centre_r) + abs(rc[1] - centre_c),
            reverse=True,
        )
        return background

    def _distance(cell: Tuple[int, int]) -> int:
        r, c = cell
        return min(abs(r - or_) + abs(c - oc) for or_, oc in objects)

    background.sort(key=_distance, reverse=True)
    return background


def _ensure_palette_superset(grid: Grid, missing_colors: Sequence[int]) -> Tuple[Grid, bool]:
    if not missing_colors:
        return grid, False
    working = copy_grid(grid)
    placements = 0
    ranked = _rank_background_cells(working)
    idx = 0
    for color in missing_colors:
        if idx < len(ranked):
            r, c = ranked[idx]
            if working[r][c] != int(color):
                working[r][c] = int(color)
                placements += 1
            idx += 1
        else:
            break
    if placements < len(missing_colors):
        comps = [comp for comp in connected_components(working, bg=0) if len(comp) <= 2]
        comps.sort(key=len)
        comp_iter = iter(comps)
        for color in missing_colors[placements:]:
            try:
                comp = next(comp_iter)
            except StopIteration:
                break
            for r, c in comp:
                if working[r][c] != int(color):
                    working[r][c] = int(color)
                    placements += 1
    return (working if placements else grid), bool(placements)


def _shape_preview_score(grid: Grid, reference: Optional[Grid]) -> Tuple[float, bool]:
    if not grid or not isinstance(grid, list) or not grid[0]:
        return 0.0, True
    cand_mask = {(r, c) for r, row in enumerate(grid) for c, val in enumerate(row) if val != 0}
    if reference and isinstance(reference, list) and reference and reference[0]:
        if grid_shape(reference) != grid_shape(grid):
            inter = 0
            union = len(cand_mask)
        else:
            ref_mask = {
                (r, c)
                for r, row in enumerate(reference)
                for c, val in enumerate(row)
                if val != 0
            }
            inter = len(cand_mask & ref_mask)
            union = len(cand_mask | ref_mask)
    else:
        ref_mask = set()
        inter = 0
        union = len(cand_mask)
    iou = inter / float(union) if union else 1.0

    cand_components = connected_components(grid, bg=0)
    ref_components = connected_components(reference, bg=0) if reference else []
    dcount = abs(len(cand_components) - len(ref_components)) / float(max(1, len(ref_components)))
    dcount = min(dcount, 1.0)

    def _centroids(comps: List[List[Tuple[int, int]]]) -> List[Tuple[float, float]]:
        centres: List[Tuple[float, float]] = []
        for comp in comps:
            if not comp:
                continue
            total = float(len(comp))
            avg_r = sum(r for r, _ in comp) / total
            avg_c = sum(c for _, c in comp) / total
            centres.append((avg_r, avg_c))
        return centres

    cand_centroids = _centroids(cand_components)
    ref_centroids = _centroids(ref_components)
    avg_dist = 0.0
    if cand_centroids and ref_centroids:
        distances: List[float] = []
        for cr, cc in cand_centroids:
            best = min(abs(cr - rr) + abs(cc - rc) for rr, rc in ref_centroids)
            distances.append(best)
        avg_dist = sum(distances) / float(len(distances))
    elif cand_centroids:
        avg_dist = sum(abs(cr) + abs(cc) for cr, cc in cand_centroids) / float(len(cand_centroids))

    scale = max(1.0, max(len(grid), len(grid[0]) if grid and grid[0] else 1) / 2.0)
    centroid_term = math.exp(-(avg_dist / scale)) if avg_dist else 1.0
    shape_score = 0.6 * iou + 0.2 * (1.0 - dcount) + 0.2 * centroid_term
    shape_score = max(0.0, min(1.0, shape_score))
    degenerate = not cand_mask or (iou < 0.02 and len(cand_components) == 0)
    return shape_score, degenerate


def _grid_difference_ratio(grid: Grid, reference: Optional[Grid]) -> float:
    if not isinstance(grid, list) or not grid or not isinstance(grid[0], list):
        return 1.0
    if not reference or not isinstance(reference, list) or not reference or not isinstance(reference[0], list):
        return 1.0
    if grid_shape(grid) != grid_shape(reference):
        return 1.0
    h, w = grid_shape(grid)
    total = h * w
    if total <= 0:
        return 1.0
    mismatches = 0
    for r in range(h):
        for c in range(w):
            if grid[r][c] != reference[r][c]:
                mismatches += 1
    return mismatches / float(total)


def _relax_gate_context(
    gate_ctx: GateContext,
    *,
    candidate_palette: Iterable[int],
    expand_palette: bool,
    erode_frozen: bool,
    test_input_colors: Optional[Iterable[int]] = None,
) -> GateContext:
    relaxed_palette = set(gate_ctx.allowed_palette)
    relaxed_supplemental = dict(getattr(gate_ctx, "supplemental_palette", {}))
    base_palette = set(getattr(gate_ctx, "core_palette", set()))
    seeded_test_colors: Set[int] = set(getattr(gate_ctx, "test_input_colors", set()))

    def _promote_color(color: Any, *, score: float) -> None:
        try:
            color_int = int(color)
        except Exception:
            return
        relaxed_palette.add(color_int)
        if color_int not in base_palette:
            relaxed_supplemental[color_int] = max(
                relaxed_supplemental.get(color_int, 0.0), score
            )

    if expand_palette:
        for color in candidate_palette:
            _promote_color(color, score=0.75)
        if test_input_colors:
            for color in test_input_colors:
                _promote_color(color, score=0.6)
                try:
                    seeded_test_colors.add(int(color))
                except Exception:
                    continue
        if gate_ctx.frozen_density >= 0.6 and getattr(gate_ctx, "palette_order", ()):  # type: ignore[attr-defined]
            relaxed_palette.update(int(color) for color in gate_ctx.palette_order)
    relaxed_mask = _copy_mask(gate_ctx.frozen_mask)
    if erode_frozen and relaxed_mask:
        relaxed_mask = _erode_mask(relaxed_mask)
    relaxed_density = _mask_density(relaxed_mask)
    return GateContext(
        shape_preserving=gate_ctx.shape_preserving,
        allowed_shapes=set(gate_ctx.allowed_shapes),
        allow_growth=gate_ctx.allow_growth,
        allowed_palette=relaxed_palette,
        frozen_mask=relaxed_mask,
        reference_input=copy_grid(gate_ctx.reference_input) if gate_ctx.reference_input else None,
        frozen_density=relaxed_density,
        palette_order=getattr(gate_ctx, "palette_order", ()),
        supplemental_palette=relaxed_supplemental,
        core_palette=set(base_palette) if base_palette else set(gate_ctx.allowed_palette),
        test_input_colors=seeded_test_colors,
        emergent_palette=set(getattr(gate_ctx, "emergent_palette", set())),
        vanishing_palette=set(getattr(gate_ctx, "vanishing_palette", set())),
        palette_floor=getattr(gate_ctx, "palette_floor", DEFAULT_PALETTE_FLOOR),
        palette_rescue_keep_ratio=getattr(
            gate_ctx, "palette_rescue_keep_ratio", DEFAULT_PALETTE_RESCUE_KEEP
        ),
        gate_soft_topk=getattr(gate_ctx, "gate_soft_topk", DEFAULT_GATE_SOFT_TOPK),
        palette_soft_floor=getattr(gate_ctx, "palette_soft_floor", 0),
        palette_soft_ratio=getattr(gate_ctx, "palette_soft_ratio", 0.0),
        min_beam_after_gate=getattr(
            gate_ctx,
            "min_beam_after_gate",
            DEFAULT_MIN_BEAM_AFTER_GATE,
        ),
        region_gate_mode=getattr(gate_ctx, "region_gate_mode", "strict"),
        train_output_palettes=[
            [int(cell) for cell in palette]
            for palette in getattr(gate_ctx, "train_output_palettes", [])
            if isinstance(palette, (list, tuple))
        ],
        recolour_mapping_hint={
            int(color): int(target)
            for color, target in getattr(gate_ctx, "recolour_mapping_hint", {}).items()
            if isinstance(color, (int, float)) and isinstance(target, (int, float))
        },
        recolour_only_task=bool(getattr(gate_ctx, "recolour_only_task", False)),
    )


def _compute_frozen_mask(train_examples: List[Dict]) -> Optional[List[List[bool]]]:
    mask: Optional[List[List[bool]]] = None
    shape: Optional[Tuple[int, int]] = None
    for example in train_examples or []:
        if not isinstance(example, dict):
            return None
        inp = example.get("input")
        out = example.get("output")
        if not isinstance(inp, list) or not isinstance(out, list):
            return None
        if not inp or not inp[0] or not out or not out[0]:
            return None
        if grid_shape(inp) != grid_shape(out):
            return None
        if shape is None:
            shape = grid_shape(inp)
            mask = [[True for _ in range(shape[1])] for _ in range(shape[0])]
        elif shape != grid_shape(inp):
            return None
        assert mask is not None  # for type checker
        for r in range(shape[0]):
            for c in range(shape[1]):
                if inp[r][c] != out[r][c]:
                    mask[r][c] = False
    if mask is None:
        return None
    if all(not any(row) for row in mask):
        return None
    return mask


def _build_gate_context(
    rules: Dict[str, Any],
    train_examples: List[Dict],
    test_input: Grid,
    *,
    provenance_mode: str = "train_or_test",
    dead_color_policy: str = "ignore",
    attr_weight: float = 0.65,
    repulse_weight: float = 0.35,
) -> GateContext:
    training_pairs: List[Tuple[Grid, Grid]] = []
    for example in train_examples or []:
        if not isinstance(example, dict):
            continue
        inp = example.get("input")
        out = example.get("output")
        if isinstance(inp, list) and isinstance(out, list):
            training_pairs.append((inp, out))

    train_output_palettes: List[List[int]] = []
    mapping_hint: Dict[int, int] = {}
    mapping_detected = False
    mapping_consistent = True
    shapes_consistent = True
    for inp, out in training_pairs:
        palette_vals: List[int] = []
        for row in out or []:
            if not isinstance(row, list):
                palette_vals = []
                break
            for cell in row:
                try:
                    palette_vals.append(int(cell))
                except Exception:
                    continue
        if palette_vals:
            train_output_palettes.append(palette_vals)

        if not isinstance(inp, list) or not isinstance(out, list):
            mapping_consistent = False
            continue
        if grid_shape(inp) != grid_shape(out):
            shapes_consistent = False
            continue
        if not inp or not inp[0]:
            continue
        try:
            width = len(inp[0])
        except Exception:
            mapping_consistent = False
            break
        for r in range(len(inp)):
            if not isinstance(inp[r], list) or not isinstance(out[r], list):
                mapping_consistent = False
                break
            for c in range(width):
                try:
                    src = int(inp[r][c])
                    dst = int(out[r][c])
                except Exception:
                    mapping_consistent = False
                    break
                if src == dst:
                    continue
                mapping_detected = True
                if src in mapping_hint and mapping_hint[src] != dst:
                    mapping_consistent = False
                    break
                mapping_hint[src] = dst
            if not mapping_consistent:
                break
        if not mapping_consistent:
            break

    recolour_only_task = False
    if mapping_detected and mapping_consistent and shapes_consistent:
        changed_targets = [dst for src, dst in mapping_hint.items() if src != dst]
        if changed_targets and len(set(changed_targets)) == len(changed_targets):
            recolour_only_task = True
        else:
            mapping_hint.clear()
    else:
        mapping_hint.clear()

    input_shapes = {
        tuple(shape)
        for shape in rules.get("input_shapes", [])
        if isinstance(shape, (tuple, list)) and len(shape) == 2
    }
    output_shapes = {
        tuple(shape)
        for shape in rules.get("output_shapes", [])
        if isinstance(shape, (tuple, list)) and len(shape) == 2
    }

    shape_pairs = list(
        zip(
            [tuple(s) for s in rules.get("input_shapes", []) if isinstance(s, (tuple, list)) and len(s) == 2],
            [tuple(s) for s in rules.get("output_shapes", []) if isinstance(s, (tuple, list)) and len(s) == 2],
        )
    )
    base_shape_preserving = bool(shape_pairs) and all(inp == out for inp, out in shape_pairs)

    ratios = [tuple(pair) for pair in rules.get("size_ratios", []) if isinstance(pair, (tuple, list)) and len(pair) == 2]
    rounded_ratios = {tuple(round(val, 3) for val in ratio) for ratio in ratios}
    allow_growth = not base_shape_preserving and len(rounded_ratios) == 1 and rounded_ratios != {(1.0, 1.0)}

    allowed_shapes: set[Tuple[int, int]] = set(output_shapes)
    if base_shape_preserving and input_shapes:
        allowed_shapes |= set(input_shapes)

    palette_counter = palette_frequencies(train_examples)
    palette_order = tuple(color for color, _count in palette_counter.most_common())

    base_palette: set[int] = {int(color) for color in palette_counter.keys()}
    example_hist: Dict[int, int] = {int(color): int(count) for color, count in palette_counter.items()}
    allowed_palette: set[int] = set(base_palette)
    supplemental_palette: Dict[int, float] = {}

    provenance = str(provenance_mode or "train_or_test").lower()
    dead_policy = str(dead_color_policy or "ignore").lower()

    output_presence, _ = _collect_example_palette(train_examples, "output")
    example_colors: Set[int] = set()
    for color in output_presence.keys():
        try:
            example_colors.add(int(color))
        except Exception:
            continue

    input_presence, input_total = _collect_example_palette(train_examples, "input")
    dead_colors: Set[int] = set()
    if input_total:
        denom = float(max(1, input_total))
        for color, count in input_presence.items():
            try:
                color_int = int(color)
            except Exception:
                continue
            if dead_policy in {"suppress", "block", "strict"} and color_int not in example_colors:
                dead_colors.add(color_int)
                continue
            share = float(count) / denom
            if color_int not in base_palette:
                supplemental_palette[color_int] = max(
                    supplemental_palette.get(color_int, 0.0), share
                )
                if share >= 0.75:
                    allowed_palette.add(color_int)

    test_counter: Counter[int] = Counter()
    if isinstance(test_input, list):
        for row in test_input:
            if not isinstance(row, list):
                continue
            for value in row:
                try:
                    test_counter.update([int(value)])
                except Exception:
                    continue

    test_colors = set(test_counter.keys())
    strong_test_colors: Set[int] = set()
    tot = sum(test_counter.values()) or 1
    if test_counter:
        for color, count in test_counter.items():
            try:
                color_int = int(color)
            except Exception:
                continue
            if (float(count) / float(tot)) >= ALLOW_TEST_COLOR_IF:
                strong_test_colors.add(color_int)

    if provenance in {"train_or_test", "train_and_test", "examples_and_test"}:
        evidence_test_colors: Set[int] = set(strong_test_colors)
        evidence_palette = set(example_colors) | set(evidence_test_colors)
    else:
        evidence_test_colors = set()
        evidence_palette = set(example_colors)
    if dead_colors:
        evidence_palette.difference_update(dead_colors)
        evidence_test_colors.difference_update(dead_colors)

    if test_colors and provenance in {"train_or_test", "train_and_test", "examples_and_test"}:
        test_score = 0.6 if len(test_colors) <= 3 else 0.5
        for color in evidence_test_colors:
            if color in base_palette or color in dead_colors:
                continue
            supplemental_palette[color] = max(
                supplemental_palette.get(color, 0.0), test_score
            )
            if test_score >= 0.75:
                allowed_palette.add(color)

    evidence_weights: Dict[int, float] = {}
    for color in evidence_palette:
        weight = 1.0 if color in example_colors else 0.7
        evidence_weights[int(color)] = float(weight)

    allowed_palette.update(evidence_palette)
    allowed_palette = {color for color in allowed_palette if 0 <= int(color) <= 9}

    def _grid_size_safe(grid: Any) -> Optional[Tuple[int, int]]:
        if not isinstance(grid, list):
            return None
        height = len(grid)
        if height == 0:
            return (0, 0)
        first_row = grid[0]
        if not isinstance(first_row, list):
            return (height, 0)
        return (height, len(first_row))

    allow_size_change_hint = False
    for inp, out in training_pairs:
        inp_shape = _grid_size_safe(inp)
        out_shape = _grid_size_safe(out)
        if inp_shape is None or out_shape is None:
            continue
        if inp_shape != out_shape:
            allow_size_change_hint = True
            break

    shape_preserving = base_shape_preserving and not allow_size_change_hint
    if allow_size_change_hint:
        allow_growth = True

    attr_weight = max(0.0, float(attr_weight))
    repulse_weight = max(0.0, float(repulse_weight))
    weight_total = max(0.01, attr_weight + repulse_weight)
    attr_weight /= weight_total
    repulse_weight /= weight_total

    frozen_mask = _compute_frozen_mask(train_examples)

    return GateContext(
        shape_preserving=shape_preserving,
        allowed_shapes={tuple(map(int, shape)) for shape in allowed_shapes if len(shape) == 2},
        allow_growth=allow_growth,
        allowed_palette=allowed_palette,
        frozen_mask=frozen_mask,
        reference_input=copy_grid(test_input),
        frozen_density=_mask_density(frozen_mask),
        palette_order=palette_order,
        supplemental_palette=supplemental_palette,
        core_palette=set(base_palette),
        test_input_colors=set(test_colors),
        emergent_palette=set(),
        vanishing_palette=set(),
        evidence_example_colors=example_colors,
        evidence_test_colors=evidence_test_colors,
        evidence_weights=evidence_weights,
        dead_colors=dead_colors,
        palette_attr_weight=attr_weight,
        palette_repulse_weight=repulse_weight,
        palette_provenance_mode=provenance,
        dead_color_policy=dead_policy,
        palette_floor=DEFAULT_PALETTE_FLOOR,
        palette_rescue_keep_ratio=DEFAULT_PALETTE_RESCUE_KEEP,
        gate_soft_topk=DEFAULT_GATE_SOFT_TOPK,
        min_beam_after_gate=DEFAULT_MIN_BEAM_AFTER_GATE,
        palette_gate_mode="hard",
        shape_gate_mode="strict",
        shape_soft_floor=0,
        shape_soft_ratio=0.0,
        shape_preview_tau_drop=0.0,
        train_output_palettes=train_output_palettes,
        recolour_mapping_hint={int(src): int(dst) for src, dst in mapping_hint.items()},
        recolour_only_task=recolour_only_task,
    )


def _apply_candidate_gates(
    candidates: List[Dict[str, Any]],
    gate_ctx: GateContext,
    *,
    metrics: Optional[Metrics],
    palette_support_min: float = 0.85,
    **_unused: Any,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Apply structural gates to candidates and collect profiling info."""

    totals: Counter[str] = Counter()
    accepted_counts: Counter[str] = Counter()
    gate_accept: Counter[str] = Counter()
    gate_reject: Counter[str] = Counter()
    per_gate_by_type: Dict[str, Counter[str]] = {}
    filtered: List[Dict[str, Any]] = []
    soft_palette_pool: List[Dict[str, Any]] = []
    min_after_gate = max(0, int(getattr(gate_ctx, "min_beam_after_gate", 0)))

    allowed_palette = set(getattr(gate_ctx, "allowed_palette", set()))
    base_palette = set(getattr(gate_ctx, "core_palette", set())) or set(allowed_palette)
    supplemental_palette = dict(getattr(gate_ctx, "supplemental_palette", {}))
    frozen_mask = gate_ctx.frozen_mask
    ref_input = gate_ctx.reference_input
    recolour_only_task = bool(getattr(gate_ctx, "recolour_only_task", False))
    recolour_mapping_hint_raw = getattr(gate_ctx, "recolour_mapping_hint", {})
    recolour_mapping_hint: Dict[int, int] = {}
    if isinstance(recolour_mapping_hint_raw, Mapping):
        for key, value in recolour_mapping_hint_raw.items():
            try:
                recolour_mapping_hint[int(key)] = int(value)
            except Exception:
                continue
    train_output_palettes: List[List[int]] = []
    raw_train_palettes = getattr(gate_ctx, "train_output_palettes", [])
    if isinstance(raw_train_palettes, Iterable):
        for palette in raw_train_palettes:
            if not isinstance(palette, (list, tuple)):
                continue
            values: List[int] = []
            for cell in palette:
                try:
                    values.append(int(cell))
                except Exception:
                    continue
            if values:
                train_output_palettes.append(values)

    evidence_example_colors: Set[int] = set()
    for color in getattr(gate_ctx, "evidence_example_colors", set()):
        try:
            evidence_example_colors.add(int(color))
        except Exception:
            continue
    evidence_test_colors: Set[int] = set()
    for color in getattr(gate_ctx, "evidence_test_colors", set()):
        try:
            evidence_test_colors.add(int(color))
        except Exception:
            continue
    dead_colors: Set[int] = set()
    for color in getattr(gate_ctx, "dead_colors", set()):
        try:
            dead_colors.add(int(color))
        except Exception:
            continue

    evidence_palette: Set[int] = set(evidence_example_colors) | set(evidence_test_colors)
    if dead_colors:
        evidence_palette.difference_update(dead_colors)

    evidence_core_colors: Set[int] = set(evidence_example_colors)
    if not evidence_core_colors and base_palette:
        evidence_core_colors.update(int(color) for color in base_palette)
    if dead_colors:
        evidence_core_colors.difference_update(dead_colors)

    evidence_weights: Dict[int, float] = {
        int(color): float(weight)
        for color, weight in getattr(gate_ctx, "evidence_weights", {}).items()
    }
    if not evidence_weights and evidence_palette:
        for color in evidence_palette:
            evidence_weights[int(color)] = 1.0

    attr_weight = max(0.0, float(getattr(gate_ctx, "palette_attr_weight", 0.65)))
    repulse_weight = max(0.0, float(getattr(gate_ctx, "palette_repulse_weight", 0.35)))
    weight_total = max(0.01, attr_weight + repulse_weight)
    attr_weight /= weight_total
    repulse_weight /= weight_total

    feasible_bonus = float(getattr(gate_ctx, "palette_feasible_bonus", 0.0))
    cost_weight = max(0.0, float(getattr(gate_ctx, "palette_cost_weight", 0.0)))
    spectral_weight = max(0.0, float(getattr(gate_ctx, "palette_spectral_weight", 0.0)))
    spectral_reference = float(
        getattr(
            gate_ctx,
            "palette_spectral_reference",
            max(1, len(evidence_core_colors) or len(base_palette) or 1),
        )
    )
    euler_weight = max(0.0, float(getattr(gate_ctx, "palette_euler_weight", 0.0)))
    coverage_bonus = float(getattr(gate_ctx, "palette_coverage_bonus", 0.0))
    coverage_penalty_weight = max(
        0.0, float(getattr(gate_ctx, "palette_coverage_penalty_weight", 0.0))
    )

    palette_denominator = float(
        max(
            1,
            len(evidence_palette | set(getattr(gate_ctx, "test_input_colors", set()))),
            len(base_palette),
        )
    )

    candidate_color_sets: List[set[int]] = []
    candidate_color_counts: Counter[int] = Counter()
    candidate_total = 0
    candidate_recolour_mappings: List[Optional[Dict[int, int]]] = []
    palette_synth_count = 0
    palette_extras_total = 0
    dead_color_revived_total = 0
    test_input_palette: Set[int] = set()
    for color in getattr(gate_ctx, "test_input_colors", set()):
        try:
            test_input_palette.add(int(color))
        except Exception:
            continue
    palette_mode = str(getattr(gate_ctx, "palette_gate_mode", "hard") or "hard").lower()
    shape_gate_mode = str(getattr(gate_ctx, "shape_gate_mode", "strict") or "strict").lower()
    shape_drop_tau = max(0.0, float(getattr(gate_ctx, "shape_preview_tau_drop", 0.0)))

    for cand in candidates:
        grid = cand.get("grid") or []
        meta = cand.setdefault("meta", {})
        candidate_mapping: Optional[Dict[int, int]] = None
        allow_missing_for_recolour = False
        if (
            recolour_only_task
            and ref_input is not None
            and grid
            and isinstance(grid, list)
            and isinstance(ref_input, list)
            and grid_shape(grid) == grid_shape(ref_input)
        ):
            candidate_mapping = _detect_palette_recolour(
                grid,
                ref_input,
                train_output_palettes,
                recolour_mapping_hint,
            )
            if candidate_mapping:
                allow_missing_for_recolour = True
                changed_mapping = {
                    int(src): int(dst)
                    for src, dst in candidate_mapping.items()
                    if int(src) != int(dst)
                }
                if changed_mapping:
                    meta.setdefault("palette_recolour_mapping", changed_mapping)
                meta.setdefault("palette_strict", False)

        colors = _grid_colors(grid)
        missing_colors = [color for color in test_input_palette if color not in colors]
        if missing_colors and not allow_missing_for_recolour:
            fixed_grid, modified = _ensure_palette_superset(grid, missing_colors)
            if modified:
                cand["grid"] = fixed_grid
                grid = fixed_grid
                colors = _grid_colors(grid)
                meta["palette_synthesized"] = True
                meta["palette_missing_before"] = len(missing_colors)
                meta["palette_missing_after"] = len(
                    [color for color in test_input_palette if color not in colors]
                )
                palette_synth_count += 1
            else:
                meta.setdefault("palette_synthesized", False)
        else:
            meta.setdefault("palette_synthesized", False)
            if allow_missing_for_recolour:
                meta.setdefault("palette_missing_before", len(missing_colors))
                meta.setdefault("palette_missing_after", len(missing_colors))
        colors = _grid_colors(grid)
        candidate_color_sets.append(colors)
        if colors:
            candidate_total += 1
            candidate_color_counts.update(colors)
        candidate_recolour_mappings.append(candidate_mapping if candidate_mapping else None)

    candidate_soft_palette: Dict[int, float] = {}
    palette_floor = max(1, getattr(gate_ctx, "palette_floor", DEFAULT_PALETTE_FLOOR))
    rescue_boost = max(
        0.5,
        float(getattr(gate_ctx, "palette_rescue_keep_ratio", DEFAULT_PALETTE_RESCUE_KEEP)),
    )
    palette_rescue_applied = False
    colors_before = {color for color in _pool_distinct_colors(candidates) if color != 0}
    if (candidate_total == 0 or len(colors_before) < palette_floor) and test_input_palette:
        palette_rescue_applied = True
        for color in test_input_palette:
            if color == 0:
                continue
            allowed_palette.add(color)
            supplemental_palette[color] = max(
                supplemental_palette.get(color, 0.0), rescue_boost
            )
    if candidate_total:
        denom = float(candidate_total)
        for color, count in candidate_color_counts.items():
            try:
                color_int = int(color)
            except Exception:
                continue
            share = float(count) / denom
            candidate_soft_palette[color_int] = share
            if color_int not in allowed_palette and share >= palette_support_min:
                allowed_palette.add(color_int)
            if color_int not in base_palette:
                supplemental_palette[color_int] = max(
                    supplemental_palette.get(color_int, 0.0), share
                )

    palette_gate_active = bool(allowed_palette or supplemental_palette or candidate_soft_palette)
    pool_before = len(candidates)
    gate_soft_cap = max(0, int(getattr(gate_ctx, "gate_soft_topk", DEFAULT_GATE_SOFT_TOPK)))
    palette_soft_floor = max(0, int(getattr(gate_ctx, "palette_soft_floor", 0)))
    palette_soft_ratio = max(0.0, float(getattr(gate_ctx, "palette_soft_ratio", 0.0)))
    palette_soft_limit = max(
        palette_soft_floor,
        int(math.ceil(palette_soft_ratio * max(1, pool_before))),
    )
    soft_limit = max(gate_soft_cap, palette_soft_limit)
    soft_overrides = 0
    palette_overrides = 0
    palette_soft_promoted = 0
    shape_pass_count = 0
    palette_stage_keep = 0
    region_stage_keep = 0
    palette_penalty_sum = 0.0
    palette_missing_total = 0
    shape_preview_scores: List[float] = []
    region_preview_scores: List[float] = []
    palette_preview_scores: List[float] = []
    palette_rescue_requests: List[Dict[str, Any]] = []
    region_mode = str(getattr(gate_ctx, "region_gate_mode", "strict") or "strict").lower()

    for idx, cand in enumerate(candidates):
        ctype = str(cand.get("type", "unknown"))
        totals[ctype] += 1
        grid = cand.get("grid") or []
        meta = cand.setdefault("meta", {})
        gate_sequence: List[GateResult] = []
        palette_allowed_for_candidate: set[int] = set()
        palette_support_details: Dict[int, str] = {}
        palette_hints = _extract_gate_allowed_palette(cand)
        candidate_mapping = (
            candidate_recolour_mappings[idx]
            if idx < len(candidate_recolour_mappings)
            else None
        )
        if candidate_mapping:
            meta.setdefault("palette_strict", False)
            tags = meta.setdefault("gate_tags", [])
            if isinstance(tags, list) and "recolour" not in tags:
                tags.append("recolour")
        colors = candidate_color_sets[idx]

        def _record(name: str, accepted: bool, reason: Optional[str] = None) -> None:
            gate_sequence.append(GateResult(name=name, accepted=accepted, reason=reason))
            if accepted:
                gate_accept[name] += 1
            else:
                gate_reject[name] += 1
                per_gate_by_type.setdefault(name, Counter())[ctype] += 1

        structural_ok = True
        structural_reason: Optional[str] = None
        hard_reject = False

        meta.setdefault("palette_gate_mode", palette_mode)

        shape_reason: Optional[str] = None
        strict_shape_violation = False
        if gate_ctx.allowed_shapes:
            shape = grid_shape(grid)
            if (
                gate_ctx.shape_preserving
                and gate_ctx.allowed_shapes
                and shape not in gate_ctx.allowed_shapes
                and not gate_ctx.allow_growth
            ):
                strict_shape_violation = True
                shape_reason = (
                    f"shape={grid_shape(grid)} not in {sorted(gate_ctx.allowed_shapes)}"
                )

        shape_score, shape_degenerate = _shape_preview_score(grid, ref_input)
        shape_preview_scores.append(shape_score)
        meta["shape_preview_score"] = round(shape_score, 4)
        if shape_degenerate:
            meta["shape_degenerate"] = True

        shape_pass = True
        if shape_gate_mode == "preview":
            drop_for_preview = shape_degenerate or (shape_score < shape_drop_tau and shape_drop_tau > 0.0)
            if drop_for_preview:
                shape_pass = False
                structural_ok = False
                structural_reason = structural_reason or "shape"
                if shape_reason is None:
                    shape_reason = f"score={shape_score:.2f}"
        else:
            if strict_shape_violation:
                shape_pass = False
                structural_ok = False
                structural_reason = "shape"
        _record(
            "shape",
            shape_pass,
            None
            if shape_pass
            else f"shape={grid_shape(grid)} not in {sorted(gate_ctx.allowed_shapes)}",
        )
        if shape_pass:
            shape_pass_count += 1

        # Region gate with preview score
        region_pass = True
        mismatches = 0
        locked = 0
        preserved = 0
        erased = 0
        if frozen_mask and ref_input and grid_shape(grid) == grid_shape(frozen_mask):
            try:
                for r, mask_row in enumerate(frozen_mask):
                    for c, keep in enumerate(mask_row):
                        if not keep:
                            continue
                        locked += 1
                        if grid[r][c] == ref_input[r][c]:
                            preserved += 1
                        else:
                            mismatches += 1
                            if grid[r][c] == 0:
                                erased += 1
                            if mismatches >= 8:
                                raise StopIteration
            except StopIteration:
                pass
        region_preview = 1.0
        erased_ratio = 0.0
        if locked:
            region_preview = preserved / float(max(1, locked))
            erased_ratio = erased / float(max(1, locked))
        region_reason: Optional[str] = None
        if mismatches and shape_pass:
            if region_mode == "preview":
                nonzero = _grid_nonzero_area(grid)
                total_cells = len(grid) * (len(grid[0]) if grid and grid[0] else 0)
                blank_ratio = 1.0
                if total_cells:
                    blank_ratio = 1.0 - (nonzero / float(total_cells))
                if nonzero == 0 or blank_ratio >= 0.98:
                    region_pass = False
                    region_reason = "blank"
                elif erased_ratio >= 0.65:
                    region_pass = False
                    region_reason = f"erased={erased_ratio:.2f}"
                elif region_preview <= 0.25:
                    region_pass = False
                    region_reason = f"preview={region_preview:.2f}"
            else:
                region_pass = False
                region_reason = f"mismatch={mismatches}"
        if not region_pass:
            structural_ok = False
            if structural_reason is None:
                structural_reason = "region"
        else:
            region_stage_keep += int(shape_pass)
        region_preview_scores.append(region_preview)
        meta["region_preview_score"] = round(region_preview, 4)
        _record("region", region_pass, region_reason)

        palette_extras = [color for color in colors if color not in evidence_palette and color != 0]
        palette_extras_total += len(palette_extras)
        meta["palette_extras_colors"] = len(palette_extras)

        dead_revived = any(color in dead_colors for color in palette_extras)
        if dead_revived:
            meta["dead_color_revived"] = True
            dead_color_revived_total += 1
        else:
            meta.setdefault("dead_color_revived", False)

        dual_weights = {int(color): float(weight) for color, weight in meta.get("pft_dual_weights", {}).items()}
        evidence_weight_total = sum(
            dual_weights.get(color, evidence_weights.get(color, 1.0))
            for color in evidence_core_colors
        )
        if evidence_weight_total <= 0:
            evidence_weight_total = float(
                max(1, len(evidence_core_colors) or len(test_input_palette) or 1)
            )
        missing_weight = sum(
            dual_weights.get(color, evidence_weights.get(color, 1.0))
            for color in evidence_core_colors
            if color not in colors
        )
        repulsive_term = max(
            0.0,
            min(1.0, 1.0 - (len(palette_extras) / float(palette_denominator))),
        )
        attractive_term = max(0.0, min(1.0, 1.0 - (missing_weight / evidence_weight_total)))
        palette_score = (repulse_weight * repulsive_term) + (attr_weight * attractive_term)
        if meta.get("pft_feasible"):
            palette_score += feasible_bonus
        palette_cost_norm = min(
            1.0,
            float(meta.get("pft_cost", 0.0)) / float(palette_denominator),
        )
        palette_score -= cost_weight * palette_cost_norm
        spectral_norm = min(
            1.0,
            float(meta.get("spectral_delta", 0.0)) / max(1.0, spectral_reference + 1.0),
        )
        palette_score -= spectral_weight * spectral_norm
        euler_penalty = min(
            1.0,
            float(meta.get("euler_violations", 0))
            / float(palette_denominator),
        )
        palette_score -= euler_weight * euler_penalty
        coverage_gap_norm = min(
            1.0,
            float(meta.get("palette_coverage_gap", 0)) / float(palette_denominator),
        )
        if meta.get("palette_coverage_ok"):
            palette_score += coverage_bonus
        else:
            palette_score -= coverage_penalty_weight * coverage_gap_norm
        palette_score = max(0.0, min(1.0, palette_score))
        palette_preview_scores.append(palette_score)
        palette_penalty = max(0.0, min(1.0, 1.0 - palette_score))
        meta["palette_preview_score"] = round(palette_score, 4)
        meta["palette_penalty"] = round(palette_penalty, 4)
        meta["palette_cost_norm"] = round(palette_cost_norm, 4)
        meta["spectral_norm"] = round(spectral_norm, 4)
        meta["euler_penalty"] = round(euler_penalty, 4)
        meta["palette_repulsive_term"] = round(repulsive_term, 4)
        meta["palette_attractive_term"] = round(attractive_term, 4)
        meta["palette_coverage_penalty"] = round(coverage_gap_norm, 4)

        palette_pass = not palette_extras
        if palette_gate_active:
            if palette_pass:
                _record("color", True, None)
            else:
                preview = sorted(palette_extras)[:4]
                suffix = "..." if len(palette_extras) > 4 else ""
                if palette_mode != "score":
                    structural_ok = False
                    if structural_reason is None:
                        structural_reason = "palette"
                _record(
                    "color",
                    False,
                    f"penalty={palette_penalty:.2f};extra={preview}{suffix}",
                )
        raw_palette_extras = (
            cand.get("palette_extras")
            or meta.get("palette_extras")
            or (cand.get("extra") or {}).get("palette_extras")
        )
        palette_extras: List[int] = []
        if isinstance(raw_palette_extras, dict):
            palette_extras = [
                int(color) for color in list(raw_palette_extras.keys()) + list(raw_palette_extras.values())
            ]
        elif isinstance(raw_palette_extras, (list, tuple, set)):
            palette_extras = [int(color) for color in raw_palette_extras]
        elif raw_palette_extras is not None:
            try:
                palette_extras = [int(raw_palette_extras)]
            except Exception:
                palette_extras = []

        palette_pass = True
        palette_penalty = 0.0
        mapping_targets: Set[int] = set()
        if candidate_mapping:
            mapping_targets = {int(dst) for dst in candidate_mapping.values()}
            for target in mapping_targets:
                if target not in allowed_palette:
                    palette_allowed_for_candidate.add(target)
                    palette_support_details.setdefault(target, "recolour")

        missing_from_test: List[int] = []
        for color in test_input_palette:
            if color in colors:
                continue
            if candidate_mapping and color in candidate_mapping:
                mapped = candidate_mapping.get(color)
                if mapped is not None and mapped in colors:
                    continue
            missing_from_test.append(color)
        palette_missing_count = len(missing_from_test)
        palette_missing_total += palette_missing_count
        meta["palette_missing_colors"] = palette_missing_count

        if candidate_mapping and recolour_only_task:
            base_conf = _safe01(cand.get("confidence", cand.get("score", 0.0)))
            boost_factor = 1.2 if "recolour" in ctype.lower() or "recolor" in ctype.lower() else 1.1
            boosted = min(1.0, max(base_conf, (base_conf * boost_factor) + 0.05))
            if boosted > base_conf:
                cand["confidence"] = boosted
                cand["score"] = boosted
                meta["recolour_confidence_boost"] = round(boosted - base_conf, 4)
                scores_map = cand.get("scores")
                if isinstance(scores_map, dict):
                    scores_map["recolour_boost"] = boosted
                else:
                    cand["scores"] = {"recolour_boost": boosted}

        palette_pass = not palette_extras
        if palette_gate_active:
            novel_disallowed: List[int] = []
            novel_colors_in_grid = {color for color in colors if color not in base_palette}
            cand_conf = _safe01(cand.get("confidence", cand.get("score", 0.0)))
            for color in colors:
                if color in palette_allowed_for_candidate or color in allowed_palette:
                    continue
                allow_reason: Optional[str] = None
                if color in palette_hints:
                    allow_reason = "hint"
                elif color in test_input_palette:
                    allow_reason = "test_input"
                else:
                    support = max(
                        supplemental_palette.get(color, 0.0),
                        candidate_soft_palette.get(color, 0.0),
                    )
                    if support >= palette_support_min:
                        allow_reason = "pool"
                    elif support >= 0.5 and cand_conf >= 0.45:
                        allow_reason = "confidence"
                    elif support >= 0.35 and cand_conf >= 0.7:
                        allow_reason = "high_conf"
                if allow_reason:
                    palette_allowed_for_candidate.add(color)
                    palette_support_details[color] = allow_reason
                    continue
                novel_disallowed.append(color)
            if novel_disallowed:
                palette_pass = False
                structural_ok = False
                if structural_reason is None:
                    structural_reason = "palette"
                base = float(max(1, len(colors) or len(test_input_palette) or len(novel_disallowed)))
                palette_penalty = min(1.0, len(novel_disallowed) / base)
                preview = sorted(novel_disallowed)[:4]
                suffix = "..." if len(novel_disallowed) > 4 else ""
                _record(
                    "color",
                    False,
                    f"penalty={palette_penalty:.2f};extra={preview}{suffix}",
                )
            else:
                _record("color", True, None)
        elif palette_extras:
            base = float(max(1, len(colors) or len(test_input_palette) or len(palette_extras)))
            palette_penalty = min(1.0, len(palette_extras) / base)
            preview = sorted(palette_extras)[:4]
            suffix = "..." if len(palette_extras) > 4 else ""
            if palette_mode != "score":
                palette_pass = False
                structural_ok = False
                if structural_reason is None:
                    structural_reason = "palette"
            _record(
                "color",
                False,
                f"penalty={palette_penalty:.2f};extra={preview}{suffix}",
            )
        else:
            _record("color", True, None)

        if palette_hints:
            ordered_palette = sorted(palette_hints)
            meta["gate_allowed_palette"] = ordered_palette
            cand["gate_allowed_palette"] = ordered_palette
            for color in ordered_palette:
                supplemental_palette[color] = max(
                    supplemental_palette.get(color, 0.0), rescue_boost
                )
            allowed_palette.update(ordered_palette)
        if palette_support_details:
            support_meta = meta.setdefault("gate_palette_support", {})
            for color, reason in palette_support_details.items():
                support_meta[int(color)] = reason

        palette_penalty_sum += palette_penalty
        if shape_pass and region_pass and not palette_pass and palette_missing_count:
            meta["palette_rescue_requested"] = True
            palette_rescue_requests.append(
                {
                    "index": idx,
                    "missing": palette_missing_count,
                }
            )
        if shape_pass and region_pass and palette_pass:
            palette_stage_keep += 1

        meta["palette_penalty"] = round(palette_penalty, 4)
        palette_penalty_sum += palette_penalty
        if shape_pass and region_pass and not palette_pass and palette_missing_count:
            meta["palette_rescue_requested"] = True
            palette_rescue_requests.append(
                {
                    "index": idx,
                    "missing": palette_missing_count,
                }
            )
        if shape_pass and region_pass and palette_pass:
            palette_stage_keep += 1

        palette_violation = (not palette_pass) and (shape_pass or region_pass)
        if palette_violation:
            meta["palette_gate_soft"] = True
            tags = meta.setdefault("gate_tags", [])
            if isinstance(tags, list) and "palette_violation" not in tags:
                tags.append("palette_violation")
            scores_map = cand.get("scores") if isinstance(cand.get("scores"), dict) else {}
            if not isinstance(cand.get("scores"), dict):
                cand["scores"] = scores_map
            penalty_value = float(scores_map.get("penalty_palette", 0.0) or 0.0) + 0.25
            scores_map["penalty_palette"] = penalty_value
            meta["palette_penalty_soft"] = round(penalty_value, 4)

        if hard_reject:
            continue

        keep_candidate = structural_ok
        if not keep_candidate:
            palette_only = palette_violation
            if palette_only and palette_overrides < palette_soft_limit:
                keep_candidate = True
                palette_overrides += 1
                palette_stage_keep += 1
                meta.setdefault("gate_soft_override", "palette")
            elif soft_overrides < gate_soft_cap:
                keep_candidate = True
                soft_overrides += 1
                meta.setdefault("gate_soft_override", structural_reason or "structure")

        if not keep_candidate:
            if palette_violation:
                meta.setdefault("gate_soft_override", "palette_pool")
                soft_palette_pool.append(cand)
            continue

        accepted_counts[ctype] += 1
        meta["gate_results"] = [
            {k: v for k, v in {"name": gr.name, "accepted": gr.accepted, "reason": gr.reason}.items() if v is not None}
            for gr in gate_sequence
        ]
        filtered.append(cand)
    if len(filtered) < min_after_gate and soft_palette_pool:
        deficit = max(0, min_after_gate - len(filtered))

        def _soft_priority(cand: Dict[str, Any]) -> Tuple[float, float, float]:
            meta = cand.get("meta") if isinstance(cand.get("meta"), dict) else {}
            structure = float(meta.get("region_preview_score", 0.0) or 0.0)
            shape_score = float(meta.get("shape_preview_score", 0.0) or 0.0)
            palette_preview = float(meta.get("palette_preview_score", shape_score) or 0.0)
            blend = (0.6 * structure) + (0.4 * max(shape_score, palette_preview))
            scores_map = cand.get("scores") if isinstance(cand.get("scores"), dict) else {}
            penalty = float(scores_map.get("penalty_palette", 0.0) or 0.0)
            return (blend - penalty, blend, -penalty)

        soft_palette_pool.sort(key=_soft_priority, reverse=True)
        for cand in soft_palette_pool[:deficit]:
            meta = cand.setdefault("meta", {}) if isinstance(cand, dict) else {}
            if isinstance(meta, dict):
                meta.setdefault("gate_soft_override", "palette_pool")
            filtered.append(cand)
            accepted_counts[str(cand.get("type", "unknown"))] += 1
            palette_soft_promoted += 1

    total_candidates = sum(totals.values())
    accepted_total = len(filtered)
    rejected_total = total_candidates - accepted_total

    if metrics is not None:
        metrics.observe("gates.total", float(total_candidates))
        metrics.observe("gates.accepted", float(accepted_total))
        metrics.observe("gates.rejected", float(rejected_total))
        for name, count in gate_accept.items():
            metrics.observe(f"gate.{name}.accepted", float(count))
        for name, count in gate_reject.items():
            metrics.observe(f"gate.{name}.rejected", float(count))

    by_type = {
        cand_type: {
            "total": totals[cand_type],
            "accepted": accepted_counts.get(cand_type, 0),
            "rejected": totals[cand_type] - accepted_counts.get(cand_type, 0),
        }
        for cand_type in totals
    }

    gate_stats = {
        "total": total_candidates,
        "accepted": accepted_total,
        "rejected": rejected_total,
        "by_type": by_type,
        "gate_counts": {
            name: {"accepted": gate_accept.get(name, 0), "rejected": gate_reject.get(name, 0)}
            for name in set(list(gate_accept.keys()) + list(gate_reject.keys()))
        },
        "rejections_by_type": {name: dict(counter) for name, counter in per_gate_by_type.items()},
    }

    gate_stats.update(
        {
            "palette_distinct_before": len(colors_before),
            "palette_distinct_after": len({color for color in _pool_distinct_colors(filtered) if color != 0}),
            "component_retention": _component_retention(candidates, filtered),
            "palette_rescue_seeded": palette_rescue_applied,
            "gate_soft_overrides": soft_overrides,
            "palette_soft_overrides": palette_overrides,
            "palette_soft_promoted": palette_soft_promoted,
            "pool_before_gates": pool_before,
            "after_shape": shape_pass_count,
            "after_region": region_stage_keep,
            "after_palette": palette_stage_keep,
            "palette_penalty_mean": (
                palette_penalty_sum / float(max(1, pool_before))
            ),
            "palette_missing_mean": (
                palette_missing_total / float(max(1, pool_before))
            ),
            "region_preview_mean": (
                sum(region_preview_scores) / float(len(region_preview_scores))
                if region_preview_scores
                else 1.0
            ),
            "palette_rescue_requests": len(palette_rescue_requests),
            "palette_rescue_missing_total": palette_missing_total,
        }
    )

    return filtered, gate_stats

# --------------------------- RIL Solver --------------------------------------
class RILSolver:
    """Rule Induction Layer solver implementing 5-stage pipeline"""

    def __init__(self, seed: int = 1337):
        self.seed = seed
        random.seed(seed)

        # Environment configuration
        self.palette_mapper_on = os.getenv("RIL_PALETTE_MAPPER_ON", "0") == "1"
        self.csp_evidence_min = float(os.getenv("RIL_CSP_EVIDENCE_MIN", str(DEFAULT_CSP_EVIDENCE)))
        self.palette_support_min = float(
            os.getenv("RIL_PALETTE_SUPPORT_MIN", str(DEFAULT_PALETTE_SUPPORT))
        )
        self.palette_floor = int(os.getenv("RIL_PALETTE_FLOOR", str(DEFAULT_PALETTE_FLOOR)))
        self.palette_rescue_keep_ratio = float(
            os.getenv("RIL_PALETTE_RESCUE_KEEP_RATIO", str(DEFAULT_PALETTE_RESCUE_KEEP))
        )
        self.gate_soft_topk = int(os.getenv("RIL_GATE_SOFT_TOPK", str(DEFAULT_GATE_SOFT_TOPK)))
        self.palette_soft_keep_ratio = float(os.getenv("RIL_PALETTE_SOFT_RATIO", "0.1"))
        self.palette_soft_min = int(os.getenv("RIL_PALETTE_SOFT_MIN", "8"))
        self.min_beam_after_gate = int(
            os.getenv("RIL_MIN_BEAM_AFTER_GATE", str(DEFAULT_MIN_BEAM_AFTER_GATE))
        )
        self.region_gate_mode = os.getenv("RIL_REGION_GATE_MODE", "strict")
        self.min_area = int(os.getenv("RIL_SCORING_MIN_AREA", str(DEFAULT_SCORING_MIN_AREA)))
        self.topk = int(os.getenv("RIL_PALETTE_TOPK", "2"))  # final outputs to return
        self._gate_rescue_enabled = os.getenv("RIL_GATE_RESCUE", "1") != "0"
        self._gate_rescue_expand_palette = os.getenv("RIL_GATE_RESCUE_EXPAND_PALETTE", "1") != "0"
        self._gate_rescue_erode = os.getenv("RIL_GATE_RESCUE_ERODE", "1") != "0"
        self._gate_logging = os.getenv("RIL_GATE_LOGGING", "0") == "1"

        # Scoring knobs
        self.perfect_fit_bonus = float(os.getenv("RIL_PERFECT_FIT_BONUS", "2.5"))
        self.partial_fit_scale = float(os.getenv("RIL_PARTIAL_FIT_SCALE", "0.5"))
        self.never_fit_penalty = float(os.getenv("RIL_NEVER_FIT_PENALTY", "0.3"))
        self.shape_match_bonus = float(os.getenv("RIL_SHAPE_MATCH_BONUS", "1.5"))
        raw_policy = os.getenv("ARC_ROUTER_POLICY", ROUTER_POLICY_DEFAULT)
        policy = str(raw_policy or ROUTER_POLICY_DEFAULT).strip().lower()
        if policy == "adapters_first":
            policy = "adapter_first"
        self.policy_name = policy or ROUTER_POLICY_DEFAULT
        self.current_task_id = "unknown"
        self.last_trace: Dict[str, Any] = {}
        self._trace_history: List[Dict[str, Any]] = []
        self._last_output_metrics: Optional[Dict[str, Any]] = None
        self.router_config: Dict[str, Any] = {}
        self._scorecard_every = max(1, int(os.getenv("ARC_SCORECARD_EVERY", "10")))
        self._next_scorecard_at = self._scorecard_every
        self._trace_log_path = os.getenv("ARC_TRACE_LOG")
        self._trace_log_error = False
        self._recent_results = []
        self._gate_totals: Counter[str] = Counter()
        self._candidate_type_totals: Counter[str] = Counter()
        self._last_train_check: Dict[str, Any] = {"hits": 0, "cases": 0, "hamming": []}
        self._last_gate_stats: Dict[str, Any] = {}
        self._gate_profile_enabled = os.getenv("RIL_GATE_PROFILE", "0") == "1"
        self.current_test_index = 0
        self._solve_invocation = 0
        self._border_diagnostics: Dict[str, Any] = {}
        self._current_gate_context: Optional[GateContext] = None
        self._current_allowed_palette: Optional[Set[int]] = None
        self._current_canvas_dims: Optional[Tuple[int, int]] = None
        self._current_test_input: Optional[Grid] = None
        self._current_train_pairs: List[Tuple[Grid, Grid]] = []
        self._input_cc_count = 0
        self._train_output_colors: Set[int] = set()
        self._train_output_multicolor = False
        self._finishers_short_circuit: Dict[str, float] = {
            "palette_missing": 1.0,
            "region_preview": 0.25,
        }
        self._last_finishers_stats: Dict[str, int] = {"tried": 0, "applied": 0}
        print(f"[ROUTER] policy={self.policy_name}")

    # --------------------------- Candidate utils ----------------------------
    def _ensure_grid(self, grid_like: Any) -> Optional[Grid]:
        """Best-effort conversion of ``grid_like`` into a grid.

        External helpers occasionally return NumPy arrays or tuples instead of
        a plain list-of-lists.  Normalising everything early ensures the rest of
        the pipeline (beam scoring, hashing, etc.) can rely on a consistent
        representation.  ``None`` is returned if coercion fails so callers can
        drop the candidate.
        """

        if grid_like is None:
            return None

        # Convert numpy arrays or other array-like objects via ``asgrid``.  If
        # that fails we fall back to trying ``tolist``/``list`` coercions.
        if not isinstance(grid_like, list):
            try:
                grid_like = asgrid(grid_like)
            except Exception:
                try:
                    grid_like = list(grid_like)  # type: ignore[arg-type]
                except Exception:
                    return None

        coerced: Grid = []
        for row in grid_like:
            if isinstance(row, list):
                values = row
            else:
                try:
                    values = list(row)  # type: ignore[arg-type]
                except Exception:
                    return None
            try:
                coerced.append([int(cell) for cell in values])
            except Exception:
                return None
        return coerced

    def _hash_grid(self, grid: Any) -> str:
        """Stable hash for candidate grids used in deduping & traces."""

        normalized = self._ensure_grid(grid)
        if normalized is None:
            return "invalid"
        try:
            payload = json.dumps(normalized, separators=(",", ":"))
        except Exception:
            payload = repr(normalized)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _make_candidate(
        self,
        grid: Any,
        cand_type: Optional[str] = None,
        confidence: Optional[float] = None,
        *,
        source: str = "adapter",
        param_hash: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Candidate:
        """Create a structured candidate object for downstream stages."""

        normalized_grid = self._ensure_grid(grid)
        if normalized_grid is None:
            normalized_grid = []

        kind = (
            cand_type
            or kwargs.pop("kind", None)
            or kwargs.pop("type", None)
            or "unknown"
        )

        score_value: float
        raw_score = confidence if confidence is not None else kwargs.pop("score", None)
        try:
            score_value = float(raw_score) if raw_score is not None else 0.0
        except Exception:
            score_value = 0.0

        meta_dict: Dict[str, Any] = dict(meta) if isinstance(meta, dict) else {}

        extras: Dict[str, Any] = {}
        if isinstance(extra, dict):
            extras.update(extra)
        if kwargs:
            extras.update(kwargs)

        candidate = Candidate(
            grid=normalized_grid,
            kind=str(kind),
            score=score_value,
            source=source or "adapter",
            meta=meta_dict,
            param_hash=param_hash if isinstance(param_hash, str) else None,
        )

        if extras:
            candidate.extra.update(extras)

        candidate.meta.setdefault("src", candidate.source)
        return candidate

    def _apply_palette_rescue_prior(
        self,
        candidate: Candidate,
        target_palette: Iterable[int],
    ) -> None:
        """Raise palette rescue priors when the candidate covers the palette."""

        try:
            palette = {int(color) for color in target_palette if color is not None}
        except Exception:
            palette = set()

        palette_score = _palette_completion_score(candidate.grid, palette)
        candidate.scores["palette_score"] = palette_score

        if not palette or not isinstance(candidate.grid, list):
            return

        try:
            predicted = {int(value) for row in candidate.grid for value in row}
        except Exception:
            return

        if not predicted:
            return

        coverage = len(predicted & palette) / (len(palette) or 1)
        extra_ratio = len(predicted - palette) / max(1, len(predicted))

        base = candidate.scores.get("prior_conf", candidate.score)
        try:
            base = float(base)
        except Exception:
            base = candidate.score
        if not isinstance(base, (int, float)) or not math.isfinite(base):
            base = 0.0
        base = max(0.0, float(base))

        boosted = base
        if coverage >= 1.0 and extra_ratio == 0.0:
            boosted = max(base, 0.42)
        elif coverage >= 0.5 and extra_ratio <= 0.25:
            boosted = max(base, 0.30)

        candidate.scores["prior_conf"] = boosted
        candidate.score = max(candidate.score, boosted)

    def _plausible_candidate(self, cand: Dict[str, Any]) -> bool:
        grid = cand.get("grid")
        if not isinstance(grid, list) or not grid:
            return False
        if _grid_nonzero_area(grid) < self.min_area:
            return False
        gate_ctx = getattr(self, "_current_gate_context", None)
        required_colors: Set[int] = set()
        if gate_ctx is not None:
            for color in getattr(gate_ctx, "test_input_colors", set()):
                try:
                    color_int = int(color)
                except Exception:
                    continue
                if color_int != 0:
                    required_colors.add(color_int)
        candidate_colors = {
            int(color)
            for color in _grid_colors(grid)
            if isinstance(color, int) and color != 0
        }
        if self._train_output_multicolor and required_colors:
            if not required_colors.issubset(candidate_colors):
                return False
        if self._input_cc_count > 0 and _count_components_grid(grid) == 0:
            return False
        meta = cand.get("meta") if isinstance(cand.get("meta"), dict) else {}
        palette_missing = int(meta.get("palette_missing_colors", 0) or 0)
        palette_penalty = float(meta.get("palette_penalty", 0.0) or 0.0)
        region_preview = float(meta.get("region_preview_score", 1.0) or 1.0)
        if palette_missing and palette_penalty >= self._finishers_short_circuit.get("palette_missing", 1.0):
            return False
        if region_preview < self._finishers_short_circuit.get("region_preview", 0.0):
            return False
        return True

    def _micro_finishers(
        self, candidates: List[Dict[str, Any]], gate_ctx: GateContext
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        max_per = max(0, int(getattr(self, "finishers_max_per_candidate", 0)))
        if not candidates or max_per <= 0:
            return candidates, {"micro_tried": 0, "micro_applied": 0}
        reference = gate_ctx.reference_input if gate_ctx else None
        test_palette = {
            int(color)
            for color in getattr(gate_ctx, "test_input_colors", set())
            if isinstance(color, int)
        }
        augmented: List[Dict[str, Any]] = list(candidates)
        tried = 0
        applied = 0
        for cand in candidates:
            grid = cand.get("grid")
            if not isinstance(grid, list):
                continue
            meta = cand.get("meta") if isinstance(cand.get("meta"), dict) else {}
            palette_missing = int(meta.get("palette_missing_colors", 0) or 0)
            if getattr(self, "finishers_skip_if_palette_missing", True) and palette_missing:
                continue
            shape_score = float(meta.get("shape_preview_score", 1.0) or 1.0)
            if shape_score < getattr(self, "finishers_min_shape_score", 0.2):
                continue
            if shape_score > getattr(self, "finishers_max_shape_score", 0.9):
                continue
            near_miss = _grid_difference_ratio(grid, reference)
            if near_miss > 0.4:
                continue
            variants: List[Tuple[str, Grid]] = []
            centroid_variant = self._micro_centroid_snap(grid, reference)
            if centroid_variant is not None:
                variants.append(("centroid_snap", centroid_variant))
            dilated_variant = self._micro_dilate_trim(grid)
            if dilated_variant is not None:
                variants.append(("dilate_trim", dilated_variant))
            stitched_variant = self._micro_component_stitch(grid)
            if stitched_variant is not None:
                variants.append(("component_stitch", stitched_variant))
            if not variants:
                continue
            base_conf = _safe01(cand.get("confidence", 0.0))
            for name, variant_grid in variants[:max_per]:
                tried += 1
                if variant_grid == grid:
                    continue
                variant_meta = dict(meta)
                variant_meta["finisher"] = name
                variant_meta["finisher_parent"] = cand.get("type") or cand.get("op_id")
                variant_meta.setdefault("src", cand.get("source", "adapter"))
                variant_meta["shape_preview_score"], degenerate = _shape_preview_score(
                    variant_grid, reference
                )
                variant_meta["shape_preview_score"] = round(
                    float(variant_meta["shape_preview_score"]), 4
                )
                if degenerate:
                    variant_meta["shape_degenerate"] = True
                if test_palette:
                    current_colors = _grid_colors(variant_grid)
                    variant_meta["palette_missing_colors"] = len(
                        [color for color in test_palette if color not in current_colors]
                    )
                variant_payload = dict(cand)
                variant_payload["grid"] = variant_grid
                variant_payload["source"] = "finisher"
                variant_payload["type"] = f"{cand.get('type', 'unknown')}_micro_{name}"
                variant_payload["confidence"] = max(0.0, min(1.0, base_conf * 0.97))
                variant_payload["meta"] = variant_meta
                augmented.append(variant_payload)
                applied += 1
        return augmented, {"micro_tried": tried, "micro_applied": applied}

    def _collect_diagonal_runs(
        self, grid: Grid, dr: int, dc: int
    ) -> List[Tuple[int, List[Tuple[int, int]]]]:
        if not grid or not isinstance(grid, list) or not grid[0]:
            return []
        h, w = grid_shape(grid)
        runs: List[Tuple[int, List[Tuple[int, int]]]] = []
        for r in range(h):
            for c in range(w):
                color = grid[r][c]
                if color == 0:
                    continue
                prev_r, prev_c = r - dr, c - dc
                if in_bounds(prev_r, prev_c, h, w) and grid[prev_r][prev_c] == color:
                    continue
                run: List[Tuple[int, int]] = []
                nr, nc = r, c
                while in_bounds(nr, nc, h, w) and grid[nr][nc] == color:
                    run.append((nr, nc))
                    nr += dr
                    nc += dc
                if len(run) >= 2:
                    runs.append((color, run))
        return runs

    def _candidate_looks_like_scaffold(self, cand: Mapping[str, Any]) -> bool:
        grid = cand.get("grid") if isinstance(cand, Mapping) else None
        if not isinstance(grid, list) or not grid or not isinstance(grid[0], list):
            return False
        h, w = grid_shape(grid)
        if h < 3 or w < 3:
            return False
        total_nonzero = sum(1 for row in grid for cell in row if cell != 0)
        if total_nonzero < max(6, (h * w) // 10):
            return False
        diag_runs: List[Tuple[int, List[Tuple[int, int]]]] = []
        for dr, dc in ((1, 1), (1, -1)):
            diag_runs.extend(self._collect_diagonal_runs(grid, dr, dc))
        if not any(len(run) >= 3 for _, run in diag_runs):
            return False
        diag_counts: Counter[int] = Counter()
        anti_counts: Counter[int] = Counter()
        for r, row in enumerate(grid):
            for c, val in enumerate(row):
                if val == 0:
                    continue
                diag_counts[r - c] += 1
                anti_counts[r + c] += 1
        diag_peak = max(diag_counts.values(), default=0)
        anti_peak = max(anti_counts.values(), default=0)
        diag_strength = max(diag_peak, anti_peak) / float(total_nonzero) if total_nonzero else 0.0
        if diag_strength < 0.30:
            return False
        row_tail_start = max(0, h - max(1, h // 3))
        col_tail_start = max(0, w - max(1, w // 3))
        left_tail_end = max(0, max(1, w // 3))
        tail_rows = sum(1 for r in range(row_tail_start, h) for cell in grid[r] if cell != 0)
        tail_right = sum(
            1 for r in range(h) for c in range(col_tail_start, w) if grid[r][c] != 0
        )
        tail_left = sum(1 for r in range(h) for c in range(0, left_tail_end) if grid[r][c] != 0)
        head_rows = sum(1 for r in range(0, max(1, h // 3)) for cell in grid[r] if cell != 0)
        if head_rows <= 0:
            return False
        tail_ratio = min(tail_rows, max(tail_left, tail_right)) / float(total_nonzero)
        if tail_ratio > 0.4:
            return False
        return True

    def _maybe_extend_diagonals(
        self, grid: Grid, ref_stats: Optional[Mapping[str, Any]]
    ) -> Grid:
        _ = ref_stats  # context reserved for future tweaks
        if not isinstance(grid, list) or not grid or not isinstance(grid[0], list):
            return grid
        h, w = grid_shape(grid)
        if h < 2 or w < 2:
            return grid
        total_nonzero = sum(1 for row in grid for cell in row if cell != 0)
        if total_nonzero == 0:
            return grid
        row_tail_start = max(0, h - max(1, h // 3))
        col_tail_start = max(0, w - max(1, w // 3))
        left_tail_end = max(0, max(1, w // 3))
        tail_rows = sum(1 for r in range(row_tail_start, h) for cell in grid[r] if cell != 0)
        tail_right = sum(
            1 for r in range(h) for c in range(col_tail_start, w) if grid[r][c] != 0
        )
        tail_left = sum(1 for r in range(h) for c in range(0, left_tail_end) if grid[r][c] != 0)
        tail_pressure = min(tail_rows, max(tail_left, tail_right)) / float(total_nonzero)
        if tail_pressure > 0.45:
            return grid
        canvas = copy_grid(grid)
        applied_any = False
        passes = 0
        while passes < 3:
            extended = False
            for dr, dc in ((1, 1), (1, -1)):
                for color, run in self._collect_diagonal_runs(canvas, dr, dc):
                    if len(run) < 3:
                        continue
                    last_r, last_c = run[-1]
                    nr, nc = last_r + dr, last_c + dc
                    if not in_bounds(nr, nc, h, w):
                        continue
                    if canvas[nr][nc] != 0:
                        continue
                    if nr < row_tail_start:
                        continue
                    if dc == 1 and nc < col_tail_start:
                        continue
                    if dc == -1 and nc > left_tail_end:
                        continue
                    canvas[nr][nc] = color
                    extended = True
            if not extended:
                break
            applied_any = True
            passes += 1
        return canvas if applied_any else grid

    def _micro_centroid_snap(
        self, grid: Grid, reference: Optional[Grid]
    ) -> Optional[Grid]:
        if not reference or grid_shape(grid) != grid_shape(reference):
            return None
        cand_points = [(r, c) for r, row in enumerate(grid) for c, val in enumerate(row) if val != 0]
        ref_points = [
            (r, c) for r, row in enumerate(reference) for c, val in enumerate(row) if val != 0
        ]
        if not cand_points or not ref_points:
            return None
        cand_avg_r = sum(r for r, _ in cand_points) / float(len(cand_points))
        cand_avg_c = sum(c for _, c in cand_points) / float(len(cand_points))
        ref_avg_r = sum(r for r, _ in ref_points) / float(len(ref_points))
        ref_avg_c = sum(c for _, c in ref_points) / float(len(ref_points))
        dy = int(round(ref_avg_r - cand_avg_r))
        dx = int(round(ref_avg_c - cand_avg_c))
        if dy == 0 and dx == 0:
            return None
        if abs(dy) > 2 or abs(dx) > 2:
            return None
        return self._shift_grid(grid, dx=dx, dy=dy)

    def _micro_dilate_trim(self, grid: Grid) -> Optional[Grid]:
        if not grid or not grid[0]:
            return None
        bbox = rescue_utils.get_bounding_box(grid)
        min_r, min_c, max_r, max_c = bbox
        if max_r <= min_r and max_c <= min_c:
            return None
        h, w = grid_shape(grid)
        out = copy_grid(grid)
        changed = False
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if grid[r][c] == 0:
                    continue
                color = grid[r][c]
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if in_bounds(nr, nc, h, w) and min_r <= nr <= max_r and min_c <= nc <= max_c:
                        if out[nr][nc] == 0:
                            out[nr][nc] = color
                            changed = True
        return out if changed else None

    def _micro_component_stitch(self, grid: Grid) -> Optional[Grid]:
        comps = [comp for comp in connected_components(grid, bg=0) if comp]
        if len(comps) < 2:
            return None
        small_comps = [comp for comp in comps if len(comp) <= 4]
        if len(small_comps) < 2:
            return None
        best_pair: Optional[Tuple[int, int, int, int, int]] = None
        best_dist = 999
        for idx, comp_a in enumerate(small_comps):
            for comp_b in small_comps[idx + 1 :]:
                for ra, ca in comp_a:
                    for rb, cb in comp_b:
                        dist = abs(ra - rb) + abs(ca - cb)
                        if dist < best_dist:
                            best_dist = dist
                            best_pair = (ra, ca, rb, cb, idx)
        if best_pair is None or best_dist > 2:
            return None
        ra, ca, rb, cb, comp_idx = best_pair
        color = grid[small_comps[comp_idx][0][0]][small_comps[comp_idx][0][1]]
        out = copy_grid(grid)
        rr, cc = ra, ca
        changed = False
        while (rr, cc) != (rb, cb):
            if rr < rb:
                rr += 1
            elif rr > rb:
                rr -= 1
            if cc < cb:
                cc += 1
            elif cc > cb:
                cc -= 1
            if out[rr][cc] == 0:
                out[rr][cc] = color
                changed = True
        return out if changed else None


    def _arbitrate_candidate_families(
        self,
        candidates: Iterable[Dict[str, Any]],
        *,
        metrics: Optional[Metrics] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
        """Partition candidates by source and enforce the router policy."""

        buckets: Dict[str, List[Dict[str, Any]]] = {"adapter": [], "external": [], "echo": []}
        for cand in candidates:
            src = cand.get("source", "adapter")
            if src not in buckets:
                src = "adapter"
                cand["source"] = src
                cand.setdefault("meta", {}).setdefault("src", src)
            buckets[src].append(cand)

        decision = "mixed"
        primary_sources: List[str] = []
        rescue_sources: List[str] = []

        if self.policy_name == "external_first":
            if buckets["external"]:
                primary_sources = ["external"]
                if buckets["adapter"]:
                    rescue_sources = ["adapter"]
                decision = "external"
            elif buckets["adapter"]:
                primary_sources = ["adapter"]
                decision = "adapter_fallback"
            else:
                decision = "empty"
        elif self.policy_name == "adapter_first":
            if buckets["adapter"]:
                primary_sources = ["adapter"]
                if buckets["external"]:
                    rescue_sources = ["external"]
                decision = "adapter"
            elif buckets["external"]:
                primary_sources = ["external"]
                decision = "external_fallback"
            else:
                decision = "empty"
        elif self.policy_name == "external_off":
            if buckets["adapter"]:
                primary_sources = ["adapter"]
                decision = "adapter_only"
            elif buckets["external"]:
                primary_sources = ["external"]
                decision = "external_fallback"
            else:
                decision = "empty"
        else:
            for src in ("adapter", "external"):
                if buckets[src]:
                    primary_sources.append(src)
            decision = self.policy_name or "mixed"

        if not primary_sources and buckets["adapter"]:
            primary_sources = ["adapter"]
            decision = f"{decision}_adapter"
        elif not primary_sources and buckets["external"]:
            primary_sources = ["external"]
            decision = f"{decision}_external"

        held_back_sources = [
            src for src in ("adapter", "external") if src not in primary_sources and buckets[src]
        ]
        if not rescue_sources:
            rescue_sources = held_back_sources

        cfg_obj = getattr(self, "router_config", {})
        if not isinstance(cfg_obj, Mapping):
            cfg_obj = {}

        hybrid_pool: Optional[List[Dict[str, Any]]] = None
        hybrid_used = False
        if (
            "external" in primary_sources
            and "adapter" not in primary_sources
            and buckets["external"]
        ):
            external_pool = buckets.get("external", [])
            adapter_pool = buckets.get("adapter", [])
            cand_pool = _route_candidates(external_pool, adapter_pool, cfg_obj)
            if len(cand_pool) > len(external_pool) and adapter_pool:
                hybrid_pool = list(cand_pool)
                hybrid_used = True
                primary_sources = primary_sources + ["adapter"]
                rescue_sources = [src for src in rescue_sources if src != "adapter"]
                decision = f"{decision}+hybrid"

        def _gather(srcs: Iterable[str]) -> List[Dict[str, Any]]:
            gathered: List[Dict[str, Any]] = []
            for src in srcs:
                gathered.extend(buckets.get(src, []))
            return gathered

        if hybrid_pool is not None:
            primary_candidates = hybrid_pool
        else:
            primary_candidates = _gather(primary_sources)
        rescue_pool = _gather(src for src in rescue_sources if src not in primary_sources)

        primary_candidates.extend(buckets.get("echo", []))

        telemetry = {
            "policy": self.policy_name,
            "decision": decision,
            "routed_sources": primary_sources + (["echo"] if buckets.get("echo") else []),
            "held_out_sources": [src for src in held_back_sources if src not in primary_sources],
            "source_counts": {src: len(payload) for src, payload in buckets.items()},
            "router_hybrid": hybrid_used,
        }

        if metrics is not None:
            decision_tag = decision or "none"
            metrics.inc(f"gate.arbitration.{decision_tag}")
            for src, count in telemetry["source_counts"].items():
                metrics.observe(f"router.source_pool.{src}", float(count))

        return primary_candidates, rescue_pool, telemetry, cfg_obj

    def _gate_candidates_with_rescue(
        self,
        candidates: Iterable[Dict[str, Any]],
        gate_ctx: GateContext,
        metrics: Optional[Metrics],
        rescue_seed: Optional[Iterable[Dict[str, Any]]] = None,
        train_examples: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        pool = list(candidates)
        _ = train_examples  # kept for compatibility with upstream rescue API
        palette_seeds: List[Dict[str, Any]] = []
        if ensure_palette_rescue is not None and np is not None:
            try:
                palette_floor = max(1, int(getattr(gate_ctx, "palette_floor", self.palette_floor)))
            except Exception:
                palette_floor = self.palette_floor
            baseline_grid = getattr(gate_ctx, "reference_input", None)
            test_colors = sorted(
                int(c) for c in getattr(gate_ctx, "test_input_colors", set()) if int(c) != 0
            )
            target_palette: Set[int] = set(test_colors)
            if baseline_grid and np is not None:
                try:
                    baseline_arr = np.asarray(baseline_grid, dtype=np.uint8)
                except Exception:
                    baseline_arr = None
                if baseline_arr is not None:
                    pool_arrays: List[Any] = []
                    for cand in pool:
                        grid = cand.get("grid") if isinstance(cand, dict) else None
                        if not isinstance(grid, list):
                            continue
                        try:
                            pool_arrays.append(np.asarray(grid, dtype=np.uint8))
                        except Exception:
                            continue

                    ctx_payload = {"test_input_colors": test_colors, "baseline_pred": baseline_arr}
                    distinct_palette = {
                        int(v)
                        for arr in pool_arrays
                        for v in np.unique(arr)
                        if int(v) != 0
                    }
                    if len(distinct_palette) < palette_floor:
                        rescued_arrays = ensure_palette_rescue(
                            pool_arrays or [baseline_arr],
                            ctx_payload,
                            k_floor=palette_floor,
                            keep_ratio=self.palette_rescue_keep_ratio,
                        )
                        existing_hashes = {
                            self._hash_grid(c.get("grid"))
                            for c in pool
                            if isinstance(c, dict)
                        }
                        for arr in rescued_arrays:
                            grid_list = arr.tolist()
                            gh = self._hash_grid(grid_list)
                            if gh in existing_hashes:
                                continue
                            existing_hashes.add(gh)
                            rescue_candidate = self._make_candidate(
                                grid_list,
                                cand_type="palette_rescue",
                                confidence=0.0,
                                source="echo",
                                meta={"src": "palette_rescue", "palette_seed": True},
                            )
                            self._apply_palette_rescue_prior(rescue_candidate, target_palette)
                            palette_seeds.append(rescue_candidate.as_payload())

                    baseline_hash = self._hash_grid(baseline_grid)
                    if not any(
                        self._hash_grid(c.get("grid")) == baseline_hash
                        for c in pool
                        if isinstance(c, dict)
                    ):
                        base_candidate = self._make_candidate(
                            copy_grid(baseline_grid),
                            cand_type="baseline_echo",
                            confidence=0.0,
                            source="echo",
                            meta={"src": "palette_rescue", "palette_seed": "baseline"},
                        )
                        self._apply_palette_rescue_prior(base_candidate, target_palette)
                        palette_seeds.insert(0, base_candidate.as_payload())

        if palette_seeds:
            pool = palette_seeds + pool
        gate_ctx.palette_soft_floor = max(gate_ctx.palette_soft_floor, self.palette_soft_min)
        gate_ctx.palette_soft_ratio = max(0.0, self.palette_soft_keep_ratio)
        gate_ctx.region_gate_mode = self.region_gate_mode
        gate_ctx.min_beam_after_gate = max(
            getattr(gate_ctx, "min_beam_after_gate", 0),
            self.min_beam_after_gate,
        )
        filtered, strict_stats = _apply_candidate_gates(
            pool, gate_ctx, metrics=metrics, palette_support_min=self.palette_support_min
        )
        strict_stats.setdefault("mode", "strict")
        strict_stats.setdefault("rescue", {"attempted": False})

        accepted = strict_stats.get("accepted", 0)
        total = strict_stats.get("total", 0)
        if (
            self._gate_rescue_enabled
            and total
            and accepted == 0
            and pool
        ):
            if metrics is not None:
                metrics.inc("gate.rescue_attempts")
            seed_preview: List[Dict[str, Any]] = []
            for cand in list(rescue_seed or []):
                if isinstance(cand, dict):
                    seed_preview = [cand]
                    break
            palette_source = pool + seed_preview
            if not palette_source and gate_ctx.reference_input:
                palette_source = [
                    {"grid": copy_grid(gate_ctx.reference_input), "source": "gate_rescue_seed"}
                ]
            candidate_palette = _candidate_palette(palette_source)
            relaxed_ctx = _relax_gate_context(
                gate_ctx,
                candidate_palette=candidate_palette,
                expand_palette=self._gate_rescue_expand_palette,
                erode_frozen=self._gate_rescue_erode,
                test_input_colors=getattr(gate_ctx, "test_input_colors", set()),
            )
            relaxed_ctx.min_beam_after_gate = max(
                getattr(relaxed_ctx, "min_beam_after_gate", 0), self.min_beam_after_gate
            )
            support_schedule = [self.palette_support_min]
            if self.palette_support_min > 0.75:
                support_schedule.append(0.75)
            if support_schedule[-1] > 0.65:
                support_schedule.append(0.65)
            relaxed_filtered: List[Dict[str, Any]] = []
            relaxed_stats: Dict[str, Any] = {}
            applied_floor = self.palette_support_min
            for floor in support_schedule:
                relaxed_filtered, relaxed_stats = _apply_candidate_gates(
                    palette_source,
                    relaxed_ctx,
                    metrics=metrics,
                    palette_support_min=floor,
                )
                applied_floor = floor
                if relaxed_filtered:
                    break
            relaxed_stats.setdefault("mode", "relaxed")
            relaxed_stats["strict_stats"] = strict_stats
            relaxed_stats["rescue"] = {
                "attempted": True,
                "palette_before": len(gate_ctx.allowed_palette or []),
                "palette_after": len(relaxed_ctx.allowed_palette or []),
                "frozen_before": round(gate_ctx.frozen_density, 6),
                "frozen_after": round(relaxed_ctx.frozen_density, 6),
                "seeded": bool(seed_preview),
                "seed_source": seed_preview[0].get("source") if seed_preview else None,
                "support_floor": applied_floor,
            }
            accepted_relaxed = relaxed_stats.get("accepted", 0)
            print(
                "[GATE-RESCUE] triggered strict=0 "
                f"palette={len(gate_ctx.allowed_palette or [])}→{len(relaxed_ctx.allowed_palette or [])} "
                f"frozen={gate_ctx.frozen_density:.3f}→{relaxed_ctx.frozen_density:.3f} "
                f"accepted={accepted_relaxed}"
            )
            return relaxed_filtered, relaxed_stats

        return filtered, strict_stats

    def _normalize_candidate(
        self, cand: Any, *, default_source: str = "adapter"
    ) -> Optional[Dict[str, Any]]:
        """Fold heterogeneous candidate payloads into a single schema."""

        if isinstance(cand, Candidate):
            cand = cand.as_payload()

        if not isinstance(cand, dict):
            return None

        grid_like = cand.get("grid")
        if grid_like is None and "program" in cand:
            grid_like = cand.get("program")

        normalized_grid = self._ensure_grid(grid_like)
        if normalized_grid is None:
            return None

        target_dims = self._current_canvas_dims
        if target_dims and len(target_dims) == 2:
            try:
                target_h = int(target_dims[0])
                target_w = int(target_dims[1])
            except Exception:
                target_h, target_w = _dims(normalized_grid)
        else:
            target_h, target_w = _dims(normalized_grid)
        enforced_grid = _enforce_canvas_size(normalized_grid, target_h, target_w, fill=0)
        allowed_palette = self._current_allowed_palette
        if allowed_palette is not None and not _has_only_allowed_colors(enforced_grid, allowed_palette):
            return None
        normalized_grid = enforced_grid

        raw_conf = cand.get("confidence", cand.get("score", 0.0))
        try:
            confidence = _safe01(raw_conf)
        except Exception:
            confidence = 0.0

        cand_type = (
            cand.get("type")
            or cand.get("cand_type")
            or cand.get("op_id")
            or cand.get("name")
            or "unknown"
        )

        meta = cand.get("meta") if isinstance(cand.get("meta"), dict) else {}
        source = cand.get("source") or meta.get("src") or default_source

        normalized: Dict[str, Any] = {
            "grid": normalized_grid,
            "confidence": confidence,
            "type": cand_type,
            "source": source,
            "meta": {**meta},
            "op_id": cand.get("op_id") or cand_type,
        }
        normalized["meta"].setdefault("src", source)

        param_hash = cand.get("param_hash") or cand.get("hash")
        if isinstance(param_hash, str):
            normalized["param_hash"] = param_hash

        for key in (
            "pattern_name",
            "pattern_context",
            "event_debug",
            "consensus_color_map",
        ):
            if key in cand:
                normalized[key] = cand[key]

        # Preserve any additional debugging aids without overwriting core keys.
        for key, value in cand.items():
            if key in normalized or key in {"grid", "confidence", "score", "type", "cand_type", "source", "meta", "program"}:
                continue
            normalized.setdefault(key, value)

        return normalized

    # --------------------------- Public API ----------------------------------
    def solve_arc_task(self, train_examples: List[Dict], test_input: Grid) -> List[Dict[str, Any]]:
        metrics = Metrics()
        metrics.inc("pipeline.start")
        metrics.inc(f"pipeline.policy.{self.policy_name}")
        metrics.observe("parameters.topk", float(self.topk))
        metrics.inc("runtime.numpy_enabled" if USE_NUMPY else "runtime.numpy_disabled")

        fast_mode = os.environ.get("ARC_FAST_MODE", "0") == "1"
        if fast_mode:
            os.environ.setdefault("ARC_DISABLE_AXIS_PROJECTOR", "1")
        try:
            budget_ms = max(0, int(os.environ.get("ARC_TASK_BUDGET_MS", "600")))
        except ValueError:
            budget_ms = 600
        start_ms = _now_ms()
        budget_logging = os.environ.get("RIL_GATE_LOGGING") not in (None, "", "0")

        def _log_budget(stage: str, elapsed: int) -> None:
            if budget_logging:
                print(f"[BUDGET] stop after {stage} ({elapsed}ms)", flush=False)

        def _elapsed_ms(start: float) -> float:
            return (time.perf_counter() - start) * 1000.0

        stage_start = time.perf_counter()
        output_grids: List[Dict[str, Any]] = []
        prev_allowed_palette = self._current_allowed_palette
        prev_canvas_dims = self._current_canvas_dims
        prev_test_input = self._current_test_input
        prev_train_pairs = list(self._current_train_pairs)
        try:
            self._solve_invocation += 1
            # Stage 1: Abstract Encoder
            abstract_rules = self._abstract_encoder(train_examples)
            metrics.observe("timing.abstract_encoder_ms", _elapsed_ms(stage_start))
            metrics.inc("stage.abstract_encoder_done")

            gate_ctx = _build_gate_context(
                abstract_rules,
                train_examples,
                test_input,
                provenance_mode=getattr(self, "palette_provenance_mode", "train_or_test"),
                dead_color_policy=getattr(self, "palette_dead_color_policy", "ignore"),
                attr_weight=getattr(self, "palette_attr_weight", 0.65),
                repulse_weight=getattr(self, "palette_repulse_weight", 0.35),
            )
            gate_ctx.palette_floor = self.palette_floor
            gate_ctx.palette_rescue_keep_ratio = self.palette_rescue_keep_ratio
            gate_ctx.gate_soft_topk = self.gate_soft_topk
            gate_ctx.palette_soft_floor = self.palette_soft_min
            gate_ctx.palette_soft_ratio = self.palette_soft_keep_ratio
            gate_ctx.region_gate_mode = self.region_gate_mode
            self._current_gate_context = gate_ctx
            self._input_cc_count = _count_components_grid(test_input) if isinstance(test_input, list) else 0
            output_presence, _ = _collect_example_palette(train_examples, "output")
            collected: Set[int] = set()
            for color in output_presence.keys():
                try:
                    collected.add(int(color))
                except Exception:
                    continue
            self._train_output_colors = collected
            self._train_output_multicolor = len({c for c in collected if c != 0}) > 1
            metrics.observe("ae.shape_preserving", 1.0 if gate_ctx.shape_preserving else 0.0)
            metrics.observe("ae.allowed_shapes", float(len(gate_ctx.allowed_shapes)))
            metrics.observe("ae.palette_size", float(len(gate_ctx.allowed_palette)))
            if gate_ctx.frozen_density > 0.0:
                metrics.observe("ae.frozen_density", float(gate_ctx.frozen_density))

            border_diag = self._prepare_output_diagnostics(train_examples, test_input, gate_ctx)

            ae_input_shapes = [
                f"{h}x{w}" for h, w in abstract_rules.get("input_shapes", []) if h and w
            ]
            ae_output_shapes = [
                f"{h}x{w}" for h, w in abstract_rules.get("output_shapes", []) if h and w
            ]
            trace: Dict[str, Any] = {
                "task": self.current_task_id,
                "idx": getattr(self, "current_test_index", 0),
                "ae": {
                    "input": ae_input_shapes,
                    "output": ae_output_shapes,
                    "shape_preserving": gate_ctx.shape_preserving,
                    "allowed_shapes": [f"{h}x{w}" for h, w in sorted(gate_ctx.allowed_shapes)],
                    "palette": sorted(gate_ctx.allowed_palette),
                    "frozen_density": round(gate_ctx.frozen_density, 6),
                },
                "diagnostics": {
                    "border": {
                        "consistent": bool(border_diag.get("consistent_output_border")),
                        "change_ratio": round(float(border_diag.get("border_change_ratio", 0.0)), 3),
                        "should_snap": bool(border_diag.get("should_snap")),
                    }
                },
            }
            self.last_trace = trace

            expected_shape = self._expected_output_shape(abstract_rules, gate_ctx, test_input)
            if isinstance(self.last_trace, dict):
                self.last_trace.setdefault("ae", {}).update(
                    {
                        "expected_output_shape": (
                            f"{expected_shape[0]}x{expected_shape[1]}"
                            if expected_shape
                            else None
                        )
                    }
                )

            # Stage 2: Candidate Generation
            stage_start = time.perf_counter()
            train_pairs: List[Tuple[Grid, Grid]] = []
            for example in train_examples or []:
                if not isinstance(example, dict):
                    continue
                inp = example.get("input")
                out = example.get("output")
                if isinstance(inp, list) and isinstance(out, list):
                    train_pairs.append((inp, out))

            target_h, target_w = _dims(test_input)
            allowed_palette = {int(value) for value in _allowed_palette(train_pairs, test_input)}
            self._current_canvas_dims = (target_h, target_w)
            self._current_allowed_palette = set(allowed_palette)
            self._current_test_input = test_input
            self._current_train_pairs = list(train_pairs)

            candidates = self._generate_candidates(
                abstract_rules,
                train_examples,
                test_input,
                expected_shape,
                train_pairs=train_pairs,
                allowed_palette=allowed_palette,
                target_dims=(target_h, target_w),
            )
            if fast_mode and candidates:
                try:
                    fast_cap = max(1, int(os.environ.get("ARC_FAST_MAX_CANDIDATES", "40")))
                except ValueError:
                    fast_cap = 40
                if len(candidates) > fast_cap:
                    candidates = candidates[:fast_cap]
            if budget_ms:
                elapsed_ms = _now_ms() - start_ms
                if elapsed_ms > budget_ms:
                    _log_budget("candidate gen", elapsed_ms)
                    return candidates[: max(1, self.topk)] if candidates else []
            metrics.observe("timing.candidate_gen_ms", _elapsed_ms(stage_start))
            metrics.inc("stage.candidates_done")

            cand_grids = [cand.get("grid") for cand in candidates]
            cand_scores = [_safe01(cand.get("confidence", 0.0)) for cand in candidates]

            metrics.observe("candidates.per_task", float(len(cand_grids)))
            metrics.observe("adapter.topk", float(min(self.topk, len(cand_grids))))
            best_adapter_score = max(cand_scores) if cand_scores else 0.0
            metrics.observe("adapter.best_score", float(best_adapter_score))

            def _shape_str(grid: Optional[Grid]) -> str:
                if isinstance(grid, list) and grid:
                    first = grid[0]
                    if isinstance(first, list):
                        return f"{len(grid)}x{len(first)}"
                return "0x0"

            def _precheck_payload(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
                if not isinstance(payload, dict):
                    return None
                normalized_grid = self._ensure_grid(payload.get("grid"))
                if normalized_grid is None:
                    return None
                enforced_grid = _enforce_canvas_size(
                    normalized_grid, target_h, target_w, fill=0
                )
                if not _has_only_allowed_colors(enforced_grid, allowed_palette):
                    return None
                result = dict(payload)
                meta_dict = dict(result.get("meta", {}))
                meta_dict.setdefault("prechecked", True)
                result["meta"] = meta_dict
                result["grid"] = enforced_grid
                return result

            def _wrap(tag: str, grids: Iterable[Any]) -> List[Dict[str, Any]]:
                wrapped: List[Dict[str, Any]] = []
                for idx, payload in enumerate(grids or []):
                    if isinstance(payload, dict):
                        candidate_payload = dict(payload)
                    else:
                        candidate_obj = self._make_candidate(
                            payload,
                            cand_type=tag,
                            source="external",
                            confidence=0.6,
                            meta={"src": tag},
                        )
                        candidate_payload = candidate_obj.as_payload()
                    candidate_payload.setdefault("source", "external")
                    candidate_payload.setdefault("type", tag)
                    candidate_payload.setdefault("op_id", tag)
                    meta_ref = candidate_payload.setdefault("meta", {})
                    meta_ref.setdefault("src", candidate_payload["source"])
                    meta_ref.setdefault("family", tag)
                    meta_ref.setdefault("tag", tag)
                    meta_ref.setdefault("rank", idx)
                    checked = _precheck_payload(candidate_payload)
                    if checked is not None:
                        wrapped.append(checked)
                return wrapped

            shape_preview = [_shape_str(grid) for grid in cand_grids[:5]]
            score_preview = [round(float(score), 4) for score in cand_scores[:5]]
            print(
                f"[ADAPTER] produced={len(cand_grids)} shapes={shape_preview} scores={score_preview}"
            )

            event_injected = False
            if os.getenv("ENABLE_EVENTS", "0") == "1":
                try:
                    from ril.emergence import solve_with_events

                    pred, dbg = solve_with_events(train_examples, test_input)
                    normalized_pred = self._ensure_grid(pred)
                    if normalized_pred is not None:
                        enforced_pred = _enforce_canvas_size(
                            normalized_pred, target_h, target_w, fill=0
                        )
                        if not _has_only_allowed_colors(enforced_pred, allowed_palette):
                            normalized_pred = None
                    if normalized_pred is not None:
                        event_info = {}
                        if isinstance(dbg, dict):
                            picked = dbg.get("picked_ops")
                            if picked is not None:
                                event_info["picked_ops"] = picked
                        event_candidate = self._make_candidate(
                            enforced_pred,
                            cand_type="event_candidate",
                            confidence=0.95,
                            source="external",
                            extra={"event_debug": event_info} if event_info else None,
                        ).as_payload()
                        event_candidate.setdefault("meta", {}).setdefault(
                            "prechecked", True
                        )
                        candidates.insert(0, event_candidate)
                        event_injected = True
                        if os.getenv("EVENT_VERBOSE", "0") == "1":
                            print("[EVENT] candidate injected; ops:", dbg.get("picked_ops"))
                except Exception as _e:
                    if os.getenv("EVENT_VERBOSE", "0") == "1":
                        print("[EVENT] disabled or failed softly:", type(_e).__name__)
            if event_injected:
                metrics.inc("emergence.injected")

            numpy_mode = "on" if np is not None else "off"
            print(f"[EXT-PROBE] topk={self.topk} policy={self.policy_name} numpy={numpy_mode}")

            pool_size = len(candidates)
            metrics.observe("candidates.initial_pool", float(pool_size))

            external_used = False
            ext_candidates: List[Dict] = []
            if os.getenv("ARC_USE_EXTERNAL", "0") == "1":
                try:
                    ext_candidates = self._external_candidates(
                        train_examples,
                        test_input,
                        self.topk,
                        expected_shape,
                    )
                except Exception as exc:
                    print(f"[EXT-PROBE ERROR] external solver failed: {exc}")
                    ext_candidates = []

            metrics.observe("external.proposed", float(len(ext_candidates)))

            if ext_candidates:
                sanitized_ext_candidates: List[Dict[str, Any]] = []
                for cand in ext_candidates:
                    if not isinstance(cand, dict):
                        continue
                    cand.setdefault("source", "external")
                    checked = _precheck_payload(cand)
                    if checked is not None:
                        checked.setdefault("source", "external")
                        sanitized_ext_candidates.append(checked)
                ext_candidates = sanitized_ext_candidates
                if ext_candidates:
                    scores = self._score_external_candidates(ext_candidates)
                    if not scores or len(scores) != len(ext_candidates):
                        missing = 0 if not scores else len(scores)
                        print(
                            f"[EXT-WARN] scores_missing len_cands={len(ext_candidates)} len_scores={missing}"
                        )
                        log_fallback(
                            "no_external_scores", 0.0, self.csp_evidence_min, pool_size
                        )
                    else:
                        ext_max = max(scores)
                        if ext_max > 0:
                            candidates.extend(ext_candidates)
                            external_used = True
                            metrics.observe("external.max_conf", float(ext_max))

            if not external_used:
                heuristic_cands = self._heuristic_candidates(train_examples, test_input)
                if heuristic_cands:
                    sanitized_heuristics: List[Dict[str, Any]] = []
                    for cand in heuristic_cands:
                        checked = _precheck_payload(cand)
                        if checked is not None:
                            sanitized_heuristics.append(checked)
                    if sanitized_heuristics:
                        print(f"[EXT] heuristics_k={len(sanitized_heuristics)}")
                        metrics.observe(
                            "external.heuristics", float(len(sanitized_heuristics))
                        )
                        candidates.extend(sanitized_heuristics)
                    else:
                        print("[EXT] heuristics_k=0 (filtered)")
                else:
                    if not ext_candidates:
                        print("[EXT] produced=0 candidates")
                    log_fallback("no_external_candidates", 0.0, self.csp_evidence_min, pool_size)
            else:
                metrics.inc("external.used")

            normalized_candidates: List[Dict[str, Any]] = []
            for cand in candidates:
                normalized = self._normalize_candidate(
                    cand, default_source=cand.get("source", "adapter")
                )
                if normalized is not None:
                    normalized_candidates.append(normalized)
            candidates = normalized_candidates
            metrics.observe("candidates.normalized", float(len(candidates)))

            if is_tiny_grid(test_input, threshold=3):
                tiny_payloads = solve_tiny_grid_direct(test_input, train_examples)
                tiny_wrapped = _wrap("tiny_specialist", tiny_payloads)
                if tiny_wrapped:
                    candidates.extend(tiny_wrapped)
                    metrics.observe("specialist.tiny", float(len(tiny_wrapped)))

            transform_payloads = generate_transform_candidates(
                test_input,
                train_examples,
                max_candidates=8,
            )
            transform_wrapped = _wrap("train_to_test", transform_payloads)
            if transform_wrapped:
                candidates.extend(transform_wrapped)
                metrics.observe("specialist.train_to_test", float(len(transform_wrapped)))

            (
                routed_candidates,
                rescue_candidates,
                arbitration_info,
                cfg_obj,
            ) = self._arbitrate_candidate_families(candidates, metrics=metrics)

            pre_gate_pool = list(routed_candidates)

            decision_tag = (
                arbitration_info.get("decision")
                if isinstance(arbitration_info, Mapping)
                else None
            )
            if decision_tag not in {"adapter_confident", "adapter_low_conf"}:
                max_prior = max(
                    (_candidate_prior_conf(c) for c in routed_candidates),
                    default=0.0,
                )
                if max_prior < 0.30 and rescue_candidates:
                    routed_candidates.extend(rescue_candidates)
                    if isinstance(arbitration_info, dict):
                        arbitration_info.setdefault("failover_rescue", True)

            trace["router_policy"] = arbitration_info.get("policy")
            trace["pipeline.policy"] = arbitration_info.get("policy")
            trace["gate.arbitration"] = arbitration_info.get("decision")

            pre_gate_source_counts: Counter[str] = Counter()
            for cand in routed_candidates:
                src = cand.get("source", "adapter")
                if src not in ("adapter", "external", "echo"):
                    src = "adapter"
                    cand["source"] = src
                pre_gate_source_counts[src] += 1

            arbitration_info["routed_counts_initial"] = dict(pre_gate_source_counts)

            rescue_enabled = bool(cfg_obj.get("rescue_enabled", True)) if isinstance(cfg_obj, Mapping) else True
            gated_candidates, gate_stats = _apply_candidate_gates(
                routed_candidates,
                gate_ctx,
                metrics=metrics,
                palette_support_min=self.palette_support_min,
            )
            self._last_gate_stats = gate_stats
            for name, counts in gate_stats.get("gate_counts", {}).items():
                self._gate_totals[f"{name}.accepted"] += counts.get("accepted", 0)
                self._gate_totals[f"{name}.rejected"] += counts.get("rejected", 0)
            for cand_type, stats in gate_stats.get("by_type", {}).items():
                self._candidate_type_totals[f"{cand_type}.total"] += stats.get("total", 0)
                self._candidate_type_totals[f"{cand_type}.accepted"] += stats.get("accepted", 0)

            arbitration_info.setdefault("routed_counts_final", dict(pre_gate_source_counts))
            arbitration_info["rescue_triggered"] = False
            if not gated_candidates and rescue_enabled:
                metrics.inc("gate.arbitration_rescue")
                rescue_filtered, rescue_stats = self._gate_candidates_with_rescue(
                    routed_candidates,
                    gate_ctx,
                    metrics,
                    rescue_seed=rescue_candidates,
                    train_examples=train_examples,
                )
                if rescue_filtered:
                    gated_candidates = rescue_filtered
                    gate_stats = rescue_stats
                    self._last_gate_stats = gate_stats
                    arbitration_info["rescue_triggered"] = True
                    trace["gate.arbitration"] = f"{arbitration_info.get('decision')}+rescue"

            trace["gen"] = {
                "total": gate_stats.get("total", 0),
                "accepted": gate_stats.get("accepted", 0),
                "rejected": gate_stats.get("rejected", 0),
                "by_type": gate_stats.get("by_type", {}),
            }
            trace["gates"] = gate_stats.get("gate_counts", {})
            trace["gate_in"] = {
                "external": int(pre_gate_source_counts.get("external", 0)),
                "adapter": int(pre_gate_source_counts.get("adapter", 0)),
                "echo": int(pre_gate_source_counts.get("echo", 0)),
            }
            trace["arbitration"] = arbitration_info

            candidates = gated_candidates

            if budget_ms:
                elapsed_ms = _now_ms() - start_ms
                if elapsed_ms > budget_ms:
                    _log_budget("gating", elapsed_ms)
                    fallback_pool = candidates or pre_gate_pool or routed_candidates or []
                    return fallback_pool[: max(1, self.topk)] if fallback_pool else []

            if fast_mode:
                micro_stats = {"micro_tried": 0, "micro_applied": 0}
            else:
                candidates, micro_stats = self._micro_finishers(candidates, gate_ctx)
            self._last_finishers_stats = micro_stats
            if metrics is not None:
                metrics.observe("finishers.micro_tried", float(micro_stats.get("micro_tried", 0)))
                metrics.observe("finishers.micro_applied", float(micro_stats.get("micro_applied", 0)))

            source_conf: Dict[str, List[float]] = {"adapter": [], "external": [], "echo": []}
            source_counts: Counter = Counter()
            shape_counter: Counter = Counter()
            candidate_hashes: List[str] = []

            for cand in candidates:
                src = cand.get("source", "adapter")
                if src not in source_conf:
                    src = "adapter"
                    cand["source"] = src
                    cand.setdefault("meta", {}).setdefault("src", src)
                conf = _safe01(cand.get("confidence", 0.0))
                source_conf.setdefault(src, []).append(conf)
                source_counts[src] += 1
                candidate_hashes.append(self._hash_grid(cand.get("grid")))
                h, w = grid_shape(cand.get("grid"))
                shape_counter[f"{h}x{w}"] += 1

            unique_candidates = len(set(candidate_hashes))
            total_candidates = len(candidate_hashes)
            dup_ratio = (unique_candidates / total_candidates) if total_candidates else 0.0

            shape_entropy = 0.0
            if shape_counter:
                total_shapes = sum(shape_counter.values())
                if total_shapes:
                    for count in shape_counter.values():
                        p = count / total_shapes
                        if p > 0:
                            shape_entropy -= p * math.log(p, 2)

            duplicate_share = 1.0 - dup_ratio if total_candidates else 0.0
            if duplicate_share > 0.4 and gate_ctx.reference_input:
                base_palette = {
                    int(cell)
                    for row in gate_ctx.reference_input
                    for cell in row
                }
                for cand in candidates:
                    grid = cand.get("grid") or []
                    try:
                        cand_colors = {int(cell) for row in grid for cell in row}
                    except Exception:
                        cand_colors = set()
                    new_colors = cand_colors - base_palette
                    if not new_colors:
                        continue
                    boost = 0.04 + 0.01 * min(len(new_colors), 3)
                    boosted_conf = min(1.0, _safe01(cand.get("confidence", 0.0)) + boost)
                    cand["confidence"] = boosted_conf
                    meta_ref = cand.setdefault("meta", {})
                    meta_ref.setdefault("diversity_bonus", round(boost, 4))
                    if new_colors:
                        meta_ref.setdefault("new_colors", sorted(int(v) for v in list(new_colors))[:4])

            ext_max_conf = max(source_conf.get("external") or [0.0])
            adapt_max_conf = max(source_conf.get("adapter") or [0.0])
            echo_max_conf = max(source_conf.get("echo") or [0.0])

            metrics.observe("candidates.total", float(total_candidates))
            metrics.observe("candidates.unique", float(unique_candidates))
            metrics.observe("gate.dup_ratio", float(dup_ratio))
            metrics.observe("gate.shape_entropy", float(shape_entropy))
            metrics.observe("gate.external_max_conf", float(ext_max_conf))
            metrics.observe("gate.adapter_max_conf", float(adapt_max_conf))
            metrics.observe("gate.echo_max_conf", float(echo_max_conf))
            for src, count in source_counts.items():
                metrics.observe(f"gate.source_count.{src}", float(count))

            trace.update(
                {
                    "candidate_total": total_candidates,
                    "candidate_unique": unique_candidates,
                    "source_counts": dict(source_counts),
                    "ext_max_conf": ext_max_conf,
                    "adapt_max_conf": adapt_max_conf,
                    "echo_max_conf": echo_max_conf,
                    "dup_ratio": dup_ratio,
                    "shape_entropy": shape_entropy,
                    "source_conf": {
                        src: [round(val, 4) for val in sorted(vals, reverse=True)[:5]]
                        for src, vals in source_conf.items()
                        if vals
                    },
                    "shape_counts": dict(shape_counter),
                    "exact_match": None,
                    "em_at2": None,
                    "min_hamming": None,
                    "hamming_at1": None,
                    "near_miss": None,
                }
            )

            self.last_trace = trace

            # Stage 3: Beam Search
            stage_start = time.perf_counter()
            beam_results = self._beam_search(candidates, train_examples)
            if budget_ms:
                elapsed_ms = _now_ms() - start_ms
                if elapsed_ms > budget_ms:
                    _log_budget("beam", elapsed_ms)
                    fallback_pool = beam_results or candidates or pre_gate_pool or []
                    return fallback_pool[: max(1, self.topk)] if fallback_pool else []
            metrics.observe("timing.beam_ms", _elapsed_ms(stage_start))
            metrics.inc("stage.beam_done")
            metrics.observe("search.beam_total", float(len(beam_results)))
            metrics.observe(
                "search.best_score",
                float(beam_results[0].get("beam_score", 0.0)) if beam_results else 0.0,
            )

            beam_top: List[Dict[str, Any]] = []
            for rank, cand in enumerate(beam_results[:5]):
                beam_top.append(
                    {
                        "rank": rank + 1,
                        "source": cand.get("source") or cand.get("meta", {}).get("src"),
                        "op_id": cand.get("op_id") or cand.get("type"),
                        "param_hash": cand.get("param_hash"),
                        "beam_score": float(cand.get("beam_score", 0.0)),
                        "confidence": _safe01(cand.get("confidence", 0.0)),
                        "grid_hash": self._hash_grid(cand.get("grid")),
                        "shape": grid_shape(cand.get("grid")),
                    }
                )

            chosen_src = beam_top[0].get("source") if beam_top else None
            if chosen_src:
                metrics.inc(f"gate.choice.{chosen_src}")
            else:
                metrics.inc("gate.choice.none")
            trace["beam"] = {
                "k_in": gate_stats.get("accepted", len(candidates)),
                "k_out": len(beam_results),
                "top": beam_top,
            }
            trace["chosen_src"] = chosen_src
            trace["gate_out"] = chosen_src
            trace["top_op"] = beam_top[0].get("op_id") if beam_top else None
            trace["beam_total"] = len(beam_results)

            train_check = dict(self._last_train_check or {})
            hammings = train_check.get("hamming")
            if isinstance(hammings, list):
                try:
                    train_check["hamming"] = [round(float(val), 6) for val in hammings[:5]]
                except Exception:
                    train_check["hamming"] = hammings[:5]
            trace["train_check"] = train_check

            # Stage 4: CSP Solver
            stage_start = time.perf_counter()
            csp_solutions = self._csp_solver(beam_results)
            metrics.observe("timing.csp_ms", _elapsed_ms(stage_start))
            metrics.inc("stage.csp_done")
            metrics.observe("csp.pool", float(len(csp_solutions)))
            metrics.observe(
                "csp.approved",
                float(sum(1 for cand in csp_solutions if cand.get("csp_approved", False))),
            )
            self._boost_csp_scores(csp_solutions)

            # Stage 5: Output Formatting
            stage_start = time.perf_counter()
            output_grids = self._format_outputs(
                csp_solutions,
                test_input,
                train_examples,
                expected_shape,
            )
            metrics.observe("timing.output_ms", _elapsed_ms(stage_start))
            metrics.inc("stage.output_done")

            last_metrics = self._last_output_metrics or {}
            produced = float(last_metrics.get("produced", len(output_grids)))
            metrics.observe("outputs.count", produced)
            metrics.observe("outputs.requested", float(self.topk))
            best_score = float(last_metrics.get("best_score", 0.0))
            metrics.observe("outputs.best_score", best_score)

            best_loss = last_metrics.get("best_loss")
            metrics.inc("sniper.attempts")
            if isinstance(best_loss, (int, float)):
                metrics.observe("em.hamming", float(best_loss))
                if best_loss <= 0.0:
                    metrics.inc("em.hit")
                    metrics.inc("sniper.hits")
                else:
                    metrics.inc("em.miss")
            else:
                metrics.inc("em.miss")

            hit_flag = isinstance(best_loss, (int, float)) and best_loss <= 0.0
            hamming_value = float(best_loss) if isinstance(best_loss, (int, float)) else float("nan")
            self.note_task_result(hit=hit_flag, hamming=hamming_value)

            final_hashes = [self._hash_grid(grid) for grid in output_grids]
            pred_shape: Optional[Tuple[int, int]] = None
            if output_grids and isinstance(output_grids[0].get("grid"), list):
                pred_grid = output_grids[0]["grid"]
                pred_shape = grid_shape(pred_grid)
            trace.update({
                "final_hashes": final_hashes,
                "pred": {
                    "shape": pred_shape,
                    "hamming": float(best_loss) if isinstance(best_loss, (int, float)) else None,
                    "produced": len(output_grids),
                },
            })
            self._trace_history.append(trace)
            self.last_trace = trace
            self._write_trace_line(trace)
            self._dump_structured_trace(trace)

            ext_preview = trace.get("source_conf", {}).get("external", [])
            adapt_preview = trace.get("source_conf", {}).get("adapter", [])
            echo_preview = trace.get("source_conf", {}).get("echo", [])
            gate_counts = gate_stats.get("gate_counts", {}) if isinstance(gate_stats, dict) else {}
            shape_rej = gate_counts.get("shape", {}).get("rejected", 0)
            color_rej = gate_counts.get("color", {}).get("rejected", 0)
            region_rej = gate_counts.get("region", {}).get("rejected", 0)
            top_preview = [
                f"{entry.get('op_id','?')}/{entry.get('beam_score',0.0):.2f}"
                for entry in beam_top[:2]
            ]
            beam_in = gate_stats.get("accepted", len(candidates)) if isinstance(gate_stats, dict) else len(candidates)
            print(
                f"[GATE] task={self.current_task_id} gen={gate_stats.get('total', len(candidates)) if isinstance(gate_stats, dict) else len(candidates)} "
                f"pruned={gate_stats.get('rejected', 0) if isinstance(gate_stats, dict) else 0} "
                f"beam={beam_in}→{len(beam_results)} gates(shape={shape_rej},color={color_rej},region={region_rej}) "
                f"top={top_preview}"
            )
            if ext_preview:
                print(f"[GATE] ext_conf_top={ext_preview}")
            if adapt_preview:
                print(f"[GATE] adapt_conf_top={adapt_preview}")
            if echo_preview:
                print(f"[GATE] echo_conf_top={echo_preview}")

            if self._gate_profile_enabled:
                profile_payload = {
                    "gate": {
                        name: {
                            "accepted": gate_counts.get(name, {}).get("accepted", 0),
                            "rejected": gate_counts.get(name, {}).get("rejected", 0),
                        }
                        for name in sorted(gate_counts.keys())
                    },
                    "types": gate_stats.get("by_type", {}) if isinstance(gate_stats, dict) else {},
                }
                print(f"[GATE-PROFILE] {json.dumps(profile_payload, sort_keys=True)}")

            metrics.inc("pipeline.complete")
            return output_grids
        except Exception as exc:
            log_pipeline_error(metrics, exc)
            raise
        finally:
            self._current_allowed_palette = prev_allowed_palette
            self._current_canvas_dims = prev_canvas_dims
            self._current_test_input = prev_test_input
            self._current_train_pairs = prev_train_pairs
            metrics.dump()

    # -------------------------- Trace helpers --------------------------------
    def _write_trace_line(self, trace: Dict[str, Any]) -> None:
        """Persist ``trace`` to the configured log file, if any."""

        if not self._trace_log_path or self._trace_log_error:
            return

        try:
            path = Path(self._trace_log_path)
            if not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(trace, sort_keys=True, default=repr))
                handle.write("\n")
        except Exception as exc:  # pragma: no cover - defensive logging guard
            if not self._trace_log_error:
                print(f"[TRACE-WARN] failed to write trace log: {exc}")
                self._trace_log_error = True

    def _dump_structured_trace(self, trace: Dict[str, Any]) -> None:
        """Emit per-task traces to RIL_TRACE directory when enabled."""

        if os.getenv("RIL_TRACE", "0") != "1":
            return

        try:
            out_dir = Path(os.getenv("RIL_TRACE_DIR", "/tmp/ril_traces"))
            out_dir.mkdir(parents=True, exist_ok=True)
            task_id = str(trace.get("task") or self.current_task_id or "unknown")
            idx = trace.get("idx", getattr(self, "current_test_index", 0))
            try:
                idx_int = int(idx)
            except Exception:
                idx_int = 0
            filename = f"{task_id}_{idx_int}.json"
            path = out_dir / filename
            with path.open("w", encoding="utf-8") as handle:
                json.dump(trace, handle, sort_keys=True, indent=2, default=repr)
        except Exception as exc:
            if os.getenv("RIL_TRACE_DEBUG", "0") == "1":
                print(f"[TRACE-WARN] failed to write structured trace: {exc}")

    def annotate_last_trace(self, **metrics: Any) -> None:
        """Update the most recent trace entry with evaluation metrics."""

        if not isinstance(self.last_trace, dict):
            self.last_trace = {}

        updates: Dict[str, Any] = {}
        for key, value in metrics.items():
            if key in {"exact_match", "em_at2", "near_miss"}:
                updates[key] = bool(value)
            elif key in {"min_hamming", "hamming_at1"}:
                try:
                    updates[key] = float(value)
                except (TypeError, ValueError):
                    continue
            elif value is not None:
                updates[key] = value

        if not updates:
            return

        self.last_trace.update(updates)
        if self._trace_history:
            self._trace_history[-1].update(updates)

        # Re-emit the updated trace for external log consumers.
        self._write_trace_line(self.last_trace)
        self._dump_structured_trace(self.last_trace)

    def note_task_result(self, *, hit: bool, hamming: float) -> None:
        """Record the outcome of the most recent task."""

        if not isinstance(self._recent_results, list):
            self._recent_results = []
        try:
            value = float(hamming)
        except Exception:
            value = float("nan")
        self._recent_results.append({"hit": bool(hit), "hamming": value})
        if len(self._recent_results) > max(self._scorecard_every, 50):
            self._recent_results.pop(0)

    def maybe_emit_scorecard(self, final: bool = False) -> None:
        """Emit a lightweight scorecard summary if available."""

        if not isinstance(self._recent_results, list) or not self._recent_results:
            return
        if not final and len(self._recent_results) < self._scorecard_every:
            return

        total = len(self._recent_results)
        hits = sum(1 for entry in self._recent_results if entry.get("hit"))
        finite_vals = []
        for entry in self._recent_results:
            value = entry.get("hamming")
            if isinstance(value, (int, float)) and math.isfinite(value):
                finite_vals.append(float(value))
        mean_hamming = (sum(finite_vals) / len(finite_vals)) if finite_vals else None
        payload = {
            "final": bool(final),
            "window": total,
            "hits": hits,
            "hit_rate": round(hits / total, 6) if total else 0.0,
            "mean_hamming": (round(mean_hamming, 6) if mean_hamming is not None else None),
        }
        gate_payload: Dict[str, Dict[str, int]] = {}
        for key, value in self._gate_totals.items():
            if "." not in key:
                continue
            gate_name, bucket = key.split(".", 1)
            gate_payload.setdefault(gate_name, {})[bucket] = int(value)
        if gate_payload:
            payload["gates"] = gate_payload
        print(f"[SCORECARD] {json.dumps(payload, sort_keys=True)}")
        self._recent_results.clear()

    def _expected_output_shape(
        self,
        rules: Dict[str, Any],
        gate_ctx: Optional[GateContext],
        test_input: Grid,
    ) -> Optional[Tuple[int, int]]:
        shapes: List[Tuple[int, int]] = []
        for shape in rules.get("output_shapes", []) or []:
            if isinstance(shape, (list, tuple)) and len(shape) == 2:
                try:
                    h, w = int(shape[0]), int(shape[1])
                except Exception:
                    continue
                if h > 0 and w > 0:
                    shapes.append((h, w))

        counts: Counter[Tuple[int, int]] = Counter(shapes)
        candidate_shapes: set[Tuple[int, int]] = set(shapes)

        if gate_ctx and gate_ctx.allowed_shapes:
            for shape in gate_ctx.allowed_shapes:
                if len(shape) != 2:
                    continue
                h, w = int(shape[0]), int(shape[1])
                if h > 0 and w > 0:
                    candidate_shapes.add((h, w))

        test_shape = grid_shape(test_input) if test_input and test_input[0] else (0, 0)

        if (
            gate_ctx
            and gate_ctx.shape_preserving
            and test_shape[0] > 0
            and test_shape[1] > 0
        ):
            return test_shape

        if not candidate_shapes:
            return test_shape if test_shape[0] > 0 and test_shape[1] > 0 else None

        best_shape: Optional[Tuple[int, int]] = None
        best_key: Optional[Tuple[int, ...]] = None
        test_area = test_shape[0] * test_shape[1]

        for shape in sorted(candidate_shapes):
            h, w = shape
            if h <= 0 or w <= 0:
                continue
            count = counts.get(shape, 0)
            area = h * w
            area_delta = abs(area - test_area) if test_area else area
            dims_delta = abs(h - test_shape[0]) + abs(w - test_shape[1])
            key = (-count, area_delta, dims_delta, abs(h - w), h + w, h, w)
            if best_key is None or key < best_key:
                best_shape = shape
                best_key = key

        return best_shape

    def _external_candidates(
        self,
        train_examples: List[Dict],
        test_input: Grid,
        topk: int,
        expected_shape: Optional[Tuple[int, int]] = None,
    ) -> List[Dict]:
        results: List[Candidate] = []
        target_shape = expected_shape if expected_shape else grid_shape(test_input)
        palette = palette_frequencies(train_examples)
        base_conf = 0.75
        decay = 0.08
        for idx, example in enumerate(train_examples or []):
            if not isinstance(example, dict):
                continue
            out = example.get("output")
            if not isinstance(out, list) or not out:
                continue
            candidate_grid = fit_to_shape(copy_grid(out), target_shape)
            conf = max(0.3, base_conf - decay * idx)
            results.append({
                "type": f"train_output_{idx}",
                "grid": clamp_to_palette(candidate_grid, palette) if palette else candidate_grid,
                "confidence": conf,
            })
        if not results and palette:
            clamped = clamp_to_palette(test_input, palette)
            results.append({
                "type": "palette_clamped_input",
                "grid": clamped,
                "confidence": 0.55,
            })
        return results[: max(topk, 4)]

    def _score_external_candidates(self, candidates: List[Dict]) -> List[float]:
        return [_safe01(c.get("confidence", 0.0)) for c in candidates]

    def _heuristic_candidates(self, train_examples: List[Dict], test_input: Grid) -> List[Dict]:
        try:
            from ril.heuristics import heuristic_candidates as build_heuristics
        except Exception as exc:
            print(f"[EXT-WARN] heuristics unavailable: {exc}")
            return []

        try:
            grids = build_heuristics(train_examples, test_input)
        except Exception as exc:
            print(f"[EXT-WARN] heuristics failed: {exc}")
            return []

        results: List[Candidate] = []
        base_conf = 0.22
        decay = 0.04
        for idx, grid in enumerate(grids):
            if not isinstance(grid, list):
                continue
            conf = max(0.1, base_conf - decay * idx)
            results.append(
                self._make_candidate(
                    grid,
                    cand_type=f"heuristic_{idx}",
                    confidence=conf,
                    source="external",
                )
            )
        return [cand.as_payload() for cand in results]

    # ---------------------- Color-map helpers ----------------------
    def _infer_color_map_from_pair(self, inp: Grid, out: Grid) -> Dict[int, int]:
        """Infer a per-example color map by aligning cells 1:1 where shapes match."""
        H, W = min(len(inp), len(out)), min(len(inp[0]), len(out[0]))
        votes: Dict[int, Counter] = {}
        for r in range(H):
            for c in range(W):
                a, b = inp[r][c], out[r][c]
                votes.setdefault(a, Counter()).update([b])
        m: Dict[int, int] = {}
        for a, cnt in votes.items():
            b, n = cnt.most_common(1)[0]
            # require some dominance
            if n >= 2 or len(cnt) == 1:
                m[a] = b
        return m

    def _merge_color_maps(self, maps: List[Dict[int,int]]) -> Dict[int,int]:
        """Majority merge of several per-example maps."""
        tally: Dict[int, Counter] = {}
        for m in maps:
            for a, b in m.items():
                tally.setdefault(a, Counter()).update([b])
        merged: Dict[int,int] = {}
        for a, cnt in tally.items():
            merged[a] = cnt.most_common(1)[0][0]
        return merged

    # ----------------------- Stage 1: Abstract Encoder -----------------------
    def _abstract_encoder(self, train_examples: List[Dict]) -> Dict[str, Any]:
        rules = {
            "input_shapes": [],
            "output_shapes": [],
            "color_mappings": [],   # sets per pair
            "pair_mappings": [],    # positional input->output pairs
            "tiling_patterns": [],
            "size_ratios": []
        }

        for ex in train_examples:
            inp, out = ex["input"], ex["output"]
            in_h, in_w = grid_shape(inp)
            out_h, out_w = grid_shape(out)
            rules["input_shapes"].append((in_h, in_w))
            rules["output_shapes"].append((out_h, out_w))

            in_colors = set(cell for row in inp for cell in row)
            out_colors = set(cell for row in out for cell in row)
            rules["color_mappings"].append({
                "input_colors": in_colors,
                "output_colors": out_colors
            })

            # crude position pair sample (downsample if big)
            pairs = []
            H, W = min(in_h, out_h), min(in_w, out_w)
            step_r = max(1, H // 10)
            step_c = max(1, W // 10)
            for r in range(0, H, step_r):
                for c in range(0, W, step_c):
                    pairs.append((inp[r][c], out[r][c]))
            rules["pair_mappings"].append(pairs)

            if in_h and in_w:
                h_ratio = out_h / in_h if in_h else 1
                w_ratio = out_w / in_w if in_w else 1
                rules["size_ratios"].append((h_ratio, w_ratio))
                if out_h % in_h == 0 and out_w % in_w == 0 and in_h and in_w:
                    ht, wt = out_h // in_h, out_w // in_w
                    if (ht > 1) or (wt > 1):
                        if self._is_tiled(inp, out, ht, wt):
                            rules["tiling_patterns"].append((ht, wt))
        return rules

    def _prepare_output_diagnostics(
        self,
        train_examples: List[Dict],
        test_input: Grid,
        gate_ctx: Optional[GateContext] = None,
    ) -> Dict[str, Any]:
        """Summarise palette/border evidence for downstream heuristics."""

        train_count = len([ex for ex in train_examples or [] if isinstance(ex, dict)])

        output_signatures: List[Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]] = []
        input_signatures: List[Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]] = []
        border_changes = 0
        border_palette = Counter()

        for example in train_examples or []:
            if not isinstance(example, dict):
                continue
            inp = example.get("input")
            out = example.get("output")
            in_sig = border_signature(inp)
            out_sig = border_signature(out)
            if in_sig:
                input_signatures.append(in_sig)
            if out_sig:
                output_signatures.append(out_sig)
                border_palette.update(border_palette_counter(out))
            if in_sig and out_sig and in_sig != out_sig:
                border_changes += 1

        sig_set = {sig for sig in output_signatures}
        consistent_output_border = bool(sig_set) and len(sig_set) == 1
        canonical_border = output_signatures[0] if consistent_output_border else None
        border_change_ratio = (border_changes / train_count) if train_count else 0.0

        train_palette_counter = palette_frequencies(train_examples)
        diag: Dict[str, Any] = {
            "train_example_count": train_count,
            "train_border_signatures": output_signatures,
            "train_input_border_signatures": input_signatures,
            "train_border_change_count": border_changes,
            "border_change_ratio": border_change_ratio,
            "consistent_output_border": consistent_output_border,
            "canonical_output_border": canonical_border,
            "train_border_palette": border_palette,
            "train_palette_counter": train_palette_counter,
            "test_input_border": border_signature(test_input),
            "snap_log": [],
        }

        diag["test_border_matches_canonical"] = (
            canonical_border is not None and canonical_border == diag["test_input_border"]
        )

        if gate_ctx is not None:
            diag["allowed_palette"] = set(gate_ctx.allowed_palette)
            diag["palette_order"] = tuple(gate_ctx.palette_order)
            diag["shape_preserving"] = bool(gate_ctx.shape_preserving)
        else:
            prev = self._border_diagnostics if isinstance(self._border_diagnostics, dict) else {}
            if isinstance(prev, dict):
                for key in ("allowed_palette", "palette_order", "shape_preserving"):
                    if key in prev and key not in diag:
                        diag[key] = prev[key]

        # Prefer snapping only when there is strong evidence that outputs
        # deliberately adjust the border across the training set.
        diag["should_snap"] = bool(consistent_output_border and border_changes and border_change_ratio >= 0.5)

        self._border_diagnostics = diag
        return diag

    def _is_tiled(self, inp: Grid, out: Grid, h_tiles: int, w_tiles: int) -> bool:
        in_h, in_w = grid_shape(inp)
        for ti in range(h_tiles):
            for tj in range(w_tiles):
                for i in range(in_h):
                    for j in range(in_w):
                        oi = ti*in_h + i
                        oj = tj*in_w + j
                        if oi >= len(out) or oj >= len(out[0]): return False
                        if out[oi][oj] != inp[i][j]: return False
        return True

    # -------------------- Stage 2: Candidate Generation ----------------------
    def _generate_candidates(
        self,
        rules: Dict,
        train_examples: List[Dict],
        test_input: Grid,
        expected_shape: Optional[Tuple[int, int]] = None,
        *,
        train_pairs: Optional[List[Tuple[Grid, Grid]]] = None,
        allowed_palette: Optional[Set[int]] = None,
        target_dims: Optional[Tuple[int, int]] = None,
    ) -> List[Dict]:
        candidates: List[Candidate] = []

        canvas_h: int
        canvas_w: int
        if target_dims and len(target_dims) == 2:
            canvas_h, canvas_w = int(target_dims[0]), int(target_dims[1])
        else:
            canvas_h, canvas_w = _dims(test_input)

        if (
            expected_shape is None
            or expected_shape[0] <= 0
            or expected_shape[1] <= 0
        ):
            expected_shape = (canvas_h, canvas_w)

        src_h, src_w = grid_shape(test_input)

        if train_pairs is None:
            train_pairs = []
            for example in train_examples:
                if not isinstance(example, dict):
                    continue
                inp = example.get("input")
                out = example.get("output")
                if isinstance(inp, list) and isinstance(out, list):
                    train_pairs.append((inp, out))

        palette_set: Set[int]
        if allowed_palette is not None:
            palette_set = set(int(v) for v in allowed_palette)
        else:
            palette_set = _allowed_palette(train_pairs, test_input)

        def prepare_grid(grid_like: Any) -> Optional[Grid]:
            normalized = self._ensure_grid(grid_like)
            if normalized is None:
                return None
            enforced = _enforce_canvas_size(
                normalized, canvas_h, canvas_w, fill=0
            )
            if not _has_only_allowed_colors(enforced, palette_set):
                return None
            return enforced

        motif_pairs: List[Tuple[Grid, Grid]] = list(train_pairs)

        motif_seed = _motif_crop_candidate(motif_pairs, test_input)
        if motif_seed is not None:
            prepared_seed = prepare_grid(motif_seed)
            if prepared_seed is not None:
                cand = self._make_candidate(
                    prepared_seed,
                    cand_type="motif_crop",
                    confidence=0.54,
                    source="motif_crop_seed",
                    meta={"strategy": "deterministic_bbox"},
                )
                cand.meta.setdefault("prechecked", True)
                candidates.append(cand)

        def add(
            grid: Any,
            t: str,
            conf: float,
            *,
            source: str = "adapter",
            param_hash: Optional[str] = None,
            extra: Optional[Dict[str, Any]] = None,
            meta: Optional[Dict[str, Any]] = None,
        ) -> Optional[Candidate]:
            prepared = prepare_grid(grid)
            if prepared is None:
                return None
            cand = self._make_candidate(
                prepared,
                cand_type=t,
                confidence=conf,
                source=source,
                param_hash=param_hash,
                meta=meta,
                extra=extra,
            )
            cand.meta.setdefault("prechecked", True)
            candidates.append(cand)
            return cand

        # Dimensional hypothesis bank (prepend to pool)
        dimensional_candidates: List[Candidate] = []
        try:
            dim_grids = dimensional_hypotheses(test_input)
        except Exception:
            dim_grids = []
        for rank, arr in enumerate(dim_grids):
            if hasattr(arr, "tolist"):
                downsampled = arr.tolist()
                shape = getattr(arr, "shape", ())
                cand_h = int(shape[0]) if len(shape) >= 1 else len(downsampled)
                cand_w = int(shape[1]) if len(shape) >= 2 else (len(downsampled[0]) if downsampled else 0)
            else:
                downsampled = arr  # type: ignore[assignment]
                cand_h = len(downsampled)
                cand_w = len(downsampled[0]) if downsampled else 0
            if cand_h <= 0 or cand_w <= 0:
                continue
            downsample_factor = [
                int(max(1, src_h) // max(1, cand_h)),
                int(max(1, src_w) // max(1, cand_w)),
            ]
            confidence = max(0.50, 0.72 - 0.04 * rank)
            prepared_downsampled = prepare_grid(downsampled)
            if prepared_downsampled is None:
                continue
            cand = self._make_candidate(
                prepared_downsampled,
                cand_type=f"dimensional_bank_{cand_h}x{cand_w}",
                confidence=confidence,
                source="dimensional_bank",
                meta={
                    "target_shape": [cand_h, cand_w],
                    "downsample_factor": downsample_factor,
                },
                extra={
                    "structure_score": 0.86,
                    "structure": 0.86,
                },
            )
            cand.meta.setdefault("prechecked", True)
            dimensional_candidates.append(cand)
        if dimensional_candidates:
            candidates[:0] = dimensional_candidates

        # Identity
        add(test_input, "identity", 0.30)

        for crop in generate_motif_crop(motif_pairs, test_input):
            add(crop, "motif_crop", 0.52)

        # Simple color ops
        add([[1 - x if x in (0,1) else x for x in row] for row in test_input], "color_flip_0_1", 0.50)
        add([[(x + 1) % 10 for x in row] for row in test_input], "color_plus_1", 0.35)
        add([[(x - 1) % 10 for x in row] for row in test_input], "color_minus_1", 0.35)

        # Mirrors / rotations / transpose
        add([row[::-1] for row in test_input], "mirror_h", 0.40)
        add(test_input[::-1], "mirror_v", 0.40)
        if test_input and test_input[0]:
            h, w = grid_shape(test_input)
            add([[test_input[h-1-j][i] for j in range(h)] for i in range(w)], "rot_90_cw", 0.40)
            add([[test_input[j][w-1-i] for j in range(h)] for i in range(w)], "rot_90_ccw", 0.40)
            add([[test_input[i][j] for i in range(h)] for j in range(w)], "transpose", 0.35)
            add([row[::-1] for row in test_input[::-1]], "rot_180", 0.45)

        # Learned color map (set-wise 1-1)
        cm_set = self._detect_color_mapping_setwise(rules.get("color_mappings", []))
        if cm_set:
            add([[cm_set.get(x, x) for x in row] for row in test_input], "learned_color_map_set", 0.60)

        # Stronger color map (pair frequency from sampled IO pairs)
        cm_pairs = self._detect_color_mapping_pairs(rules.get("pair_mappings", []))
        if cm_pairs:
            add([[cm_pairs.get(x, x) for x in row] for row in test_input], "learned_color_map_pairs", 0.70)

        # Background normalize (paint bg = mode color)
        bgc = mode_color(test_input)
        add([[bgc if x == 0 else x for x in row] for row in test_input], "bg_to_mode_if_zero", 0.35)

        # Component relocation
        add(translate_object_to(test_input, "topleft", bg=bgc), "largest_obj_to_topleft", 0.55)
        add(translate_object_to(test_input, "center", bg=bgc), "largest_obj_to_center", 0.50)

        # Axis translations (heuristic: shift contiguous mass up/left)
        add(self._shift_grid(test_input, dx=0, dy=-self._first_nonempty_row(test_input)), "shift_to_top", 0.45)
        add(self._shift_grid(test_input, dx=-self._first_nonempty_col(test_input), dy=0), "shift_to_left", 0.45)

        # Axis projector – learn principal translation axis from training pairs
        projector_result = None
        try:
            projector = AxisProjector()
            projector_result = projector.generate(train_examples, test_input)
        except Exception as exc:  # pragma: no cover - diagnostic logging only
            print(f"[ADAPTER-WARN] axis_projector failed: {type(exc).__name__}: {exc}")

        if projector_result and projector_result.candidates:
            metrics_payload = asdict(projector_result.metrics)
            metrics_payload.update(
                {
                    "axis": [round(projector_result.axis[0], 6), round(projector_result.axis[1], 6)],
                    "step": round(projector_result.step, 4),
                }
            )
            nonzero_total = sum(1 for row in test_input for cell in row if cell != 0) or 1
            for rank, candidate_info in enumerate(projector_result.candidates):
                clipped_ratio = min(1.0, candidate_info.clipped / float(nonzero_total))
                confidence = 0.58 - 0.04 * rank - 0.08 * clipped_ratio
                if candidate_info.mean_ang_error > 0.35:
                    confidence -= 0.05
                confidence = max(0.46, confidence)
                axis_meta = {
                    "axis": list(metrics_payload["axis"]),
                    "step": metrics_payload["step"],
                    "translation": [int(candidate_info.translation[0]), int(candidate_info.translation[1])],
                    "clipped": int(candidate_info.clipped),
                    "rank": int(rank),
                    "metrics": dict(metrics_payload),
                    "candidate_metrics": {
                        "min_replay_iou": round(candidate_info.min_replay_iou, 4),
                        "mean_replay_iou": round(candidate_info.mean_replay_iou, 4),
                        "mean_ang_error": round(candidate_info.mean_ang_error, 4),
                        "mean_length_error": round(candidate_info.mean_length_error, 4),
                    },
                }
                cand = add(
                    candidate_info.grid,
                    "axis_projection",
                    confidence,
                    source="axis_projector",
                    extra={"axis_projector": axis_meta},
                )
                if cand is not None:
                    cand.extra.setdefault("axis_projector", axis_meta)

        # Centroid snap to training output centroids
        def _centroid_ratio(grid: Grid) -> Optional[Tuple[float, float]]:
            if not grid or not grid[0]:
                return None
            pts: List[Tuple[int, int]] = [
                (r, c)
                for r, row in enumerate(grid)
                for c, val in enumerate(row)
                if val != 0
            ]
            if not pts:
                return None
            total = len(pts)
            mean_r = sum(r for r, _ in pts) / total
            mean_c = sum(c for _, c in pts) / total
            h, w = grid_shape(grid)
            if h <= 0 or w <= 0:
                return None
            return (mean_r / float(h), mean_c / float(w))

        def _centroid_absolute(grid: Grid) -> Optional[Tuple[float, float]]:
            if not grid or not grid[0]:
                return None
            pts = [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v != 0]
            if not pts:
                return None
            total = len(pts)
            return (sum(r for r, _ in pts) / total, sum(c for _, c in pts) / total)

        src_centroid = _centroid_absolute(test_input)
        centroid_targets: List[Tuple[float, float]] = []
        for example in train_examples:
            out_grid = example.get("output")
            if not isinstance(out_grid, list):
                continue
            ratio = _centroid_ratio(out_grid)
            if ratio is None:
                continue
            centroid_targets.append(ratio)
        if src_centroid and centroid_targets:
            H, W = grid_shape(test_input)
            src_points = [
                (r, c, test_input[r][c])
                for r in range(H)
                for c in range(W)
                if test_input[r][c] != 0
            ]
            for ratio_r, ratio_c in centroid_targets[:4]:
                target_r = ratio_r * max(1, H - 1)
                target_c = ratio_c * max(1, W - 1)
                delta_r = target_r - src_centroid[0]
                delta_c = target_c - src_centroid[1]
                canvas = [[0 for _ in range(W)] for __ in range(H)]
                for r, c, color in src_points:
                    nr = int(round(r + delta_r))
                    nc = int(round(c + delta_c))
                    if in_bounds(nr, nc, H, W):
                        canvas[nr][nc] = color
                add(canvas, "centroid_snap", 0.48)

        # Dilate small components then trim to bbox
        H, W = grid_shape(test_input)
        if H and W:
            dilated = [[0 for _ in range(W)] for __ in range(H)]
            for r in range(H):
                for c in range(W):
                    val = test_input[r][c]
                    if val == 0:
                        continue
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            nr, nc = r + dr, c + dc
                            if in_bounds(nr, nc, H, W):
                                dilated[nr][nc] = val
            min_r, min_c, max_r, max_c = rescue_utils.get_bounding_box(test_input)
            if max_r >= min_r and max_c >= min_c:
                trimmed = [[0 for _ in range(W)] for __ in range(H)]
                for r in range(min_r, max_r + 1):
                    for c in range(min_c, max_c + 1):
                        trimmed[r][c] = dilated[r][c]
                add(trimmed, "dilate_trim_bbox", 0.42)

        # Component stitch: fill gaps of length 1 horizontally/vertically
        stitched = [row[:] for row in test_input]
        for r in range(H):
            for c in range(W - 2):
                left = stitched[r][c]
                right = stitched[r][c + 2]
                if left != 0 and left == right and stitched[r][c + 1] == 0:
                    stitched[r][c + 1] = left
        for c in range(W):
            for r in range(H - 2):
                top = stitched[r][c]
                bottom = stitched[r + 2][c]
                if top != 0 and top == bottom and stitched[r + 1][c] == 0:
                    stitched[r + 1][c] = top
        if stitched != test_input:
            add(stitched, "component_stitch", 0.44)

        # Border add/remove (thin 1px)
        add(self._add_border(test_input, color=bgc, thickness=1), "add_border_1", 0.30)
        add(self._remove_border(test_input, thickness=1), "remove_border_1", 0.30)

        # Tiling from rules
        for (ht, wt) in rules.get("tiling_patterns", []):
            tiled = self._tile_grid(test_input, ht, wt)
            if len(tiled) <= 30 and (len(tiled[0]) if tiled else 0) <= 30:
                add(tiled, f"tile_{ht}x{wt}", 0.80)
                alt_row = self._tile_grid_with_flips(test_input, ht, wt, "rowflip")
                if alt_row != tiled:
                    add(alt_row, f"tile_{ht}x{wt}_rowflip", 0.72)
                alt_checker = self._tile_grid_with_flips(test_input, ht, wt, "checker")
                if alt_checker not in (tiled, alt_row):
                    add(alt_checker, f"tile_{ht}x{wt}_checker", 0.70)

        # Speculative tilings
        if test_input and test_input[0]:
            for ht, wt in [(1,2),(2,1),(2,2),(3,1),(1,3),(3,3)]:
                tiled = self._tile_grid(test_input, ht, wt)
                if len(tiled) <= 30 and len(tiled[0]) <= 30:
                    add(tiled, f"tile_{ht}x{wt}_spec", 0.25)
                    alt_row = self._tile_grid_with_flips(test_input, ht, wt, "rowflip")
                    if alt_row != tiled:
                        add(alt_row, f"tile_{ht}x{wt}_rowflip_spec", 0.22)
                    alt_checker = self._tile_grid_with_flips(test_input, ht, wt, "checker")
                    if alt_checker not in (tiled, alt_row):
                        add(alt_checker, f"tile_{ht}x{wt}_checker_spec", 0.21)

        # Optional pattern-based generators
        if os.getenv("ARC_PATTERNS", "0") == "1":
            pattern_ctx = build_pattern_context(train_examples, test_input)
            expected_shape = grid_shape(test_input)
            pattern_specs = [
                ("outline", generate_outline_patterns, 0.62),
                ("union", generate_union_patterns, 0.58),
                ("chasm", generate_chasm_patterns, 0.55),
            ]
            for name, fn, base_conf in pattern_specs:
                try:
                    generated = fn(train_examples, test_input, pattern_ctx)
                except Exception as exc:
                    print(f"[ADAPTER-WARN] pattern={name} failed: {type(exc).__name__}")
                    continue
                kept_shapes: List[str] = []
                for arr in generated:
                    variant_meta: Dict[str, Any] = {}
                    variant_extra: Dict[str, Any] = {}
                    grid_like: Any = arr
                    if isinstance(arr, dict):
                        grid_like = arr.get("grid")
                        if isinstance(arr.get("meta"), dict):
                            variant_meta = dict(arr["meta"])  # type: ignore[index]
                        if isinstance(arr.get("extra"), dict):
                            variant_extra = dict(arr["extra"])  # type: ignore[index]
                    elif isinstance(arr, tuple) and len(arr) == 2:
                        grid_like, candidate_meta = arr
                        if isinstance(candidate_meta, dict):
                            variant_meta = dict(candidate_meta)
                    grid = asgrid(grid_like)
                    shape = grid_shape(grid)
                    if shape != expected_shape:
                        print(
                            f"[WARN] undersized output {shape} expected {expected_shape} "
                            f"(task {getattr(self, 'current_task_id', 'unknown')})"
                        )
                    extra_fields = {"pattern_name": name, "pattern_context": pattern_ctx}
                    if variant_meta.get("pattern_variant"):
                        extra_fields["pattern_variant"] = variant_meta["pattern_variant"]
                    if "bar_width" in variant_meta:
                        extra_fields["bar_width"] = variant_meta["bar_width"]
                    extra_fields.update(variant_extra)
                    cand = add(
                        grid,
                        f"pattern_{name}",
                        base_conf,
                        extra=extra_fields,
                        meta=variant_meta,
                    )
                    if cand is None:
                        continue
                    cand.extra.setdefault("pattern_name", name)
                    cand.extra.setdefault("pattern_context", pattern_ctx)
                    kept_shapes.append(f"{shape[0]}x{shape[1]}")
                if kept_shapes:
                    print(
                        f"[ADAPTER][pattern={name}] k={len(kept_shapes)} best={base_conf:.2f} "
                        f"shapes={kept_shapes}"
                    )

            hints = pattern_ctx.get("hints", {}) if isinstance(pattern_ctx, dict) else {}

            bar_variants = self._vertical_bar_clones(test_input)
            for variant in bar_variants:
                grid_like = variant.get("grid")
                meta = dict(variant.get("meta", {}))
                if grid_like is None:
                    continue
                grid = asgrid(grid_like)
                extra_fields = {"pattern_name": "bar_clone", "pattern_context": pattern_ctx}
                extra_fields.update(meta)
                cand = add(
                    grid,
                    "pattern_bar_clone",
                    0.56,
                    extra=extra_fields,
                    meta=meta,
                )
                if cand is None:
                    continue
                cand.extra.setdefault("pattern_name", "bar_clone")
                cand.extra.setdefault("pattern_context", pattern_ctx)

            if hints.get("cross"):
                for variant in self._central_cross_variants(test_input):
                    meta = dict(variant.get("meta", {}))
                    grid_like = variant.get("grid")
                    if grid_like is None:
                        continue
                    grid = asgrid(grid_like)
                    extra_fields = {"pattern_name": "cross_flip", "pattern_context": pattern_ctx}
                    extra_fields.update(meta)
                    cand = add(
                        grid,
                        "pattern_cross_flip",
                        0.54,
                        extra=extra_fields,
                        meta=meta,
                    )
                    if cand is None:
                        continue
                    cand.extra.setdefault("pattern_name", "cross_flip")
                    cand.extra.setdefault("pattern_context", pattern_ctx)

            notch_offsets = pattern_ctx.get("notch_offsets") if isinstance(pattern_ctx, dict) else None
            if notch_offsets:
                for variant in self._notch_variants(test_input, notch_offsets):
                    meta = dict(variant.get("meta", {}))
                    grid_like = variant.get("grid")
                    if grid_like is None:
                        continue
                    grid = asgrid(grid_like)
                    extra_fields = {"pattern_name": "notch", "pattern_context": pattern_ctx}
                    extra_fields.update(meta)
                    cand = add(
                        grid,
                        "pattern_notch",
                        0.53,
                        extra=extra_fields,
                        meta=meta,
                    )
                    if cand is None:
                        continue
                    cand.extra.setdefault("pattern_name", "notch")
                    cand.extra.setdefault("pattern_context", pattern_ctx)

        return [cand.as_payload() for cand in candidates]

    def _tile_grid(self, g: Grid, ht: int, wt: int) -> Grid:
        if not g or not g[0]: return g
        in_h, in_w = grid_shape(g)
        res: Grid = []
        for ti in range(ht):
            for i in range(in_h):
                row = []
                for tj in range(wt):
                    row.extend(g[i])
                res.append(row)
        return res

    def _tile_grid_with_flips(self, g: Grid, ht: int, wt: int, mode: str) -> Grid:
        """Tiling variants that flip stripes to match training patterns."""

        if not g or not g[0]:
            return g

        mode = mode.lower().strip()
        in_h, in_w = grid_shape(g)
        result: Grid = []

        for ty in range(ht):
            row_flip = mode in {"rowflip", "checker"} and (ty % 2 == 1)
            for r in range(in_h):
                base_row = list(g[r])
                if row_flip:
                    base_row = list(reversed(base_row))
                row: List[int] = []
                for tx in range(wt):
                    segment = base_row
                    if mode == "checker" and ((ty + tx) % 2 == 1):
                        segment = list(reversed(base_row))
                    row.extend(segment)
                result.append(row)

        return result

    def _vertical_bar_clones(self, grid: Grid) -> List[Dict[str, Any]]:
        h, w = grid_shape(grid)
        if h == 0 or w == 0:
            return []
        bg = mode_color(grid)
        stripes: List[Tuple[int, List[List[int]], int]] = []
        c = 0
        while c < w:
            column_values = [grid[r][c] for r in range(h)]
            if column_values and all(val == column_values[0] for val in column_values) and column_values[0] != bg:
                color = int(column_values[0])
                start = c
                while c < w:
                    if any(grid[r][c] != color for r in range(h)):
                        break
                    c += 1
                width = c - start
                if 0 < width < w:
                    patch = [row[start:start + width] for row in grid]
                    stripes.append((width, patch, color))
                continue
            c += 1

        results: List[Dict[str, Any]] = []
        for width, patch, color in stripes:
            tiled: Grid = []
            for r in range(h):
                row: List[int] = []
                while len(row) < w:
                    row.extend(int(val) for val in patch[r])
                tiled.append(row[:w])
            results.append(
                {
                    "grid": tiled,
                    "meta": {
                        "pattern_variant": "bar_clone",
                        "bar_width": int(width),
                        "bar_color": int(color),
                    },
                }
            )
        return results

    def _central_cross_variants(self, grid: Grid) -> List[Dict[str, Any]]:
        h, w = grid_shape(grid)
        if h < 2 or w < 2:
            return []
        top = max(0, (h - 2) // 2)
        left = max(0, (w - 2) // 2)
        block = [row[left:left + 2] for row in grid[top:top + 2]]
        if len(block) < 2 or len(block[0]) < 2:
            return []

        def _replace(patch: List[List[int]]) -> Grid:
            canvas = copy_grid(grid)
            for dy in range(min(2, len(patch))):
                for dx in range(min(2, len(patch[dy]))):
                    canvas[top + dy][left + dx] = int(patch[dy][dx])
            return canvas

        cw = [[block[1][0], block[0][0]], [block[1][1], block[0][1]]]
        ccw = [[block[0][1], block[1][1]], [block[0][0], block[1][0]]]
        diag_swap = [[block[1][1], block[1][0]], [block[0][1], block[0][0]]]

        variants = [
            (cw, "cross_flip_cw"),
            (ccw, "cross_flip_ccw"),
            (diag_swap, "cross_flip_diag"),
        ]
        results: List[Dict[str, Any]] = []
        for patch, variant_name in variants:
            results.append(
                {
                    "grid": _replace(patch),
                    "meta": {"pattern_variant": variant_name},
                }
            )
        return results

    def _notch_variants(
        self, grid: Grid, offsets: Optional[List[Tuple[int, int, int]]]
    ) -> List[Dict[str, Any]]:
        if not offsets:
            return []
        h, w = grid_shape(grid)
        if h == 0 or w == 0:
            return []
        centre_r = (h - 1) / 2.0
        centre_c = (w - 1) / 2.0
        orientations = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        results: List[Dict[str, Any]] = []
        for oy, ox in orientations:
            canvas = copy_grid(grid)
            applied = False
            for dy, dx, color in offsets:
                ty = int(round(centre_r + oy * dy))
                tx = int(round(centre_c + ox * dx))
                if 0 <= ty < h and 0 <= tx < w:
                    canvas[ty][tx] = int(color)
                    applied = True
            if applied:
                results.append(
                    {
                        "grid": canvas,
                        "meta": {
                            "pattern_variant": "notch",
                            "orientation": (oy, ox),
                        },
                    }
                )
        return results

    def _detect_color_mapping_setwise(self, color_mappings: List[Dict]) -> Dict[int,int]:
        """Simple 1-1 mapping by sorted sets; returns {} if inconsistent."""
        if not color_mappings: return {}
        mapping: Dict[int,int] = {}
        for cm in color_mappings:
            ins = sorted(cm.get("input_colors", set()))
            outs = sorted(cm.get("output_colors", set()))
            if len(ins) != len(outs): 
                continue
            for i, inc in enumerate(ins):
                outc = outs[i]
                if inc in mapping and mapping[inc] != outc:
                    return {}
                mapping[inc] = outc
        return mapping

    def _detect_color_mapping_pairs(self, pair_lists: List[List[Tuple[int,int]]]) -> Dict[int,int]:
        """Map each input color to most frequent output color seen in sampled pairs."""
        freq: Dict[int, Counter] = {}
        for pairs in pair_lists:
            for a,b in pairs:
                freq.setdefault(a, Counter()).update([b])
        mapping = {}
        for a, cnts in freq.items():
            outc, n = cnts.most_common(1)[0]
            # require some minimum dominance to avoid noise
            if n >= 2 or len(cnts) == 1:
                mapping[a] = outc
        return mapping

    def _first_nonempty_row(self, g: Grid) -> int:
        """Rows above the topmost non-zero pixel (approx)."""
        if not g: return 0
        H, W = grid_shape(g)
        for r in range(H):
            if any(g[r][c] != 0 for c in range(W)):
                return r
        return 0

    def _first_nonempty_col(self, g: Grid) -> int:
        if not g or not g[0]: return 0
        H, W = grid_shape(g)
        for c in range(W):
            if any(g[r][c] != 0 for r in range(H)):
                return c
        return 0

    def _shift_grid(self, g: Grid, dx: int, dy: int) -> Grid:
        """dx: left(-)/right(+), dy: up(-)/down(+) shift with zero-fill."""
        if not g or not g[0]: return g
        H, W = grid_shape(g)
        out = [[0 for _ in range(W)] for __ in range(H)]
        for r in range(H):
            for c in range(W):
                nr, nc = r + dy, c + dx
                if in_bounds(nr, nc, H, W):
                    out[nr][nc] = g[r][c]
        return out

    def _add_border(self, g: Grid, color: int, thickness: int = 1) -> Grid:
        if not g or not g[0]: return g
        H, W = grid_shape(g)
        nh, nw = H + 2*thickness, W + 2*thickness
        out = [[color for _ in range(nw)] for __ in range(nh)]
        for r in range(H):
            for c in range(W):
                out[r+thickness][c+thickness] = g[r][c]
        return out

    def _remove_border(self, g: Grid, thickness: int = 1) -> Grid:
        if not g or not g[0]: return g
        H, W = grid_shape(g)
        if H <= 2*thickness or W <= 2*thickness: return g
        return [row[thickness:W-thickness] for row in g[thickness:H-thickness]]

    # ----------------------- Stage 3: Beam Search -----------------------------
    def _beam_search(self, candidates: List[Dict], train_examples: List[Dict]) -> List[Dict]:
        tried = len(candidates)
        plausible_candidates = [cand for cand in candidates if self._plausible_candidate(cand)]
        if plausible_candidates:
            candidates = plausible_candidates
            applied = len(plausible_candidates)
        else:
            applied = tried
            if self._gate_logging:
                print("[PLAUSIBILITY] all candidates filtered; retaining originals")
        self._last_finishers_stats = {"tried": tried, "applied": applied}

        scored = []

        train_summary = {"hits": 0, "cases": len(train_examples or []), "hamming": []}

        # target output shape prior (if consistent)
        tgt_h = tgt_w = None
        if train_examples:
            ohs = [len(ex["output"]) for ex in train_examples]
            ows = [len(ex["output"][0]) for ex in train_examples if ex["output"]]
            if ohs and ows and len(set(ohs)) == 1 and len(set(ows)) == 1:
                tgt_h, tgt_w = ohs[0], ows[0]

        current_palette = getattr(self, "_current_allowed_palette", None)
        target_palette: Set[int]
        if isinstance(current_palette, set) and current_palette:
            try:
                target_palette = {int(color) for color in current_palette}
            except Exception:
                target_palette = set(current_palette)
        else:
            train_pairs_attr = getattr(self, "_current_train_pairs", [])
            test_grid = getattr(self, "_current_test_input", [])
            try:
                palette_source = _allowed_palette(train_pairs_attr, test_grid if isinstance(test_grid, list) else [])
            except Exception:
                palette_source = set()
            try:
                target_palette = {int(color) for color in palette_source}
            except Exception:
                target_palette = set()

        def _safe_float(value: Any, default: float = 0.0) -> float:
            try:
                return float(value)
            except Exception:
                return default

        for cand in candidates:
            score = _safe_float(cand.get("confidence", 0.0), 0.0)
            cand_type = cand.get("type", "")
            is_pattern = cand_type.startswith("pattern_")

            # train validation (skip penalty for pattern heuristics)
            if is_pattern:
                v = 0.0
            else:
                v = self._validate_transformation(cand, train_examples)
            if v == 1.0:
                score += self.perfect_fit_bonus
            elif v > 0:
                score += self.partial_fit_scale * v
            elif not is_pattern:
                score -= self.never_fit_penalty

            # shape prior
            ch, cw = grid_shape(cand["grid"])
            if tgt_h is not None and tgt_w is not None:
                if ch == tgt_h and cw == tgt_w:
                    score += self.shape_match_bonus
                else:
                    # small graded penalty by distance
                    score += 0.3 * (1.0 / (1.0 + abs(ch - tgt_h) + abs(cw - tgt_w)))

            if is_pattern:
                bonus = heuristic_pattern_priors(train_examples, cand, cand.get("pattern_context"))
                score += bonus

            train_loss = grid_loss_against_train(cand.get("grid", []), train_examples)
            cand["train_loss"] = train_loss
            if isinstance(train_loss, (int, float)) and math.isfinite(train_loss):
                train_summary["hamming"].append(float(train_loss))
                if train_loss <= 0.0:
                    train_summary["hits"] += 1

            raw_scores = cand.get("scores")
            if isinstance(raw_scores, dict):
                scores_map = raw_scores
            elif isinstance(raw_scores, Mapping):
                scores_map = dict(raw_scores)
                cand["scores"] = scores_map
            else:
                scores_map = {}
                cand["scores"] = scores_map

            palette_score = _palette_completion_score(cand.get("grid"), target_palette)
            scores_map["palette_score"] = palette_score

            prior_seed = scores_map.get("prior_conf", score)
            prior_base = _safe_float(prior_seed, 0.0)
            if prior_base <= 0.0:
                prior_base = max(score, 0.0)
            prior_conf = max(prior_base, 0.0) + 0.35 * palette_score
            scores_map["prior_conf"] = prior_conf

            palette_penalty = _safe_float(scores_map.get("penalty_palette", 0.0), 0.0)
            shape_penalty = _safe_float(scores_map.get("penalty_shape", 0.0), 0.0)

            score += 0.35 * palette_score
            if palette_penalty:
                score -= palette_penalty
            if shape_penalty:
                score -= shape_penalty

            rank_value = max(prior_conf, 0.0) + 0.20 * palette_score - palette_penalty - shape_penalty

            scored.append({**cand, "beam_score": score, "_beam_rank": rank_value})

        scored.sort(
            key=lambda entry: (
                entry.get("_beam_rank", entry.get("beam_score", 0.0)),
                entry.get("beam_score", 0.0),
            ),
            reverse=True,
        )

        for entry in scored:
            if "_beam_rank" in entry:
                entry.pop("_beam_rank", None)
        # keep a bit more than topk for CSP
        self._last_train_check = train_summary
        return scored[: max(8, self.topk * 3)]

    def _validate_transformation(self, cand: Dict, train_examples: List[Dict]) -> float:
        """Score by applying same transform 'type' to train. For learned maps,
        infer per-train color maps and use them for validation. Also build a
        consensus map we can trust later."""
        t = cand.get("type", "")
        total = len(train_examples) if train_examples else 0
        if total == 0:
            return 0.0

        # Special handling for learned color maps
        if t in ("learned_color_map_set", "learned_color_map_pairs"):
            matches = 0
            per_maps = []
            for ex in train_examples:
                m = self._infer_color_map_from_pair(ex["input"], ex["output"])
                per_maps.append(m)
                remapped = [[m.get(x, x) for x in row] for row in ex["input"]]
                if remapped == ex["output"]:
                    matches += 1
            cand["consensus_color_map"] = self._merge_color_maps(per_maps) if per_maps else {}
            return matches / total

        if t == "event_candidate":
            # No deterministic replay; proxy with shape equality
            matches = 0
            for ex in train_examples:
                ch, cw = grid_shape(cand["grid"])
                oh, ow = grid_shape(ex["output"])
                if ch == oh and cw == ow:
                    matches += 1
            return matches / total

        # default path for deterministic transforms
        matches = 0
        for ex in train_examples:
            try:
                predicted = self._apply_transformation(ex["input"], t)
                if predicted == ex["output"]:
                    matches += 1
            except Exception:
                pass
        return matches / total


    def _apply_transformation(self, g: Grid, t: str) -> Grid:
        if t == "identity": return g
        if t == "color_flip_0_1":
            return [[1 - x if x in (0,1) else x for x in row] for row in g]
        if t == "color_plus_1":
            return [[(x + 1) % 10 for x in row] for row in g]
        if t == "color_minus_1":
            return [[(x - 1) % 10 for x in row] for row in g]
        if t == "mirror_h": return [row[::-1] for row in g]
        if t == "mirror_v": return g[::-1]
        if t == "rot_180": return [row[::-1] for row in g[::-1]]
        if t == "transpose":
            if g and g[0]:
                H, W = grid_shape(g)
                return [[g[i][j] for i in range(H)] for j in range(W)]
            return g
        if t == "rot_90_cw":
            if g and g[0]:
                H, W = grid_shape(g)
                return [[g[H-1-j][i] for j in range(H)] for i in range(W)]
            return g
        if t == "rot_90_ccw":
            if g and g[0]:
                H, W = grid_shape(g)
                return [[g[j][W-1-i] for j in range(H)] for i in range(W)]
            return g
        if t == "learned_color_map_set" or t == "learned_color_map_pairs":
            # these require per-instance maps; in validation we can't reconstruct them here,
            # so fall back to identity (validation uses precomputed cand grid anyway)
            return g
        if t == "bg_to_mode_if_zero":
            bgc = mode_color(g)
            return [[bgc if x == 0 else x for x in row] for row in g]
        if t == "largest_obj_to_topleft":
            return translate_object_to(g, "topleft", bg=mode_color(g))
        if t == "largest_obj_to_center":
            return translate_object_to(g, "center", bg=mode_color(g))
        if t == "shift_to_top":
            return self._shift_grid(g, dx=0, dy=-self._first_nonempty_row(g))
        if t == "shift_to_left":
            return self._shift_grid(g, dx=-self._first_nonempty_col(g), dy=0)
        if t.startswith("add_border_"):
            return self._add_border(g, color=mode_color(g), thickness=1)
        if t.startswith("remove_border_"):
            return self._remove_border(g, thickness=1)
        if t.startswith("tile_"):
            try:
                spec = t.split("_")[1]  # e.g., "2x3"
                ht, wt = map(int, spec.split("x"))
                return self._tile_grid(g, ht, wt)
            except Exception:
                return g
        return g

    # -------------------------- Stage 4: CSP ---------------------------------
    def _csp_solver(self, beam_results: List[Dict]) -> List[Dict]:
        out = []
        for cand in beam_results:
            grid = cand["grid"]
            # size constraints
            if not (1 <= len(grid) <= 30): continue
            if grid and not (1 <= len(grid[0]) <= 30): continue
            # values in [0..9]
            if any((cell < 0 or cell > 9) for row in grid for cell in row): continue
            # rectangular
            if not is_rectangular(grid): continue

            evidence = cand.get("beam_score", 0.5)
            cand["csp_approved"] = (evidence >= self.csp_evidence_min)
            cand["csp_evidence"] = evidence
            out.append(cand)
        return out

    def _boost_csp_scores(self, csp_solutions: List[Dict[str, Any]]) -> None:
        for cand in csp_solutions:
            scores_map = cand.get("scores") if isinstance(cand.get("scores"), dict) else {}
            if not isinstance(scores_map, dict):
                scores_map = {}
                cand["scores"] = scores_map
            prior = _safe01(cand.get("beam_score", cand.get("confidence", 0.0)))
            scores_map["prior_conf"] = prior
            try:
                raw_csp = float(cand.get("csp_evidence", cand.get("beam_score", 0.0)))
            except Exception:
                raw_csp = 0.0
            csp_conf = _safe01(raw_csp)
            scores_map["csp_conf"] = csp_conf
            if cand.get("csp_approved", False):
                final = (0.5 * max(prior, 0.6)) + (0.5 * csp_conf)
            else:
                final = 0.8 * prior
            cand["final_score"] = final
            scores_map["final"] = final

    # ---------------------- Stage 5: Output formatting -----------------------
    def _format_outputs(
        self,
        csp_solutions: List[Dict],
        test_input: Grid,
        train_examples: List[Dict],
        expected_shape: Optional[Tuple[int, int]] = None,
    ) -> List[Dict]:
        self._last_output_metrics = None
        border_diag = self._prepare_output_diagnostics(train_examples, test_input)
        # if a candidate carries a consensus color map, apply it to its grid
        for s in csp_solutions:
            cm = s.get("consensus_color_map")
            if cm:
                s["grid"] = [[cm.get(x, x) for x in row] for row in s["grid"]]

        score_stats: Optional[Dict[str, Any]] = None
        if csp_solutions:
            def _percentile(sorted_vals: List[float], frac: float) -> float:
                if not sorted_vals:
                    return float("nan")
                if len(sorted_vals) == 1:
                    return sorted_vals[0]
                pos = max(0.0, min(float(len(sorted_vals) - 1), frac * (len(sorted_vals) - 1)))
                lower = int(math.floor(pos))
                upper = int(math.ceil(pos))
                if lower == upper:
                    return sorted_vals[lower]
                weight = pos - lower
                return sorted_vals[lower] * (1.0 - weight) + sorted_vals[upper] * weight

            beam_scores: List[float] = []
            below_floor = 0
            approved_count = 0
            for cand in csp_solutions:
                final_score = cand.get("final_score", cand.get("beam_score"))
                if isinstance(final_score, (int, float)) and math.isfinite(final_score):
                    val = float(final_score)
                    beam_scores.append(val)
                    if val < self.csp_evidence_min:
                        below_floor += 1
                if cand.get("csp_approved", False):
                    approved_count += 1

            if beam_scores:
                sorted_scores = sorted(beam_scores)

                hist_edges = [
                    float("-inf"),
                    -1.0,
                    -0.5,
                    0.0,
                    0.5,
                    0.6,
                    0.7,
                    0.74,
                    0.8,
                    0.9,
                    1.0,
                    float("inf"),
                ]

                def _label(start: float, end: float) -> str:
                    if math.isinf(start) and start < 0:
                        return f"<{end:.2f}"
                    if math.isinf(end) and end > 0:
                        return f">={start:.2f}"
                    return f"{start:.2f}-{end:.2f}"

                hist_labels = [_label(hist_edges[i], hist_edges[i + 1]) for i in range(len(hist_edges) - 1)]
                hist_counts = {label: 0 for label in hist_labels}
                for val in beam_scores:
                    for edge_idx in range(len(hist_edges) - 1):
                        start = hist_edges[edge_idx]
                        end = hist_edges[edge_idx + 1]
                        if (val >= start or math.isinf(start)) and (val < end or math.isinf(end)):
                            hist_counts[hist_labels[edge_idx]] += 1
                            break

                approved_ratio = (
                    approved_count / len(beam_scores) if beam_scores else float("nan")
                )
                below_ratio = below_floor / len(beam_scores) if beam_scores else float("nan")
                score_stats = {
                    "task": str(getattr(self, "current_task_id", "unknown")),
                    "idx": int(getattr(self, "current_test_index", 0)),
                    "topk": int(getattr(self, "topk", 0)),
                    "threshold": float(self.csp_evidence_min),
                    "count": len(beam_scores),
                    "approved_count": approved_count,
                    "below_threshold": below_floor,
                    "min": min(sorted_scores),
                    "p25": _percentile(sorted_scores, 0.25),
                    "median": statistics.median(sorted_scores),
                    "p75": _percentile(sorted_scores, 0.75),
                    "max": max(sorted_scores),
                    "mean": statistics.fmean(sorted_scores)
                    if hasattr(statistics, "fmean")
                    else sum(sorted_scores) / len(sorted_scores),
                    "hist": hist_counts,
                    "approved_ratio": approved_ratio,
                    "below_ratio": below_ratio,
                }

        approved = [s for s in csp_solutions if s.get("csp_approved", False)]
        pool = approved if len(approved) >= self.topk else csp_solutions
        if not pool:
            ext_max = max(
                (s.get("final_score", s.get("beam_score", 0.0)) for s in csp_solutions),
                default=0.0,
            )
            log_fallback("no_pool_candidates", ext_max, self.csp_evidence_min, len(csp_solutions))

        if score_stats is not None:
            score_stats["approved_pool"] = len(approved)
            score_stats["pool_size"] = len(pool)
            try:
                print(f"[CSP-SCORES] {json.dumps(score_stats, sort_keys=True)}")
            except Exception:
                print(
                    "[CSP-SCORES] failed to serialize stats", "task=", score_stats.get("task"),
                )
            if isinstance(self.last_trace, dict):
                self.last_trace.setdefault("csp", {}).update({"beam_stats": score_stats})
            if self._trace_history and isinstance(self._trace_history[-1], dict):
                self._trace_history[-1].setdefault("csp", {}).update({"beam_stats": score_stats})

        def _shape_of(grid: Grid) -> Tuple[int, int]:
            return (len(grid), len(grid[0]) if grid else 0)

        resolved_shape: Optional[Tuple[int, int]] = expected_shape
        if (
            resolved_shape is None
            or resolved_shape[0] <= 0
            or resolved_shape[1] <= 0
        ):
            train_shapes = [
                _shape_of(ex["output"])
                for ex in train_examples or []
                if isinstance(ex, dict)
                and isinstance(ex.get("output"), list)
                and ex["output"]
                and ex["output"][0]
            ]
            if train_shapes:
                unique_shapes = {
                    shape for shape in train_shapes if shape[0] > 0 and shape[1] > 0
                }
                if len(unique_shapes) == 1:
                    resolved_shape = unique_shapes.pop()

        if (
            resolved_shape is None or resolved_shape[0] <= 0 or resolved_shape[1] <= 0
        ) and test_input and test_input[0]:
            resolved_shape = _shape_of(test_input)

        palette = palette_frequencies(train_examples)
        train_in_palette, _ = _collect_example_palette(train_examples, "input")
        train_out_palette, _ = _collect_example_palette(train_examples, "output")
        test_palette = _grid_colors(test_input)
        base_allowed_palette: set[int] = {
            int(color)
            for color in set(train_in_palette.keys())
            | set(train_out_palette.keys())
            | set(test_palette)
        }
        birth_palette: list[int] = [
            int(color)
            for color in train_out_palette.keys()
            if color not in train_in_palette
        ]

        train_outputs = [
            ex.get("output")
            for ex in train_examples or []
            if isinstance(ex, dict)
            and isinstance(ex.get("output"), list)
            and ex.get("output")
            and isinstance(ex.get("output")[0], list)
        ]
        exemplar_signatures = collect_exemplar_signatures(train_outputs)

        ordered_pool = sorted(
            pool,
            key=lambda x: x.get("final_score", x.get("beam_score", 0.0)),
            reverse=True,
        )
        if ordered_pool:
            diag_ctx = {
                "train_summary": dict(getattr(self, "_last_train_check", {}) or {}),
                "border": border_diag,
            }
            top_candidate = ordered_pool[0]
            if self._candidate_looks_like_scaffold(top_candidate):
                extended = self._maybe_extend_diagonals(top_candidate.get("grid"), diag_ctx)
                original_grid = top_candidate.get("grid")
                if isinstance(extended, list) and extended is not original_grid:
                    top_candidate["grid"] = extended
                    meta_ref = top_candidate.setdefault("meta", {})
                    meta_ref.setdefault("post_diag_extend", True)

        matching = [
            cand
            for cand in ordered_pool
            if resolved_shape and _shape_of(cand["grid"]) == resolved_shape
        ]
        mismatched = [
            cand for cand in ordered_pool
            if not resolved_shape or _shape_of(cand["grid"]) != resolved_shape
        ]

        outputs: List[Dict] = []
        used_scores: List[float] = []
        snap_log: Optional[List[Dict[str, Any]]] = None
        canonical_border: Optional[Tuple[int, int, int, int]] = None
        if isinstance(border_diag, dict):
            snap_entries = border_diag.get("snap_log")
            if isinstance(snap_entries, list):
                snap_log = snap_entries
            canonical_border = border_diag.get("canonical_output_border")

        def _postprocess(
            grid: Grid,
            target: Optional[Tuple[int, int]],
            cand: Optional[Dict[str, Any]] = None,
            stage: str = "output",
            idx: int = 0,
        ) -> Grid:
            extras = _extract_gate_allowed_palette(cand) if cand else set()
            palette_counter = palette if palette else Counter()
            # Preserve novel colours introduced by the candidate even when the training
            # palette does not contain them. Previously these colours were clamped to the
            # minimum training colour which erased legitimate hues (e.g. public task
            # 00576224 where the test palette differs from the training outputs).
            if isinstance(grid, list):
                candidate_colors: set[int] = set()
                for row in grid:
                    if not isinstance(row, list):
                        continue
                    for cell in row:
                        try:
                            candidate_colors.add(int(cell))
                        except Exception:
                            continue
                allowed_palette = {int(color) for color in palette_counter.keys()}
                missing_colors = candidate_colors - allowed_palette - set(extras)
                if missing_colors:
                    extras = set(extras) | missing_colors
            has_extras = bool(extras)
            if palette_counter or extras:
                candidate = clamp_to_palette(grid, palette_counter, extras=extras)
                candidate = recolor_components_by_palette(
                    candidate, palette_counter, extras=extras
                )
            else:
                candidate = copy_grid(grid)
            candidate_signature_before = border_signature(candidate)
            action = "keep"
            reason: Optional[str] = None
            snapped_signature: Optional[Tuple[int, int, int, int]] = None
            best_border = candidate
            best_loss = grid_loss_against_train(candidate, train_examples)
            if not has_extras:
                for ex in train_examples or []:
                    if not isinstance(ex, dict):
                        continue
                    gold = ex.get("output")
                    if not isinstance(gold, list):
                        continue
                    snapped = snap_border_like(gold, candidate)
                    if snapped is candidate:
                        continue
                    loss = grid_loss_against_train(snapped, train_examples)
                    if loss < best_loss:
                        best_loss = loss
                        best_border = snapped
                        action = "snap"
                        snapped_signature = border_signature(snapped)
                        reason = "train_border_alignment"
            candidate = best_border
            if snap_log is not None:
                log_entry: Dict[str, Any] = {
                    "candidate": candidate_signature_before,
                    "expected": canonical_border,
                    "action": action,
                }
                if reason:
                    log_entry["reason"] = reason
                if action == "snap":
                    log_entry["result"] = snapped_signature or border_signature(candidate)
                snap_log.append(log_entry)

            if target and target[0] > 0 and target[1] > 0:
                candidate = tight_crop_uncrop(candidate, target)
            orient_best = candidate
            orient_loss = grid_loss_against_train(candidate, train_examples)
            if not has_extras:
                for variant in dihedral_variants(candidate):
                    loss = grid_loss_against_train(variant, train_examples)
                    if loss < orient_loss - 1e-9:
                        orient_loss = loss
                        orient_best = variant
                candidate = orient_best
            if target and target[0] > 0 and target[1] > 0:
                candidate = fit_to_shape(candidate, target)

            allowed_palette = set(base_allowed_palette)
            allowed_palette.update(int(color) for color in extras)
            palette_context: Dict[str, Any] = {
                "allowed_palette": allowed_palette,
                "birth_palette": birth_palette,
            }
            if exemplar_signatures:
                palette_context["exemplar_signatures"] = exemplar_signatures
            if train_examples:
                palette_context["train_examples"] = train_examples
            if test_input:
                palette_context["test_input"] = test_input
            candidate = palette_finalize_candidate(candidate, palette_context)
            log_candidate(stage, idx, candidate, palette_context)
            metrics_map = palette_context.get("_metrics", {})
            if cand is not None and isinstance(metrics_map, dict):
                meta_ref = cand.setdefault("meta", {})
                for key, value in metrics_map.items():
                    metric_key = f"palette_{key}"
                    if isinstance(value, (int, float)):
                        meta_ref[metric_key] = round(float(value), 4)
                    else:
                        meta_ref[metric_key] = value
            missing_births = palette_context.get("_missing_birth_colours")
            if cand is not None and missing_births:
                meta_ref = cand.setdefault("meta", {})
                meta_ref["palette_missing_births"] = sorted(
                    int(value) for value in missing_births if isinstance(value, (int, float))
                )
            return candidate

        def _consume(candidates: Iterable[Dict]) -> None:
            for cand in candidates:
                if len(outputs) >= self.topk:
                    return
                grid = _postprocess(
                    cand["grid"],
                    resolved_shape,
                    cand,
                    stage="output",
                    idx=len(outputs),
                )
                outputs.append(
                    {
                        "grid": grid,
                        "confidence": _safe01(
                            cand.get(
                                "final_score",
                                cand.get("beam_score", cand.get("confidence", 0.0)),
                            )
                        ),
                        "source": cand.get("type", "unknown"),
                    }
                )
                try:
                    used_scores.append(
                        float(
                            cand.get(
                                "final_score",
                                cand.get("beam_score", cand.get("confidence", 0.0)),
                            )
                        )
                    )
                except Exception:
                    pass

        _consume(matching)
        if len(outputs) < self.topk:
            _consume(mismatched)

        produced_k = len(outputs)
        if produced_k < self.topk:
            fallback_shape = resolved_shape or (_shape_of(test_input) if test_input else None)
            if fallback_shape and fallback_shape[0] > 0 and fallback_shape[1] > 0 and test_input:
                fallback = fit_to_shape(test_input, fallback_shape)
            else:
                fallback = test_input if test_input else [[0]]
            if mismatched and not outputs:
                print(
                    f"[ADAPTER-WARN] unable to adapt {len(mismatched)} mismatched outputs; "
                    "padding with resized test input"
                )
            elif produced_k < self.topk:
                print(f"[ADAPTER-WARN] produced_k={produced_k} requested_k={self.topk}")
            while len(outputs) < self.topk:
                outputs.append(
                    {
                        "grid": copy_grid(fallback),
                        "confidence": 0.0,
                        "source": "fallback_test_input",
                    }
                )

        best_score = 0.0
        if used_scores:
            best_score = _safe01(max(used_scores))
        elif ordered_pool:
            best_score = _safe01(
                ordered_pool[0].get(
                    "final_score", ordered_pool[0].get("beam_score", 0.0)
                )
            )

        shapes = [f"{len(entry['grid'])}x{len(entry['grid'][0]) if entry['grid'] else 0}" for entry in outputs]
        disp = max(0.0, min(best_score, 1.0))
        best_loss: Optional[float] = None
        if outputs and isinstance(outputs[0].get("grid"), list):
            best_loss = grid_loss_against_train(outputs[0]["grid"], train_examples)
        metrics_payload: Dict[str, Any] = {
            "best_score": float(best_score),
            "best_loss": best_loss,
            "produced": len(outputs),
        }
        if score_stats is not None:
            metrics_payload["beam_stats"] = score_stats
        self._last_output_metrics = metrics_payload
        print(f"[ADAPTER] k={len(outputs)} best={disp:.2f} shapes={shapes}")
        return outputs[: self.topk]

# ------------------------ Convenience functions ------------------------------
def _solve_arc_task_impl(
    train_examples: List[Dict], test_input: Grid, topk: int = 2
) -> List[Dict[str, Any]]:
    solver = RILSolver(seed=int(os.getenv("RIL_SEED", "1337")))
    solver.topk = int(os.getenv("RIL_PALETTE_TOPK", str(topk)))
    return solver.solve_arc_task(train_examples, test_input)


def solve_arc_task(
    train_examples: List[Dict], test_input: Grid, topk: int = 2
) -> List[Dict[str, Any]]:
    """Adapter entrypoint expected by the Kaggle wrapper."""
    try:
        return _solve_arc_task_impl(train_examples, test_input, topk)
    except Exception as exc:  # pragma: no cover - defensive guard for stdlib smoke
        print(f"[EXT-PROBE ERROR] ril/solver failed: {exc}")
        return []

def load_task_from_file(task_file: Path) -> Task:
    with open(task_file) as f:
        return json.load(f)

def solve_task_file(task_file: Path, topk: int = 2) -> List[Grid]:
    task = load_task_from_file(task_file)
    train_examples = task.get("train", [])
    test_examples = task.get("test", [])
    if not test_examples:
        raise ValueError(f"No test examples in task {task_file}")
    test_input = test_examples[0]["input"]
    attempts = solve_arc_task(train_examples, test_input, topk)
    grids: List[Grid] = []
    for entry in attempts:
        if isinstance(entry, dict) and "grid" in entry:
            grids.append(entry["grid"])
        else:
            grids.append(entry)
    return grids

# ------------------------------ CLI -----------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        task_path = Path(sys.argv[1])
        sols = solve_task_file(task_path)
        print(f"Generated {len(sols)} solutions:")
        for i, sol in enumerate(sols, 1):
            h, w = grid_shape(sol)
            print(f"Solution {i}: {h}x{w}")
    else:
        print("Usage: python solver.py <task_file.json>")
