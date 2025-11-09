"""Lightweight helpers for exercising the RIL solver in unit tests.

The real solver is fairly monolithic which makes it awkward to probe specific
pipeline stages (candidate generation, gating, beam padding, etc.) from a unit
test.  The utilities in this module provide a thin, well-behaved façade that
the tests can use to simulate carefully controlled scenarios without having to
spin up the entire solver.  They intentionally bias towards determinism and
clarity rather than raw fidelity – the goal is to expose the failure modes that
showed up in the logs (empty beams, over-eager gates, duplicate collapse,
format regressions) and keep them from regressing again.

The helpers favour normal ``ril.solver`` building blocks where possible so the
behaviour stays aligned with the production code path, but they also add the
“safety rails” discussed in the bug report: relaxed gate retries, candidate
padding, and diversity enforcement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import json
from collections import Counter

from ril.solver import Candidate, RILSolver, grid_shape, mode_color
from src.common.config import get
from src.metrics.candidate_trace import CandidateTrace
from src.pipeline.beam_progressive import progressive_beam
from src.pipeline.beam_seed import structure_seed
from src.pipeline.csp_expand import csp_generate_colorings
from src.gates.entropy_gate import gate_candidates as entropy_gate_candidates, pixel_entropy
from src.pipeline.gates import palette_gate
from src.pipeline.palette_seeding import seed_absent_colors


def _candidate_id(candidate: Any) -> str:
    """Best-effort identifier for trace logging."""

    direct = getattr(candidate, "id", None)
    if direct is not None:
        return str(direct)

    meta = getattr(candidate, "meta", {}) or {}
    if isinstance(meta, dict):
        meta_id = meta.get("id") or meta.get("candidate_id")
        if meta_id is not None:
            return str(meta_id)

    getter = getattr(candidate, "get", None)
    if callable(getter):
        lookup = getter("id") or getter("candidate_id") or getter("op_id")
        if lookup is not None:
            return str(lookup)

    kind = getattr(candidate, "kind", None)
    if kind is not None:
        return str(kind)

    source = getattr(candidate, "source", None)
    if source is not None:
        return str(source)

    return "candidate"


def _candidate_entropy(candidate: Any) -> float:
    """Extract the pixel entropy estimate for a candidate."""

    sources = [
        getattr(candidate, "entropy", None),
        getattr(getattr(candidate, "extra", None), "get", lambda *_: None)("entropy"),
    ]

    meta = getattr(candidate, "meta", {}) or {}
    if isinstance(meta, dict):
        sources.append(meta.get("entropy"))

    getter = getattr(candidate, "get", None)
    if callable(getter):
        sources.append(getter("entropy"))
        sources.append(getter("H_pixel"))

    for raw in sources:
        if raw is None:
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    grid = getattr(candidate, "grid", None)
    if isinstance(grid, list):
        try:
            return float(pixel_entropy(grid))
        except Exception:
            return 0.0
    return 0.0


def compute_tau(candidate: Any) -> float:
    """Return the τ readiness score recorded on the candidate, if any."""

    keys = ("tau", "tau_readiness", "tau_score")

    for key in keys:
        value = getattr(candidate, key, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue

    meta = getattr(candidate, "meta", {}) or {}
    if isinstance(meta, dict):
        for key in keys:
            value = meta.get(key)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue

    getter = getattr(candidate, "get", None)
    if callable(getter):
        for key in keys:
            value = getter(key, None)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue

    grid = getattr(candidate, "grid", None)
    if isinstance(grid, list) and grid and isinstance(grid[0], list):
        height = len(grid)
        width = len(grid[0]) if grid[0] else 0
        area = max(1, height * max(1, width))
        return 1.0 / (1.0 + float(area))

    return 0.0


def _candidate_palette_completeness(candidate: Any) -> float:
    sources = [
        getattr(candidate, "palette_completeness", None),
        getattr(candidate, "palette_score", None),
    ]
    meta = getattr(candidate, "meta", {}) or {}
    if isinstance(meta, dict):
        sources.extend(
            meta.get(key)
            for key in ("palette_completeness", "palette_score", "palette_match")
        )
    getter = getattr(candidate, "get", None)
    if callable(getter):
        for key in ("palette_completeness", "palette_score"):
            try:
                sources.append(getter(key))
            except Exception:
                continue
    for raw in sources:
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        return max(0.0, min(1.0, value))
    return 0.0


def _candidate_structural_iou(candidate: Any) -> float | None:
    sources = [
        getattr(candidate, "structural_iou", None),
        getattr(candidate, "approx_iou", None),
    ]
    meta = getattr(candidate, "meta", {}) or {}
    if isinstance(meta, dict):
        sources.extend(
            meta.get(key)
            for key in ("structural_iou", "approx_iou", "iou", "sIoU")
        )
    getter = getattr(candidate, "get", None)
    if callable(getter):
        for key in ("structural_iou", "approx_iou", "iou"):
            try:
                sources.append(getter(key))
            except Exception:
                continue
    for raw in sources:
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        return max(0.0, min(1.0, value))
    return None


def _apply_entropy_penalty(candidate: Any, penalty: float) -> None:
    try:
        penalty = float(penalty)
    except (TypeError, ValueError):
        return
    if penalty >= 0.999:
        return
    penalty = max(0.0, min(1.0, penalty))

    for attr in ("score", "confidence"):
        try:
            value = getattr(candidate, attr)
        except Exception:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        try:
            setattr(candidate, attr, numeric * penalty)
        except Exception:
            pass

    meta = getattr(candidate, "meta", None)
    if isinstance(meta, dict):
        for key in ("score", "confidence"):
            if key in meta and meta[key] is not None:
                try:
                    meta[key] = float(meta[key]) * penalty
                except (TypeError, ValueError):
                    continue


def _log_candidate_trace(task_id: str, candidate: Any, stage: str, **payload: Any) -> None:
    trace = CandidateTrace(
        task_id=task_id,
        candidate_id=_candidate_id(candidate),
        stage=stage,
        **payload,
    )
    try:
        trace.save(None)
    except Exception:
        # Logging failures should never break the gating flow.
        return


def _log_generate_and_rescue(task_id: str, candidate: Any) -> None:
    entropy = float(_candidate_entropy(candidate))
    tau_score = compute_tau(candidate)
    palette_score = _candidate_palette_completeness(candidate)
    s_iou = _candidate_structural_iou(candidate)
    base_payload = {
        "H_pixel": entropy,
        "tau": tau_score,
        "structural_iou": s_iou,
        "palette_completeness": palette_score,
        "survived_entropy": None,
        "survived_tau": None,
        "rejected": False,
        "reject_reason": None,
        "was_final_answer": False,
    }
    _log_candidate_trace(task_id, candidate, "generate", **base_payload)
    rescue_payload = dict(base_payload)
    rescue_payload["csp_applied"] = bool(getattr(candidate, "structure", None))
    _log_candidate_trace(task_id, candidate, "rescue_pre", **rescue_payload)


def _log_entropy_soft_stub(task_id: str, candidate: Any) -> None:
    entropy = float(_candidate_entropy(candidate))
    tau_score = compute_tau(candidate)
    palette_score = _candidate_palette_completeness(candidate)
    s_iou = _candidate_structural_iou(candidate)
    payload = {
        "H_pixel": entropy,
        "tau": tau_score,
        "structural_iou": s_iou,
        "palette_completeness": palette_score,
        "survived_entropy": True,
        "survived_tau": True,
        "rejected": False,
        "reject_reason": None,
        "extra": {"entropy_penalty": 1.0},
    }
    setattr(candidate, "_trace_survived_entropy", True)
    setattr(candidate, "_trace_survived_tau", True)
    setattr(candidate, "_trace_entropy_penalty", 1.0)
    _log_candidate_trace(task_id, candidate, "entropy_soft", **payload)

Grid = List[List[int]]


@dataclass
class AEContext:
    """Minimal abstraction of the abstract encoder outputs used by gates."""

    shape_preserving: bool
    allowed_shapes: set[Tuple[int, int]]
    palette_size: int
    frozen_mask_density: float
    confidence: float
    allowed_palette: Optional[Sequence[int]] = None


@dataclass
class MockTask:
    """Synthetic ARC-like task used by the stdlib tests."""

    task_id: str
    train_examples: List[Dict[str, Grid]]
    test_inputs: List[Grid]
    solutions: List[Grid]
    palette_in: Sequence[int]
    palette_out: Sequence[int]
    _training_palette_cache: Set[int] | None = None

    def is_exact(self, grid: Grid) -> bool:
        return any(grid == sol for sol in self.solutions)

    @property
    def test_ids(self) -> List[str]:  # pragma: no cover - trivial
        return [f"{self.task_id}_test_{idx}" for idx, _ in enumerate(self.test_inputs)]

    @property
    def training_palette(self) -> Set[int]:
        cached = getattr(self, "_training_palette_cache", None)
        if isinstance(cached, set) and cached:
            return set(cached)

        colours: Set[int] = set()
        for raw in self.palette_out:
            try:
                colours.add(int(raw))
            except Exception:
                continue
        if not colours:
            for raw in self.palette_in:
                try:
                    colours.add(int(raw))
                except Exception:
                    continue

        for example in self.train_examples:
            if not isinstance(example, dict):
                continue
            for key in ("output", "input"):
                grid = example.get(key)
                if not isinstance(grid, list):
                    continue
                for row in grid:
                    if not isinstance(row, list):
                        continue
                    for value in row:
                        try:
                            colours.add(int(value))
                        except Exception:
                            continue

        self._training_palette_cache = set(colours)
        return set(colours)

    def baseline_bank(self) -> List[Candidate]:
        solver = RILSolver()
        test_input = self.test_inputs[0]
        majority = mode_color(test_input)
        outline_color = int(self.palette_out[-1]) if self.palette_out else majority
        outline = _add_outline(test_input, outline_color)
        filled = _fill_constant(test_input, outline_color)
        solution = self.solutions[0] if self.solutions else outline

        return [
            solver._make_candidate(test_input, "identity", 0.35),
            solver._make_candidate(filled, "majority_fill", 0.32),
            solver._make_candidate(outline, "outline_seed", 0.40),
            solver._make_candidate(solution, "solution_seed", 0.95),
        ]


def _grid_palette(grid: Optional[Sequence[Sequence[int]]]) -> set[int]:
    palette: set[int] = set()
    if not isinstance(grid, Sequence):
        return palette
    for row in grid:
        if not isinstance(row, Sequence):
            continue
        for value in row:
            try:
                palette.add(int(value))
            except Exception:
                continue
    return palette


def _candidate_palette(candidate: Any) -> set[int]:
    grid = getattr(candidate, "grid", None)
    if grid is None:
        getter = getattr(candidate, "get", None)
        if callable(getter):
            grid = getter("grid", None)
    return _grid_palette(grid)


def _expected_palette(ae: AEContext, task: MockTask | None) -> set[int]:
    palette: set[int] = set()

    allowed = getattr(ae, "allowed_palette", None) or []
    for colour in allowed:
        try:
            palette.add(int(colour))
        except Exception:
            continue

    if not palette and ae.palette_size:
        palette.update(range(int(ae.palette_size)))

    if task is not None:
        for colour in getattr(task, "palette_out", []) or []:
            try:
                palette.add(int(colour))
            except Exception:
                continue
        for colour in getattr(task, "palette_in", []) or []:
            try:
                palette.add(int(colour))
            except Exception:
                continue
        for grid in getattr(task, "test_inputs", []) or []:
            palette.update(_grid_palette(grid))
        for example in getattr(task, "train_examples", []) or []:
            if not isinstance(example, dict):
                continue
            out_grid = example.get("output")
            palette.update(_grid_palette(out_grid))

    return palette


def rescue_palette(
    candidate: Any,
    *,
    expected_palette: set[int],
    task_id: str,
    threshold: float,
) -> Any:
    """Annotate a candidate with palette-completeness signals prior to gating."""

    candidate_palette = _candidate_palette(candidate)
    if expected_palette:
        matched = candidate_palette & expected_palette
        completeness = len(matched) / len(expected_palette)
        missing = sorted(expected_palette - candidate_palette)
    else:
        completeness = 1.0
        missing = []

    payload_missing = missing[:8]

    # Ensure downstream consumers can read the signal from either attribute or meta.
    try:
        candidate.palette_completeness = float(completeness)
    except Exception:
        pass

    meta: dict[str, Any]
    if hasattr(candidate, "meta"):
        existing = getattr(candidate, "meta", {}) or {}
        if isinstance(existing, dict):
            meta = dict(existing)
        else:
            meta = {}
        meta.setdefault("palette_completeness", round(completeness, 4))
        if missing and "palette_missing" not in meta:
            meta["palette_missing"] = payload_missing
        if completeness < threshold:
            meta.setdefault("rescue_flagged", True)
        candidate.meta = meta
    elif isinstance(candidate, dict):
        existing = candidate.get("meta", {})
        meta = dict(existing) if isinstance(existing, dict) else {}
        meta.setdefault("palette_completeness", round(completeness, 4))
        if missing and "palette_missing" not in meta:
            meta["palette_missing"] = payload_missing
        if completeness < threshold:
            meta.setdefault("rescue_flagged", True)
        candidate["meta"] = meta

    _log_candidate_trace(
        task_id,
        candidate,
        "rescue_pre",
        H_pixel=_candidate_entropy(candidate),
        palette_completeness=float(completeness),
        extra={
            "missing_colors": payload_missing,
            "flagged": completeness < threshold,
        },
    )

    return candidate


def make_mock_task(
    *,
    in_shape: Tuple[int, int] = (12, 12),
    out_shape: Tuple[int, int] = (12, 12),
    train_examples: int = 3,
    palette_in: Sequence[int] = (0, 1, 2, 3),
    palette_out: Sequence[int] = (0, 1, 2, 3, 4),
    frozen_density: float = 0.5,
    exact_solution_ops: Sequence[str] = ("outline", "prior_fill"),
) -> MockTask:
    """Construct a deterministic toy task.

    The task mirrors the structure of an ARC instance but keeps the content
    predictable: we build a checkerboard-ish input using ``palette_in`` and
    derive the outputs by applying a small stack of operations that introduces
    one additional colour from ``palette_out``.  That means the adapter has to
    allow a new colour through the gates to find an exact match – precisely the
    failure mode the regression tests are targeting.
    """

    in_h, in_w = in_shape
    out_h, out_w = out_shape

    def _make_input(seed: int) -> Grid:
        colors = list(palette_in)
        if not colors:
            colors = [0]
        grid: Grid = []
        for r in range(in_h):
            row: List[int] = []
            for c in range(in_w):
                row.append(colors[(r + c + seed) % len(colors)])
            grid.append(row)
        return grid

    def _apply_ops(grid: Grid) -> Grid:
        result = [row[:] for row in grid]
        for op in exact_solution_ops:
            if op == "outline":
                result = _add_outline(result, int(palette_out[-1]))
            elif op == "prior_fill":
                result = _fill_constant(result, int(palette_out[-1]))
            elif op == "identity":
                continue
            else:
                # Unknown op – fall back to a gentle brighten so tests stay deterministic.
                result = [[(cell + 1) % 10 for cell in row] for row in result]
        # Ensure the output respects the requested shape.
        return _resize_grid(result, out_h, out_w, fill=int(palette_out[-1]))

    train: List[Dict[str, Grid]] = []
    for idx in range(train_examples):
        inp = _make_input(idx)
        out = _apply_ops(inp)
        train.append({"input": inp, "output": out})

    test_input = _make_input(999)
    solutions = [_apply_ops(test_input)]

    return MockTask(
        task_id="mock",
        train_examples=train,
        test_inputs=[test_input],
        solutions=solutions,
        palette_in=palette_in,
        palette_out=palette_out,
    )


def generate_candidates(
    task: MockTask,
    ae: AEContext,
    *,
    max_per_type: int = 6,
    total_cap: int = 32,
    force_adapter_count: Optional[int] = None,
    baseline_only: bool = False,
) -> List[Candidate]:
    """Return a diverse pool of candidates for the toy task.

    We seed a collection of operator families (identity/copy, symmetry,
    translations, colour tweaks, pattern fills).  Each family contributes at
    most ``max_per_type`` candidates and we keep the ``total_cap`` newest ones.
    ``force_adapter_count`` is used by the regression test to simulate an
    under-producing adapter.
    """

    solver = RILSolver()
    base = task.test_inputs[0]
    solution = task.solutions[0]
    palette_extra = int(task.palette_out[-1]) if task.palette_out else mode_color(base)

    families: Dict[str, List[Candidate]] = {
        "identity": [
            solver._make_candidate(base, "identity", 0.40),
            solver._make_candidate([row[:] for row in base], "identity", 0.38),
        ],
        "symmetry": [
            solver._make_candidate(base[::-1], "mirror", 0.36),
            solver._make_candidate([row[::-1] for row in base], "mirror", 0.36),
        ],
        "translation": [
            solver._make_candidate(_shift_grid(base, dx=1, dy=0), "shift", 0.33),
            solver._make_candidate(_shift_grid(base, dx=0, dy=1), "shift", 0.33),
        ],
        "colour": [
            solver._make_candidate([[ (cell + 1) % 10 for cell in row] for row in base], "color_shift", 0.34),
            solver._make_candidate([[ (cell - 1) % 10 for cell in row] for row in base], "color_shift", 0.34),
        ],
        "pattern": [
            solver._make_candidate(_add_outline(base, palette_extra), "pattern_outline", 0.62),
            solver._make_candidate(_fill_constant(base, palette_extra), "pattern_fill", 0.60),
            solver._make_candidate(solution, "pattern_solution", 0.98),
        ],
    }

    if baseline_only:
        baseline = task.baseline_bank()
        return baseline[: total_cap]

    ordered: List[Candidate] = []
    for family in families.values():
        ordered.extend(family[:max_per_type])

    # Simulate the adapter under-producing when requested.
    if force_adapter_count is not None:
        ordered = ordered[: force_adapter_count]

    # Ensure we always have at least the baseline solutions available for padding.
    ordered.extend(task.baseline_bank())

    if bool(get("palette.pre_seed_absent", False)):
        max_seed = int(get("palette.max_seed_colors", 3) or 3)
        ordered = seed_absent_colors(ordered, task, max_colors=max_seed)

    # NEW: early CSP expansion to couple structure & colors
    enable_csp_expand = bool(get("pipeline.csp_expand.early_expand", True))
    if enable_csp_expand:
        base_palette: List[int] = [int(col) for col in getattr(task, "palette_out", []) or []]
        if not base_palette:
            base_palette = [int(col) for col in getattr(task, "palette_in", []) or []]

        max_colorings = int(get("pipeline.csp_expand.max_colorings", 8) or 8)
        expanded: List[Candidate] = []
        for cand in ordered:
            structure = getattr(cand, "structure", None)
            palette_hint = getattr(cand, "palette", None)
            if palette_hint:
                try:
                    palette_seq = [int(col) for col in palette_hint]
                except Exception:
                    palette_seq = base_palette
            else:
                palette_seq = base_palette

            variants: List[Candidate] = []
            if structure is not None and palette_seq:
                try:
                    variants = csp_generate_colorings(
                        structure,
                        palette_seq,
                        max_colorings=max_colorings,
                    )
                except Exception:
                    variants = []

            if variants:
                expanded.extend(variants)
            else:
                expanded.append(cand)

        ordered = expanded

    task_id = str(getattr(task, "task_id", "task"))
    if bool(get("pipeline.trace.enable", True)):
        for cand in ordered:
            _log_generate_and_rescue(task_id, cand)

    # Clip to the requested cap while retaining deterministic order.
    return ordered[: total_cap]


def apply_gates(
    candidates: Iterable[Candidate],
    ae: AEContext,
    *,
    task: MockTask | None = None,
    mode: str = "strict",
) -> Tuple[List[Candidate], Dict[str, Any]]:
    """Apply soft/strict gate logic to the provided candidate pool."""

    mode = mode or "strict"
    allowed_shapes = set(ae.allowed_shapes or set())
    allowed_palette = set(int(c) for c in (ae.allowed_palette or []))
    if not allowed_palette and ae.palette_size:
        allowed_palette.update(range(ae.palette_size))

    relaxed_palette = set(allowed_palette)
    if ae.palette_size:
        # Always let one new colour through during relaxed passes.
        relaxed_palette.update(range(ae.palette_size + 1))

    accepted: List[Candidate] = []
    totals: Counter[str] = Counter()
    gate_counts: Dict[str, Counter[str]] = {
        "shape": Counter({"accepted": 0, "rejected": 0}),
        "color": Counter({"accepted": 0, "rejected": 0}),
        "region": Counter({"accepted": 0, "rejected": 0}),
    }
    palette_extra_rates: List[float] = []

    cands = list(candidates)
    task_id = str(getattr(task, "task_id", getattr(task, "id", "task")) if task else "task")

    entropy_settings = get("gates.entropy", {})
    if isinstance(entropy_settings, Mapping):
        entropy_enabled = bool(entropy_settings.get("enabled", True))
    else:
        entropy_enabled = bool(entropy_settings)
        entropy_settings = {}
    medium_min = float(get("gates.entropy_medium_min", 0.25) or 0.25)
    medium_max = float(get("gates.entropy_medium_max", 0.40) or 0.40)
    def _entropy_config_value(key: str, default: float | int) -> float | int:
        if isinstance(entropy_settings, Mapping) and key in entropy_settings:
            value = entropy_settings.get(key, default)
        else:
            value = get(f"gates.{key}", default)
        return default if value is None else value

    entropy_cfg = {
        "entropy_min_margin": float(_entropy_config_value("entropy_min_margin", 0.10) or 0.10),
        "entropy_pctl": float(_entropy_config_value("entropy_pctl", 75) or 75),
        "entropy_hard_min": float(_entropy_config_value("entropy_hard_min", 0.30) or 0.30),
        "entropy_hard_max": float(_entropy_config_value("entropy_hard_max", 0.95) or 0.95),
        "entropy_keep_topk_high": int(_entropy_config_value("entropy_keep_topk_high", 2) or 2),
        "min_beam_width_after_entropy": int(
            _entropy_config_value("min_beam_width_after_entropy", 4) or 4
        ),
    }

    tau_gate_enabled = bool(get("gates.tau_readiness", False))
    tau_threshold = float(get("gates.tau_threshold", 0.65) or 0.65)

    entropy_stats = {"threshold": None, "kept": len(cands), "total": len(cands)}
    kept_ids: Set[int]
    if entropy_enabled:
        kept_candidates, entropy_stats = entropy_gate_candidates(cands, entropy_cfg)
        kept_ids = {id(cand) for cand in kept_candidates}
    else:
        kept_ids = {id(cand) for cand in cands}

    entropy_threshold = entropy_stats.get("threshold")
    try:
        entropy_threshold_val = float(entropy_threshold) if entropy_threshold is not None else None
    except (TypeError, ValueError):
        entropy_threshold_val = None

    survivors: List[Candidate] = []
    for cand in cands:
        entropy = float(_candidate_entropy(cand))
        tau_score = compute_tau(cand)
        palette_score = _candidate_palette_completeness(cand)
        s_iou = _candidate_structural_iou(cand)

        entropy_pass = (not entropy_enabled) or (id(cand) in kept_ids)
        penalty = 1.0
        rejected = False
        reject_reason: str | None = None
        medium_band = medium_min <= entropy <= medium_max

        forced_keep = (
            entropy_enabled
            and entropy_threshold_val is not None
            and entropy > entropy_threshold_val
            and entropy_pass
        )

        if entropy_enabled and not entropy_pass:
            rejected = True
            reject_reason = "entropy_threshold"

        tau_pass = True
        if tau_gate_enabled:
            tau_pass = tau_score >= tau_threshold
            if not tau_pass and not rejected:
                rejected = True
                reject_reason = "tau_gate"

        setattr(cand, "_trace_survived_entropy", entropy_pass)
        setattr(cand, "_trace_survived_tau", tau_pass)
        setattr(cand, "_trace_entropy_penalty", penalty)

        payload = {
            "H_pixel": entropy,
            "tau": tau_score,
            "structural_iou": s_iou,
            "palette_completeness": palette_score,
            "survived_entropy": entropy_pass,
            "survived_tau": tau_pass,
            "rejected": rejected,
            "reject_reason": reject_reason,
            "extra": {
                "entropy_penalty": penalty,
                "medium_band": medium_band,
                "entropy_threshold": entropy_threshold_val,
                "entropy_forced_keep": forced_keep,
            },
        }

        _log_candidate_trace(task_id, cand, "entropy_soft", **payload)

        if not rejected:
            survivors.append(cand)

    cands = survivors

    reference = cands[0].grid if cands else []

    for cand in cands:
        totals[cand.kind] += 1

        shape = grid_shape(cand.grid)
        shape_ok = not allowed_shapes or shape in allowed_shapes
        if shape_ok or mode == "relaxed":
            gate_counts["shape"]["accepted"] += 1
        else:
            gate_counts["shape"]["rejected"] += 1
            continue

        colors = {int(val) for row in cand.grid for val in row}
        palette = relaxed_palette if mode == "relaxed" or ae.confidence < 0.65 else allowed_palette

        palette_values: Set[int] = set()
        for value in palette:
            try:
                palette_values.add(int(value))
            except Exception:
                continue
        total_cells = 0
        extra_cells = 0
        for row in cand.grid:
            if not isinstance(row, list):
                continue
            total_cells += len(row)
            for value in row:
                try:
                    color_val = int(value)
                except Exception:
                    color_val = value
                if color_val not in palette_values:
                    extra_cells += 1
        extra_rate = float(extra_cells) / float(total_cells or 1)
        palette_extra_rates.append(extra_rate)

        enforce_hard = mode == "strict" and ae.confidence >= 0.65
        candidate_stats = {
            "extra_color_rate": extra_rate,
            "extra_color_cells": extra_cells,
            "total_cells": total_cells,
        }
        if not palette_gate(cand, candidate_stats, enforce_hard=enforce_hard):
            gate_counts["color"]["rejected"] += 1
            continue
        gate_counts["color"]["accepted"] += 1

        # Region gating: treat high frozen density as advisory unless we are in
        # strict mode with high confidence.  We approximate by counting how many
        # cells differ from the reference; if a large frozen area is claimed we
        # only reject if the delta is substantial.
        if reference and ae.frozen_mask_density > 0.6 and mode == "strict" and ae.confidence >= 0.7:
            mismatches = _grid_delta(reference, cand.grid)
            if mismatches > int(ae.frozen_mask_density * len(reference) * len(reference[0])) // 4:
                gate_counts["region"]["rejected"] += 1
                continue
        gate_counts["region"]["accepted"] += 1

        accepted.append(cand)

    total = len(cands)
    unique = len({json.dumps(c.grid) for c in cands})
    dup_ratio = 1.0 - (unique / total) if total else 0.0

    stats = {
        "gates.total": total,
        "gates.rejected": total - len(accepted),
        "gates.accepted": len(accepted),
        "gate_counts": {
            name: {"accepted": int(counts.get("accepted", 0)), "rejected": int(counts.get("rejected", 0))}
            for name, counts in gate_counts.items()
        },
        "by_type": {
            kind: {
                "total": totals[kind],
                "accepted": sum(1 for cand in accepted if cand.kind == kind),
                "rejected": totals[kind] - sum(1 for cand in accepted if cand.kind == kind),
            }
            for kind in totals
        },
        "gate.dup_ratio": dup_ratio,
    }

    stats["entropy.threshold"] = entropy_threshold_val
    stats["entropy.kept"] = int(entropy_stats.get("kept", 0))
    stats["entropy.total"] = int(entropy_stats.get("total", 0))

    if palette_extra_rates:
        stats["palette.extra_color_rate.mean"] = sum(palette_extra_rates) / len(palette_extra_rates)
        stats["palette.extra_color_rate.max"] = max(palette_extra_rates)

    return accepted, stats


def run_beam(
    candidates: Sequence[Candidate], task: MockTask, *, k: Optional[int] = None
) -> List[Candidate]:
    """Rank candidates and ensure the beam is filled to ``k``."""

    if k is None:
        try:
            k = int(get("beam.width", 6) or 6)
        except Exception:
            k = 6

    if k <= 0:
        return []

    def _score(candidate: Candidate) -> float:
        try:
            return float(getattr(candidate, "score", 0.0))
        except Exception:
            return 0.0

    task_id = str(getattr(task, "task_id", "task"))
    trace_enabled = bool(get("pipeline.trace.enable", True))

    original_pool = list(candidates)
    seeded_pool = structure_seed(original_pool)
    refined = progressive_beam(seeded_pool)
    pool = refined if refined else seeded_pool
    pool = structure_seed(pool)

    baseline_iter = iter(task.baseline_bank())
    while len(pool) < k:
        try:
            new_cand = next(baseline_iter)
        except StopIteration:
            solver = RILSolver()
            new_cand = solver._make_candidate(task.test_inputs[0], "identity_pad", 0.20)
        if trace_enabled and not hasattr(new_cand, "_trace_survived_entropy"):
            _log_generate_and_rescue(task_id, new_cand)
            _log_entropy_soft_stub(task_id, new_cand)
        original_pool.append(new_cand)
        pool.append(new_cand)
        pool = structure_seed(pool)

    selected = pool[:k]

    if trace_enabled:
        rank_map = {id(cand): idx + 1 for idx, cand in enumerate(selected)}
        selected_ids = set(rank_map)

        for cand in original_pool:
            cand_id = id(cand)
            in_beam = cand_id in selected_ids
            rank = rank_map.get(cand_id)
            setattr(cand, "_trace_entered_beam", in_beam)

            entropy = float(_candidate_entropy(cand))
            tau_score = compute_tau(cand)
            palette_score = _candidate_palette_completeness(cand)
            s_iou = _candidate_structural_iou(cand)
            survived_entropy = getattr(cand, "_trace_survived_entropy", None)
            survived_tau = getattr(cand, "_trace_survived_tau", None)
            penalty = getattr(cand, "_trace_entropy_penalty", None)

            extra_payload: Dict[str, Any] = {}
            if penalty is not None:
                extra_payload["entropy_penalty"] = penalty

            payload = {
                "H_pixel": entropy,
                "tau": tau_score,
                "structural_iou": s_iou,
                "palette_completeness": palette_score,
                "survived_entropy": survived_entropy,
                "survived_tau": survived_tau,
                "beam_rank": rank,
                "entered_beam": in_beam,
                "rejected": not in_beam,
                "reject_reason": None if in_beam else "beam_pruned",
                "was_final_answer": False,
            }
            if extra_payload:
                payload["extra"] = extra_payload

            _log_candidate_trace(task_id, cand, "beam", **payload)

            refine_payload = dict(payload)
            refine_payload["rejected"] = not in_beam
            refine_payload["reject_reason"] = None if in_beam else "beam_pruned"
            _log_candidate_trace(task_id, cand, "csp_refine", **refine_payload)

    return selected


def save_competition_json(task: MockTask, predictions: Sequence[Grid], path: Path) -> None:
    """Write Kaggle-style competition outputs for the toy task."""

    rows: List[Dict[str, Any]] = []
    for task_id in task.test_ids:
        rows.append(
            {
                "id": task_id,
                "attempts": [
                    {"attempt": idx + 1, "grid": grid}
                    for idx, grid in enumerate(predictions)
                ],
            }
        )
    path.write_text(json.dumps(rows), encoding="utf-8")


def save_private_jsonl(task: MockTask, predictions: Sequence[Grid], path: Path) -> None:
    """Write the private evaluation JSONL shim used in CI."""

    with path.open("w", encoding="utf-8") as handle:
        for idx, task_id in enumerate(task.test_ids):
            grid = predictions[min(idx, len(predictions) - 1)] if predictions else []
            handle.write(json.dumps({"id": task_id, "y": grid}))
            handle.write("\n")


# ---------------------------------------------------------------------------
# Helper utilities


def _add_outline(grid: Grid, color: int) -> Grid:
    if not grid or not grid[0]:
        return grid
    h, w = grid_shape(grid)
    outlined = [row[:] for row in grid]
    for c in range(w):
        outlined[0][c] = color
        outlined[h - 1][c] = color
    for r in range(h):
        outlined[r][0] = color
        outlined[r][w - 1] = color
    return outlined


def _fill_constant(grid: Grid, color: int) -> Grid:
    return [[color for _ in row] for row in grid]


def _shift_grid(grid: Grid, dx: int, dy: int) -> Grid:
    if not grid or not grid[0]:
        return grid
    h, w = grid_shape(grid)
    out = [[0 for _ in range(w)] for _ in range(h)]
    for r in range(h):
        for c in range(w):
            nr, nc = r + dy, c + dx
            if 0 <= nr < h and 0 <= nc < w:
                out[nr][nc] = grid[r][c]
    return out


def _grid_delta(a: Grid, b: Grid) -> int:
    total = 0
    for r in range(min(len(a), len(b))):
        row_a = a[r]
        row_b = b[r]
        for c in range(min(len(row_a), len(row_b))):
            if row_a[c] != row_b[c]:
                total += 1
    return total


def _resize_grid(grid: Grid, h: int, w: int, *, fill: int) -> Grid:
    if h <= 0 or w <= 0:
        return []
    resized = [[fill for _ in range(w)] for _ in range(h)]
    for r in range(min(h, len(grid))):
        row = grid[r]
        for c in range(min(w, len(row))):
            resized[r][c] = row[c]
    return resized


__all__ = [
    "AEContext",
    "MockTask",
    "make_mock_task",
    "generate_candidates",
    "apply_gates",
    "run_beam",
    "save_competition_json",
    "save_private_jsonl",
]

