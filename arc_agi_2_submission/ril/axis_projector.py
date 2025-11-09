"""Axis-aligned translation projector used by the RIL adapter.

This helper learns a dominant translation axis from the training examples and
emits a small bundle of integer translations for the test input.  The
implementation is intentionally NumPy-optional so it continues to run in the
Kaggle inference environment.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
import itertools
import math
from typing import Dict, List, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency detection
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover - maintain graceful degradation
    _np = None

HAVE_NUMPY = _np is not None

from .assembly_ops import extract_components

Grid = List[List[int]]
Example = Dict[str, object]


class _AxisProjectorBudgetExceeded(RuntimeError):
    """Internal sentinel exception raised when the time budget is exceeded."""


def _bbox_int(grid: Grid) -> Tuple[int, int, int, int]:
    H, W = len(grid), len(grid[0]) if grid and grid[0] else 0
    ys: List[int] = []
    xs: List[int] = []
    for y in range(H):
        for x in range(W):
            if grid[y][x] != 0:
                ys.append(y)
                xs.append(x)
    if not ys:
        return (0, 0, 0, 0)
    y0 = min(ys)
    y1 = max(ys)
    x0 = min(xs)
    x1 = max(xs)
    return (y0, x0, y1, x1)


def _centroid_int(grid: Grid) -> Tuple[int, int]:
    y0, x0, y1, x1 = _bbox_int(grid)
    return ((y0 + y1) // 2, (x0 + x1) // 2)


@dataclass(frozen=True)
class AxisProjectionMetrics:
    """Diagnostic metrics describing the learnt axis."""

    vector_count: int
    ang_var: float
    step_median: float
    step_mad: float
    explained_frac: float
    preferred_sign: int
    sign_confidence: float
    best_min_replay_iou: float
    best_mean_replay_iou: float


@dataclass(frozen=True)
class AxisProjectionCandidate:
    """Candidate grid alongside metadata for solver integration."""

    grid: Grid
    translation: Tuple[int, int]
    clipped: int
    min_replay_iou: float
    mean_replay_iou: float
    mean_ang_error: float
    mean_length_error: float


@dataclass(frozen=True)
class AxisProjectionResult:
    """Bundle containing the learnt axis and generated candidates."""

    axis: Tuple[float, float]
    step: float
    translations: List[Tuple[int, int]]
    metrics: AxisProjectionMetrics
    candidates: List[AxisProjectionCandidate]


def _centroid(pixels: Sequence[Tuple[int, int]]) -> Tuple[float, float]:
    total = len(pixels)
    if total == 0:
        return (0.0, 0.0)
    sum_r = sum(r for r, _ in pixels)
    sum_c = sum(c for _, c in pixels)
    return (sum_r / float(total), sum_c / float(total))


def _vector_length(vec: Tuple[float, float]) -> float:
    return math.hypot(vec[0], vec[1])


def _dot(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _normalise(vec: Tuple[float, float]) -> Optional[Tuple[float, float]]:
    length = _vector_length(vec)
    if length <= 1e-9:
        return None
    return (vec[0] / length, vec[1] / length)


def _match_components(
    inputs: Sequence[Dict[str, object]],
    outputs: Sequence[Dict[str, object]],
    *,
    deadline: Optional[float] = None,
) -> List[Tuple[Dict[str, object], Dict[str, object]]]:
    """Brute-force component matching for tiny populations (≤7).

    The ARC tasks that rely on the projector rarely exceed six components per
    colour family.  Brute force keeps the implementation simple and avoids a
    heavy Hungarian dependency while still providing globally optimal matches.
    """

    n = min(len(inputs), len(outputs))
    if n == 0:
        return []
    if n == 1:
        return [(inputs[0], outputs[0])]

    # Distance + size penalty discourages accidental cross matches.
    def cost(idx: int, jdx: int) -> float:
        inp = inputs[idx]
        out = outputs[jdx]
        cin = _centroid(inp["pixels"])  # type: ignore[arg-type]
        cout = _centroid(out["pixels"])  # type: ignore[arg-type]
        dist = abs(cin[0] - cout[0]) + abs(cin[1] - cout[1])
        size_in = float(inp.get("size", 1))
        size_out = float(out.get("size", 1))
        rel = abs(size_in - size_out) / max(size_in, size_out, 1.0)
        return dist + 0.35 * rel

    indices = range(n)
    best_perm: Optional[Tuple[int, ...]] = None
    best_cost = float("inf")
    for perm in itertools.permutations(indices):
        if deadline is not None and time.perf_counter() > deadline:
            raise _AxisProjectorBudgetExceeded
        total = 0.0
        for idx, jdx in enumerate(perm):
            total += cost(idx, jdx)
            if total >= best_cost:
                break
        else:
            if total < best_cost:
                best_cost = total
                best_perm = perm

    if best_perm is None:
        # Fallback to greedy nearest-neighbour if an unexpected overflow occurs.
        matches: List[Tuple[Dict[str, object], Dict[str, object]]] = []
        remaining = list(outputs)
        for inp in inputs[:n]:
            if deadline is not None and time.perf_counter() > deadline:
                raise _AxisProjectorBudgetExceeded
            cin = _centroid(inp["pixels"])  # type: ignore[arg-type]
            best_j = 0
            best_d = float("inf")
            for j, out in enumerate(remaining):
                cout = _centroid(out["pixels"])  # type: ignore[arg-type]
                dist = abs(cin[0] - cout[0]) + abs(cin[1] - cout[1])
                if dist < best_d:
                    best_d = dist
                    best_j = j
            matches.append((inp, remaining.pop(best_j)))
        return matches

    return [(inputs[idx], outputs[jdx]) for idx, jdx in enumerate(best_perm)]


def _principal_axis(
    vectors: Sequence[Tuple[float, float]],
    weights: Sequence[float],
) -> Optional[Tuple[float, float]]:
    if not vectors:
        return None

    if len(vectors) == 1:
        return _normalise(vectors[0])

    total_w = sum(weights) or float(len(vectors))
    mean_r = sum(vec[0] * w for vec, w in zip(vectors, weights)) / total_w
    mean_c = sum(vec[1] * w for vec, w in zip(vectors, weights)) / total_w

    # Weighted covariance matrix
    s_rr = 0.0
    s_cc = 0.0
    s_rc = 0.0
    for vec, weight in zip(vectors, weights):
        dr = vec[0] - mean_r
        dc = vec[1] - mean_c
        w = weight or 1.0
        s_rr += w * dr * dr
        s_cc += w * dc * dc
        s_rc += w * dr * dc

    if total_w:
        s_rr /= total_w
        s_cc /= total_w
        s_rc /= total_w

    trace = s_rr + s_cc
    det = s_rr * s_cc - s_rc * s_rc
    disc = max(trace * trace - 4.0 * det, 0.0)
    eig = 0.5 * (trace + math.sqrt(disc))

    if abs(s_rc) > 1e-9:
        axis = (eig - s_cc, s_rc)
    elif s_rr >= s_cc:
        axis = (1.0, 0.0)
    else:
        axis = (0.0, 1.0)

    normalised = _normalise(axis)
    if normalised is not None:
        return normalised

    # Degenerate case – fall back to the mean translation direction.
    return _normalise((mean_r, mean_c))


def _angular_variance(
    vectors: Sequence[Tuple[float, float]],
    axis: Tuple[float, float],
) -> float:
    projections: List[float] = []
    for vec in vectors:
        length = _vector_length(vec)
        if length <= 1e-9:
            continue
        cos_theta = abs(_dot(vec, axis) / length)
        projections.append(cos_theta)
    if not projections:
        return 1.0
    mean = sum(projections) / len(projections)
    return sum((val - mean) ** 2 for val in projections) / len(projections)


def _weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    pairs = sorted(zip(values, weights), key=lambda item: item[0])
    total = sum(weights) or float(len(values))
    acc = 0.0
    for value, weight in pairs:
        acc += weight or 1.0
        if acc >= total / 2.0:
            return value
    return pairs[-1][0] if pairs else 0.0


def _median_absolute_deviation(values: Sequence[float], weights: Sequence[float]) -> float:
    if not values:
        return 0.0
    median = _weighted_median(values, weights)
    deviations = [abs(val - median) for val in values]
    return _weighted_median(deviations, weights)


def _candidate_translations(
    axis: Tuple[float, float], step: float, preferred_sign: int
) -> List[Tuple[int, int]]:
    base: List[Tuple[int, int]] = []
    for sign in (1.0, -1.0):
        if preferred_sign and sign * preferred_sign < 0:
            continue
        for rounder in (round, math.floor, math.ceil):
            dr = int(rounder(sign * axis[0] * step))
            dc = int(rounder(sign * axis[1] * step))
            base.append((dr, dc))

    # Deduplicate while preserving order.
    seen = set()
    ordered: List[Tuple[int, int]] = []
    for vec in base:
        if vec not in seen:
            seen.add(vec)
            ordered.append(vec)

    jitters: List[Tuple[int, int]] = []
    for dr, dc in ordered:
        jitters.extend([(dr + 1, dc), (dr - 1, dc), (dr, dc + 1), (dr, dc - 1)])

    for vec in jitters:
        if vec not in seen:
            seen.add(vec)
            ordered.append(vec)

    return ordered


def _translate_grid(grid: Grid, dr: int, dc: int) -> Tuple[Grid, int]:
    if not grid or not grid[0]:
        return grid, 0
    height = len(grid)
    width = len(grid[0])
    result: Grid = [[0 for _ in range(width)] for _ in range(height)]
    clipped = 0
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            if value == 0:
                continue
            nr = r + dr
            nc = c + dc
            if 0 <= nr < height and 0 <= nc < width:
                result[nr][nc] = value
            else:
                clipped += 1
    return result, clipped


def _mask_iou(a: Grid, b: Grid) -> float:
    if not a or not a[0] or not b or not b[0]:
        return 0.0
    height = min(len(a), len(b))
    width = min(len(a[0]), len(b[0]))
    inter = union = 0
    for r in range(height):
        for c in range(width):
            av = 1 if a[r][c] != 0 else 0
            bv = 1 if b[r][c] != 0 else 0
            if av and bv:
                inter += 1
            if av or bv:
                union += 1
    if union == 0:
        return 0.0
    return inter / float(union)


class AxisProjector:
    """Learn a dominant translation axis from training examples."""

    def __init__(
        self,
        *,
        min_vectors: int = 2,
        max_ang_var: float = 0.12,
        max_step_mad: float = 1.0,
        min_replay_iou: float = 0.9,
        min_explained_frac: float = 0.6,
    ) -> None:
        self.min_vectors = min_vectors
        self.max_ang_var = max_ang_var
        self.max_step_mad = max_step_mad
        self.min_replay_iou = min_replay_iou
        self.min_explained_frac = min_explained_frac

    def _collect_vectors(
        self,
        train_examples: Sequence[Example],
        *,
        deadline: Optional[float] = None,
    ) -> Tuple[List[Tuple[float, float]], List[float]]:
        vectors: List[Tuple[float, float]] = []
        weights: List[float] = []

        for example in train_examples:
            if deadline is not None and time.perf_counter() > deadline:
                raise _AxisProjectorBudgetExceeded
            if not isinstance(example, dict):
                continue
            inp = example.get("input")
            out = example.get("output")
            if not isinstance(inp, list) or not isinstance(out, list):
                continue

            in_comps = extract_components(inp)
            out_comps = extract_components(out)

            families: Dict[int, Tuple[List[Dict[str, object]], List[Dict[str, object]]]] = {}
            for comp in in_comps:
                families.setdefault(int(comp["color"]), ([], []))[0].append(comp)
            for comp in out_comps:
                families.setdefault(int(comp["color"]), ([], []))[1].append(comp)

            for comps_in, comps_out in families.values():
                if deadline is not None and time.perf_counter() > deadline:
                    raise _AxisProjectorBudgetExceeded
                if not comps_in or not comps_out:
                    continue
                matches = _match_components(comps_in, comps_out, deadline=deadline)
                for cin, cout in matches:
                    src = _centroid(cin["pixels"])  # type: ignore[arg-type]
                    dst = _centroid(cout["pixels"])  # type: ignore[arg-type]
                    vec = (dst[0] - src[0], dst[1] - src[1])
                    vectors.append(vec)
                    weights.append(float(cin.get("size", 1)))

        return vectors, weights

    def generate(
        self,
        train_examples: Sequence[Example],
        test_input: Grid,
        *_,
        **kwargs: object,
    ) -> Optional[AxisProjectionResult]:
        disable_flag = kwargs.get("disable_axis_projector")
        if isinstance(disable_flag, str):
            disable_requested = disable_flag.strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        elif disable_flag is not None:
            disable_requested = bool(disable_flag)
        else:
            disable_requested = False

        fast_env = os.environ.get("ARC_DISABLE_AXIS_PROJECTOR") == "1"
        if fast_env or (not HAVE_NUMPY) or disable_requested or (
            os.environ.get("RIL_DISABLE_AXIS") == "1"
        ):
            if (fast_env or not HAVE_NUMPY) and os.environ.get("RIL_GATE_LOGGING"):
                reason_parts = []
                if not HAVE_NUMPY:
                    reason_parts.append("no NumPy")
                if fast_env:
                    reason_parts.append("fast mode")
                reason = " and ".join(reason_parts) if reason_parts else "disabled"
                print(f"[AXIS-PROJECTOR] skipped ({reason})", flush=False)
            return None

        env_budget = os.environ.get("AXIS_PROJECTOR_BUDGET_MS", "250")
        try:
            default_budget = int(env_budget)
        except (TypeError, ValueError):
            default_budget = 250

        budget_override = kwargs.get("axis_projector_budget_ms")
        budget_ms: int
        if budget_override is not None:
            try:
                budget_ms = int(budget_override)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                budget_ms = default_budget
        else:
            budget_ms = default_budget
        if budget_ms < 0:
            budget_ms = 0

        start_time = time.perf_counter()
        deadline = start_time + (budget_ms / 1000.0)

        if budget_ms == 0:
            return self._fallback_result(train_examples, test_input)

        try:
            vectors, weights = self._collect_vectors(
                train_examples, deadline=deadline
            )
            if len(vectors) < self.min_vectors:
                return None

            if deadline is not None and time.perf_counter() > deadline:
                raise _AxisProjectorBudgetExceeded

            axis = _principal_axis(vectors, weights)
            if axis is None:
                return None

            projections = [_dot(vec, axis) for vec in vectors]
            abs_proj = [abs(val) for val in projections]
            step = _weighted_median(abs_proj, weights)
            if step <= 0.0:
                return None

            ang_var = _angular_variance(vectors, axis)
            step_mad = _median_absolute_deviation(abs_proj, weights)

            axis_energy = 0.0
            total_energy = 0.0
            pos_weight = 0.0
            neg_weight = 0.0
            for vec, proj, weight in zip(vectors, projections, weights):
                if deadline is not None and time.perf_counter() > deadline:
                    raise _AxisProjectorBudgetExceeded
                w = weight or 1.0
                axis_energy += w * proj * proj
                total_energy += w * _dot(vec, vec)
                if proj > 1e-6:
                    pos_weight += w
                elif proj < -1e-6:
                    neg_weight += w

            explained = axis_energy / total_energy if total_energy else 0.0

            total_sign_weight = pos_weight + neg_weight
            preferred_sign = 0
            sign_confidence = 0.0
            if total_sign_weight > 0.0:
                if pos_weight >= 0.6 * total_sign_weight:
                    preferred_sign = 1
                    sign_confidence = pos_weight / total_sign_weight
                elif neg_weight >= 0.6 * total_sign_weight:
                    preferred_sign = -1
                    sign_confidence = neg_weight / total_sign_weight

            if (
                ang_var > self.max_ang_var
                or step_mad > self.max_step_mad
                or explained < self.min_explained_frac
            ):
                return None

            translations = [
                vec
                for vec in _candidate_translations(axis, step, preferred_sign)
                if vec != (0, 0)
            ]
            if not translations:
                return None

            candidates: List[AxisProjectionCandidate] = []
            best_min_replay_iou = 0.0
            best_mean_replay_iou = 0.0
            for dr, dc in translations:
                if deadline is not None and time.perf_counter() > deadline:
                    raise _AxisProjectorBudgetExceeded
                grid, clipped = _translate_grid(test_input, dr, dc)
                # Replay on train examples to validate the move.
                iou_scores = []
                for example in train_examples:
                    if deadline is not None and time.perf_counter() > deadline:
                        raise _AxisProjectorBudgetExceeded
                    inp = example.get("input") if isinstance(example, dict) else None
                    out = example.get("output") if isinstance(example, dict) else None
                    if not isinstance(inp, list) or not isinstance(out, list):
                        continue
                    replay, _ = _translate_grid(inp, dr, dc)
                    iou_scores.append(_mask_iou(replay, out))
                min_iou = min(iou_scores) if iou_scores else 0.0
                mean_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0.0
                if min_iou < self.min_replay_iou:
                    continue

                cand_vec = (float(dr), float(dc))
                cand_len = _vector_length(cand_vec)
                ang_total = 0.0
                len_total = 0.0
                weight_total = 0.0
                for vec, weight in zip(vectors, weights):
                    if deadline is not None and time.perf_counter() > deadline:
                        raise _AxisProjectorBudgetExceeded
                    w = weight or 1.0
                    weight_total += w
                    vec_len = _vector_length(vec)
                    len_total += w * abs(cand_len - vec_len)
                    if cand_len <= 1e-9 or vec_len <= 1e-9:
                        # Treat degenerate vectors as orthogonal to penalise mismatch.
                        ang_total += w * (math.pi / 2.0)
                        continue
                    cos_theta = max(
                        -1.0, min(1.0, _dot(cand_vec, vec) / (cand_len * vec_len))
                    )
                    ang_total += w * math.acos(cos_theta)
                if weight_total > 0.0:
                    mean_ang_error = ang_total / weight_total
                    mean_length_error = len_total / weight_total
                else:
                    mean_ang_error = 0.0
                    mean_length_error = 0.0

                candidates.append(
                    AxisProjectionCandidate(
                        grid=grid,
                        translation=(dr, dc),
                        clipped=clipped,
                        min_replay_iou=min_iou,
                        mean_replay_iou=mean_iou,
                        mean_ang_error=mean_ang_error,
                        mean_length_error=mean_length_error,
                    )
                )

                best_min_replay_iou = max(best_min_replay_iou, min_iou)
                best_mean_replay_iou = max(best_mean_replay_iou, mean_iou)

            if not candidates:
                return None

            metrics = AxisProjectionMetrics(
                vector_count=len(vectors),
                ang_var=ang_var,
                step_median=step,
                step_mad=step_mad,
                explained_frac=float(max(0.0, min(1.0, explained))),
                preferred_sign=preferred_sign,
                sign_confidence=sign_confidence,
                best_min_replay_iou=best_min_replay_iou,
                best_mean_replay_iou=best_mean_replay_iou,
            )

            return AxisProjectionResult(
                axis=axis,
                step=step,
                translations=[cand.translation for cand in candidates],
                metrics=metrics,
                candidates=candidates,
            )
        except _AxisProjectorBudgetExceeded:
            return self._fallback_result(train_examples, test_input)

    def _fallback_result(
        self,
        train_examples: Sequence[Example],
        test_input: Grid,
    ) -> Optional[AxisProjectionResult]:
        try:
            c_in = _centroid_int(test_input)
        except Exception:
            c_in = (0, 0)

        c_ref = c_in
        for example in train_examples:
            out = example.get("output") if isinstance(example, dict) else None
            if isinstance(out, list) and out and isinstance(out[0], list):
                try:
                    c_ref = _centroid_int(out)  # type: ignore[arg-type]
                    break
                except Exception:
                    continue

        dy = c_ref[0] - c_in[0]
        dx = c_ref[1] - c_in[1]

        try:
            grid, clipped = _translate_grid(test_input, dy, dx)
        except Exception:
            grid = test_input
            clipped = 0

        metrics = AxisProjectionMetrics(
            vector_count=0,
            ang_var=0.0,
            step_median=0.0,
            step_mad=0.0,
            explained_frac=0.0,
            preferred_sign=0,
            sign_confidence=0.0,
            best_min_replay_iou=0.0,
            best_mean_replay_iou=0.0,
        )

        candidate = AxisProjectionCandidate(
            grid=grid,
            translation=(dy, dx),
            clipped=clipped,
            min_replay_iou=0.0,
            mean_replay_iou=0.0,
            mean_ang_error=0.0,
            mean_length_error=0.0,
        )

        return AxisProjectionResult(
            axis=(0.0, 0.0),
            step=0.0,
            translations=[candidate.translation],
            metrics=metrics,
            candidates=[candidate],
        )


__all__ = [
    "AxisProjector",
    "AxisProjectionResult",
    "AxisProjectionCandidate",
    "AxisProjectionMetrics",
]

