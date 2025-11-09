"""Palette feasibility utilities used by the gating pipeline.

The solver only needs a tiny optimisation routine to determine whether a
candidate grid can realise the evidenced palette using legal recolours and the
background seeding heuristic.  The routines here intentionally avoid heavy
dependencies (no NumPy / network simplex) so they can run inside the Kaggle
sandbox and the lightweight eval harness.
"""

from __future__ import annotations

import heapq

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple


@dataclass(frozen=True)
class PalettePlanStep:
    """Instruction describing how to fix a missing evidenced colour."""

    mode: str  # "seed" or "recolor"
    source: int
    target: int
    count: int = 1


@dataclass
class PaletteFeasibilityResult:
    feasible: bool
    cost: float
    dual: Dict[int, float]
    plan: List[PalettePlanStep]
    missing: List[int]


def _demand_weight(example_hist: Dict[int, int], color: int) -> float:
    """Return a softness weight for an evidenced colour.

    Colours that appear frequently in the training outputs carry higher weight
    because failing to realise them is usually fatal.  The weight is purposely
    capped to keep the solver numerically stable.
    """

    count = float(example_hist.get(color, 0))
    if count <= 0.0:
        return 1.0
    # Scale gently and cap to avoid exploding weights on large boards.
    return min(1.0 + (count * 0.1), 2.5)


def _allowed_sources_by_target(
    allowed_edges: Dict[int, Set[int]], evidence_palette: Set[int]
) -> Dict[int, Set[int]]:
    mapping: Dict[int, Set[int]] = defaultdict(set)
    for source, targets in allowed_edges.items():
        if not targets:
            continue
        for target in targets:
            if target in evidence_palette:
                mapping[target].add(source)
    return mapping


def compute_palette_feasibility(
    candidate_counts: Counter,
    evidence_palette: Set[int],
    allowed_edges: Dict[int, Set[int]],
    example_hist: Dict[int, int],
    *,
    background_slots: int = 0,
    max_seed_use: int | None = None,
    seed_cost: float = 1.5,
    recolor_cost: float = 1.0,
) -> PaletteFeasibilityResult:
    """Check whether the evidenced palette can be achieved legally.

    Parameters
    ----------
    candidate_counts:
        Histogram of colours present in the candidate grid.
    evidence_palette:
        Set of colours that must appear in the final solution (excludes 0).
    allowed_edges:
        Directed recolour graph describing legal source→target transitions.
    example_hist:
        Histogram of colours observed in the training outputs.  Used to weight
        dual penalties.
    background_slots:
        Number of background cells that can be seeded with new colours.
    max_seed_use:
        Optional cap on how many background seeds we are willing to use.
    seed_cost / recolor_cost:
        Relative costs attached to each repair action.  Only the ratios matter.
    """

    demand_colors = sorted(color for color in evidence_palette if color != 0)
    if not demand_colors:
        return PaletteFeasibilityResult(True, 0.0, {}, [], [])

    supply: Counter[int] = Counter()
    for color, count in candidate_counts.items():
        try:
            color_int = int(color)
        except Exception:
            continue
        if color_int != 0:
            supply[color_int] += int(count)

    seeds_available = int(max(0, background_slots))
    if max_seed_use is not None:
        seeds_available = min(seeds_available, int(max_seed_use))

    allowed_sources = _allowed_sources_by_target(allowed_edges, evidence_palette)

    plan: List[PalettePlanStep] = []
    dual: Dict[int, float] = {}
    missing: List[int] = []
    total_cost = 0.0
    feasible = True

    for color in demand_colors:
        weight = _demand_weight(example_hist, color)
        if supply.get(color, 0) > 0:
            supply[color] -= 1
            dual[color] = weight
            continue

        # Try recolouring a legal source colour.
        best_choice: Tuple[float, int] | None = None
        for source in sorted(allowed_sources.get(color, set())):
            if source == color:
                continue
            if supply.get(source, 0) <= 0:
                continue
            # Mild bonus if the recolour is supported directly by examples.
            cost = recolor_cost
            if source in evidence_palette:
                cost = max(0.2, recolor_cost * 0.8)
            candidate = (cost, source)
            if best_choice is None or candidate < best_choice:
                best_choice = candidate

        if best_choice is not None:
            cost, source = best_choice
            supply[source] -= 1
            plan.append(PalettePlanStep("recolor", source=source, target=color))
            total_cost += float(cost)
            dual[color] = weight * (1.0 + float(cost))
            continue

        if seeds_available > 0:
            seeds_available -= 1
            plan.append(PalettePlanStep("seed", source=0, target=color))
            total_cost += float(seed_cost)
            dual[color] = weight * (1.0 + float(seed_cost))
            continue

        feasible = False
        missing.append(color)
        dual[color] = weight * (1.0 + float(seed_cost) * 2.0)

    return PaletteFeasibilityResult(feasible, total_cost, dual, plan, missing)


__all__ = [
    "PalettePlanStep",
    "PaletteFeasibilityResult",
    "compute_palette_feasibility",
]
