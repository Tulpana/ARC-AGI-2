"""
Enhancement wrapper for RILSolver integrating adaptive constraints.
Adds: SmallGridSolver fast-path, PaletteOracle, ShapeOracle, ComponentAnalyzer.
Kaggle-compatible: stdlib-only.
"""
from __future__ import annotations
import os
from typing import Any, Dict, List, Optional

Grid = List[List[int]]
Example = Dict[str, Any]


def enhance_solve_arc_task(
    original_solve_fn,
    train_examples: List[Example],
    test_input: Grid,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Enhanced wrapper around RILSolver.solve_arc_task.
    Adds fast-path optimizations and adaptive constraints.
    """
    # Check feature flags
    enable_small_grid = os.getenv("RIL_SMALL_GRID_FAST_PATH", "1") == "1"
    enable_palette_oracle = os.getenv("RIL_ADAPTIVE_PALETTE", "1") == "1"
    enable_shape_oracle = os.getenv("RIL_ADAPTIVE_SHAPE", "1") == "1"
    enable_component_scoring = os.getenv("RIL_COMPONENT_SCORING", "1") == "1"
    enable_perfect_cache = os.getenv("RIL_PERFECT_MATCH_CACHE", "1") == "1"

    # Perfect match cache (cheapest check - do first)
    if enable_perfect_cache:
        perfect_match = check_perfect_match(train_examples, test_input)
        if perfect_match is not None:
            print("[ENHANCEMENT] Perfect match found in training data!")
            return [
                {
                    "grid": perfect_match,
                    "confidence": 1.0,
                    "score": 1.0,
                    "type": "perfect_match",
                    "source": "cache",
                    "meta": {"strategy": "training_memorization"}
                }
            ]

    # Color extraction pattern fast path (for high-confidence patterns)
    enable_color_extraction = os.getenv("RIL_COLOR_EXTRACTION_FAST_PATH", "1") == "1"
    if enable_color_extraction:
        try:
            from ril.color_extraction import detect_color_extraction_pattern, apply_color_extraction

            pattern = detect_color_extraction_pattern(train_examples)
            if pattern and pattern.get('confidence', 0) == 1.0:
                # 100% pattern confidence - bypass solver entirely!
                extracted = apply_color_extraction(test_input, pattern)
                print(f"[ENHANCEMENT] Color extraction pattern detected: {pattern['type']} (conf=1.0)")
                return [
                    {
                        "grid": extracted,
                        "confidence": 0.98,
                        "score": 0.98,
                        "type": f"color_extraction_{pattern['type']}",
                        "source": "color_extraction_fastpath",
                        "meta": {"pattern": pattern, "strategy": "pattern_bypass"}
                    }
                ]
        except Exception as e:
            print(f"[ENHANCEMENT] Color extraction fast path failed: {e}, continuing")

    # Small grid fast path
    if enable_small_grid:
        try:
            from ril.small_grid_solver import SmallGridSolver

            solver = SmallGridSolver()
            if solver.can_handle(test_input, train_examples):
                topk = kwargs.get('topk', 2)
                print(f"[ENHANCEMENT] Small grid detected ({solver.grid_size(test_input)}px), using specialist solver")
                candidates = solver.solve(train_examples, test_input, topk=topk)

                if candidates:
                    # Convert to expected format
                    results = []
                    for i, cand in enumerate(candidates):
                        results.append({
                            "grid": cand,
                            "confidence": max(0.9 - i * 0.1, 0.5),  # Descending confidence
                            "score": max(0.9 - i * 0.1, 0.5),
                            "type": "small_grid_specialist",
                            "source": "small_grid_solver",
                            "meta": {"rank": i + 1}
                        })
                    return results
        except Exception as e:
            print(f"[ENHANCEMENT] Small grid solver failed: {e}, falling back to main solver")

    # Run original solver
    candidates = original_solve_fn(train_examples, test_input, **kwargs)

    # Post-process with adaptive constraints
    if enable_palette_oracle or enable_shape_oracle or enable_component_scoring:
        candidates = apply_adaptive_filters(
            candidates,
            train_examples,
            test_input,
            enable_palette=enable_palette_oracle,
            enable_shape=enable_shape_oracle,
            enable_component=enable_component_scoring
        )

    return candidates


def check_perfect_match(train_examples: List[Example], test_input: Grid) -> Optional[Grid]:
    """Check if test input exactly matches any training input (instant EM!)."""
    for ex in train_examples:
        if grids_equal(ex['input'], test_input):
            return ex['output']
    return None


def grids_equal(g1: Grid, g2: Grid) -> bool:
    """Check if two grids are identical."""
    if len(g1) != len(g2):
        return False
    if not g1:
        return not g2
    if len(g1[0]) != len(g2[0]):
        return False

    for r in range(len(g1)):
        for c in range(len(g1[0])):
            if g1[r][c] != g2[r][c]:
                return False
    return True


def apply_adaptive_filters(
    candidates: List[Dict[str, Any]],
    train_examples: List[Example],
    test_input: Grid,
    enable_palette: bool = True,
    enable_shape: bool = True,
    enable_component: bool = True
) -> List[Dict[str, Any]]:
    """
    Apply adaptive constraints to filter and re-score candidates.
    """
    if not candidates:
        return candidates

    # Initialize oracles
    palette_oracle = None
    shape_oracle = None
    component_analyzer = None

    if enable_palette:
        try:
            from ril.adaptive_constraints import AdaptivePaletteOracle
            palette_oracle = AdaptivePaletteOracle(train_examples)
        except Exception as e:
            print(f"[ENHANCEMENT] Failed to init PaletteOracle: {e}")

    if enable_shape:
        try:
            from ril.adaptive_constraints import AdaptiveShapeOracle
            shape_oracle = AdaptiveShapeOracle(train_examples)
        except Exception as e:
            print(f"[ENHANCEMENT] Failed to init ShapeOracle: {e}")

    if enable_component:
        try:
            from ril.adaptive_constraints import ComponentAnalyzer
            component_analyzer = ComponentAnalyzer(train_examples)
        except Exception as e:
            print(f"[ENHANCEMENT] Failed to init ComponentAnalyzer: {e}")

    # Filter and re-score
    filtered = []
    rejections = []

    for cand in candidates:
        grid = cand.get("grid")
        if not grid:
            continue

        # Validate palette
        if palette_oracle:
            valid, reason = palette_oracle.validate_candidate(grid, test_input)
            if not valid:
                rejections.append(("palette", reason))
                continue

        # Validate shape
        if shape_oracle:
            cand_shape = (len(grid), len(grid[0]) if grid else 0)
            test_shape = (len(test_input), len(test_input[0]) if test_input else 0)
            valid, reason = shape_oracle.validate_candidate(cand_shape, test_shape)
            if not valid:
                rejections.append(("shape", reason))
                continue

        # Re-score based on component structure
        if component_analyzer:
            component_score = component_analyzer.score_candidate(grid, test_input)
            # Multiply original confidence by component score
            original_conf = cand.get("confidence", 0.0)
            adjusted_conf = original_conf * component_score
            cand["confidence"] = adjusted_conf
            cand["score"] = adjusted_conf

            if "meta" not in cand:
                cand["meta"] = {}
            cand["meta"]["component_multiplier"] = round(component_score, 3)

        filtered.append(cand)

    # Log rejections
    if rejections:
        palette_rejects = sum(1 for t, _ in rejections if t == "palette")
        shape_rejects = sum(1 for t, _ in rejections if t == "shape")
        print(f"[ENHANCEMENT] Filtered out {len(rejections)} candidates: palette={palette_rejects}, shape={shape_rejects}")

    # Re-sort by adjusted confidence
    filtered.sort(key=lambda c: c.get("confidence", 0.0), reverse=True)

    return filtered


def get_enhanced_solver_wrapper(solver_instance):
    """
    Create an enhanced wrapper around a RILSolver instance.
    Usage:
        solver = RILSolver(...)
        enhanced_solver = get_enhanced_solver_wrapper(solver)
        results = enhanced_solver.solve_arc_task(train, test_input)
    """
    original_solve = solver_instance.solve_arc_task

    def enhanced_solve(train_examples: List[Example], test_input: Grid, **kwargs) -> List[Dict[str, Any]]:
        return enhance_solve_arc_task(original_solve, train_examples, test_input, **kwargs)

    # Monkey-patch the instance
    solver_instance.solve_arc_task = enhanced_solve
    return solver_instance
