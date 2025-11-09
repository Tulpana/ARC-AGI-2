"""Specialist solver for tiny grids (≤ 3x3) and palette-sensitive tasks.

The diagnostic shows that half of worst failures are tiny grids that the
general pipeline overcomplicates. This module provides focused strategies
for small-grid and palette-constrained problems.
"""

from typing import List, Dict, Any, Set
from collections import Counter

Grid = List[List[int]]


def is_tiny_grid(grid: Grid, threshold: int = 3) -> bool:
    """Check if grid is tiny (height ≤ threshold or width ≤ threshold)."""
    if not grid or not grid[0]:
        return False
    return len(grid) <= threshold or len(grid[0]) <= threshold


def extract_palette(grid: Grid) -> Set[int]:
    """Extract all colors used in a grid."""
    colors = set()
    for row in grid:
        for cell in row:
            colors.add(int(cell))
    return colors


def learn_palette_completeness(train_examples: List[Dict[str, Any]]) -> Set[int]:
    """Learn which colors must appear in output from training examples."""
    if not train_examples:
        return set()

    # Find colors that appear in ALL training outputs
    output_palettes = []
    for ex in train_examples:
        ex_out = ex.get("output", [])
        if ex_out:
            output_palettes.append(extract_palette(ex_out))

    if not output_palettes:
        return set()

    # Intersection of all output palettes = required colors
    required = output_palettes[0].copy()
    for pal in output_palettes[1:]:
        required &= pal

    return required


def validate_palette_completeness(
    candidate_grid: Grid,
    train_examples: List[Dict[str, Any]]
) -> bool:
    """Check if candidate has all required colors from training."""
    required_colors = learn_palette_completeness(train_examples)
    if not required_colors:
        return True  # No requirement

    candidate_colors = extract_palette(candidate_grid)
    missing = required_colors - candidate_colors

    return len(missing) == 0


def solve_tiny_grid_direct(
    test_input: Grid,
    train_examples: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Direct solver for tiny grids using exact pattern matching.

    For tiny grids, try:
    1. Identity (output = input)
    2. Color swap patterns from training
    3. Simple transformations (transpose, rotate, flip)
    """
    candidates = []

    if not test_input:
        return candidates

    # Strategy 1: Identity for very small grids
    if len(test_input) == 1 and len(test_input[0]) <= 6:
        candidates.append({
            "grid": [row[:] for row in test_input],
            "source": "tiny_specialist",
            "meta": {
                "strategy": "identity",
                "confidence": 0.5,
            },
            "confidence": 0.5,
            "score": 0.5,
        })

    # Strategy 2: Learn exact transformations from training
    if train_examples:
        for ex in train_examples:
            ex_in = ex.get("input", [])
            ex_out = ex.get("output", [])

            if not ex_in or not ex_out:
                continue

            # If train input matches test input exactly, use its output
            if ex_in == test_input:
                candidates.append({
                    "grid": [row[:] for row in ex_out],
                    "source": "tiny_specialist",
                    "meta": {
                        "strategy": "exact_match",
                        "confidence": 0.95,
                    },
                    "confidence": 0.95,
                    "score": 0.95,
                })
                break

    # Strategy 3: Transpose if training shows it
    if train_examples and all(
        len(ex.get("input", [])) == len(ex.get("output", [[]])[0]) if ex.get("output") else False
        for ex in train_examples
    ):
        # Transpose pattern detected
        transposed = [[test_input[r][c] for r in range(len(test_input))]
                      for c in range(len(test_input[0]) if test_input else 0)]
        if transposed:
            candidates.append({
                "grid": transposed,
                "source": "tiny_specialist",
                "meta": {
                    "strategy": "transpose",
                    "confidence": 0.7,
                },
                "confidence": 0.7,
                "score": 0.7,
            })

    # Strategy 4: Color remapping for single-cell outputs
    if train_examples and all(
        len(ex.get("output", [])) == 1 and len(ex.get("output", [[]])[0]) == 1
        for ex in train_examples if ex.get("output")
    ):
        # All outputs are 1x1 - find most common output value
        output_vals = []
        for ex in train_examples:
            ex_out = ex.get("output", [])
            if ex_out and ex_out[0]:
                output_vals.append(ex_out[0][0])

        if output_vals:
            most_common = Counter(output_vals).most_common(1)[0][0]
            candidates.append({
                "grid": [[most_common]],
                "source": "tiny_specialist",
                "meta": {
                    "strategy": "constant_1x1",
                    "confidence": 0.6,
                },
                "confidence": 0.6,
                "score": 0.6,
            })

    return candidates


def enforce_palette_discipline(
    candidate_grid: Grid,
    train_examples: List[Dict[str, Any]],
    test_input: Grid
) -> Dict[str, Any]:
    """Validate and fix palette issues in candidate.

    Returns metadata about palette compliance and suggested fixes.
    """
    required_colors = learn_palette_completeness(train_examples)
    candidate_colors = extract_palette(candidate_grid)
    test_colors = extract_palette(test_input)

    # Collect training input/output palettes
    train_input_palettes = []
    train_output_palettes = []
    for ex in train_examples:
        ex_in = ex.get("input", [])
        ex_out = ex.get("output", [])
        if ex_in:
            train_input_palettes.append(extract_palette(ex_in))
        if ex_out:
            train_output_palettes.append(extract_palette(ex_out))

    # Check for violations
    issues = []
    missing_required = required_colors - candidate_colors
    if missing_required:
        issues.append(f"missing_required={sorted(missing_required)}")

    # Check for hallucinated colors (not in any training output)
    all_train_colors = set()
    for pal in train_output_palettes:
        all_train_colors.update(pal)

    hallucinated = candidate_colors - all_train_colors - test_colors
    if hallucinated:
        issues.append(f"hallucinated={sorted(hallucinated)}")

    return {
        "palette_ok": len(issues) == 0,
        "issues": issues,
        "required_colors": sorted(required_colors),
        "candidate_colors": sorted(candidate_colors),
        "missing": sorted(missing_required),
        "hallucinated": sorted(hallucinated),
    }


def generate_palette_aware_candidates(
    test_input: Grid,
    train_examples: List[Dict[str, Any]],
    max_candidates: int = 3
) -> List[Dict[str, Any]]:
    """Generate candidates that respect palette constraints from training."""
    candidates = []

    # Get required palette
    required_colors = learn_palette_completeness(train_examples)
    if not required_colors:
        return candidates

    # Strategy: Start with test input and ensure all required colors appear
    candidate_grid = [row[:] for row in test_input]
    candidate_colors = extract_palette(candidate_grid)

    missing = required_colors - candidate_colors
    if missing:
        # Try to add missing colors by replacing background (0) cells
        for r, row in enumerate(candidate_grid):
            for c, cell in enumerate(row):
                if cell == 0 and missing:
                    # Replace with a missing color
                    candidate_grid[r][c] = missing.pop()
                if not missing:
                    break
            if not missing:
                break

    # Validate result
    palette_check = enforce_palette_discipline(candidate_grid, train_examples, test_input)
    if palette_check["palette_ok"]:
        candidates.append({
            "grid": candidate_grid,
            "source": "palette_specialist",
            "meta": {
                "strategy": "palette_completion",
                "palette_check": palette_check,
                "confidence": 0.65,
            },
            "confidence": 0.65,
            "score": 0.65,
        })

    return candidates[:max_candidates]
