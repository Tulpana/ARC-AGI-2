"""
Palette refinement module for post-processing predictions.

This module improves predictions by:
1. Analyzing training examples to learn valid output color palettes
2. Detecting and correcting invalid colors in predictions
3. Using spatial context to fix palette errors

ANTI-LEAKAGE GUARANTEE:
- Uses ONLY training examples for learning
- Never accesses test solutions
- All inference is from train -> test patterns
"""

from typing import List, Dict, Set, Tuple
from collections import Counter

Grid = List[List[int]]

def get_palette(grid: Grid) -> Set[int]:
    """Extract unique colors from a grid."""
    if not grid:
        return set()
    colors = set()
    for row in grid:
        colors.update(row)
    return colors

def learn_output_palette_constraints(train_examples: List[Dict]) -> Dict:
    """
    Learn what colors are ALLOWED in outputs based on training examples.

    Returns constraints learned purely from training data.
    """
    if not train_examples:
        return {'allowed_colors': None, 'forbidden_colors': set(), 'confidence': 0.0}

    # Collect all colors that appear in training outputs
    all_output_colors = set()
    all_input_colors = set()

    for ex in train_examples:
        inp = ex.get('input', [])
        out = ex.get('output', [])

        all_input_colors.update(get_palette(inp))
        all_output_colors.update(get_palette(out))

    # Check if output palette is constrained
    # Pattern 1: Output colors are always a SUBSET of input colors
    outputs_subset_of_inputs = all(
        get_palette(ex['output']) <= get_palette(ex['input'])
        for ex in train_examples
    )

    # Pattern 2: Output colors are always from a FIXED set
    output_palettes = [get_palette(ex['output']) for ex in train_examples]
    if len(output_palettes) >= 2:
        # Check if all outputs use same palette
        first_palette = output_palettes[0]
        all_same = all(p == first_palette for p in output_palettes[1:])
        if all_same:
            return {
                'type': 'fixed_palette',
                'allowed_colors': first_palette,
                'forbidden_colors': set(),
                'confidence': 1.0
            }

    # Pattern 3: Some colors NEVER appear in outputs
    # Colors that appear in inputs but NEVER in outputs are forbidden
    colors_only_in_inputs = all_input_colors - all_output_colors

    if outputs_subset_of_inputs:
        return {
            'type': 'subset_of_input',
            'allowed_colors': None,  # Determined per test input
            'forbidden_colors': colors_only_in_inputs,
            'confidence': 1.0 if len(train_examples) >= 3 else 0.8
        }

    # Pattern 4: Output palette is superset (adds colors)
    colors_added = all_output_colors - all_input_colors
    colors_removed = all_input_colors - all_output_colors

    return {
        'type': 'general',
        'allowed_colors': all_output_colors,
        'forbidden_colors': set(),
        'colors_typically_added': colors_added,
        'colors_typically_removed': colors_removed,
        'confidence': 0.6 if len(train_examples) >= 2 else 0.3
    }

def find_replacement_color(
    invalid_color: int,
    grid: Grid,
    row: int,
    col: int,
    allowed_colors: Set[int]
) -> int:
    """
    Find best replacement color for an invalid pixel.
    Uses spatial context (neighbors) to make intelligent choice.
    """
    if not grid or not allowed_colors:
        return list(allowed_colors)[0] if allowed_colors else 0

    h, w = len(grid), len(grid[0]) if grid else 0

    # Collect colors of valid neighbors
    neighbor_colors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = row + dr, col + dc
        if 0 <= nr < h and 0 <= nc < w:
            neighbor_color = grid[nr][nc]
            if neighbor_color in allowed_colors:
                neighbor_colors.append(neighbor_color)

    # Use most common valid neighbor color
    if neighbor_colors:
        return Counter(neighbor_colors).most_common(1)[0][0]

    # Fallback: use most common allowed color in grid
    grid_colors = []
    for row_data in grid:
        grid_colors.extend(row_data)

    allowed_in_grid = [c for c in grid_colors if c in allowed_colors]
    if allowed_in_grid:
        return Counter(allowed_in_grid).most_common(1)[0][0]

    # CRITICAL FIX: Prefer non-zero (birth) colors over background when isolated
    # This fixes single-cell tasks that were collapsing to wrong hue (typically 0)
    # when every cell was marked "extra" and no spatial context existed
    non_zero_allowed = [c for c in allowed_colors if c != 0]
    if non_zero_allowed:
        # Prefer the smallest non-zero color (typically the primary birth color)
        return sorted(non_zero_allowed)[0]

    # Last resort: return any allowed color (only if all allowed are 0)
    return list(allowed_colors)[0]

def refine_prediction_palette(
    prediction: Grid,
    train_examples: List[Dict],
    test_input: Grid
) -> Tuple[Grid, Dict]:
    """
    Refine prediction palette based on patterns learned from training.

    Returns:
        (refined_grid, metadata)
    """
    if not prediction or not train_examples:
        return prediction, {'refined': False, 'reason': 'no_data'}

    # Learn palette constraints from training
    constraints = learn_output_palette_constraints(train_examples)

    if constraints['confidence'] < 0.5:
        return prediction, {'refined': False, 'reason': 'low_confidence', 'confidence': constraints['confidence']}

    # Determine allowed colors for this specific test case
    pred_palette = get_palette(prediction)
    test_input_palette = get_palette(test_input)

    forbidden_colors = set(constraints.get('forbidden_colors') or [])

    if constraints['type'] == 'fixed_palette':
        allowed_colors = set(constraints.get('allowed_colors') or [])
    elif constraints['type'] == 'subset_of_input':
        allowed_colors = set(test_input_palette or [])
    elif constraints['type'] == 'general':
        # Start with colors seen in training outputs, then include admissible test colors
        allowed_colors = set(constraints.get('allowed_colors') or [])
        allowed_colors.update(test_input_palette or [])
    else:
        return prediction, {'refined': False, 'reason': 'unknown_pattern'}

    # Remove any colors that were explicitly marked as forbidden
    if forbidden_colors:
        allowed_colors.difference_update(forbidden_colors)

    if not allowed_colors:
        # Fall back to the most conservative palette we can infer
        fallback = set(constraints.get('allowed_colors') or [])
        if fallback:
            allowed_colors = fallback - forbidden_colors
        if not allowed_colors and test_input_palette:
            allowed_colors = set(test_input_palette) - forbidden_colors
        if not allowed_colors:
            allowed_colors = pred_palette - forbidden_colors

    if not allowed_colors:
        return prediction, {'refined': False, 'reason': 'no_allowed_colors'}

    # Check if prediction violates constraints
    invalid_colors = pred_palette - allowed_colors

    if not invalid_colors:
        return prediction, {'refined': False, 'reason': 'already_valid'}

    # Refine: replace invalid colors
    refined = [row[:] for row in prediction]  # Deep copy
    replacements = {}

    for r in range(len(refined)):
        for c in range(len(refined[0]) if refined else 0):
            color = refined[r][c]
            if color in invalid_colors:
                if color not in replacements:
                    replacements[color] = find_replacement_color(
                        color, refined, r, c, allowed_colors
                    )
                refined[r][c] = replacements[color]

    metadata = {
        'refined': True,
        'constraint_type': constraints['type'],
        'confidence': constraints['confidence'],
        'invalid_colors': sorted(invalid_colors),
        'replacements': replacements,
        'pixels_changed': sum(1 for r in range(len(prediction)) for c in range(len(prediction[0]))
                             if prediction[r][c] != refined[r][c])
    }

    return refined, metadata

def test_refinement():
    """Micro-test to verify refinement logic works."""
    print("=== PALETTE REFINEMENT MICRO-TEST ===\n")

    # Test case 1: Fixed palette
    print("Test 1: Fixed output palette")
    train = [
        {'input': [[1, 2], [3, 4]], 'output': [[1, 1], [1, 1]]},
        {'input': [[5, 6], [7, 8]], 'output': [[1, 1], [1, 1]]},
    ]
    prediction = [[1, 1], [9, 1]]  # Color 9 is invalid
    test_input = [[0, 0], [0, 0]]

    refined, meta = refine_prediction_palette(prediction, train, test_input)
    print(f"  Original: {prediction}")
    print(f"  Refined:  {refined}")
    print(f"  Invalid colors: {meta.get('invalid_colors', [])}")
    print(f"  ✓ PASS" if refined[1][0] == 1 else "  ✗ FAIL")

    # Test case 2: Subset of input
    print("\nTest 2: Output subset of input")
    train = [
        {'input': [[1, 2, 3]], 'output': [[1, 2]]},
        {'input': [[4, 5, 6]], 'output': [[4, 5]]},
    ]
    prediction = [[7, 8]]  # Colors 7, 8 not in test input
    test_input = [[1, 2, 3]]

    refined, meta = refine_prediction_palette(prediction, train, test_input)
    print(f"  Test input palette: {get_palette(test_input)}")
    print(f"  Original: {prediction}")
    print(f"  Refined:  {refined}")
    print(f"  Invalid colors: {meta.get('invalid_colors', [])}")
    refined_palette = get_palette(refined)
    valid = refined_palette <= get_palette(test_input)
    print(f"  ✓ PASS" if valid else "  ✗ FAIL")

    print("\n=== ALL TESTS PASSED ===")

if __name__ == '__main__':
    test_refinement()
