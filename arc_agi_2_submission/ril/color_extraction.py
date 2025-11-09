"""
Color extraction and aggregation patterns for ARC-AGI.
Critical for tasks that extract specific colors from input grids.
Stdlib-only, Kaggle-compatible.
"""
from __future__ import annotations
from typing import List, Dict, Tuple, Optional
from collections import Counter

Grid = List[List[int]]
Example = Dict[str, Grid]


def get_mode_color(grid: Grid, ignore_background: bool = True) -> int:
    """
    Get the most common color in grid.

    Args:
        grid: Input grid
        ignore_background: If True, ignore color 0

    Returns:
        Most common color value
    """
    if not grid or not grid[0]:
        return 0

    colors = []
    for row in grid:
        colors.extend(row)

    if ignore_background:
        colors = [c for c in colors if c != 0]

    if not colors:
        return 0

    counter = Counter(colors)
    return counter.most_common(1)[0][0]


def get_median_color(grid: Grid, ignore_background: bool = True) -> int:
    """
    Get the median color value in grid.

    Args:
        grid: Input grid
        ignore_background: If True, ignore color 0

    Returns:
        Median color value
    """
    if not grid or not grid[0]:
        return 0

    colors = []
    for row in grid:
        colors.extend(row)

    if ignore_background:
        colors = [c for c in colors if c != 0]

    if not colors:
        return 0

    colors.sort()
    mid = len(colors) // 2
    return colors[mid]


def get_rarest_color(grid: Grid, ignore_background: bool = True) -> int:
    """
    Get the least common color in grid.

    Args:
        grid: Input grid
        ignore_background: If True, ignore color 0

    Returns:
        Rarest color value
    """
    if not grid or not grid[0]:
        return 0

    colors = []
    for row in grid:
        colors.extend(row)

    if ignore_background:
        colors = [c for c in colors if c != 0]

    if not colors:
        return 0

    counter = Counter(colors)
    return counter.most_common()[-1][0]


def get_color_at_position(grid: Grid, row: int, col: int) -> int:
    """
    Get color at specific position.

    Args:
        grid: Input grid
        row: Row index
        col: Column index

    Returns:
        Color at position, or 0 if out of bounds
    """
    if not grid or row < 0 or col < 0:
        return 0
    if row >= len(grid) or col >= len(grid[0]):
        return 0
    return grid[row][col]


def get_color_at_corner(grid: Grid, corner: str) -> int:
    """
    Get color at corner position.

    Args:
        grid: Input grid
        corner: One of 'topleft', 'topright', 'bottomleft', 'bottomright'

    Returns:
        Color at corner
    """
    if not grid or not grid[0]:
        return 0

    h = len(grid)
    w = len(grid[0])

    if corner == 'topleft':
        return grid[0][0]
    elif corner == 'topright':
        return grid[0][w-1]
    elif corner == 'bottomleft':
        return grid[h-1][0]
    elif corner == 'bottomright':
        return grid[h-1][w-1]
    else:
        return grid[0][0]


def get_color_at_center(grid: Grid) -> int:
    """
    Get color at center of grid.

    Args:
        grid: Input grid

    Returns:
        Color at center
    """
    if not grid or not grid[0]:
        return 0

    h = len(grid)
    w = len(grid[0])

    return grid[h//2][w//2]


def detect_color_extraction_pattern(train_examples: List[Example]) -> Optional[Dict]:
    """
    Detect if training examples show a color extraction pattern.

    Returns pattern dict with:
    - type: 'mode', 'median', 'rarest', 'corner', 'center', 'specific_position'
    - confidence: 0.0-1.0
    - params: additional parameters
    """
    if not train_examples:
        return None

    # Check if all outputs are small (1x1, 1xN, Nx1)
    small_outputs = all(
        len(ex['output']) == 1 or len(ex['output'][0]) == 1
        for ex in train_examples
    )

    if not small_outputs:
        return None

    # Try mode color
    mode_matches = 0
    for ex in train_examples:
        mode_color = get_mode_color(ex['input'])
        output_color = ex['output'][0][0] if len(ex['output']) == 1 else ex['output'][0][0]
        if mode_color == output_color:
            mode_matches += 1

    if mode_matches == len(train_examples):
        return {'type': 'mode', 'confidence': 1.0, 'ignore_bg': True}

    # Try median color
    median_matches = 0
    for ex in train_examples:
        median_color = get_median_color(ex['input'])
        output_color = ex['output'][0][0]
        if median_color == output_color:
            median_matches += 1

    if median_matches == len(train_examples):
        return {'type': 'median', 'confidence': 1.0, 'ignore_bg': True}

    # Try rarest color
    rarest_matches = 0
    for ex in train_examples:
        rarest_color = get_rarest_color(ex['input'])
        output_color = ex['output'][0][0]
        if rarest_color == output_color:
            rarest_matches += 1

    if rarest_matches == len(train_examples):
        return {'type': 'rarest', 'confidence': 1.0, 'ignore_bg': True}

    # Try corner positions
    for corner in ['topleft', 'topright', 'bottomleft', 'bottomright']:
        corner_matches = 0
        for ex in train_examples:
            corner_color = get_color_at_corner(ex['input'], corner)
            output_color = ex['output'][0][0]
            if corner_color == output_color:
                corner_matches += 1

        if corner_matches == len(train_examples):
            return {'type': 'corner', 'confidence': 1.0, 'corner': corner}

    # Try center
    center_matches = 0
    for ex in train_examples:
        center_color = get_color_at_center(ex['input'])
        output_color = ex['output'][0][0]
        if center_color == output_color:
            center_matches += 1

    if center_matches == len(train_examples):
        return {'type': 'center', 'confidence': 1.0}

    return None


def apply_color_extraction(grid: Grid, pattern: Dict) -> Grid:
    """
    Apply color extraction pattern to grid.

    Args:
        grid: Input grid
        pattern: Pattern dict from detect_color_extraction_pattern

    Returns:
        1x1 grid with extracted color
    """
    pattern_type = pattern.get('type')

    if pattern_type == 'mode':
        color = get_mode_color(grid, pattern.get('ignore_bg', True))
    elif pattern_type == 'median':
        color = get_median_color(grid, pattern.get('ignore_bg', True))
    elif pattern_type == 'rarest':
        color = get_rarest_color(grid, pattern.get('ignore_bg', True))
    elif pattern_type == 'corner':
        color = get_color_at_corner(grid, pattern.get('corner', 'topleft'))
    elif pattern_type == 'center':
        color = get_color_at_center(grid)
    else:
        color = 0

    return [[color]]


def generate_color_extraction_candidates(
    test_input: Grid,
    train_examples: List[Example] = None
) -> List[Dict]:
    """
    Generate color extraction candidates.

    Returns list of candidate dicts with:
    - grid: Output grid (1x1)
    - type: Candidate type
    - confidence: Confidence score
    - source: 'color_extraction'
    """
    candidates = []

    # If we have training examples, detect pattern
    if train_examples:
        pattern = detect_color_extraction_pattern(train_examples)
        if pattern:
            extracted_grid = apply_color_extraction(test_input, pattern)
            # CRITICAL: High confidence (0.98) when pattern detected with 100% consistency
            # This ensures pattern-based candidates beat learned color maps (0.70-0.95)
            confidence = 0.98 if pattern['confidence'] == 1.0 else 0.85
            candidates.append({
                'grid': extracted_grid,
                'type': f"color_extract_{pattern['type']}",
                'confidence': confidence,
                'source': 'color_extraction_pattern',  # Different source to track
                'meta': {'pattern': pattern, 'pattern_confidence': pattern['confidence']}
            })
            return candidates  # If pattern detected, only return that

    # Otherwise, try all extraction methods speculatively

    # Mode color (most common)
    mode_color = get_mode_color(test_input, ignore_background=True)
    candidates.append({
        'grid': [[mode_color]],
        'type': 'color_extract_mode',
        'confidence': 0.55,
        'source': 'color_extraction',
        'meta': {'method': 'mode', 'ignore_bg': True}
    })

    # Mode color including background
    mode_color_with_bg = get_mode_color(test_input, ignore_background=False)
    if mode_color_with_bg != mode_color:
        candidates.append({
            'grid': [[mode_color_with_bg]],
            'type': 'color_extract_mode_with_bg',
            'confidence': 0.45,
            'source': 'color_extraction',
            'meta': {'method': 'mode', 'ignore_bg': False}
        })

    # Median color
    median_color = get_median_color(test_input, ignore_background=True)
    candidates.append({
        'grid': [[median_color]],
        'type': 'color_extract_median',
        'confidence': 0.50,
        'source': 'color_extraction',
        'meta': {'method': 'median'}
    })

    # Rarest color
    rarest_color = get_rarest_color(test_input, ignore_background=True)
    candidates.append({
        'grid': [[rarest_color]],
        'type': 'color_extract_rarest',
        'confidence': 0.50,
        'source': 'color_extraction',
        'meta': {'method': 'rarest'}
    })

    # Corner colors
    for corner in ['topleft', 'topright', 'bottomleft', 'bottomright']:
        corner_color = get_color_at_corner(test_input, corner)
        candidates.append({
            'grid': [[corner_color]],
            'type': f'color_extract_corner_{corner}',
            'confidence': 0.48,
            'source': 'color_extraction',
            'meta': {'method': 'corner', 'corner': corner}
        })

    # Center color
    center_color = get_color_at_center(test_input)
    candidates.append({
        'grid': [[center_color]],
        'type': 'color_extract_center',
        'confidence': 0.52,
        'source': 'color_extraction',
        'meta': {'method': 'center'}
    })

    return candidates
