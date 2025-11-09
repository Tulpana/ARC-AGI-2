"""
Spatial Ordering and Sequencing Operators.
Handles tile ordering, spatial sorting, sequence assembly.
Critical for tiny grid tasks and 23% of failures.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

Grid = List[List[int]]


def extract_tiles(grid: Grid) -> List[Dict]:
    """
    Extract individual non-background pixels as tiles.
    Returns list of tile dicts with position, color.
    """
    if not grid or not grid[0]:
        return []

    h, w = len(grid), len(grid[0])
    tiles = []

    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0:
                tiles.append({
                    'pos': (r, c),
                    'color': grid[r][c],
                    'row': r,
                    'col': c
                })

    return tiles


def order_tiles(tiles: List[Dict], order_by: str = 'left-to-right') -> List[Dict]:
    """
    Order tiles by spatial criteria.
    Order types: 'left-to-right', 'top-to-bottom', 'by-color', 'by-distance'
    """
    if not tiles:
        return []

    if order_by == 'left-to-right':
        # Sort by column (x), then row (y)
        return sorted(tiles, key=lambda t: (t['col'], t['row']))

    elif order_by == 'top-to-bottom':
        # Sort by row (y), then column (x)
        return sorted(tiles, key=lambda t: (t['row'], t['col']))

    elif order_by == 'right-to-left':
        return sorted(tiles, key=lambda t: (-t['col'], t['row']))

    elif order_by == 'bottom-to-top':
        return sorted(tiles, key=lambda t: (-t['row'], t['col']))

    elif order_by == 'by-color':
        # Sort by color value
        return sorted(tiles, key=lambda t: t['color'])

    elif order_by == 'by-distance':
        # Sort by distance from origin (0, 0)
        return sorted(tiles, key=lambda t: t['row']**2 + t['col']**2)

    elif order_by == 'spiral':
        # Spiral order from center
        return order_spiral(tiles)

    return tiles


def order_spiral(tiles: List[Dict]) -> List[Dict]:
    """Order tiles in spiral pattern from outside to inside."""
    if not tiles:
        return []

    # Group by distance from center
    if tiles:
        avg_r = sum(t['row'] for t in tiles) / len(tiles)
        avg_c = sum(t['col'] for t in tiles) / len(tiles)

        return sorted(tiles, key=lambda t: (
            abs(t['row'] - avg_r) + abs(t['col'] - avg_c),
            t['row'],
            t['col']
        ))

    return tiles


def assemble_sequence(tiles: List[Dict], layout: str = 'horizontal', spacing: int = 0) -> Grid:
    """
    Assemble ordered tiles into a sequence.
    Layouts: 'horizontal' (1xN), 'vertical' (Nx1), 'diagonal'
    """
    if not tiles:
        return [[0]]

    if layout == 'horizontal':
        # Create 1 x N grid
        w = len(tiles) * (1 + spacing)
        result = [[0] * w]

        for i, tile in enumerate(tiles):
            result[0][i * (1 + spacing)] = tile['color']

        return result

    elif layout == 'vertical':
        # Create N x 1 grid
        h = len(tiles) * (1 + spacing)
        result = [[0] for _ in range(h)]

        for i, tile in enumerate(tiles):
            result[i * (1 + spacing)][0] = tile['color']

        return result

    elif layout == 'diagonal':
        # Create diagonal sequence
        size = len(tiles)
        result = [[0] * size for _ in range(size)]

        for i, tile in enumerate(tiles):
            if i < size:
                result[i][i] = tile['color']

        return result

    elif layout == 'compact':
        # Find minimal grid that fits all tiles in their relative positions
        if not tiles:
            return [[0]]

        min_r = min(t['row'] for t in tiles)
        max_r = max(t['row'] for t in tiles)
        min_c = min(t['col'] for t in tiles)
        max_c = max(t['col'] for t in tiles)

        h = max_r - min_r + 1
        w = max_c - min_c + 1
        result = [[0] * w for _ in range(h)]

        for tile in tiles:
            r, c = tile['row'] - min_r, tile['col'] - min_c
            result[r][c] = tile['color']

        return result

    return [[0]]


def detect_ordering_pattern(train_examples: List[Dict]) -> Optional[str]:
    """
    Detect if task involves spatial ordering/sequencing.
    Returns: ordering type or None
    """
    if not train_examples:
        return None

    # Check if outputs are sequences (1xN or Nx1)
    sequence_count = 0
    horizontal_count = 0
    vertical_count = 0

    for ex in train_examples:
        out = ex['output']
        h, w = len(out), len(out[0]) if out else 0

        # Check if output is a sequence
        if h == 1 and w > 1:
            sequence_count += 1
            horizontal_count += 1
        elif w == 1 and h > 1:
            sequence_count += 1
            vertical_count += 1

    n = len(train_examples)
    if sequence_count >= n * 0.7:
        if horizontal_count > vertical_count:
            return 'horizontal'
        else:
            return 'vertical'

    return None


def collect_and_sequence(grid: Grid, order_by: str = 'left-to-right', layout: str = 'horizontal') -> Grid:
    """
    Extract all tiles, order them, and assemble into sequence.
    One-shot operation for "collect scattered tiles and arrange" tasks.
    """
    tiles = extract_tiles(grid)
    if not tiles:
        return [[0]]

    ordered = order_tiles(tiles, order_by)
    result = assemble_sequence(ordered, layout)

    return result


def generate_ordering_candidates(grid: Grid, train_examples: List[Dict] = None) -> List[Dict]:
    """
    Generate ordering/sequencing candidate transformations.
    Returns list of candidate dicts.
    """
    candidates = []

    # Extract tiles
    tiles = extract_tiles(grid)
    if not tiles:
        return candidates

    # Detect pattern from training
    detected_pattern = None
    if train_examples:
        detected_pattern = detect_ordering_pattern(train_examples)

    # Generate candidates for different orderings
    orderings = ['left-to-right', 'top-to-bottom', 'by-color']

    if detected_pattern:
        # Prioritize detected pattern
        orderings.insert(0, detected_pattern)

    for order in orderings:
        ordered = order_tiles(tiles, order)

        # Try different layouts
        layouts = ['horizontal', 'vertical'] if not detected_pattern else [detected_pattern]

        for layout in layouts:
            result = assemble_sequence(ordered, layout)
            candidates.append({
                'grid': result,
                'type': f'sequence_{order}_{layout}',
                'confidence': 0.7 if order == detected_pattern else 0.5,
                'source': 'ordering',
                'meta': {'order': order, 'layout': layout, 'tiles': len(tiles)}
            })

    # Also try compact assembly (preserve relative positions)
    compact = assemble_sequence(tiles, 'compact')
    candidates.append({
        'grid': compact,
        'type': 'compact_assembly',
        'confidence': 0.6,
        'source': 'ordering',
        'meta': {'tiles': len(tiles)}
    })

    return candidates


def extract_unique_colors_ordered(grid: Grid, order_by: str = 'first-appearance') -> List[int]:
    """
    Extract unique colors in specific order.
    Order types: 'first-appearance', 'frequency', 'value'
    """
    if not grid or not grid[0]:
        return []

    h, w = len(grid), len(grid[0])

    if order_by == 'first-appearance':
        seen = []
        for r in range(h):
            for c in range(w):
                if grid[r][c] != 0 and grid[r][c] not in seen:
                    seen.append(grid[r][c])
        return seen

    elif order_by == 'frequency':
        from collections import Counter
        colors = [grid[r][c] for r in range(h) for c in range(w) if grid[r][c] != 0]
        counts = Counter(colors)
        return [color for color, _ in counts.most_common()]

    elif order_by == 'value':
        colors = {grid[r][c] for r in range(h) for c in range(w) if grid[r][c] != 0}
        return sorted(colors)

    return []


def create_color_sequence(colors: List[int], layout: str = 'horizontal') -> Grid:
    """
    Create a grid sequence from list of colors.
    Useful for "collect all colors and arrange" tasks.
    """
    if not colors:
        return [[0]]

    if layout == 'horizontal':
        return [colors]

    elif layout == 'vertical':
        return [[c] for c in colors]

    elif layout == 'diagonal':
        n = len(colors)
        result = [[0] * n for _ in range(n)]
        for i, color in enumerate(colors):
            result[i][i] = color
        return result

    return [[0]]
