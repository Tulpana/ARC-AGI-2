"""
Tile Assembly and Spatial Packing Operators.
Handles scatter→assemble, gravity, packing transformations.
Critical for 24% of failures (tile collection tasks).
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

Grid = List[List[int]]


def apply_gravity(grid: Grid, direction: str = 'down') -> Grid:
    """
    Apply gravity to move all non-background tiles in direction.
    Simulates tiles falling, packing, consolidation.
    """
    if not grid or not grid[0]:
        return grid

    h, w = len(grid), len(grid[0])
    result = [[0] * w for _ in range(h)]

    if direction == 'down':
        # Process each column
        for c in range(w):
            # Collect all non-zero cells
            tiles = [grid[r][c] for r in range(h) if grid[r][c] != 0]
            # Place them at bottom
            for i, tile in enumerate(tiles):
                result[h - len(tiles) + i][c] = tile

    elif direction == 'up':
        for c in range(w):
            tiles = [grid[r][c] for r in range(h) if grid[r][c] != 0]
            for i, tile in enumerate(tiles):
                result[i][c] = tile

    elif direction == 'left':
        for r in range(h):
            tiles = [grid[r][c] for c in range(w) if grid[r][c] != 0]
            for i, tile in enumerate(tiles):
                result[r][i] = tile

    elif direction == 'right':
        for r in range(h):
            tiles = [grid[r][c] for c in range(w) if grid[r][c] != 0]
            for i, tile in enumerate(tiles):
                result[r][w - len(tiles) + i] = tile

    return result


def pack_components_tight(grid: Grid) -> Grid:
    """
    Pack all non-background cells with minimal bounding box.
    Removes empty space, consolidates scattered tiles.
    """
    if not grid or not grid[0]:
        return [[]]

    h, w = len(grid), len(grid[0])

    # Find all non-background pixels
    pixels = [(r, c, grid[r][c]) for r in range(h) for c in range(w) if grid[r][c] != 0]

    if not pixels:
        return [[0]]

    # Find bounding box
    min_r = min(r for r, c, _ in pixels)
    max_r = max(r for r, c, _ in pixels)
    min_c = min(c for r, c, _ in pixels)
    max_c = max(c for r, c, _ in pixels)

    # Create packed grid
    new_h = max_r - min_r + 1
    new_w = max_c - min_c + 1
    result = [[0] * new_w for _ in range(new_h)]

    for r, c, color in pixels:
        new_r, new_c = r - min_r, c - min_c
        result[new_r][new_c] = color

    return result


def extract_components(grid: Grid) -> List[Dict]:
    """
    Extract connected components as separate entities.
    Returns list of component dicts with pixels, color, bounds.
    """
    if not grid or not grid[0]:
        return []

    h, w = len(grid), len(grid[0])
    visited = [[False] * w for _ in range(h)]
    components = []

    for r in range(h):
        for c in range(w):
            if visited[r][c] or grid[r][c] == 0:
                continue

            # BFS to find connected component
            color = grid[r][c]
            pixels = []
            stack = [(r, c)]
            visited[r][c] = True

            while stack:
                cr, cc = stack.pop()
                pixels.append((cr, cc))

                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and grid[nr][nc] == color:
                        visited[nr][nc] = True
                        stack.append((nr, nc))

            # Calculate bounds
            min_r = min(r for r, c in pixels)
            max_r = max(r for r, c in pixels)
            min_c = min(c for r, c in pixels)
            max_c = max(c for r, c in pixels)

            components.append({
                'pixels': pixels,
                'color': color,
                'bounds': (min_r, min_c, max_r, max_c),
                'height': max_r - min_r + 1,
                'width': max_c - min_c + 1,
                'size': len(pixels)
            })

    return components


def assemble_components(components: List[Dict], layout: str = 'horizontal') -> Grid:
    """
    Assemble components into a single grid with specified layout.
    Layouts: 'horizontal', 'vertical', 'grid', 'tight'
    """
    if not components:
        return [[]]

    if layout == 'tight':
        # Pack all components with minimal bounding box
        all_pixels = []
        for comp in components:
            for r, c in comp['pixels']:
                all_pixels.append((r, c, comp['color']))

        if not all_pixels:
            return [[0]]

        min_r = min(r for r, c, _ in all_pixels)
        max_r = max(r for r, c, _ in all_pixels)
        min_c = min(c for r, c, _ in all_pixels)
        max_c = max(c for r, c, _ in all_pixels)

        h = max_r - min_r + 1
        w = max_c - min_c + 1
        result = [[0] * w for _ in range(h)]

        for r, c, color in all_pixels:
            result[r - min_r][c - min_c] = color

        return result

    elif layout == 'horizontal':
        # Place components left-to-right
        max_h = max(comp['height'] for comp in components)
        total_w = sum(comp['width'] for comp in components)
        result = [[0] * total_w for _ in range(max_h)]

        x_offset = 0
        for comp in components:
            min_r, min_c, max_r, max_c = comp['bounds']
            for r, c in comp['pixels']:
                local_r = r - min_r
                local_c = c - min_c
                if local_r < max_h and x_offset + local_c < total_w:
                    result[local_r][x_offset + local_c] = comp['color']
            x_offset += comp['width']

        return result

    elif layout == 'vertical':
        # Place components top-to-bottom
        max_w = max(comp['width'] for comp in components)
        total_h = sum(comp['height'] for comp in components)
        result = [[0] * max_w for _ in range(total_h)]

        y_offset = 0
        for comp in components:
            min_r, min_c, max_r, max_c = comp['bounds']
            for r, c in comp['pixels']:
                local_r = r - min_r
                local_c = c - min_c
                if y_offset + local_r < total_h and local_c < max_w:
                    result[y_offset + local_r][local_c] = comp['color']
            y_offset += comp['height']

        return result

    return [[]]


def compress_grid(grid: Grid, direction: str = 'both') -> Grid:
    """
    Remove empty rows/columns from grid.
    Direction: 'horizontal', 'vertical', 'both'
    """
    if not grid or not grid[0]:
        return grid

    h, w = len(grid), len(grid[0])

    # Find non-empty rows
    non_empty_rows = [r for r in range(h) if any(grid[r][c] != 0 for c in range(w))]

    # Find non-empty columns
    non_empty_cols = [c for c in range(w) if any(grid[r][c] != 0 for r in range(h))]

    if not non_empty_rows or not non_empty_cols:
        return [[0]]

    if direction == 'horizontal' or direction == 'both':
        grid = [[grid[r][c] for c in non_empty_cols] for r in range(h)]

    if direction == 'vertical' or direction == 'both':
        grid = [grid[r] for r in non_empty_rows]

    return grid


def generate_assembly_candidates(grid: Grid, train_examples: List[Dict] = None) -> List[Dict]:
    """
    Generate assembly-based candidate transformations.
    Returns list of candidate dicts with grid, type, confidence.
    """
    candidates = []

    # Candidate 1-4: Gravity in all directions
    for direction in ['down', 'up', 'left', 'right']:
        result = apply_gravity(grid, direction)
        candidates.append({
            'grid': result,
            'type': f'gravity_{direction}',
            'confidence': 0.7,
            'source': 'assembly',
            'meta': {'direction': direction}
        })

    # Candidate 5: Tight packing
    packed = pack_components_tight(grid)
    candidates.append({
        'grid': packed,
        'type': 'tight_pack',
        'confidence': 0.6,
        'source': 'assembly',
        'meta': {'operation': 'tight_pack'}
    })

    # Candidate 6: Compress (remove empty space)
    compressed = compress_grid(grid, 'both')
    candidates.append({
        'grid': compressed,
        'type': 'compress',
        'confidence': 0.65,
        'source': 'assembly',
        'meta': {'operation': 'compress'}
    })

    # Candidate 7-8: Assemble components horizontally/vertically
    comps = extract_components(grid)
    if comps:
        h_assembled = assemble_components(comps, 'horizontal')
        candidates.append({
            'grid': h_assembled,
            'type': 'assemble_horizontal',
            'confidence': 0.5,
            'source': 'assembly',
            'meta': {'layout': 'horizontal', 'components': len(comps)}
        })

        v_assembled = assemble_components(comps, 'vertical')
        candidates.append({
            'grid': v_assembled,
            'type': 'assemble_vertical',
            'confidence': 0.5,
            'source': 'assembly',
            'meta': {'layout': 'vertical', 'components': len(comps)}
        })

    return candidates


def detect_assembly_pattern(train_examples: List[Dict]) -> Optional[str]:
    """
    Detect if training examples show assembly/packing pattern.
    Returns: direction ('down', 'left', etc.) or layout ('tight', 'horizontal')
    """
    if not train_examples:
        return None

    gravity_votes = {'down': 0, 'up': 0, 'left': 0, 'right': 0}
    tight_pack_count = 0

    for ex in train_examples:
        inp = ex['input']
        out = ex['output']

        # Check if output is gravity-applied version of input
        for direction in ['down', 'up', 'left', 'right']:
            if grids_equal(apply_gravity(inp, direction), out):
                gravity_votes[direction] += 1

        # Check if output is tight-packed version
        if grids_equal(pack_components_tight(inp), out):
            tight_pack_count += 1

    # Consensus?
    max_gravity = max(gravity_votes.values())
    if max_gravity >= len(train_examples) * 0.7:  # 70% consensus
        for direction, count in gravity_votes.items():
            if count == max_gravity:
                return direction

    if tight_pack_count >= len(train_examples) * 0.7:
        return 'tight'

    return None


def grids_equal(g1: Grid, g2: Grid) -> bool:
    """Check if two grids are identical."""
    if len(g1) != len(g2):
        return False
    if not g1:
        return not g2
    if g1 and len(g1[0]) != len(g2[0]):
        return False

    for r in range(len(g1)):
        for c in range(len(g1[0]) if g1 else 0):
            if g1[r][c] != g2[r][c]:
                return False
    return True
