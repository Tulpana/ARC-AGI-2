"""
Outline and Boundary Extraction Operators.
Handles edge detection, perimeter extraction, boundary tracing.
Critical for outline tracing tasks (10% of almost-matches).
"""
from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple

Grid = List[List[int]]


def extract_outline(grid: Grid, color: Optional[int] = None, thickness: int = 1) -> Grid:
    """
    Extract outline/boundary of filled regions.
    If color specified: outline of that color only
    If None: outline of all non-background regions
    """
    if not grid or not grid[0]:
        return grid

    h, w = len(grid), len(grid[0])
    outline = [[0] * w for _ in range(h)]

    for r in range(h):
        for c in range(w):
            cell = grid[r][c]

            # Skip if not target
            if color is not None and cell != color:
                continue
            if color is None and cell == 0:
                continue

            # Check if on boundary
            is_boundary = False

            # Check all neighbors (4-connected)
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc

                # Edge of grid is boundary
                if nr < 0 or nr >= h or nc < 0 or nc >= w:
                    is_boundary = True
                    break

                neighbor_cell = grid[nr][nc]

                # Different color/background is boundary
                if neighbor_cell == 0:
                    is_boundary = True
                    break

                if color is not None and neighbor_cell != color:
                    is_boundary = True
                    break

            if is_boundary:
                outline[r][c] = cell

    return outline


def extract_outer_boundary(grid: Grid) -> Grid:
    """
    Extract only the outermost boundary pixels.
    Returns grid with just the perimeter of the shape.
    """
    if not grid or not grid[0]:
        return grid

    h, w = len(grid), len(grid[0])
    boundary = [[0] * w for _ in range(h)]

    # Find all non-background pixels
    filled = {(r, c) for r in range(h) for c in range(w) if grid[r][c] != 0}

    if not filled:
        return boundary

    # For each filled pixel, check if it's on the outer edge
    for r, c in filled:
        # Check 8-connected neighbors
        has_background_neighbor = False
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue

                nr, nc = r + dr, c + dc

                # Edge of grid counts as background
                if nr < 0 or nr >= h or nc < 0 or nc >= w:
                    has_background_neighbor = True
                    break

                if grid[nr][nc] == 0:
                    has_background_neighbor = True
                    break

            if has_background_neighbor:
                break

        if has_background_neighbor:
            boundary[r][c] = grid[r][c]

    return boundary


def extract_component_outlines(grid: Grid) -> List[Grid]:
    """
    Extract outline of each connected component separately.
    Returns list of grids, each containing one component's outline.
    """
    if not grid or not grid[0]:
        return []

    components = find_components(grid)
    outlines = []

    for comp in components:
        # Create mask for this component
        h, w = len(grid), len(grid[0])
        mask = [[0] * w for _ in range(h)]

        for r, c in comp['pixels']:
            mask[r][c] = comp['color']

        # Extract outline of this component
        outline = extract_outline(mask, comp['color'])
        outlines.append(outline)

    return outlines


def find_components(grid: Grid) -> List[Dict]:
    """Find all connected components in grid."""
    if not grid or not grid[0]:
        return []

    h, w = len(grid), len(grid[0])
    visited = [[False] * w for _ in range(h)]
    components = []

    for r in range(h):
        for c in range(w):
            if visited[r][c] or grid[r][c] == 0:
                continue

            # BFS to find component
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

            components.append({
                'pixels': pixels,
                'color': color,
                'size': len(pixels)
            })

    return components


def trace_boundary_path(grid: Grid, start_pos: Optional[Tuple[int, int]] = None) -> List[Tuple[int, int]]:
    """
    Trace the boundary as an ordered path (clockwise).
    Returns list of (row, col) positions in boundary order.
    """
    if not grid or not grid[0]:
        return []

    h, w = len(grid), len(grid[0])

    # Find starting position if not provided
    if start_pos is None:
        # Find top-left most filled pixel
        for r in range(h):
            for c in range(w):
                if grid[r][c] != 0:
                    start_pos = (r, c)
                    break
            if start_pos:
                break

    if not start_pos:
        return []

    # Extract outline first
    outline_grid = extract_outline(grid)

    # Find all boundary pixels
    boundary_pixels = {(r, c) for r in range(h) for c in range(w) if outline_grid[r][c] != 0}

    if not boundary_pixels:
        return []

    # Trace path clockwise from start
    path = []
    current = start_pos if start_pos in boundary_pixels else next(iter(boundary_pixels))
    visited = {current}
    path.append(current)

    # Directions: right, down, left, up (clockwise)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    while True:
        r, c = current
        found_next = False

        # Try each direction
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (nr, nc) in boundary_pixels and (nr, nc) not in visited:
                current = (nr, nc)
                visited.add(current)
                path.append(current)
                found_next = True
                break

        if not found_next:
            # Try 8-connected
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in boundary_pixels and (nr, nc) not in visited:
                        current = (nr, nc)
                        visited.add(current)
                        path.append(current)
                        found_next = True
                        break
                if found_next:
                    break

        if not found_next or len(visited) == len(boundary_pixels):
            break

    return path


def fill_interior(grid: Grid) -> Grid:
    """
    Fill the interior of outlined shapes.
    Useful for converting outlines back to solid shapes.
    """
    if not grid or not grid[0]:
        return grid

    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Flood fill from edges to find exterior
    exterior = [[False] * w for _ in range(h)]
    stack = []

    # Add all edge pixels that are background
    for r in range(h):
        for c in range(w):
            if (r == 0 or r == h-1 or c == 0 or c == w-1) and grid[r][c] == 0:
                stack.append((r, c))
                exterior[r][c] = True

    # Flood fill exterior
    while stack:
        r, c = stack.pop()
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not exterior[nr][nc] and grid[nr][nc] == 0:
                exterior[nr][nc] = True
                stack.append((nr, nc))

    # Fill interior (non-exterior background) with color
    # Use most common non-background color
    from collections import Counter
    colors = [grid[r][c] for r in range(h) for c in range(w) if grid[r][c] != 0]
    if colors:
        fill_color = Counter(colors).most_common(1)[0][0]
    else:
        fill_color = 1

    for r in range(h):
        for c in range(w):
            if grid[r][c] == 0 and not exterior[r][c]:
                result[r][c] = fill_color

    return result


def generate_outline_candidates(grid: Grid, train_examples: List[Dict] = None) -> List[Dict]:
    """
    Generate outline-based candidate transformations.
    Returns list of candidate dicts.
    """
    candidates = []

    # Candidate 1: Extract full outline
    outline = extract_outline(grid)
    candidates.append({
        'grid': outline,
        'type': 'outline_full',
        'confidence': 0.7,
        'source': 'outline',
        'meta': {'operation': 'extract_outline'}
    })

    # Candidate 2: Extract outer boundary only
    boundary = extract_outer_boundary(grid)
    candidates.append({
        'grid': boundary,
        'type': 'outer_boundary',
        'confidence': 0.65,
        'source': 'outline',
        'meta': {'operation': 'outer_boundary'}
    })

    # Candidate 3: Outline each component separately
    comp_outlines = extract_component_outlines(grid)
    if comp_outlines:
        # Merge all component outlines
        h, w = len(grid), len(grid[0])
        merged = [[0] * w for _ in range(h)]
        for outline in comp_outlines:
            for r in range(h):
                for c in range(w):
                    if outline[r][c] != 0:
                        merged[r][c] = outline[r][c]

        candidates.append({
            'grid': merged,
            'type': 'component_outlines',
            'confidence': 0.6,
            'source': 'outline',
            'meta': {'components': len(comp_outlines)}
        })

    # Candidate 4: Fill interior (inverse operation)
    filled = fill_interior(grid)
    candidates.append({
        'grid': filled,
        'type': 'fill_interior',
        'confidence': 0.55,
        'source': 'outline',
        'meta': {'operation': 'fill_interior'}
    })

    return candidates


def detect_outline_pattern(train_examples: List[Dict]) -> bool:
    """
    Detect if task involves outline extraction.
    Returns True if pattern detected.
    """
    if not train_examples:
        return False

    outline_count = 0

    for ex in train_examples:
        inp = ex['input']
        out = ex['output']

        # Check if output looks like outline of input
        in_filled = sum(1 for row in inp for cell in row if cell != 0)
        out_filled = sum(1 for row in out for cell in row if cell != 0)

        # Outline has fewer filled cells (just edges)
        if 0 < out_filled < in_filled * 0.5:
            # Verify it's actually an outline
            expected_outline = extract_outline(inp)
            if grids_similar(expected_outline, out):
                outline_count += 1

    return outline_count >= len(train_examples) * 0.7


def grids_similar(g1: Grid, g2: Grid, threshold: float = 0.8) -> bool:
    """Check if two grids are similar (allowing some differences)."""
    if len(g1) != len(g2):
        return False
    if not g1 or len(g1[0]) != len(g2[0]):
        return False

    h, w = len(g1), len(g1[0])
    matches = 0
    total = h * w

    for r in range(h):
        for c in range(w):
            if g1[r][c] == g2[r][c]:
                matches += 1

    return (matches / total) >= threshold
