"""
Diagonal Pattern Detection and Operations.
Handles diagonal lines, directional movement, diagonal fills.
Critical for 50% of almost-matches.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

Grid = List[List[int]]


def detect_diagonal_line(grid: Grid) -> Optional[Tuple[str, int]]:
    """
    Detect if grid contains a diagonal line pattern.
    Returns: (direction, color) or None
    Direction: 'main' (top-left to bottom-right) or 'anti' (top-right to bottom-left)
    """
    if not grid or not grid[0]:
        return None

    h, w = len(grid), len(grid[0])

    if h < 3 or w < 3:
        return None  # Too small for meaningful diagonal

    # Main diagonal (top-left to bottom-right)
    main_diag_pixels = [(r, r) for r in range(min(h, w))]
    main_diag_colors = [grid[r][c] for r, c in main_diag_pixels if r < h and c < w]

    # Check if it's a consistent non-background diagonal
    if main_diag_colors and len(set(main_diag_colors)) == 1 and main_diag_colors[0] != 0:
        return ('main', main_diag_colors[0])

    # Check if most diagonal pixels are same color (allowing some noise)
    if main_diag_colors:
        from collections import Counter
        color_counts = Counter(main_diag_colors)
        most_common = color_counts.most_common(1)[0]
        if most_common[1] >= len(main_diag_colors) * 0.8 and most_common[0] != 0:
            return ('main', most_common[0])

    # Anti-diagonal (top-right to bottom-left)
    anti_diag_pixels = [(r, w - 1 - r) for r in range(min(h, w))]
    anti_diag_colors = [grid[r][c] for r, c in anti_diag_pixels if r < h and c < w]

    if anti_diag_colors and len(set(anti_diag_colors)) == 1 and anti_diag_colors[0] != 0:
        return ('anti', anti_diag_colors[0])

    if anti_diag_colors:
        from collections import Counter
        color_counts = Counter(anti_diag_colors)
        most_common = color_counts.most_common(1)[0]
        if most_common[1] >= len(anti_diag_colors) * 0.8 and most_common[0] != 0:
            return ('anti', most_common[0])

    return None


def fill_diagonal(grid: Grid, color: int, direction: str = 'main') -> Grid:
    """
    Fill diagonal with specified color.
    Direction: 'main' or 'anti'
    """
    if not grid or not grid[0]:
        return grid

    result = [row[:] for row in grid]
    h, w = len(grid), len(grid[0])

    if direction == 'main':
        for i in range(min(h, w)):
            result[i][i] = color
    elif direction == 'anti':
        for i in range(min(h, w)):
            result[i][w - 1 - i] = color

    return result


def extract_diagonal(grid: Grid, direction: str = 'main') -> List[int]:
    """
    Extract values along diagonal.
    Returns list of colors along the diagonal.
    """
    if not grid or not grid[0]:
        return []

    h, w = len(grid), len(grid[0])

    if direction == 'main':
        return [grid[i][i] for i in range(min(h, w))]
    elif direction == 'anti':
        return [grid[i][w - 1 - i] for i in range(min(h, w))]

    return []


def reflect_over_diagonal(grid: Grid, direction: str = 'main') -> Grid:
    """
    Reflect grid over diagonal (transpose operation).
    Direction: 'main' (standard transpose) or 'anti' (anti-diagonal transpose)
    """
    if not grid or not grid[0]:
        return grid

    h, w = len(grid), len(grid[0])

    if direction == 'main':
        # Standard transpose
        result = [[grid[r][c] for r in range(h)] for c in range(w)]
        return result
    elif direction == 'anti':
        # Anti-diagonal transpose (flip both dimensions)
        result = [[grid[h - 1 - r][w - 1 - c] for r in range(h)] for c in range(w)]
        return result

    return grid


def move_along_diagonal(grid: Grid, steps: int, direction: str = 'main-down') -> Grid:
    """
    Shift components along diagonal direction.
    Direction: 'main-down', 'main-up', 'anti-down', 'anti-up'
    """
    if not grid or not grid[0]:
        return grid

    h, w = len(grid), len(grid[0])
    result = [[0] * w for _ in range(h)]

    # Determine shift direction
    if direction == 'main-down':
        dr, dc = steps, steps  # Move down-right
    elif direction == 'main-up':
        dr, dc = -steps, -steps  # Move up-left
    elif direction == 'anti-down':
        dr, dc = steps, -steps  # Move down-left
    elif direction == 'anti-up':
        dr, dc = -steps, steps  # Move up-right
    else:
        return grid

    # Move non-background pixels
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    result[nr][nc] = grid[r][c]

    return result


def generate_diagonal_candidates(grid: Grid, train_examples: List[Dict] = None) -> List[Dict]:
    """
    Generate diagonal-based candidate transformations.
    Returns list of candidate dicts.
    """
    candidates = []

    # Detect existing diagonal
    diag_info = detect_diagonal_line(grid)

    # Candidate 1-2: Fill diagonals with detected color or common colors
    common_colors = get_common_colors(grid)
    for direction in ['main', 'anti']:
        if diag_info and diag_info[0] == direction:
            color = diag_info[1]
        elif common_colors:
            color = common_colors[0]
        else:
            continue

        filled = fill_diagonal(grid, color, direction)
        candidates.append({
            'grid': filled,
            'type': f'fill_diagonal_{direction}',
            'confidence': 0.6,
            'source': 'diagonal',
            'meta': {'direction': direction, 'color': color}
        })

    # Candidate 3-4: Reflect over diagonals
    for direction in ['main', 'anti']:
        reflected = reflect_over_diagonal(grid, direction)
        candidates.append({
            'grid': reflected,
            'type': f'reflect_{direction}',
            'confidence': 0.65,
            'source': 'diagonal',
            'meta': {'direction': direction}
        })

    # Candidate 5-8: Move along diagonal directions
    for direction in ['main-down', 'main-up', 'anti-down', 'anti-up']:
        for steps in [1, 2]:
            moved = move_along_diagonal(grid, steps, direction)
            candidates.append({
                'grid': moved,
                'type': f'diagonal_move_{direction}',
                'confidence': 0.5,
                'source': 'diagonal',
                'meta': {'direction': direction, 'steps': steps}
            })

    return candidates


def detect_diagonal_pattern(train_examples: List[Dict]) -> Optional[str]:
    """
    Detect if training examples show diagonal pattern.
    Returns: pattern type or None
    """
    if not train_examples:
        return None

    diagonal_fill_count = 0
    reflect_main_count = 0
    reflect_anti_count = 0

    for ex in train_examples:
        inp = ex['input']
        out = ex['output']

        # Check for diagonal fill
        diag_info = detect_diagonal_line(out)
        if diag_info:
            diagonal_fill_count += 1

        # Check for reflection
        if grids_equal(reflect_over_diagonal(inp, 'main'), out):
            reflect_main_count += 1

        if grids_equal(reflect_over_diagonal(inp, 'anti'), out):
            reflect_anti_count += 1

    # Consensus?
    n = len(train_examples)
    if reflect_main_count >= n * 0.7:
        return 'reflect_main'
    if reflect_anti_count >= n * 0.7:
        return 'reflect_anti'
    if diagonal_fill_count >= n * 0.7:
        return 'diagonal_fill'

    return None


def get_common_colors(grid: Grid) -> List[int]:
    """Get most common non-background colors in grid."""
    from collections import Counter

    if not grid or not grid[0]:
        return []

    all_colors = [cell for row in grid for cell in row if cell != 0]

    if not all_colors:
        return []

    counts = Counter(all_colors)
    return [color for color, _ in counts.most_common(3)]


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
