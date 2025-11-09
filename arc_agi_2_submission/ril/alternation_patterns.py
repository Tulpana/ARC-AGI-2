"""
Alternating and Repeating Pattern Detection and Extension.
Handles ABAB patterns, sequence extension, pattern completion.
Critical for pattern extension tasks (27% of analyzed failures).
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

Grid = List[List[int]]


def detect_alternation(sequence: List[int]) -> Optional[Dict]:
    """
    Detect alternating pattern in sequence.
    Returns: {'period': int, 'values': List[int]} or None
    """
    if len(sequence) < 4:
        return None

    # Try period 2 (ABABAB...)
    if len(sequence) >= 4:
        if sequence[0] == sequence[2] and sequence[1] == sequence[3]:
            if sequence[0] != sequence[1]:
                # Verify pattern continues
                valid = True
                for i in range(4, len(sequence)):
                    if sequence[i] != sequence[i % 2]:
                        valid = False
                        break

                if valid or len(sequence) == 4:
                    return {'period': 2, 'values': [sequence[0], sequence[1]]}

    # Try period 3 (ABCABCABC...)
    if len(sequence) >= 6:
        if (sequence[0] == sequence[3] and
            sequence[1] == sequence[4] and
            sequence[2] == sequence[5]):

            valid = True
            for i in range(6, len(sequence)):
                if sequence[i] != sequence[i % 3]:
                    valid = False
                    break

            if valid or len(sequence) == 6:
                return {'period': 3, 'values': [sequence[0], sequence[1], sequence[2]]}

    # Try period 4
    if len(sequence) >= 8:
        if all(sequence[i] == sequence[i + 4] for i in range(4)):
            return {'period': 4, 'values': sequence[:4]}

    return None


def extend_sequence(sequence: List[int], count: int, direction: str = 'forward') -> List[int]:
    """
    Extend sequence based on detected pattern.
    Direction: 'forward' or 'backward'
    """
    if not sequence:
        return sequence

    # Detect pattern
    pattern = detect_alternation(sequence)

    if not pattern:
        # No pattern - just repeat last value
        if direction == 'forward':
            return sequence + [sequence[-1]] * count
        else:
            return [sequence[0]] * count + sequence

    # Extend using detected pattern
    period = pattern['period']
    values = pattern['values']

    if direction == 'forward':
        extended = sequence[:]
        for i in range(count):
            next_idx = (len(extended)) % period
            extended.append(values[next_idx])
        return extended
    else:
        # Backward extension
        prepended = []
        for i in range(count):
            prev_idx = (period - 1 - i) % period
            prepended.insert(0, values[prev_idx])
        return prepended + sequence


def extend_grid_pattern(grid: Grid, direction: str, count: int) -> Grid:
    """
    Extend grid pattern in specified direction.
    Direction: 'right', 'left', 'down', 'up'
    """
    if not grid or not grid[0]:
        return grid

    h, w = len(grid), len(grid[0])

    if direction == 'right':
        # Extend each row to the right
        extended = []
        for row in grid:
            pattern = detect_alternation(row)
            if pattern:
                extended_row = extend_sequence(row, count, 'forward')
            else:
                # Repeat last value
                extended_row = row + [row[-1]] * count
            extended.append(extended_row)
        return extended

    elif direction == 'left':
        # Extend each row to the left
        extended = []
        for row in grid:
            pattern = detect_alternation(row)
            if pattern:
                extended_row = extend_sequence(row, count, 'backward')
            else:
                extended_row = [row[0]] * count + row
            extended.append(extended_row)
        return extended

    elif direction == 'down':
        # Extend columns downward
        # Extract each column, extend it, then reassemble
        extended_cols = []
        for c in range(w):
            col = [grid[r][c] for r in range(h)]
            pattern = detect_alternation(col)
            if pattern:
                extended_col = extend_sequence(col, count, 'forward')
            else:
                extended_col = col + [col[-1]] * count
            extended_cols.append(extended_col)

        # Reassemble into grid
        new_h = len(extended_cols[0])
        return [[extended_cols[c][r] for c in range(w)] for r in range(new_h)]

    elif direction == 'up':
        # Extend columns upward
        extended_cols = []
        for c in range(w):
            col = [grid[r][c] for r in range(h)]
            pattern = detect_alternation(col)
            if pattern:
                extended_col = extend_sequence(col, count, 'backward')
            else:
                extended_col = [col[0]] * count + col
            extended_cols.append(extended_col)

        new_h = len(extended_cols[0])
        return [[extended_cols[c][r] for c in range(w)] for r in range(new_h)]

    return grid


def detect_grid_alternation(grid: Grid) -> Optional[Tuple[str, Dict]]:
    """
    Detect alternation in grid (rows/cols/diagonals).
    Returns: (location, pattern) where location is 'row', 'col', or 'diag'
    """
    if not grid or not grid[0]:
        return None

    h, w = len(grid), len(grid[0])

    # Check rows
    for row in grid:
        pattern = detect_alternation(row)
        if pattern:
            return ('row', pattern)

    # Check columns
    for c in range(w):
        col = [grid[r][c] for r in range(h)]
        pattern = detect_alternation(col)
        if pattern:
            return ('col', pattern)

    # Check main diagonal
    if h >= 4 and w >= 4:
        diag = [grid[i][i] for i in range(min(h, w))]
        pattern = detect_alternation(diag)
        if pattern:
            return ('main_diag', pattern)

    # Check anti-diagonal
    if h >= 4 and w >= 4:
        anti_diag = [grid[i][w-1-i] for i in range(min(h, w))]
        pattern = detect_alternation(anti_diag)
        if pattern:
            return ('anti_diag', pattern)

    return None


def complete_pattern(grid: Grid, target_shape: Optional[Tuple[int, int]] = None) -> Grid:
    """
    Complete/extend pattern to fill grid or reach target shape.
    """
    if not grid or not grid[0]:
        return grid

    # Detect pattern
    pattern_info = detect_grid_alternation(grid)

    if not pattern_info:
        return grid

    location, pattern = pattern_info

    if target_shape:
        target_h, target_w = target_shape
        current_h, current_w = len(grid), len(grid[0])

        if location == 'row':
            # Extend rows to target width
            if target_w > current_w:
                grid = extend_grid_pattern(grid, 'right', target_w - current_w)

            # Add more rows if needed
            if target_h > current_h:
                # Repeat pattern in new rows
                result = grid[:]
                for _ in range(target_h - current_h):
                    # Use pattern from first row
                    new_row = extend_sequence(grid[0], target_w, 'forward')[:target_w]
                    result.append(new_row)
                return result

        elif location == 'col':
            # Extend columns to target height
            if target_h > current_h:
                grid = extend_grid_pattern(grid, 'down', target_h - current_h)

            # Add more columns if needed
            if target_w > current_w:
                result = [row[:] for row in grid]
                for r in range(len(result)):
                    col = [result[i][0] for i in range(len(result))]
                    extended = extend_sequence(col, target_w, 'forward')
                    for c in range(current_w, target_w):
                        if c < len(extended):
                            result[r].append(extended[c])
                return result

    return grid


def fill_checkerboard(h: int, w: int, color1: int, color2: int) -> Grid:
    """
    Create checkerboard pattern (common ABAB pattern).
    """
    result = []
    for r in range(h):
        row = []
        for c in range(w):
            if (r + c) % 2 == 0:
                row.append(color1)
            else:
                row.append(color2)
        result.append(row)
    return result


def generate_alternation_candidates(grid: Grid, train_examples: List[Dict] = None) -> List[Dict]:
    """
    Generate alternation-based candidate transformations.
    Returns list of candidate dicts.
    """
    candidates = []

    # Detect existing pattern
    pattern_info = detect_grid_alternation(grid)

    # Candidate 1-4: Extend in each direction
    for direction in ['right', 'down', 'left', 'up']:
        for count in [1, 2, 3]:
            extended = extend_grid_pattern(grid, direction, count)
            candidates.append({
                'grid': extended,
                'type': f'extend_{direction}',
                'confidence': 0.6 if pattern_info else 0.4,
                'source': 'alternation',
                'meta': {'direction': direction, 'count': count}
            })

    # Candidate 5: Complete to square
    h, w = len(grid), len(grid[0])
    if h != w:
        target_size = max(h, w)
        completed = complete_pattern(grid, (target_size, target_size))
        candidates.append({
            'grid': completed,
            'type': 'complete_square',
            'confidence': 0.55,
            'source': 'alternation',
            'meta': {'target': f'{target_size}x{target_size}'}
        })

    # Candidate 6: Complete to 2x size
    if pattern_info:
        completed = complete_pattern(grid, (h * 2, w * 2))
        candidates.append({
            'grid': completed,
            'type': 'extend_2x',
            'confidence': 0.5,
            'source': 'alternation',
            'meta': {'target': f'{h*2}x{w*2}'}
        })

    return candidates


def detect_alternation_pattern(train_examples: List[Dict]) -> bool:
    """
    Detect if task involves alternation/extension patterns.
    Returns True if pattern detected.
    """
    if not train_examples:
        return False

    alternation_count = 0

    for ex in train_examples:
        out = ex['output']

        # Check if output has alternation
        pattern_info = detect_grid_alternation(out)
        if pattern_info:
            alternation_count += 1

    return alternation_count >= len(train_examples) * 0.5  # 50% threshold
