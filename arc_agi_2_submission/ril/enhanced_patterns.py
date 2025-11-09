"""
Enhanced Pattern Generators - Integration Module.
Consolidates all new pattern types for easy integration into solver.
Stdlib-only, Kaggle-compatible.
"""
from __future__ import annotations
from typing import Dict, List

Grid = List[List[int]]

# Import all new pattern generators
try:
    from .assembly_ops import generate_assembly_candidates
except ImportError:
    def generate_assembly_candidates(grid: Grid, train_examples=None) -> List[Dict]:
        return []

try:
    from .diagonal_patterns import generate_diagonal_candidates
except ImportError:
    def generate_diagonal_candidates(grid: Grid, train_examples=None) -> List[Dict]:
        return []

try:
    from .ordering_patterns import generate_ordering_candidates
except ImportError:
    def generate_ordering_candidates(grid: Grid, train_examples=None) -> List[Dict]:
        return []

try:
    from .outline_ops import generate_outline_candidates
except ImportError:
    def generate_outline_candidates(grid: Grid, train_examples=None) -> List[Dict]:
        return []

try:
    from .alternation_patterns import generate_alternation_candidates
except ImportError:
    def generate_alternation_candidates(grid: Grid, train_examples=None) -> List[Dict]:
        return []

try:
    from .color_extraction import generate_color_extraction_candidates
except ImportError:
    def generate_color_extraction_candidates(grid: Grid, train_examples=None) -> List[Dict]:
        return []


def generate_all_enhanced_candidates(
    test_input: Grid,
    train_examples: List[Dict] = None
) -> List[Dict]:
    """
    Generate all enhanced pattern candidates.

    Returns list of candidate dicts with keys:
    - 'grid': Grid
    - 'type': str
    - 'confidence': float
    - 'source': str
    - 'meta': Dict (optional)

    Args:
        test_input: Test input grid
        train_examples: Training examples for pattern detection

    Returns:
        List of candidate dicts
    """
    all_candidates = []

    # Assembly patterns (gravity, packing, tile collection)
    # Critical for 24% of failures
    try:
        assembly_cands = generate_assembly_candidates(test_input, train_examples)
        all_candidates.extend(assembly_cands)
    except Exception as e:
        print(f"[ENHANCED-WARN] assembly_ops failed: {e}")

    # Diagonal patterns (diagonal lines, reflection, movement)
    # Critical for 50% of almost-matches
    try:
        diagonal_cands = generate_diagonal_candidates(test_input, train_examples)
        all_candidates.extend(diagonal_cands)
    except Exception as e:
        print(f"[ENHANCED-WARN] diagonal_patterns failed: {e}")

    # Ordering/sequencing patterns (tile ordering, spatial assembly)
    # Critical for 23% of failures
    try:
        ordering_cands = generate_ordering_candidates(test_input, train_examples)
        all_candidates.extend(ordering_cands)
    except Exception as e:
        print(f"[ENHANCED-WARN] ordering_patterns failed: {e}")

    # Outline tracing (boundary extraction, hollow shapes)
    # Critical for 17% of failures
    try:
        outline_cands = generate_outline_candidates(test_input, train_examples)
        all_candidates.extend(outline_cands)
    except Exception as e:
        print(f"[ENHANCED-WARN] outline_ops failed: {e}")

    # Alternation patterns (ABAB, pattern extension)
    # Critical for 27% of failures
    try:
        alternation_cands = generate_alternation_candidates(test_input, train_examples)
        all_candidates.extend(alternation_cands)
    except Exception as e:
        print(f"[ENHANCED-WARN] alternation_patterns failed: {e}")

    # Color extraction (mode, median, corner, center)
    # CRITICAL: 70% of severe failures are palette errors!
    try:
        color_cands = generate_color_extraction_candidates(test_input, train_examples)
        all_candidates.extend(color_cands)
    except Exception as e:
        print(f"[ENHANCED-WARN] color_extraction failed: {e}")

    return all_candidates


def get_pattern_stats(candidates: List[Dict]) -> Dict[str, int]:
    """
    Get statistics on generated candidates by source.

    Args:
        candidates: List of candidate dicts

    Returns:
        Dict mapping source -> count
    """
    from collections import Counter

    sources = [c.get('source', 'unknown') for c in candidates]
    return dict(Counter(sources))


def filter_candidates_by_confidence(
    candidates: List[Dict],
    min_confidence: float = 0.4
) -> List[Dict]:
    """
    Filter candidates by minimum confidence threshold.

    Args:
        candidates: List of candidate dicts
        min_confidence: Minimum confidence to keep

    Returns:
        Filtered list of candidates
    """
    return [c for c in candidates if c.get('confidence', 0.0) >= min_confidence]


def rank_candidates_by_confidence(candidates: List[Dict]) -> List[Dict]:
    """
    Sort candidates by confidence (descending).

    Args:
        candidates: List of candidate dicts

    Returns:
        Sorted list of candidates
    """
    return sorted(candidates, key=lambda c: c.get('confidence', 0.0), reverse=True)


def deduplicate_candidates(candidates: List[Dict]) -> List[Dict]:
    """
    Remove duplicate grids from candidate list.
    Keeps first occurrence of each unique grid.

    Args:
        candidates: List of candidate dicts

    Returns:
        Deduplicated list of candidates
    """
    seen_grids = []
    unique_candidates = []

    for cand in candidates:
        grid = cand.get('grid')
        if grid is None:
            continue

        # Check if grid already seen
        is_duplicate = False
        for seen in seen_grids:
            if grids_equal(grid, seen):
                is_duplicate = True
                break

        if not is_duplicate:
            seen_grids.append(grid)
            unique_candidates.append(cand)

    return unique_candidates


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
