from __future__ import annotations

import sys
from pathlib import Path

ENTRY_ROOT = Path(__file__).resolve().parents[1]
if str(ENTRY_ROOT) not in sys.path:
    sys.path.insert(0, str(ENTRY_ROOT))

from ril.pattern_ops import (
    build_pattern_context,
    generate_chasm_patterns,
    generate_outline_patterns,
    generate_union_patterns,
)
from ril.np_compat import asgrid


def _extract_grid(candidate):
    if isinstance(candidate, dict) and "grid" in candidate:
        return asgrid(candidate["grid"])
    if isinstance(candidate, tuple) and len(candidate) == 2:
        return asgrid(candidate[0])
    return asgrid(candidate)


def _contains(candidates, expected):
    expected_grid = expected
    for cand in candidates:
        if _extract_grid(cand) == expected_grid:
            return True
    return False


def test_outline_generator_creates_ring():
    train_examples = []
    test_input = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ]
    ctx = build_pattern_context(train_examples, test_input)
    candidates = generate_outline_patterns(train_examples, test_input, ctx)
    expected = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ]
    assert _contains(candidates, expected)


def test_union_generator_bridges_gap():
    train_examples = []
    test_input = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    ctx = build_pattern_context(train_examples, test_input)
    candidates = generate_union_patterns(train_examples, test_input, ctx)
    expected = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    assert _contains(candidates, expected)


def test_chasm_generator_recovers_hole():
    train_examples = []
    test_input = [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ]
    ctx = build_pattern_context(train_examples, test_input)
    candidates = generate_chasm_patterns(train_examples, test_input, ctx)
    expected = [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ]
    assert _contains(candidates, expected)
