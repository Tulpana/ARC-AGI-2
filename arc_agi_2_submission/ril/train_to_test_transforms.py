"""Train-to-test transformation learners for ARC-AGI-2.

This module implements pattern learners that derive transformations from
training example pairs and apply them to test inputs, addressing the
architectural gap identified in FINAL_ANALYSIS.md where the solver assumed
"output = function(test)" rather than "output = function(train, test)".
"""

from typing import List, Dict, Any, Tuple, Set, Optional
from collections import Counter

Grid = List[List[int]]


def learn_color_mapping(train_examples: List[Dict[str, Any]]) -> Optional[Dict[int, int]]:
    """Learn deterministic color mapping from train→test pairs.

    Returns a color mapping dict if consistent across all examples, else None.
    """
    if not train_examples:
        return None

    mappings: List[Dict[int, int]] = []

    for ex in train_examples:
        ex_in = ex.get("input", [])
        ex_out = ex.get("output", [])

        if not ex_in or not ex_out:
            continue

        # Align grids if same shape
        if len(ex_in) != len(ex_out) or (ex_in and len(ex_in[0]) != len(ex_out[0])):
            continue

        mapping = {}
        for r in range(len(ex_in)):
            for c in range(len(ex_in[0])):
                in_color = int(ex_in[r][c])
                out_color = int(ex_out[r][c])

                if in_color in mapping:
                    if mapping[in_color] != out_color:
                        # Inconsistent mapping within this example
                        return None
                else:
                    mapping[in_color] = out_color

        mappings.append(mapping)

    if not mappings:
        return None

    # Check consistency across all examples
    base_mapping = mappings[0]
    for m in mappings[1:]:
        for color, target in m.items():
            if color in base_mapping and base_mapping[color] != target:
                return None
            base_mapping[color] = target

    return base_mapping


def apply_color_mapping(grid: Grid, mapping: Dict[int, int]) -> Grid:
    """Apply learned color mapping to a grid."""
    result = []
    for row in grid:
        new_row = [mapping.get(int(cell), int(cell)) for cell in row]
        result.append(new_row)
    return result


def learn_row_selection(train_examples: List[Dict[str, Any]]) -> Optional[List[int]]:
    """Learn row selection pattern (e.g., task 22425bda selects one column per example).

    Returns list of selected row indices if pattern is consistent.
    """
    if not train_examples:
        return None

    # Check if all outputs are 1xN
    for ex in train_examples:
        out = ex.get("output", [])
        if not out or len(out) != 1:
            return None

    # Pattern: each training output picks one column from its input
    # For test, we might need to combine selections
    selections = []
    for ex in train_examples:
        ex_in = ex.get("input", [])
        ex_out = ex.get("output", [])

        if not ex_in or not ex_out or len(ex_out) != 1:
            continue

        out_row = ex_out[0]
        # Find which column from input matches each position in output
        for idx, val in enumerate(out_row):
            if idx < len(ex_in[0]):
                selections.append(idx)

    return selections if selections else None


def learn_tiling_pattern(train_examples: List[Dict[str, Any]]) -> Optional[Tuple[int, int]]:
    """Learn if output is a tiling of input pattern.

    Returns (tile_h, tile_w) if consistent tiling pattern found.
    """
    if not train_examples:
        return None

    tiles = []
    for ex in train_examples:
        ex_in = ex.get("input", [])
        ex_out = ex.get("output", [])

        if not ex_in or not ex_out:
            continue

        in_h, in_w = len(ex_in), len(ex_in[0]) if ex_in else 0
        out_h, out_w = len(ex_out), len(ex_out[0]) if ex_out else 0

        if out_h == 0 or out_w == 0 or in_h == 0 or in_w == 0:
            continue

        # Check if output is exact multiple of input
        if out_h % in_h == 0 and out_w % in_w == 0:
            th = out_h // in_h
            tw = out_w // in_w
            tiles.append((th, tw))

    if not tiles:
        return None

    # Check consistency
    if len(set(tiles)) == 1:
        return tiles[0]

    return None


def learn_motif_assembly(train_examples: List[Dict[str, Any]]) -> Optional[List[Grid]]:
    """Learn if output is assembled from motifs in training inputs.

    For task 0a1d4ef5, the output is built from rows of different examples.
    Returns list of motif grids if pattern detected.
    """
    if not train_examples:
        return None

    # Check if all training outputs have same height
    output_heights = []
    for ex in train_examples:
        out = ex.get("output", [])
        if out:
            output_heights.append(len(out))

    if not output_heights or len(set(output_heights)) > 1:
        return None

    # Extract potential motifs (rows from each example)
    motifs = []
    for ex in train_examples:
        ex_in = ex.get("input", [])
        ex_out = ex.get("output", [])

        if not ex_in or not ex_out:
            continue

        # Check if output rows appear in input
        for out_row in ex_out:
            for in_row in ex_in:
                if in_row == out_row:
                    motifs.append([in_row])
                    break

    return motifs if motifs else None


def generate_transform_candidates(
    test_input: Grid,
    train_examples: List[Dict[str, Any]],
    max_candidates: int = 5
) -> List[Dict[str, Any]]:
    """Generate candidates by applying learned transformations to test input.

    This is the main entry point that tries multiple transformation strategies
    and returns candidate grids with metadata.
    """
    candidates = []

    # Strategy 1: Color mapping
    color_map = learn_color_mapping(train_examples)
    if color_map:
        mapped_grid = apply_color_mapping(test_input, color_map)
        candidates.append({
            "grid": mapped_grid,
            "source": "adapter",
            "meta": {
                "transform": "color_mapping",
                "confidence": 0.8,
                "learned_from_train": True,
            },
            "confidence": 0.8,
            "score": 0.8,
        })

    # Strategy 2: Tiling
    tiling = learn_tiling_pattern(train_examples)
    if tiling:
        th, tw = tiling
        in_h, in_w = len(test_input), len(test_input[0]) if test_input else 0

        if in_h > 0 and in_w > 0:
            tiled_grid = []
            for tr in range(th):
                for r in range(in_h):
                    row = []
                    for tc in range(tw):
                        row.extend(test_input[r])
                    tiled_grid.append(row)

            candidates.append({
                "grid": tiled_grid,
                "source": "adapter",
                "meta": {
                    "transform": "tiling",
                    "tile_factor": f"{th}x{tw}",
                    "confidence": 0.75,
                    "learned_from_train": True,
                },
                "confidence": 0.75,
                "score": 0.75,
            })

    # Strategy 3: Row selection (for 1xN outputs)
    row_pattern = learn_row_selection(train_examples)
    if row_pattern and test_input:
        # Build 1xN output by selecting columns
        selected = []
        for idx in row_pattern[:min(len(row_pattern), len(test_input[0]) if test_input else 0)]:
            if test_input and idx < len(test_input[0]):
                # Take value from first row, column idx
                selected.append(test_input[0][idx] if test_input else 0)

        if selected:
            candidates.append({
                "grid": [selected],
                "source": "adapter",
                "meta": {
                    "transform": "row_selection",
                    "confidence": 0.7,
                    "learned_from_train": True,
                },
                "confidence": 0.7,
                "score": 0.7,
            })

    # Strategy 4: Motif assembly
    motifs = learn_motif_assembly(train_examples)
    if motifs and len(motifs) > 0:
        # Assemble output from motifs
        assembled = []
        for motif in motifs[:4]:  # Limit to 4 motifs
            assembled.extend(motif)

        if assembled:
            candidates.append({
                "grid": assembled,
                "source": "adapter",
                "meta": {
                    "transform": "motif_assembly",
                    "num_motifs": len(motifs),
                    "confidence": 0.65,
                    "learned_from_train": True,
                },
                "confidence": 0.65,
                "score": 0.65,
            })

    return candidates[:max_candidates]
