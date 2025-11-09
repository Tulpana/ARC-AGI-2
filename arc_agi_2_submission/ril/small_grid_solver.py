"""
Small Grid Specialist Solver for ARC-AGI-2.
Optimized for grids ≤25 pixels (5x5, 3x3, 2x2, 1x1).
Kaggle-compatible: stdlib-only, no external dependencies.
"""
from __future__ import annotations
from collections import Counter
from typing import Dict, List, Optional, Tuple

Grid = List[List[int]]
Example = Dict[str, Grid]


class SmallGridSolver:
    """
    Specialized solver for tiny grids where complex reasoning is overkill.
    Uses: direct lookup, color mapping, position rules, exhaustive search.
    """

    THRESHOLD = 25  # pixels

    def __init__(self):
        pass

    @staticmethod
    def grid_size(grid: Grid) -> int:
        """Count total pixels in grid."""
        h = len(grid)
        w = len(grid[0]) if h > 0 else 0
        return h * w

    @staticmethod
    def can_handle(test_input: Grid, train_examples: List[Example]) -> bool:
        """
        Check if this is a small-grid task.
        Returns True if all grids (train + test) are ≤ THRESHOLD pixels.
        """
        all_grids = [test_input]

        for ex in train_examples:
            all_grids.append(ex['input'])
            all_grids.append(ex['output'])

        max_size = max(SmallGridSolver.grid_size(g) for g in all_grids)
        return max_size <= SmallGridSolver.THRESHOLD

    def solve(self, train_examples: List[Example], test_input: Grid, topk: int = 2) -> List[Grid]:
        """
        Solve small grid task using specialized strategies.
        Returns up to topk candidate solutions.
        """
        candidates = []

        # Strategy 0: Constant output (always same output regardless of input)
        constant_output = self._detect_constant_output(train_examples)
        if constant_output is not None:
            return [constant_output]  # Instant answer!

        # Strategy 1: Perfect memorization (test input = training input)
        perfect_match = self._perfect_match(train_examples, test_input)
        if perfect_match:
            return [perfect_match]  # Instant exact match!

        # Strategy 2: Color mapping detection (bijection)
        color_map_candidates = self._color_mapping_strategy(train_examples, test_input)
        candidates.extend(color_map_candidates)

        # Strategy 3: Position-wise rules (for structured patterns)
        position_rule_candidate = self._position_rule_strategy(train_examples, test_input)
        if position_rule_candidate:
            candidates.append(position_rule_candidate)

        # Strategy 4: Exhaustive search (for 1x1, 2x2, maybe 3x3)
        if self.grid_size(test_input) <= 9:  # Up to 3x3
            exhaustive_candidates = self._exhaustive_search(train_examples, test_input, topk)
            candidates.extend(exhaustive_candidates)

        # Deduplicate and rank
        unique_candidates = self._deduplicate(candidates)
        ranked = self._rank_candidates(unique_candidates, train_examples)

        return ranked[:topk]

    def _detect_constant_output(self, train_examples: List[Example]) -> Optional[Grid]:
        """
        Check if all training examples have identical output.
        Critical for tasks like '0d3d703e' and '1a2e2828'.
        """
        if not train_examples:
            return None

        first_output = train_examples[0]['output']

        for ex in train_examples[1:]:
            if not self._grids_equal(ex['output'], first_output):
                return None  # Not constant

        return first_output  # Return the constant output

    def _perfect_match(self, train_examples: List[Example], test_input: Grid) -> Optional[Grid]:
        """Check if test input exactly matches any training input."""
        for ex in train_examples:
            if self._grids_equal(ex['input'], test_input):
                return ex['output']
        return None

    def _grids_equal(self, g1: Grid, g2: Grid) -> bool:
        """Check if two grids are identical."""
        if len(g1) != len(g2):
            return False
        if not g1:
            return not g2

        if len(g1[0]) != len(g2[0]):
            return False

        for r in range(len(g1)):
            for c in range(len(g1[0])):
                if g1[r][c] != g2[r][c]:
                    return False
        return True

    def _color_mapping_strategy(self, train_examples: List[Example], test_input: Grid) -> List[Grid]:
        """
        Detect if task is a simple color substitution.
        Works well for 1x1, 2x2, 3x3 grids.
        """
        candidates = []

        # Try to find a consistent color bijection
        mappings = []

        for ex in train_examples:
            mapping = self._infer_color_mapping(ex['input'], ex['output'])
            if mapping is not None:
                mappings.append(mapping)

        if not mappings:
            return candidates

        # Check for consensus
        if all(m == mappings[0] for m in mappings):
            # Perfect consensus!
            candidate = self._apply_color_mapping(test_input, mappings[0])
            candidates.append(candidate)

        # Also try majority-vote mapping
        consensus_map = self._consensus_color_mapping(mappings)
        if consensus_map:
            candidate = self._apply_color_mapping(test_input, consensus_map)
            if not any(self._grids_equal(candidate, c) for c in candidates):
                candidates.append(candidate)

        return candidates

    def _infer_color_mapping(self, inp: Grid, out: Grid) -> Optional[Dict[int, int]]:
        """
        Infer 1:1 color mapping from input to output.
        Returns None if not a pure color substitution.
        """
        if len(inp) != len(out):
            return None
        if inp and len(inp[0]) != len(out[0]):
            return None

        # Check if spatial structure is preserved
        if not self._same_structure(inp, out):
            return None

        # Extract mapping
        mapping: Dict[int, int] = {}

        for r in range(len(inp)):
            for c in range(len(inp[0]) if inp else 0):
                in_color = inp[r][c]
                out_color = out[r][c]

                if in_color in mapping:
                    if mapping[in_color] != out_color:
                        return None  # Inconsistent mapping
                else:
                    mapping[in_color] = out_color

        return mapping

    def _same_structure(self, g1: Grid, g2: Grid) -> bool:
        """Check if two grids have same spatial pattern (ignoring colors)."""
        if len(g1) != len(g2):
            return False
        if not g1:
            return True

        if len(g1[0]) != len(g2[0]):
            return False

        # Create normalized grids (map colors to 0, 1, 2, ...)
        norm1 = self._normalize_grid(g1)
        norm2 = self._normalize_grid(g2)

        return self._grids_equal(norm1, norm2)

    def _normalize_grid(self, grid: Grid) -> Grid:
        """Map colors to sequential integers preserving pattern."""
        color_map = {}
        next_id = 0
        normalized = []

        for row in grid:
            norm_row = []
            for cell in row:
                if cell not in color_map:
                    color_map[cell] = next_id
                    next_id += 1
                norm_row.append(color_map[cell])
            normalized.append(norm_row)

        return normalized

    def _consensus_color_mapping(self, mappings: List[Dict[int, int]]) -> Dict[int, int]:
        """Find majority-vote color mapping."""
        if not mappings:
            return {}

        all_sources = set()
        for m in mappings:
            all_sources.update(m.keys())

        consensus = {}
        for src in all_sources:
            targets = [m[src] for m in mappings if src in m]
            if targets:
                most_common = Counter(targets).most_common(1)[0][0]
                consensus[src] = most_common

        return consensus

    def _apply_color_mapping(self, grid: Grid, mapping: Dict[int, int]) -> Grid:
        """Apply color substitution to grid."""
        result = []
        for row in grid:
            new_row = [mapping.get(cell, cell) for cell in row]
            result.append(new_row)
        return result

    def _position_rule_strategy(self, train_examples: List[Example], test_input: Grid) -> Optional[Grid]:
        """
        Learn position-wise rules: output[r][c] = f(input[r][c], r, c).
        Good for small grids with positional logic.
        """
        if not train_examples:
            return None

        # Extract position-based rules from training
        h = len(test_input)
        w = len(test_input[0]) if h > 0 else 0

        # Check if all examples have same shape
        if not all(
            len(ex['input']) == h and len(ex['output']) == h and
            (not ex['input'] or len(ex['input'][0]) == w) and
            (not ex['output'] or len(ex['output'][0]) == w)
            for ex in train_examples
        ):
            return None  # Shape mismatch, can't use position rules

        # For each position, learn the rule
        position_rules: Dict[Tuple[int, int], Dict[int, int]] = {}

        for r in range(h):
            for c in range(w):
                # Collect (input_color, output_color) pairs for this position
                color_pairs = []
                for ex in train_examples:
                    in_color = ex['input'][r][c]
                    out_color = ex['output'][r][c]
                    color_pairs.append((in_color, out_color))

                # Find consensus rule for this position
                # Rule: if input=X at (r,c), output=Y at (r,c)
                rule_map: Dict[int, List[int]] = {}
                for in_c, out_c in color_pairs:
                    if in_c not in rule_map:
                        rule_map[in_c] = []
                    rule_map[in_c].append(out_c)

                # Use most common output for each input
                consensus_rule = {}
                for in_c, out_cs in rule_map.items():
                    consensus_rule[in_c] = Counter(out_cs).most_common(1)[0][0]

                position_rules[(r, c)] = consensus_rule

        # Apply rules to test input
        output = []
        for r in range(h):
            row = []
            for c in range(w):
                in_color = test_input[r][c]
                rule = position_rules.get((r, c), {})
                out_color = rule.get(in_color, in_color)  # Default to same color
                row.append(out_color)
            output.append(row)

        return output

    def _exhaustive_search(self, train_examples: List[Example], test_input: Grid, topk: int) -> List[Grid]:
        """
        Brute-force search for very small grids (≤9 pixels).
        Generate all plausible outputs and rank by training fit.
        """
        h = len(test_input)
        w = len(test_input[0]) if h > 0 else 0
        total_pixels = h * w

        if total_pixels > 9:
            return []  # Too expensive

        # Collect all colors seen in training outputs
        output_palette = set()
        for ex in train_examples:
            for row in ex['output']:
                output_palette.update(row)

        if len(output_palette) > 5:
            return []  # Too many colors to search

        # Generate all possible grids
        candidates = self._generate_all_grids(h, w, sorted(output_palette))

        # Rank by similarity to training pattern
        scored = []
        for cand in candidates[:1000]:  # Cap at 1000 to avoid timeout
            score = self._score_against_training(cand, train_examples, test_input)
            scored.append((score, cand))

        scored.sort(reverse=True, key=lambda x: x[0])
        return [cand for score, cand in scored[:topk]]

    def _generate_all_grids(self, h: int, w: int, palette: List[int]) -> List[Grid]:
        """Generate all possible h×w grids using given palette."""
        total_pixels = h * w
        num_colors = len(palette)

        if num_colors == 0:
            return []

        grids = []

        # Generate all combinations (this is exponential!)
        for config in range(num_colors ** total_pixels):
            grid = []
            temp = config
            for r in range(h):
                row = []
                for c in range(w):
                    color_idx = temp % num_colors
                    row.append(palette[color_idx])
                    temp //= num_colors
                grid.append(row)
            grids.append(grid)

        return grids

    def _score_against_training(self, candidate: Grid, train_examples: List[Example], test_input: Grid) -> float:
        """
        Score candidate by how well it matches the transformation pattern in training.
        """
        if not train_examples:
            return 0.0

        scores = []

        for ex in train_examples:
            # Check if this training example matches our test scenario
            if self._grids_equal(ex['input'], test_input):
                # Exact match case
                if self._grids_equal(candidate, ex['output']):
                    return 1000.0  # Jackpot!

            # Measure pattern similarity
            score = self._pattern_similarity(ex['input'], ex['output'], test_input, candidate)
            scores.append(score)

        return sum(scores) / len(scores) if scores else 0.0

    def _pattern_similarity(self, train_in: Grid, train_out: Grid, test_in: Grid, test_out: Grid) -> float:
        """
        Measure how similar the test transformation is to training transformation.
        """
        score = 0.0

        # Palette similarity
        train_out_pal = {cell for row in train_out for cell in row}
        test_out_pal = {cell for row in test_out for cell in row}
        palette_overlap = len(train_out_pal & test_out_pal) / max(len(train_out_pal | test_out_pal), 1)
        score += palette_overlap * 10

        # Structure similarity (normalized grids)
        if self._same_structure(train_out, test_out):
            score += 50

        # Color mapping consistency
        train_map = self._infer_color_mapping(train_in, train_out)
        test_map = self._infer_color_mapping(test_in, test_out)

        if train_map and test_map:
            map_similarity = len(set(train_map.items()) & set(test_map.items())) / max(len(train_map), len(test_map), 1)
            score += map_similarity * 30

        return score

    def _deduplicate(self, candidates: List[Grid]) -> List[Grid]:
        """Remove duplicate grids."""
        unique = []
        for cand in candidates:
            if not any(self._grids_equal(cand, u) for u in unique):
                unique.append(cand)
        return unique

    def _rank_candidates(self, candidates: List[Grid], train_examples: List[Example]) -> List[Grid]:
        """Rank candidates by training fit."""
        scored = []
        for cand in candidates:
            score = sum(
                self._grid_similarity(cand, ex['output'])
                for ex in train_examples
            ) / max(len(train_examples), 1)
            scored.append((score, cand))

        scored.sort(reverse=True, key=lambda x: x[0])
        return [cand for score, cand in scored]

    def _grid_similarity(self, g1: Grid, g2: Grid) -> float:
        """Measure similarity between two grids."""
        if len(g1) != len(g2):
            return 0.0
        if not g1:
            return 1.0 if not g2 else 0.0
        if len(g1[0]) != len(g2[0]):
            return 0.0

        matches = 0
        total = len(g1) * len(g1[0])

        for r in range(len(g1)):
            for c in range(len(g1[0])):
                if g1[r][c] == g2[r][c]:
                    matches += 1

        return matches / max(total, 1)
