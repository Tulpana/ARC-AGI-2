"""
Adaptive constraint system for RIL solver.
Learns patterns from training examples rather than imposing blind restrictions.
Kaggle-compatible: stdlib-only, no external dependencies.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

Grid = List[List[int]]
Example = Dict[str, Grid]


@dataclass
class PalettePattern:
    """Learned palette transformation pattern from training examples."""

    input_colors: Set[int] = field(default_factory=set)
    output_colors: Set[int] = field(default_factory=set)
    colors_added: Set[int] = field(default_factory=set)
    colors_removed: Set[int] = field(default_factory=set)
    color_mappings: Dict[int, int] = field(default_factory=dict)

    # Pattern flags
    preserves_all: bool = False
    adds_colors: bool = False
    removes_colors: bool = False
    remaps_colors: bool = False

    # Confidence scores
    pattern_consistency: float = 0.0


class AdaptivePaletteOracle:
    """
    Learns palette transformation rules from training examples.
    Understands when tasks add/remove/preserve/remap colors.
    """

    def __init__(self, train_examples: List[Example]):
        self.patterns = self._extract_palette_patterns(train_examples)
        self.dominant_pattern = self._identify_dominant_pattern()

    def _get_palette(self, grid: Grid) -> Set[int]:
        """Extract unique colors from grid."""
        return {cell for row in grid for cell in row}

    def _extract_palette_patterns(self, train_examples: List[Example]) -> List[PalettePattern]:
        """Extract palette transformation pattern from each training example."""
        patterns = []

        for ex in train_examples:
            inp = ex['input']
            out = ex['output']

            inp_pal = self._get_palette(inp)
            out_pal = self._get_palette(out)

            pattern = PalettePattern(
                input_colors=inp_pal,
                output_colors=out_pal,
                colors_added=out_pal - inp_pal,
                colors_removed=inp_pal - out_pal,
            )

            # Detect pattern type
            pattern.preserves_all = (inp_pal == out_pal)
            pattern.adds_colors = len(pattern.colors_added) > 0
            pattern.removes_colors = len(pattern.colors_removed) > 0

            # Detect color remapping (same palette size, different colors)
            if len(inp_pal) == len(out_pal) and not pattern.preserves_all:
                pattern.remaps_colors = True
                # Try to infer 1:1 mapping via spatial analysis
                pattern.color_mappings = self._infer_color_mapping(inp, out)

            patterns.append(pattern)

        return patterns

    def _infer_color_mapping(self, inp: Grid, out: Grid) -> Dict[int, int]:
        """Infer color mapping by position correlation."""
        mapping = {}

        if len(inp) != len(out) or (inp and len(inp[0]) != len(out[0])):
            return mapping  # Shape mismatch, can't infer

        # Count co-occurrences at same positions
        cooccur: Dict[Tuple[int, int], int] = {}

        for r in range(len(inp)):
            for c in range(len(inp[0]) if inp else 0):
                in_color = inp[r][c]
                out_color = out[r][c]
                key = (in_color, out_color)
                cooccur[key] = cooccur.get(key, 0) + 1

        # Find dominant mapping for each input color
        inp_colors = {inp[r][c] for r in range(len(inp)) for c in range(len(inp[0]) if inp else 0)}

        for in_c in inp_colors:
            candidates = [(out_c, count) for (ic, out_c), count in cooccur.items() if ic == in_c]
            if candidates:
                best_out_c = max(candidates, key=lambda x: x[1])[0]
                if best_out_c != in_c:
                    mapping[in_c] = best_out_c

        return mapping

    def _identify_dominant_pattern(self) -> Optional[PalettePattern]:
        """Identify the most consistent pattern across training examples."""
        if not self.patterns:
            return None

        # Count pattern types
        preserves_count = sum(1 for p in self.patterns if p.preserves_all)
        adds_count = sum(1 for p in self.patterns if p.adds_colors)
        removes_count = sum(1 for p in self.patterns if p.removes_colors)
        remaps_count = sum(1 for p in self.patterns if p.remaps_colors)

        n = len(self.patterns)

        # Consensus: if >80% agree, it's a strong pattern
        if preserves_count >= 0.8 * n:
            pattern = PalettePattern(preserves_all=True)
            pattern.pattern_consistency = preserves_count / n
            return pattern

        if adds_count >= 0.8 * n:
            # Find which colors are consistently added
            added_colors_counts: Dict[int, int] = {}
            for p in self.patterns:
                for c in p.colors_added:
                    added_colors_counts[c] = added_colors_counts.get(c, 0) + 1

            consistent_adds = {c for c, count in added_colors_counts.items() if count >= 0.5 * n}

            pattern = PalettePattern(adds_colors=True, colors_added=consistent_adds)
            pattern.pattern_consistency = adds_count / n
            return pattern

        if removes_count >= 0.8 * n:
            # Find which colors are consistently removed
            removed_colors_counts: Dict[int, int] = {}
            for p in self.patterns:
                for c in p.colors_removed:
                    removed_colors_counts[c] = removed_colors_counts.get(c, 0) + 1

            consistent_removes = {c for c, count in removed_colors_counts.items() if count >= 0.5 * n}

            pattern = PalettePattern(removes_colors=True, colors_removed=consistent_removes)
            pattern.pattern_consistency = removes_count / n
            return pattern

        if remaps_count >= 0.8 * n:
            # Consensus color mapping
            consensus_map = self._consensus_mapping([p.color_mappings for p in self.patterns if p.remaps_colors])
            pattern = PalettePattern(remaps_colors=True, color_mappings=consensus_map)
            pattern.pattern_consistency = remaps_count / n
            return pattern

        # Mixed pattern - be permissive
        return PalettePattern(pattern_consistency=0.5)

    def _consensus_mapping(self, mappings: List[Dict[int, int]]) -> Dict[int, int]:
        """Find consensus color mapping across multiple examples."""
        if not mappings:
            return {}

        # For each source color, find most common target
        all_sources = {src for m in mappings for src in m.keys()}
        consensus = {}

        for src in all_sources:
            targets = [m[src] for m in mappings if src in m]
            if targets:
                most_common = Counter(targets).most_common(1)[0][0]
                consensus[src] = most_common

        return consensus

    def get_valid_palette(self, test_input: Grid) -> Set[int]:
        """
        Predict valid output colors based on test input and learned pattern.
        """
        test_palette = self._get_palette(test_input)

        if not self.dominant_pattern:
            # No strong pattern - allow any colors seen in training
            all_output_colors = set()
            for p in self.patterns:
                all_output_colors.update(p.output_colors)
            return all_output_colors | test_palette

        pattern = self.dominant_pattern

        # Pattern: Preserves all colors
        if pattern.preserves_all:
            return test_palette

        # Pattern: Adds specific colors
        if pattern.adds_colors:
            return test_palette | pattern.colors_added

        # Pattern: Removes specific colors
        if pattern.removes_colors:
            return test_palette - pattern.colors_removed

        # Pattern: Remaps colors
        if pattern.remaps_colors:
            remapped = set()
            for c in test_palette:
                remapped.add(pattern.color_mappings.get(c, c))
            return remapped

        # Fallback: be permissive
        all_colors = set()
        for p in self.patterns:
            all_colors.update(p.output_colors)
        return all_colors | test_palette

    def validate_candidate(self, candidate: Grid, test_input: Grid) -> Tuple[bool, Optional[str]]:
        """
        Validate candidate against learned palette pattern.
        Returns (is_valid, reason_if_invalid).
        """
        candidate_palette = self._get_palette(candidate)
        valid_palette = self.get_valid_palette(test_input)

        forbidden = candidate_palette - valid_palette

        if forbidden:
            # Only reject if pattern is very consistent (>90%)
            if self.dominant_pattern and self.dominant_pattern.pattern_consistency > 0.9:
                return False, f"Uses forbidden colors {forbidden} (pattern conf: {self.dominant_pattern.pattern_consistency:.2f})"

        return True, None


@dataclass
class ShapePattern:
    """Learned shape transformation pattern."""

    input_shape: Tuple[int, int]
    output_shape: Tuple[int, int]

    # Pattern type
    identity: bool = False
    scale: Optional[Tuple[float, float]] = None
    transpose: bool = False
    crop: Optional[Tuple[int, int, int, int]] = None  # top, left, bottom, right
    fixed: bool = False
    uniform_scale: Optional[float] = None


class AdaptiveShapeOracle:
    """
    Learns shape transformation rules from training examples.
    Handles identity, scaling, transposition, cropping, and fixed-size outputs.
    """

    def __init__(self, train_examples: List[Example]):
        self.patterns = self._extract_shape_patterns(train_examples)
        self.dominant_pattern = self._identify_dominant_pattern()

        self._output_shapes = [p.output_shape for p in self.patterns]
        self._output_heights = [shape[0] for shape in self._output_shapes if shape[0] > 0]
        self._output_widths = [shape[1] for shape in self._output_shapes if shape[1] > 0]

        self.max_output_height = max(self._output_heights) if self._output_heights else None
        self.max_output_width = max(self._output_widths) if self._output_widths else None
        self.min_output_height = min(self._output_heights) if self._output_heights else None
        self.min_output_width = min(self._output_widths) if self._output_widths else None

        self._area_ratios: List[float] = []
        self._allows_expansion = False
        for pattern in self.patterns:
            in_h, in_w = pattern.input_shape
            out_h, out_w = pattern.output_shape
            if in_h > 0 and in_w > 0:
                in_area = in_h * in_w
                out_area = out_h * out_w
                self._area_ratios.append(out_area / in_area)

            if pattern.scale:
                h_scale, w_scale = pattern.scale
                if h_scale is not None and w_scale is not None:
                    if h_scale > 1.05 or w_scale > 1.05:
                        self._allows_expansion = True

        self._max_area_ratio = max(self._area_ratios) if self._area_ratios else None
        self._min_area_ratio = min(self._area_ratios) if self._area_ratios else None

    @staticmethod
    def _is_close(a: float, b: float, tol: float = 0.05) -> bool:
        return abs(a - b) <= tol

    def _scale_close(self, a: float, b: float) -> bool:
        max_mag = max(abs(a), abs(b), 1e-6)
        tolerance = max(0.02, 0.1 * max_mag)
        return abs(a - b) <= tolerance

    def _scales_consistent(self, scales: List[Tuple[float, float]]) -> bool:
        if len(scales) <= 1:
            return True
        base_h, base_w = scales[0]
        return all(
            self._scale_close(h, base_h) and self._scale_close(w, base_w)
            for h, w in scales[1:]
        )

    def _values_consistent(self, values: List[float]) -> bool:
        if len(values) <= 1:
            return True
        base = values[0]
        return all(self._is_close(val, base) for val in values[1:])

    def _get_shape(self, grid: Grid) -> Tuple[int, int]:
        """Get (height, width) of grid."""
        h = len(grid)
        w = len(grid[0]) if h > 0 else 0
        return (h, w)

    def _extract_shape_patterns(self, train_examples: List[Example]) -> List[ShapePattern]:
        """Extract shape transformation from each training example."""
        patterns = []

        for ex in train_examples:
            in_shape = self._get_shape(ex['input'])
            out_shape = self._get_shape(ex['output'])

            pattern = ShapePattern(input_shape=in_shape, output_shape=out_shape)

            in_h, in_w = in_shape
            out_h, out_w = out_shape

            # Identity
            if in_shape == out_shape:
                pattern.identity = True

            # Scaling
            elif in_h > 0 and in_w > 0:
                h_scale = out_h / in_h
                w_scale = out_w / in_w

                pattern.scale = (h_scale, w_scale)

                if self._scale_close(h_scale, w_scale):  # Uniform scaling
                    avg_scale = (h_scale + w_scale) / 2
                    pattern.scale = (avg_scale, avg_scale)
                    pattern.uniform_scale = avg_scale
                else:
                    # Some training pairs only scale one axis (e.g. rotated examples).
                    if self._is_close(h_scale, 1.0) and w_scale > 0:
                        pattern.uniform_scale = w_scale
                    elif self._is_close(w_scale, 1.0) and h_scale > 0:
                        pattern.uniform_scale = h_scale

            # Transpose
            if (in_h, in_w) == (out_w, out_h):
                pattern.transpose = True

            patterns.append(pattern)

        return patterns

    def _identify_dominant_pattern(self) -> Optional[ShapePattern]:
        """Identify most consistent shape pattern."""
        if not self.patterns:
            return None

        n = len(self.patterns)

        # Check for identity
        identity_count = sum(1 for p in self.patterns if p.identity)
        if identity_count >= 0.8 * n:
            return ShapePattern(input_shape=(0, 0), output_shape=(0, 0), identity=True)

        # Check for fixed output shape
        output_shapes = [p.output_shape for p in self.patterns]
        if len(set(output_shapes)) == 1:
            return ShapePattern(
                input_shape=(0, 0),
                output_shape=output_shapes[0],
                fixed=True
            )

        # Check for consistent scaling
        scales = [p.scale for p in self.patterns if p.scale is not None]
        if scales and len(scales) >= 0.8 * n:
            if self._scales_consistent(scales):
                base = scales[0]
                uniform = (base[0] + base[1]) / 2 if self._scale_close(base[0], base[1]) else None
                return ShapePattern(
                    input_shape=(0, 0),
                    output_shape=(0, 0),
                    scale=base,
                    uniform_scale=uniform,
                )

        uniform_patterns = [p for p in self.patterns if p.uniform_scale is not None]
        uniform_scales = [p.uniform_scale for p in uniform_patterns]
        if uniform_scales and len(uniform_scales) >= 0.8 * n:
            if self._values_consistent(uniform_scales):
                scaled_h = any(p.scale and not self._is_close(p.scale[0], 1.0) for p in uniform_patterns)
                scaled_w = any(p.scale and not self._is_close(p.scale[1], 1.0) for p in uniform_patterns)
                if scaled_h and scaled_w:
                    factor = uniform_scales[0]
                    return ShapePattern(
                        input_shape=(0, 0),
                        output_shape=(0, 0),
                        scale=(factor, factor),
                        uniform_scale=factor
                    )

        # Check for transpose (lowered to 50% - transpose is common in ARC)
        transpose_count = sum(1 for p in self.patterns if p.transpose)
        if transpose_count >= 0.5 * n:
            return ShapePattern(input_shape=(0, 0), output_shape=(0, 0), transpose=True)

        # No strong pattern
        return None

    def predict_shape(self, test_input_shape: Tuple[int, int]) -> Tuple[Tuple[int, int], float]:
        """
        Predict output shape for test input.
        Returns (predicted_shape, confidence).
        """
        if not self.dominant_pattern:
            # No pattern - use most common output shape
            output_shapes = [p.output_shape for p in self.patterns]
            most_common, count = Counter(output_shapes).most_common(1)[0]
            confidence = 0.6 + 0.4 * (count / len(output_shapes))
            confidence = min(confidence, 0.95)
            return most_common, confidence

        pattern = self.dominant_pattern
        in_h, in_w = test_input_shape

        # Identity
        if pattern.identity:
            return (in_h, in_w), 0.95

        # Fixed size
        if pattern.fixed:
            return pattern.output_shape, 0.90

        # Scaling
        if pattern.scale:
            h_scale, w_scale = pattern.scale
            if pattern.uniform_scale is not None:
                h_scale = w_scale = pattern.uniform_scale
            out_h = max(1, round(in_h * h_scale))
            out_w = max(1, round(in_w * w_scale))
            return (out_h, out_w), 0.85

        # Transpose
        if pattern.transpose:
            return (in_w, in_h), 0.80

        # Fallback
        output_shapes = [p.output_shape for p in self.patterns]
        most_common, count = Counter(output_shapes).most_common(1)[0]
        confidence = 0.55 + 0.35 * (count / len(output_shapes))
        confidence = min(confidence, 0.9)
        return most_common, confidence

    def validate_candidate(self, candidate_shape: Tuple[int, int], test_input_shape: Tuple[int, int]) -> Tuple[bool, Optional[str]]:
        """
        Validate candidate shape against learned pattern.
        Returns (is_valid, reason_if_invalid).
        """
        predicted_shape, confidence = self.predict_shape(test_input_shape)

        # Only enforce if very confident (>85%)
        if confidence > 0.85 and candidate_shape != predicted_shape:
            return False, f"Shape mismatch: expected {predicted_shape}, got {candidate_shape} (conf: {confidence:.2f})"

        cand_h, cand_w = candidate_shape

        if not self._allows_expansion and self.max_output_height is not None and self.max_output_width is not None:
            height_buffer = 1 if self.max_output_height < test_input_shape[0] else 0
            width_buffer = 1 if self.max_output_width < test_input_shape[1] else 0
            if cand_h > self.max_output_height + height_buffer or cand_w > self.max_output_width + width_buffer:
                return False, (
                    f"Shape {candidate_shape} exceeds observed max {(self.max_output_height, self.max_output_width)} "
                    "without expansion evidence"
                )

        if self._max_area_ratio is not None and not self._allows_expansion:
            test_h, test_w = test_input_shape
            test_area = test_h * test_w
            if test_area > 0:
                candidate_ratio = (cand_h * cand_w) / test_area
                tolerance = max(0.05, 0.25 * self._max_area_ratio)
                if candidate_ratio > self._max_area_ratio + tolerance:
                    return False, (
                        f"Area ratio {candidate_ratio:.3f} exceeds observed max {self._max_area_ratio:.3f}"
                        " without expansion evidence"
                    )

        return True, None


def count_components(grid: Grid, ignore_bg: bool = True) -> int:
    """Count connected components in grid (stdlib-only, no scipy)."""
    if not grid or not grid[0]:
        return 0

    h, w = len(grid), len(grid[0])
    visited = [[False] * w for _ in range(h)]

    def flood_fill(r: int, c: int, color: int) -> None:
        """DFS flood fill."""
        stack = [(r, c)]
        visited[r][c] = True

        while stack:
            cr, cc = stack.pop()
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and grid[nr][nc] == color:
                    visited[nr][nc] = True
                    stack.append((nr, nc))

    component_count = 0
    bg_color = None
    if ignore_bg:
        color_counts = Counter(cell for row in grid for cell in row)
        if 0 in color_counts:
            bg_color = 0
        elif len(color_counts) > 1:
            bg_color = color_counts.most_common(1)[0][0]

    for r in range(h):
        for c in range(w):
            if not visited[r][c]:
                color = grid[r][c]
                if bg_color is not None and color == bg_color:
                    visited[r][c] = True
                    continue
                flood_fill(r, c, color)
                component_count += 1

    return component_count


class ComponentAnalyzer:
    """Analyzes component structure fidelity."""

    def __init__(self, train_examples: List[Example]):
        self.expected_ratio = self._compute_component_ratio(train_examples)
        self.ratios = self._extract_ratios(train_examples)

    def _compute_component_ratio(self, train_examples: List[Example]) -> float:
        """Average component count ratio output/input."""
        ratios = []

        for ex in train_examples:
            in_comps = count_components(ex['input'])
            out_comps = count_components(ex['output'])

            if in_comps > 0:
                ratios.append(out_comps / in_comps)

        return sum(ratios) / len(ratios) if ratios else 1.0

    def _extract_ratios(self, train_examples: List[Example]) -> List[float]:
        """Extract all component ratios."""
        ratios = []
        for ex in train_examples:
            in_comps = count_components(ex['input'])
            out_comps = count_components(ex['output'])
            if in_comps > 0:
                ratios.append(out_comps / in_comps)
        return ratios

    def score_candidate(self, candidate: Grid, test_input: Grid) -> float:
        """
        Score candidate based on component structure fidelity.
        Returns multiplier in range [0.3, 1.0].
        """
        test_comps = count_components(test_input)
        cand_comps = count_components(candidate)

        if test_comps == 0:
            return 1.0  # No meaningful structure

        actual_ratio = cand_comps / test_comps
        deviation = abs(actual_ratio - self.expected_ratio)

        # Heavy penalty for severe deviation
        if deviation > 1.0:  # 2x or more off
            return 0.3
        elif deviation > 0.5:
            return 0.6
        elif deviation > 0.3:
            return 0.8
        else:
            return 1.0
