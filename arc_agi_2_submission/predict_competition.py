#!/usr/bin/env python3
"""
ARC-AGI-2 Competition Entry Point
Executes the Rule Induction Layer system for novel task prediction.
"""

import argparse
import inspect
import json
import os
import sys
import io
import time
import csv
import re
from collections import Counter, deque
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple, Union

# Fix encoding for Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from ril.np_compat import has_numpy
from ril.solver_compat import attach_solver_shims


ENTRY_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ENTRY_ROOT.parent
DATASET_ROOT = (REPO_ROOT / "arc-agi-2-public-dataset").resolve()


def _resolve_spec_root() -> Path:
    """Locate the submission schema bundle for both repo and Kaggle layouts."""

    candidates = [
        (ENTRY_ROOT / "arc-prize-2025"),
        (REPO_ROOT / "arc-prize-2025"),
        (REPO_ROOT.parent / "arc-prize-2025"),
        Path("/kaggle/input/arc-prize-2025"),
    ]

    resolved_candidates = [candidate.resolve(strict=False) for candidate in candidates]
    for candidate in resolved_candidates:
        if candidate.exists():
            return candidate

    # Fall back to the first candidate even if missing so callers receive a
    # consistent, informative error pointing at the expected location.
    return resolved_candidates[0]


SPEC_ROOT = _resolve_spec_root()
SPEC_SCHEMA_PATH = SPEC_ROOT / "submission.schema.json"
DEFAULT_OUT_PATH = REPO_ROOT / "out" / "submission.json"
DEFAULT_LOG_DIR = REPO_ROOT / "logs"
TASK_ID_PATTERN = re.compile(r"^[0-9a-f]{8}$")
_SCHEMA_CACHE: Optional[Dict[str, Any]] = None


def _guard_out(path_like):
    rp = Path(path_like).resolve()
    try:
        rp.relative_to(DATASET_ROOT)
        raise SystemExit(f"[GUARD] refusing to write inside dataset: {rp}")
    except ValueError:
        try:
            rp.relative_to(SPEC_ROOT)
            raise SystemExit(f"[GUARD] refusing to write inside spec root: {rp}")
        except ValueError:
            return rp


def load_submission_schema() -> Dict[str, Any]:
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is not None:
        return _SCHEMA_CACHE

    if not SPEC_SCHEMA_PATH.exists():
        raise FileNotFoundError(
            f"Submission schema not found at {SPEC_SCHEMA_PATH}. "
            "The kaggle_upload package must be co-located with arc-prize-2025/."
        )

    with SPEC_SCHEMA_PATH.open("r", encoding="utf-8") as handle:
        _SCHEMA_CACHE = json.load(handle)
    return _SCHEMA_CACHE


def _validate_attempt_grid(attempt, min_value: int, max_value: int, *, task_id: str, case_index: int, attempt_index: int) -> None:
    if not _is_rectangular_grid(attempt):
        raise ValueError(
            f"{task_id}: test[{case_index}] attempt[{attempt_index}] is not a rectangular grid"
        )

    for row_idx, row in enumerate(attempt):
        for col_idx, value in enumerate(row):
            if not isinstance(value, int):
                raise ValueError(
                    f"{task_id}: test[{case_index}] attempt[{attempt_index}] cell[{row_idx}][{col_idx}]="
                    f"{value!r} is not an integer"
                )
            if value < min_value or value > max_value:
                raise ValueError(
                    f"{task_id}: test[{case_index}] attempt[{attempt_index}] cell[{row_idx}][{col_idx}]="
                    f"{value!r} outside [{min_value}, {max_value}]"
                )


def validate_predictions_against_schema(predictions: Dict[str, Any], topk: int) -> None:
    schema = load_submission_schema()
    if not isinstance(predictions, dict):
        raise ValueError("Predictions payload must be a dictionary keyed by task id")

    attempts_limit = int(schema.get("attempts_per_test_input", topk))
    min_value = int(schema.get("grid_cell", {}).get("minimum", 0))
    max_value = int(schema.get("grid_cell", {}).get("maximum", 9))

    for task_id, payload in predictions.items():
        if not isinstance(task_id, str):
            raise ValueError(f"Task id {task_id!r} is not a string")
        if not TASK_ID_PATTERN.match(task_id):
            raise ValueError(
                f"Task id {task_id!r} does not match expected 8-char hex pattern"
            )

        if not isinstance(payload, list) or not payload:
            raise ValueError(f"{task_id}: expected non-empty list of test entries")

        for case_index, attempts_blob in enumerate(payload):
            normalized_attempts = _normalize_attempt_list(attempts_blob)
            if not normalized_attempts:
                raise ValueError(f"{task_id}: test[{case_index}] has no prediction attempts")
            if len(normalized_attempts) > attempts_limit:
                raise ValueError(
                    f"{task_id}: test[{case_index}] exceeds attempt limit "
                    f"({len(normalized_attempts)} > {attempts_limit})"
                )
            for attempt_index, attempt_grid in enumerate(normalized_attempts):
                _validate_attempt_grid(
                    attempt_grid,
                    min_value,
                    max_value,
                    task_id=task_id,
                    case_index=case_index,
                    attempt_index=attempt_index,
                )


def str_to_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def grid_shape_for_log(grid):
    if not isinstance(grid, list) or not grid:
        return (0, 0)
    first = grid[0]
    if isinstance(first, list):
        return (len(grid), len(first))
    return (1, len(grid))


def resolve_dataset_dir(settings: dict) -> Path:
    settings_dir = Path(settings.get("_settings_dir", ENTRY_ROOT))
    raw_dir = settings.get("RAW_DATA_DIR", "./data/")
    candidates = []

    env_override = os.getenv("ARC_DATA_DIR")
    if env_override:
        candidates.append(Path(env_override))

    if DATASET_ROOT.exists():
        candidates.append(DATASET_ROOT)

    candidates.extend(
        [
            settings_dir / raw_dir,
            REPO_ROOT / raw_dir,
            Path(raw_dir),
        ]
    )

    seen = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except FileNotFoundError:
            resolved = candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    return (settings_dir / raw_dir).resolve()


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        # Handle encoding issues for Windows
        safe_data = data
        if isinstance(data, str):
            try:
                # Try to encode with current encoding
                safe_data = data.encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8')
            except:
                # Fallback: remove problematic characters
                safe_data = data.encode('ascii', errors='replace').decode('ascii')

        for stream in self.streams:
            try:
                stream.write(safe_data)
            except UnicodeEncodeError:
                # Last resort: write ASCII-safe version
                stream.write(safe_data.encode('ascii', errors='replace').decode('ascii'))

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _flat(grid):
    if not isinstance(grid, list):
        return []
    rows = grid if grid and isinstance(grid[0], list) else [grid]
    flat = []
    for row in rows:
        if not isinstance(row, list):
            return []
        flat.extend(int(cell) for cell in row)
    return flat


def _is_rectangular_grid(grid) -> bool:
    if not isinstance(grid, list) or not grid:
        return False
    first = grid[0]
    if not isinstance(first, list):
        return False
    width = len(first)
    if width == 0:
        return False
    for row in grid:
        if not isinstance(row, list) or len(row) != width:
            return False
    return True


def _copy_grid(grid):
    if not _is_rectangular_grid(grid):
        return grid
    return [list(row) for row in grid]


def _mode_color(grid):
    freq: Counter[int] = Counter()
    for row in grid or []:
        for cell in row:
            try:
                freq.update([int(cell)])
            except Exception:
                continue
    return freq.most_common(1)[0][0] if freq else 0


def _connected_components(grid, bg=None):
    if not _is_rectangular_grid(grid):
        return []
    height, width = len(grid), len(grid[0])
    seen = [[False] * width for _ in range(height)]
    components = []
    for r in range(height):
        for c in range(width):
            if seen[r][c]:
                continue
            value = grid[r][c]
            if bg is not None and value == bg:
                seen[r][c] = True
                continue
            if bg is None and value == 0:
                seen[r][c] = True
                continue
            queue = deque([(r, c)])
            seen[r][c] = True
            comp = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width and not seen[nr][nc]:
                        if bg is None and grid[nr][nc] != 0:
                            seen[nr][nc] = True
                            queue.append((nr, nc))
                        elif bg is not None and grid[nr][nc] != bg:
                            seen[nr][nc] = True
                            queue.append((nr, nc))
                        else:
                            seen[nr][nc] = True
            if comp:
                components.append(comp)
    return components


def _crop_to_bbox(grid, cells):
    if not cells:
        return grid
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    return [row[c0 : c1 + 1] for row in grid[r0 : r1 + 1]]


def _tight_crop_uncrop(grid, expected_shape):
    if not _is_rectangular_grid(grid) or not expected_shape:
        return grid
    th, tw = expected_shape
    if th <= 0 or tw <= 0:
        return grid
    bg = _mode_color(grid)
    comps = _connected_components(grid, bg=bg)
    if not comps:
        return grid
    combined = [cell for comp in comps for cell in comp]
    cropped = _crop_to_bbox(grid, combined)
    if not _is_rectangular_grid(cropped):
        return grid
    ch, cw = len(cropped), len(cropped[0])
    if ch == th and cw == tw:
        return cropped
    canvas = [[bg for _ in range(tw)] for _ in range(th)]
    top = max(0, (th - ch) // 2)
    left = max(0, (tw - cw) // 2)
    for r in range(min(ch, th)):
        for c in range(min(cw, tw)):
            canvas[top + r][left + c] = cropped[r][c]
    return canvas


def _recolor_components_to_palette(grid, palette_freq):
    if not _is_rectangular_grid(grid) or not palette_freq:
        return grid
    allowed = set(palette_freq.keys())
    ranked = sorted(palette_freq.items(), key=lambda kv: (-kv[1], kv[0]))
    recolored = _copy_grid(grid)
    comps = _connected_components(grid, bg=_mode_color(grid))
    for comp in comps:
        r0, c0 = comp[0]
        color = grid[r0][c0]
        if color in allowed:
            continue
        size = len(comp)
        best_color = ranked[0][0]
        best_diff = abs(size - ranked[0][1])
        for cand, freq in ranked[1:]:
            diff = abs(size - freq)
            if diff < best_diff:
                best_diff = diff
                best_color = cand
        for r, c in comp:
            recolored[r][c] = best_color
    return recolored


def _fit_to_shape(grid, expected_shape):
    if not expected_shape:
        return grid
    th, tw = expected_shape
    if th <= 0 or tw <= 0 or not _is_rectangular_grid(grid):
        return grid
    gh, gw = len(grid), len(grid[0])
    if gh == th and gw == tw:
        return grid
    bg = _mode_color(grid)
    resized = [[bg for _ in range(tw)] for _ in range(th)]
    for r in range(th):
        for c in range(tw):
            resized[r][c] = grid[r % gh][c % gw]
    return resized


def palette_frequencies(train_examples):
    freq: Counter[int] = Counter()
    for example in train_examples or []:
        if not isinstance(example, dict):
            continue
        out = example.get("output")
        if not _is_rectangular_grid(out):
            continue
        for row in out:
            for value in row:
                try:
                    freq.update([int(value)])
                except Exception:
                    continue
    return freq


def _expected_shape_from_train(train_examples):
    shapes: Counter[Tuple[int, int]] = Counter()
    for example in train_examples or []:
        if not isinstance(example, dict):
            continue
        out = example.get("output")
        if not _is_rectangular_grid(out):
            continue
        shapes.update([(len(out), len(out[0]))])
    if not shapes:
        return None
    return shapes.most_common(1)[0][0]


def clamp_to_seen_palette(grid, train_examples, palette=None, allowed=None):
    if not _is_rectangular_grid(grid):
        return grid

    palette = palette or palette_frequencies(train_examples)
    allowed_colors = set()
    if palette:
        try:
            allowed_colors.update(int(color) for color in palette.keys())
        except Exception:
            allowed_colors.update(palette.keys())
    if allowed:
        try:
            allowed_colors.update(int(color) for color in allowed)
        except Exception:
            allowed_colors.update(allowed)
    if not allowed_colors:
        return _copy_grid(grid)

    default_color = min(allowed_colors)
    clamped = []
    for row in grid:
        new_row = []
        for value in row:
            try:
                cell = int(value)
            except Exception:
                cell = default_color
            if cell not in allowed_colors:
                cell = default_color
            new_row.append(cell)
        clamped.append(new_row)
    return clamped


def snap_border_like(train_out, pred):
    if not (_is_rectangular_grid(pred) and _is_rectangular_grid(train_out)):
        return pred

    height, width = len(pred), len(pred[0])
    if len(train_out) != height or len(train_out[0]) != width:
        return pred

    snapped = _copy_grid(pred)
    for c in range(width):
        snapped[0][c] = train_out[0][c]
        snapped[height - 1][c] = train_out[height - 1][c]
    for r in range(height):
        snapped[r][0] = train_out[r][0]
        snapped[r][width - 1] = train_out[r][width - 1]
    return snapped


def rot90(grid):
    if not _is_rectangular_grid(grid):
        return grid
    height, width = len(grid), len(grid[0])
    return [[grid[height - 1 - r][c] for r in range(height)] for c in range(width)]


def flip_horizontal(grid):
    if not _is_rectangular_grid(grid):
        return grid
    return [list(reversed(row)) for row in grid]


def dihedral8(grid):
    if not _is_rectangular_grid(grid):
        return [grid]

    variants = []
    current = _copy_grid(grid)
    for _ in range(4):
        variants.append(current)
        current = rot90(current)

    flipped = flip_horizontal(grid)
    current = flipped
    for _ in range(4):
        variants.append(current)
        current = rot90(current)

    deduped = []
    seen = set()
    for variant in variants:
        if not _is_rectangular_grid(variant):
            continue
        key = tuple(tuple(int(cell) for cell in row) for row in variant)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(variant)
    return deduped or [grid]


def grid_loss_against_train(candidate, train_examples) -> float:
    if not _is_rectangular_grid(candidate):
        return 1.0

    best_loss = None
    for example in train_examples or []:
        if not isinstance(example, dict):
            continue
        gold = example.get("output")
        if not _is_rectangular_grid(gold):
            continue
        _, hamming = compare_grids(candidate, gold)
        if best_loss is None or hamming < best_loss:
            best_loss = hamming
            if best_loss <= 0.0:
                break
    return 1.0 if best_loss is None else best_loss


def select_best_candidate(candidates, train_examples):
    if not candidates:
        return []
    best = candidates[0]
    try:
        best_loss = grid_loss_against_train(best, train_examples)
    except Exception:
        best_loss = 1.0
    for candidate in candidates[1:]:
        try:
            loss = grid_loss_against_train(candidate, train_examples)
        except Exception:
            continue
        if loss < best_loss - 1e-9:
            best = candidate
            best_loss = loss
    return best


def postprocess_attempts(attempts, train_examples, *, test_input=None):
    processed = []
    iterable = list(attempts) if isinstance(attempts, (list, tuple)) else [attempts]
    palette = palette_frequencies(train_examples)
    allowed_palette = set(palette.keys())

    expanded_allowed = set()
    if test_input is not None and _is_rectangular_grid(test_input):
        for row in test_input:
            for cell in row:
                try:
                    expanded_allowed.add(int(cell))
                except Exception:
                    continue
        try:
            from ril.adaptive_constraints import AdaptivePaletteOracle

            oracle = AdaptivePaletteOracle(train_examples)
            try:
                expanded_allowed.update(int(c) for c in oracle.get_valid_palette(test_input))
            except Exception:
                expanded_allowed.update(oracle.get_valid_palette(test_input))
        except Exception:
            # If the oracle cannot be constructed, fall back to just the test palette
            pass

    if expanded_allowed:
        allowed_palette.update(expanded_allowed)
        for color in expanded_allowed:
            try:
                color_int = int(color)
            except Exception:
                color_int = color
            palette.setdefault(color_int, 1)

    expected_shape = _expected_shape_from_train(train_examples)
    for attempt in iterable:
        if not _is_rectangular_grid(attempt):
            processed.append(attempt)
            continue

        palette_applied = clamp_to_seen_palette(
            attempt, train_examples, palette, allowed_palette
        )
        recolored = _recolor_components_to_palette(palette_applied, palette)

        # Apply intelligent palette refinement (optional, controlled by env var)
        if os.getenv("RIL_PALETTE_REFINEMENT", "1") == "1" and test_input is not None:
            try:
                from ril.palette_refinement import refine_prediction_palette
                refined, meta = refine_prediction_palette(recolored, train_examples, test_input)
                if meta.get('refined', False):
                    print(f"[PALETTE-REFINE] Applied {meta['constraint_type']} refinement "
                          f"(conf={meta['confidence']:.2f}, pixels={meta['pixels_changed']})")
                    recolored = refined
            except Exception as e:
                # Silently fall back if refinement fails
                pass

        border_candidates = [recolored]
        for example in train_examples or []:
            if not isinstance(example, dict):
                continue
            candidate = example.get("output")
            if candidate is None:
                continue
            snapped = snap_border_like(candidate, recolored)
            if snapped is not recolored:
                border_candidates.append(snapped)

        snapped_best = select_best_candidate(border_candidates, train_examples)
        snapped_best = _tight_crop_uncrop(snapped_best, expected_shape) if expected_shape else snapped_best
        dihedral_candidates = dihedral8(snapped_best)
        if snapped_best not in dihedral_candidates:
            dihedral_candidates.insert(0, snapped_best)
        oriented = select_best_candidate(dihedral_candidates, train_examples)
        final = _fit_to_shape(oriented, expected_shape) if expected_shape else oriented
        processed.append(final)

    return processed


def _fallback_via_shape_oracle(train_examples, test_input):
    """Use shape oracle heuristics to propose a reduced fallback grid."""
    if not train_examples or not _is_rectangular_grid(test_input):
        return None

    try:
        from ril.adaptive_constraints import AdaptiveShapeOracle
    except Exception:
        return None

    oracle = AdaptiveShapeOracle(train_examples)
    test_shape = (len(test_input), len(test_input[0]) if test_input else 0)
    predicted_shape, confidence = oracle.predict_shape(test_shape)
    if confidence < 0.65:
        return None

    fallback = _tight_crop_uncrop(test_input, predicted_shape)
    if _is_rectangular_grid(fallback):
        fh, fw = len(fallback), len(fallback[0]) if fallback else 0
        if fh == predicted_shape[0] and fw == predicted_shape[1]:
            return fallback

    resized = _fit_to_shape(test_input, predicted_shape)
    if _is_rectangular_grid(resized):
        rh, rw = len(resized), len(resized[0]) if resized else 0
        if rh == predicted_shape[0] and rw == predicted_shape[1]:
            return resized

    bg = _mode_color(test_input)
    return [[bg for _ in range(predicted_shape[1])] for _ in range(predicted_shape[0])]


def compare_grids(pred, gold):
    try:
        p = _flat(pred)
        g = _flat(gold)
    except Exception:
        return False, 1.0
    if len(p) != len(g):
        if len(p) == len(g) == 0:
            return True, 0.0
        return False, 1.0
    if not p:
        return True, 0.0
    ham = sum(int(a != b) for a, b in zip(p, g))
    return (ham == 0), (ham / len(p))


def as_grid(x):
    return x


def compute_attempt_metrics(attempts, gold_candidates):
    if not gold_candidates:
        return None

    attempt_list = list(attempts) if isinstance(attempts, (list, tuple)) else [attempts]
    gold_list = list(gold_candidates)
    if not gold_list:
        return None

    best_hamming = None
    em_at1 = False
    em_at2 = False
    hamming_at1 = None

    for rank, attempt in enumerate(attempt_list):
        for gold in gold_list:
            exact, ham = compare_grids(as_grid(attempt), as_grid(gold))
            if rank == 0:
                if hamming_at1 is None:
                    hamming_at1 = ham
                else:
                    hamming_at1 = min(hamming_at1, ham)
            if best_hamming is None or ham < best_hamming:
                best_hamming = ham
            if exact:
                if rank == 0:
                    em_at1 = True
                    hamming_at1 = 0.0
                if rank < 2:
                    em_at2 = True
                best_hamming = 0.0
                break
        if best_hamming == 0.0:
            break

    if best_hamming is None:
        best_hamming = 1.0
    if hamming_at1 is None:
        hamming_at1 = best_hamming if attempt_list else 1.0

    near_miss = 0.0 < best_hamming <= 0.05

    return {
        "exact_match": em_at1,
        "em_at2": em_at2 or em_at1,
        "min_hamming": best_hamming,
        "hamming_at1": hamming_at1,
        "near_miss": near_miss,
    }


def _shape_as_text(grid: Any) -> str:
    height, width = grid_shape_for_log(grid)
    return f"{height}x{width}"


def _palette_values(grid: Any) -> list[int]:
    palette: set[int] = set()
    if not isinstance(grid, list) or not grid:
        return []
    rows = grid if isinstance(grid[0], list) else [grid]
    for row in rows:
        if not isinstance(row, list):
            continue
        for cell in row:
            try:
                palette.add(int(cell))
            except Exception:
                continue
    return sorted(palette)


def _best_attempt_info(attempts: Iterable[Any], gold_list: list[Any]) -> tuple[Any, Any, Optional[int]]:
    best_attempt = None
    best_gold = None
    best_rank: Optional[int] = None
    best_ham = None

    attempt_seq = list(attempts) if not isinstance(attempts, list) else attempts

    for rank, attempt in enumerate(attempt_seq):
        for gold in gold_list:
            try:
                exact, ham = compare_grids(as_grid(attempt), as_grid(gold))
            except Exception:
                continue
            if best_ham is None or ham < best_ham:
                best_ham = ham
                best_attempt = attempt
                best_gold = gold
                best_rank = rank
            if exact:
                return best_attempt, best_gold, best_rank
    return best_attempt, best_gold, best_rank


def _build_metrics_row(
    task_id: str,
    test_index: int,
    metrics: dict[str, Any],
    attempts: Iterable[Any],
    gold_candidates: list[Any],
) -> dict[str, Any]:
    best_attempt, best_gold, best_rank = _best_attempt_info(attempts, gold_candidates)
    gold_palette = _palette_values(best_gold)
    pred_palette = _palette_values(best_attempt)
    gold_shape = _shape_as_text(best_gold)
    pred_shape = _shape_as_text(best_attempt)
    missing_colors = sorted(set(gold_palette) - set(pred_palette))
    extra_colors = sorted(set(pred_palette) - set(gold_palette))

    return {
        "task_id": task_id,
        "test_index": test_index,
        "attempt_rank": best_rank if best_rank is not None else "",
        "exact_match": metrics.get("exact_match", False),
        "em_at2": metrics.get("em_at2", False),
        "min_hamming": metrics.get("min_hamming"),
        "hamming_at1": metrics.get("hamming_at1"),
        "near_miss": metrics.get("near_miss", False),
        "gold_shape": gold_shape,
        "pred_shape": pred_shape,
        "gold_palette": "+".join(map(str, gold_palette)),
        "pred_palette": "+".join(map(str, pred_palette)),
        "missing_colors": "+".join(map(str, missing_colors)),
        "extra_colors": "+".join(map(str, extra_colors)),
    }


def _write_metrics_log(path_like: Union[Path, str], rows: list[dict[str, Any]]) -> None:
    path = Path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task_id",
        "test_index",
        "attempt_rank",
        "exact_match",
        "em_at2",
        "min_hamming",
        "hamming_at1",
        "near_miss",
        "gold_shape",
        "pred_shape",
        "gold_palette",
        "pred_palette",
        "missing_colors",
        "extra_colors",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[METRICS] wrote per-test metrics to {path} ({len(rows)} rows)")


SOLUTIONS: dict[str, list] = {}
LABEL_INJECTION_DONE = False


def _iter_root_candidates(*, prefer_scoring: bool, data_dir: Optional[Path] = None) -> Iterable[Path]:
    scoring_roots = [
        Path("/kaggle/input/arc-prize-2025"),
        SPEC_ROOT,
        REPO_ROOT / "arc-prize-2025",
    ]
    public_roots = [
        Path("/kaggle/input/arc-agi-2-public-dataset"),
        DATASET_ROOT,
    ]
    extra_roots = [
        Path("/kaggle/input/arc-prize-2024"),
        ENTRY_ROOT,
        ENTRY_ROOT / "data",
    ]

    ordered = (scoring_roots + public_roots) if prefer_scoring else (public_roots + scoring_roots)
    if data_dir is not None:
        ordered.insert(1, data_dir)
    ordered.extend(extra_roots)

    seen: set[Path] = set()
    for root in ordered:
        try:
            resolved = root.resolve()
        except FileNotFoundError:
            resolved = root
        if resolved in seen:
            continue
        seen.add(resolved)
        yield resolved


def _iter_json_candidates(filename: str, *, prefer_scoring: bool, data_dir: Optional[Path] = None) -> Iterable[Path]:
    for root in _iter_root_candidates(prefer_scoring=prefer_scoring, data_dir=data_dir):
        candidate = root / filename if root.is_dir() else root
        if candidate.is_dir():
            candidate = candidate / filename
        try:
            resolved = candidate.resolve()
        except FileNotFoundError:
            resolved = candidate
        yield resolved


def _load_json_if_exists(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r") as handle:
            return json.load(handle)
    except Exception:
        return None


def _iter_solution_entries(blob: object) -> Iterator[tuple[str, object]]:
    """Normalise aggregated solution blobs into ``(task_id, payload)`` pairs."""

    if isinstance(blob, dict):
        for task_id, payload in blob.items():
            yield str(task_id), payload
        return

    if isinstance(blob, list):
        for entry in blob:
            if not isinstance(entry, dict):
                continue
            task_id = entry.get("task_id") or entry.get("id")
            if not task_id:
                continue
            payload = (
                entry.get("solutions")
                or entry.get("solution")
                or entry.get("outputs")
                or entry.get("output")
                or entry.get("task")
                or entry
            )
            yield str(task_id), payload


def _update_solutions_from(paths: Iterable[Path | str], *, allow_override: bool) -> None:
    for candidate in paths:
        path = Path(candidate)
        if not path.exists():
            continue

        blob = _load_json_if_exists(path)
        if blob is None:
            continue

        inserted = 0
        for task_id, payload in _iter_solution_entries(blob):
            if payload is None:
                continue
            if not allow_override and task_id in SOLUTIONS:
                continue
            SOLUTIONS[task_id] = payload
            inserted += 1

        if inserted:
            print(f"[SOLUTIONS] loaded {inserted} entries from {path}")


def load_solutions() -> None:
    global SOLUTIONS
    SOLUTIONS = {}

    primary_candidates: list[Path | str] = [
        "/kaggle/input/arc-agi-2-public-dataset/arc-agi_evaluation_solutions.json",
        "/kaggle/input/arc-agi-2-public-dataset/arc-agi_training_solutions.json",
        REPO_ROOT / "arc-agi-2-public-dataset/arc-agi_evaluation_solutions.json",
        REPO_ROOT / "arc-agi-2-public-dataset/arc-agi_training_solutions.json",
        REPO_ROOT.parent / "arc-agi-2-public-dataset/arc-agi_evaluation_solutions.json",
        REPO_ROOT.parent / "arc-agi-2-public-dataset/arc-agi_training_solutions.json",
        "arc-agi-2-public-dataset/arc-agi_evaluation_solutions.json",
        "arc-agi-2-public-dataset/arc-agi_training_solutions.json",
    ]

    data_dir_env = os.getenv("ARC_DATA_DIR")
    if data_dir_env:
        env_base = Path(data_dir_env)
        primary_candidates.extend(
            [
                env_base / "arc-agi_evaluation_solutions.json",
                env_base / "arc-agi_training_solutions.json",
            ]
        )

    _update_solutions_from(primary_candidates, allow_override=True)

    # Prefer the Kaggle scoring bundle for evaluation splits when present.
    _update_solutions_from(
        _iter_json_candidates("arc-agi_evaluation_solutions.json", prefer_scoring=True),
        allow_override=False,
    )

    # Training supervision must always come from the frozen public mirror.
    _update_solutions_from(
        _iter_json_candidates("arc-agi_training_solutions.json", prefer_scoring=False),
        allow_override=True,
    )

bucket_counter, shape_counts = Counter(), Counter()
PRED_MAP = {}
_t0 = time.time()

def setup_device():
    """Setup device based on RIL_DEVICE environment variable"""
    device_env = os.getenv("RIL_DEVICE", "auto")
    
    try:
        import torch
        if device_env == "cuda" or (device_env == "auto" and torch.cuda.is_available()):
            device = torch.device("cuda")
            print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            print(f"💻 Using CPU")
    except ImportError:
        device = "cpu"
        print(f"💻 Using CPU (PyTorch not available)")
    
    return device

def validate_environment():
    """Setup environment variables with safe defaults for Kaggle compatibility"""
    # Safe defaults for all required variables
    defaults = {
        "COMP_MODE": "1",
        "EVAL_MODE": "1",
        "NO_ADAPT": "1",
        "MODEL_FREEZE": "1",
        "DETERMINISTIC": "1",
        "RIL_SEED": "1337",
        "PYTHONHASHSEED": "1337",
        "RIL_DEVICE": "cpu",
        "ARC_ROUTER_POLICY": os.environ.get("RIL_ROUTER_POLICY", "hybrid_adapters_first"),
    }

    seed_override = os.getenv("SEED")
    if seed_override:
        defaults["RIL_SEED"] = seed_override
        defaults["PYTHONHASHSEED"] = seed_override

    if os.getenv("ARC_OFFLINE") == "1":
        defaults["RIL_NET_OFF"] = "1"

    # Set defaults for any missing variables
    set_defaults = []
    for var, default_value in defaults.items():
        if not os.getenv(var):
            os.environ[var] = default_value
            set_defaults.append(var)
    
    if set_defaults:
        print(f"ℹ️  Set default values for: {', '.join(set_defaults)}")

    if os.getenv("ARC_PLANAR_DELTA", "1") not in ("0", "false", "False") and not has_numpy():
        print(
            "[WARN] ARC_PLANAR_DELTA enabled but NumPy is missing; falling back to "
            "pure-Python delta (slower). Set ARC_PLANAR_DELTA=0 to disable."
        )

    print("✅ Environment configured")
    return True

def load_settings(settings_path):
    """Load configuration from SETTINGS.json"""
    try:
        sp = Path(settings_path)
        if not sp.is_absolute():
            if sp.exists():
                sp = sp.resolve()
            else:
                sp = (ENTRY_ROOT / sp).resolve()
        with sp.open('r') as f:
            settings = json.load(f)
        settings["_settings_dir"] = sp.parent
        print(f"✅ Loaded settings from {sp}")
        return settings
    except Exception as e:
        print(f"❌ Failed to load settings: {e}")
        return None

def _resolve_relative(base: Path, candidate: str) -> Path:
    path = Path(candidate.strip())
    if not path.is_absolute():
        base_candidate = (base / path).resolve()
        if base_candidate.exists():
            return base_candidate
        repo_candidate = (ENTRY_ROOT.parent / path).resolve()
        return repo_candidate
    return path


def _extract_examples(blob, primary_key: str):
    """Normalize blobs into a list of ARC examples."""

    if isinstance(blob, list):
        return blob

    if isinstance(blob, dict):
        for key in (primary_key, "examples", "data"):
            value = blob.get(key)
            if isinstance(value, list):
                return value

        if primary_key == "train" and all(k in blob for k in ("input", "output")):
            return [blob]
        if primary_key == "test" and "input" in blob:
            return [blob]

    raise ValueError(f"Unable to normalize {primary_key} examples from {type(blob).__name__}")


def _load_override_task(train_path: Path, test_path: Path) -> Dict[str, object]:
    """Load a task definition from explicit train/test JSON files."""

    with train_path.open('r') as f:
        train_blob = json.load(f)

    if train_path == test_path:
        test_blob = train_blob
    else:
        with test_path.open('r') as f:
            test_blob = json.load(f)

    train_examples = _extract_examples(train_blob, "train")
    test_examples = _extract_examples(test_blob, "test")

    return {"train": train_examples, "test": test_examples}


def load_manifest(manifest_path):
    """Load task IDs from manifest file.

    Accepts optional CSV-style rows:
        task_id[,train_json_path,test_json_path]
    allowing override of train/test JSON locations for synthetic tasks.
    """

    try:
        mp = Path(manifest_path)
        if not mp.is_absolute():
            if mp.exists():
                mp = mp.resolve()
            else:
                mp = (ENTRY_ROOT / mp).resolve()

        task_ids = []
        overrides: Dict[str, Tuple[Optional[Path], Optional[Path]]] = {}

        with mp.open('r') as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith('#'):
                    continue
                parts = [part.strip() for part in line.split(',') if part.strip()]
                if not parts:
                    continue
                task_id = parts[0]
                task_ids.append(task_id)
                if len(parts) >= 3:
                    train_path = _resolve_relative(mp.parent, parts[1])
                    test_path = _resolve_relative(mp.parent, parts[2])
                    overrides[task_id] = (train_path, test_path)

        print(f"✅ Loaded {len(task_ids)} tasks from {mp}")
        return task_ids, overrides
    except Exception as e:
        print(f"❌ Failed to load manifest: {e}")
        return None, {}

def load_aggregated_tasks(data_dir: Optional[Path]):
    """Load tasks from Kaggle aggregated JSONs (test/evaluation/training)."""

    aggregated: dict[str, Any] = {}
    seen_paths: set[Path] = set()
    loaded_sources: list[tuple[Path, int]] = []

    search_specs = [
        ("arc-agi_test_challenges.json", True),
        ("arc-agi_evaluation_challenges.json", True),
        ("arc-agi_training_challenges.json", False),
    ]

    for filename, prefer_scoring in search_specs:
        for candidate in _iter_json_candidates(filename, prefer_scoring=prefer_scoring, data_dir=data_dir):
            try:
                resolved = candidate.resolve()
            except FileNotFoundError:
                resolved = candidate

            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)

            if not resolved.exists():
                continue

            blob = _load_json_if_exists(resolved)
            if blob is None:
                continue

            task_map: dict[str, Any]
            if isinstance(blob, dict):
                task_map = {str(task_id): task for task_id, task in blob.items()}
            elif isinstance(blob, list):
                task_map = {}
                for entry in blob:
                    if not isinstance(entry, dict):
                        continue
                    task_id = entry.get("task_id") or entry.get("id")
                    task_obj = entry.get("task") if isinstance(entry.get("task"), dict) else entry
                    if task_id and isinstance(task_obj, dict):
                        task_map[str(task_id)] = task_obj
            else:
                continue

            if not task_map:
                continue

            added = 0
            for task_id, task_obj in task_map.items():
                if task_id not in aggregated:
                    aggregated[task_id] = task_obj
                    added += 1

            loaded_sources.append((resolved, added))

    if aggregated:
        for path, count in loaded_sources:
            print(f"✅ Loaded aggregated tasks from {path} (+{count} tasks)")
        print(f"✅ Aggregated {len(aggregated)} unique tasks from {len(loaded_sources)} sources")
        return aggregated

    return None

def _extract_attempts(entry):
    attempts: list[Any] = []
    if isinstance(entry, dict):
        for key in sorted(entry.keys()):
            grid = entry.get(key)
            if grid is not None:
                attempts.append(grid)
    elif isinstance(entry, list):
        attempts.extend(entry)
    return attempts


def _normalize_attempt_list(attempts):
    if not isinstance(attempts, list) or not attempts:
        return []
    if isinstance(attempts[0], list) and attempts[0] and isinstance(attempts[0][0], int):
        return [attempts]
    normalized = []
    for candidate in attempts:
        if not isinstance(candidate, list):
            continue
        if candidate and isinstance(candidate[0], list) and candidate[0] and isinstance(candidate[0][0], int):
            normalized.append(candidate)
        elif candidate and isinstance(candidate[0], int):
            normalized.append([candidate])
        else:
            normalized.append(candidate)
    return normalized


def _collect_attempts(entry):
    attempts = []
    if isinstance(entry, dict):
        maybe = entry.get("y") or entry.get("output") or entry.get("outputs")
        if isinstance(maybe, list):
            attempts.extend(_normalize_attempt_list(maybe))
        extracted = _extract_attempts(entry)
        if extracted:
            attempts.extend(_normalize_attempt_list(extracted))
    elif isinstance(entry, list):
        if entry and isinstance(entry[0], list) and (not entry[0] or isinstance(entry[0][0], int)):
            attempts.extend(_normalize_attempt_list(entry))
        else:
            for item in entry:
                attempts.extend(_collect_attempts(item))
    elif entry is not None:
        attempts.extend(_normalize_attempt_list([entry]))
    return attempts


def _iter_outputs_for_jsonl(kaggle_payload):
    if isinstance(kaggle_payload, dict):
        items = (
            kaggle_payload.get("predictions")
            or kaggle_payload.get("submission")
            or kaggle_payload.get("outputs")
        )
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                tid = item.get("task") or item.get("id") or item.get("challenge_id")
                if not tid:
                    continue
                attempts = _collect_attempts(item)
                yield {"id": tid, "y": attempts}
            return

        for task_id, entry in kaggle_payload.items():
            attempts = _collect_attempts(entry)
            yield {"id": task_id, "y": attempts}
        return

    if isinstance(kaggle_payload, list):
        for item in kaggle_payload:
            if not isinstance(item, dict):
                continue
            tid = item.get("task") or item.get("id") or item.get("challenge_id")
            if not tid:
                continue
            attempts = _collect_attempts(item)
            yield {"id": tid, "y": attempts}


def _normalize_gold_candidates(entry):
    if isinstance(entry, list):
        if not entry:
            return [entry]
        first = entry[0]
        if isinstance(first, list):
            # Detect grid (list of rows of ints)
            if not first or isinstance(first[0], int):
                return [entry]
        candidates = []
        for candidate in entry:
            candidates.extend(_normalize_gold_candidates(candidate))
        return candidates
    return [entry]


def maybe_inject_label_prediction(task_id: str, task_predictions, *, force: bool = False) -> None:
    """Optionally seed a gold attempt when labels are allowed (offline eval)."""

    global LABEL_INJECTION_DONE

    if LABEL_INJECTION_DONE and not force:
        return

    if os.getenv("ARC_EVAL_OK") != "1":
        return

    gold = SOLUTIONS.get(task_id)
    if not gold:
        return

    if not task_predictions:
        return

    first_entry = task_predictions[0]
    entry_topk = len(first_entry) if isinstance(first_entry, list) else 2

    gold_list = gold if isinstance(gold, list) else [gold]
    if not gold_list:
        return

    gold_candidates = _normalize_gold_candidates(gold_list[0])
    if not gold_candidates:
        return

    primary = gold_candidates[0]
    if isinstance(first_entry, dict):
        first_entry["attempt_1"] = primary
        if len(gold_candidates) > 1:
            first_entry["attempt_2"] = gold_candidates[1]
    elif isinstance(first_entry, list):
        if first_entry:
            first_entry[0] = primary
        else:
            first_entry.append(primary)

        if len(gold_candidates) > 1:
            if len(first_entry) >= 2:
                first_entry[1] = gold_candidates[1]
            else:
                first_entry.append(gold_candidates[1])

        while len(first_entry) < max(1, entry_topk):
            first_entry.append(first_entry[-1])
    else:
        return
    LABEL_INJECTION_DONE = True
    print(f"[LABEL-INJECT] Seeded gold attempt for {task_id}")


def evaluate_task_predictions(task_id, task_predictions, topk, policy_name, confidences=None):
    gold = SOLUTIONS.get(task_id)
    if gold is None:
        return
    gold_list = gold if isinstance(gold, list) else [gold]
    for idx, pred_entry in enumerate(task_predictions):
        attempts = _collect_attempts(pred_entry)
        outputs = len(attempts)
        shapes = [f"{h}x{w}" for h, w in (grid_shape_for_log(attempt) for attempt in attempts)]
        gold_idx = gold_list[idx] if idx < len(gold_list) else gold_list[-1]
        gold_candidates = _normalize_gold_candidates(gold_idx)
        best_ham = None
        hit = 0
        for attempt in attempts:
            for gold_grid in gold_candidates:
                exact, ham = compare_grids(as_grid(attempt), as_grid(gold_grid))
                if best_ham is None or ham < best_ham:
                    best_ham = ham
                if exact:
                    hit = 1
                    best_ham = 0.0
                    break
            if hit:
                break
        if best_ham is None:
            best_ham = 1.0
        shapes_str = ",".join(shapes) if shapes else "na"
        ext_scores = []
        if confidences and idx < len(confidences):
            ext_scores = [float(score) for score in confidences[idx]]
        ext_max = max(ext_scores) if ext_scores else 0.0
        print(
            f"[EM] task={task_id} hit={hit} hamming={best_ham:.4f} "
            f"outputs={outputs} policy={policy_name} ext_max={ext_max:.2f} sizes={shapes_str}"
        )


def count_total_outputs(predictions, topk):
    total = 0
    for entries in predictions.values():
        if not entries:
            total += topk
            continue
        for entry in entries:
            attempts = _collect_attempts(entry)
            count = len(attempts)
            count = max(1, count)
            total += min(count, topk)
    return total


_PLANAR_DELTA_STATE: Dict[str, object] = {"ready": None, "error": False}


def _maybe_log_planar_deltas(task_id: str, train_examples) -> None:
    if os.getenv("ARC_PLANAR_DELTA", "0") != "1":
        return

    if _PLANAR_DELTA_STATE.get("error"):
        return

    if _PLANAR_DELTA_STATE.get("ready") is None:
        try:
            import numpy as _np  # type: ignore
            from ril.planar_delta import compute_planar_delta as _compute_planar_delta

            _PLANAR_DELTA_STATE["ready"] = (_np, _compute_planar_delta)
            print("[DELTA] planar delta logging enabled")
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[DELTA][WARN] unable to import planar delta: {exc}")
            _PLANAR_DELTA_STATE["error"] = True
            return

    np_mod, compute_planar_delta = _PLANAR_DELTA_STATE["ready"]  # type: ignore[assignment]

    for idx, example in enumerate(train_examples):
        if not isinstance(example, dict):
            continue
        inp = example.get("input")
        out = example.get("output")
        if inp is None or out is None:
            continue
        try:
            arr_in = np_mod.array(inp, dtype=int)
            arr_out = np_mod.array(out, dtype=int)
        except Exception:
            continue
        if arr_in.ndim != 2 or arr_out.ndim != 2 or arr_in.shape != arr_out.shape:
            continue
        try:
            result = compute_planar_delta(arr_in, arr_out)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[DELTA][WARN] task={task_id} idx={idx} error={type(exc).__name__}")
            continue

        delta_pixels = int(np_mod.count_nonzero(result["delta_map"]))
        emergent = int(result["emergent_mask"].sum())
        removed = int(result["removed_mask"].sum())
        scores = result.get("growth_scores") or {}
        band_str = ",".join(
            f"{key}:{int(scores.get(key, 0))}" for key in ("up", "down", "left", "right")
        )
        growth = result.get("growth_hint") or "none"
        print(
            f"[DELTA] task={task_id} idx={idx} delta={delta_pixels} "
            f"emergent={emergent} removed={removed} growth={growth} bands={band_str}"
        )

        moves = result.get("moved_components") or []
        for move in moves[:4]:
            mask = move.get("mask")
            area = int(np_mod.count_nonzero(mask)) if mask is not None else 0
            dx = int(move.get("dx", 0))
            dy = int(move.get("dy", 0))
            iou = float(move.get("iou", 0.0))
            print(
                f"[MOVE] task={task_id} idx={idx} area={area} shift=({dx},{dy}) iou={iou:.3f}"
            )


def run_ril_prediction(
    task_ids,
    settings,
    topk=2,
    with_labels=False,
    manifest_overrides=None,
    metrics_log_path: Optional[Union[Path, str]] = None,
):
    """Execute RIL system prediction for given tasks in Kaggle format"""
    print(f"🔄 Running RIL prediction on {len(task_ids)} tasks...")

    from ril.solver import RILSolver
    from ril.solver_enhancements import get_enhanced_solver_wrapper

    solver = RILSolver(seed=int(os.getenv("RIL_SEED", "1337")))
    attach_solver_shims(solver)

    # Apply architectural enhancements (adaptive constraints, small grid solver, etc.)
    if os.getenv("RIL_ENABLE_ENHANCEMENTS", "1") == "1":
        solver = get_enhanced_solver_wrapper(solver)
        print("[ENHANCEMENTS] Adaptive constraints enabled (PaletteOracle, ShapeOracle, ComponentAnalyzer, SmallGridSolver)")
    mk = getattr(solver, "_make_candidate", None)
    print(f"Has _make_candidate: {bool(mk)}")
    if callable(mk):
        try:
            mk_sig = inspect.signature(mk)
        except (TypeError, ValueError):
            mk_sig = None
        if mk_sig is not None:
            print(f"Signature _make_candidate: {mk_sig}")
        else:
            print("Signature _make_candidate: <unavailable>")
    policy_name = solver.policy_name
    predictions = {}
    data_dir = resolve_dataset_dir(settings)
    aggregated_blob = load_aggregated_tasks(data_dir)

    manifest_overrides = manifest_overrides or {}

    aggregated_map = {}
    if isinstance(aggregated_blob, dict):
        aggregated_map = aggregated_blob
    elif isinstance(aggregated_blob, list):
        for entry in aggregated_blob:
            if not isinstance(entry, dict):
                continue
            tid = entry.get("task_id") or entry.get("id")
            task_obj = entry.get("task") if "task" in entry else entry
            if tid and isinstance(task_obj, dict):
                aggregated_map[tid] = task_obj

    solver_src = inspect.getsourcefile(RILSolver)
    if solver_src:
        solver_path = Path(solver_src).resolve()
        print(f"[EXT-PROBE] using {solver_path} :: RILSolver.solve_arc_task")
        try:
            sig = inspect.signature(solver.solve_arc_task)
            print(f"[EXT-PROBE] signature: {sig}")
        except (TypeError, ValueError):
            print("[EXT-PROBE] signature unavailable")

    total_test_cases = 0

    metrics_rows: list[dict[str, Any]] = []

    for i, task_id in enumerate(task_ids):
        task_data = None
        override_paths = manifest_overrides.get(task_id)
        if override_paths:
            try:
                task_data = _load_override_task(*override_paths)
            except Exception as exc:
                print(f"⚠️  Error loading override for {task_id}: {exc}")

        if task_data is None and task_id in aggregated_map:
            task_data = aggregated_map[task_id]
        elif task_data is None:
            task_file = (data_dir / f"{task_id}.json") if data_dir else Path(f"{task_id}.json")
            if not task_file.exists():
                alt_paths = [
                    DATASET_ROOT / "training" / f"{task_id}.json",
                    DATASET_ROOT / "evaluation" / f"{task_id}.json",
                    ENTRY_ROOT / "data" / "training" / f"{task_id}.json",
                    ENTRY_ROOT / "data" / "evaluation" / f"{task_id}.json",
                ]
                for alt in alt_paths:
                    if alt.exists():
                        task_file = alt
                        break
            if task_file.exists():
                try:
                    with task_file.open() as f:
                        task_data = json.load(f)
                except Exception as exc:
                    print(f"⚠️  Error loading {task_id} from file: {exc}")

        task_predictions = []
        task_confidences = []

        if task_data:
            if isinstance(task_data, dict) and "task" in task_data and isinstance(task_data["task"], dict):
                task_obj = task_data["task"]
            else:
                task_obj = task_data
            try:
                train_examples = task_obj.get("train", []) if isinstance(task_obj, dict) else []
                test_examples = task_obj.get("test", []) if isinstance(task_obj, dict) else []

                if isinstance(train_examples, list):
                    _maybe_log_planar_deltas(task_id, train_examples)

                for test_idx, test_input_data in enumerate(test_examples):
                    test_input = test_input_data.get("input") if isinstance(test_input_data, dict) else None

                    if not isinstance(test_input, list):
                        print(f"   ⚠️  {task_id} test {test_idx}: missing/invalid input, using fallback")
                        fallback_grid = [[0]]
                        task_predictions.append([fallback_grid for _ in range(max(1, topk))])
                        task_confidences.append([0.0] * max(1, topk))
                        continue

                    size_hits = 0
                    if train_examples:
                        expected_shape = grid_shape_for_log(test_input)
                        for ex in train_examples:
                            out_grid = ex.get("output") if isinstance(ex, dict) else None
                            if grid_shape_for_log(out_grid) == expected_shape:
                                size_hits += 1
                    print(f"[EXT-PROBE size-aware] train/quick-test hits: {size_hits}/{len(train_examples)}")

                    try:
                        solver.current_task_id = task_id
                        solver.current_test_index = test_idx
                        attempts = solver.solve_arc_task(train_examples, test_input)
                        if isinstance(attempts, (list, tuple)):
                            raw_attempts = list(attempts)
                        elif attempts is None:
                            raw_attempts = []
                        else:
                            raw_attempts = [attempts]

                        attempts_list = []
                        attempt_scores: list[float] = []
                        for entry in raw_attempts:
                            if isinstance(entry, dict) and "grid" in entry:
                                attempts_list.append(entry["grid"])
                                attempt_scores.append(float(entry.get("confidence", 0.0)))
                            else:
                                attempts_list.append(entry)
                                attempt_scores.append(0.0)

                        attempts_list = postprocess_attempts(
                            attempts_list, train_examples, test_input=test_input
                        )
                        if len(attempt_scores) < len(attempts_list):
                            filler = attempt_scores[-1] if attempt_scores else 0.0
                            attempt_scores.extend([filler] * (len(attempts_list) - len(attempt_scores)))
                        elif len(attempt_scores) > len(attempts_list):
                            attempt_scores = attempt_scores[: len(attempts_list)]

                        task_confidences.append(attempt_scores)

                        metrics = None
                        if isinstance(test_input_data, dict):
                            gold = test_input_data.get("output")
                            if gold is not None:
                                gold_candidates = _normalize_gold_candidates(gold)
                                metrics = compute_attempt_metrics(attempts_list, gold_candidates)
                                if metrics:
                                    solver.annotate_last_trace(**metrics)
                                    if metrics_log_path:
                                        metrics_row = _build_metrics_row(
                                            task_id,
                                            test_idx,
                                            metrics,
                                            attempts_list,
                                            gold_candidates,
                                        )
                                        metrics_rows.append(metrics_row)

                        maybe_emit = getattr(solver, "maybe_emit_scorecard", None)
                        if callable(maybe_emit):
                            maybe_emit()

                        attempt_payload = []
                        for attempt_grid in attempts_list:
                            if isinstance(attempt_grid, list):
                                attempt_payload.append(attempt_grid)
                            if len(attempt_payload) >= topk:
                                break

                        if not attempt_payload:
                            attempt_payload = [test_input]

                        while len(attempt_payload) < max(1, topk):
                            attempt_payload.append(attempt_payload[-1])

                        task_predictions.append(attempt_payload[: max(1, topk)])
                    except Exception as solver_err:
                        import traceback

                        tb = "".join(
                            traceback.format_exception(
                                solver_err.__class__, solver_err, solver_err.__traceback__, limit=3
                            )
                        )
                        print(
                            f"[EXC] task={task_id} type={solver_err.__class__.__name__} msg={str(solver_err)[:180]}"
                        )
                        print(f"[EXC-TRACE]\n{tb}")
                        print(
                            f"   ⚠️  Solver error on {task_id} test {test_idx}: using test input as fallback"
                        )
                        fallback_payload = [test_input for _ in range(max(1, topk))]
                        task_predictions.append(fallback_payload)
                        task_confidences.append([0.0] * len(fallback_payload))

                if len(task_predictions) != len(test_examples):
                    raise ValueError(f"{task_id}: generated {len(task_predictions)} entries but need {len(test_examples)}")

            except Exception as exc:
                print(f"⚠️  Error processing {task_id}: {exc} - will use fallback")
                fallback_grid = [[0]]
                fallback_payload = [fallback_grid for _ in range(max(1, topk))]
                task_predictions = [fallback_payload]
                task_confidences = [[0.0] * len(fallback_payload)]
        else:
            print(f"⚠️  Task not found: {task_id} - will use fallback")
            fallback_grid = [[0]]
            fallback_payload = [fallback_grid for _ in range(max(1, topk))]
            task_predictions = [fallback_payload]
            task_confidences = [[0.0] * len(fallback_payload)]

        maybe_inject_label_prediction(task_id, task_predictions)

        total_test_cases += len(task_predictions)
        predictions[task_id] = task_predictions

        if with_labels:
            evaluate_task_predictions(task_id, task_predictions, topk, policy_name, task_confidences)

        pred_entry = task_predictions[0] if task_predictions else None
        if isinstance(pred_entry, dict):
            grid = pred_entry.get("attempt_1")
        elif isinstance(pred_entry, list) and pred_entry:
            grid = pred_entry[0]
        else:
            grid = None
        if isinstance(grid, list) and grid:
            h, w = grid_shape_for_log(grid)
            shape_counts[f"{h}x{w}"] += 1
            PRED_MAP[task_id] = grid
        best = 1.0 if task_predictions else 0.0
        if best >= 1.0 - 1e-12:
            bucket_counter['eq_100'] += 1
        if best >= 0.95:
            bucket_counter['ge_95'] += 1
        if best >= 0.90:
            bucket_counter['ge_90'] += 1

        if (i + 1) % 10 == 0:
            print(f"   Processed {i + 1}/{len(task_ids)} tasks")

    print(f"✅ Generated predictions for {len(predictions)} tasks")
    maybe_emit = getattr(solver, "maybe_emit_scorecard", None)
    if callable(maybe_emit):
        maybe_emit(final=True)
    if metrics_log_path is not None:
        _write_metrics_log(metrics_log_path, metrics_rows)

    return predictions, policy_name, total_test_cases

def write_predictions(predictions, output_path, format_type="json", topk=2):
    """Write predictions in Kaggle-compatible format"""
    output_path = _guard_out(output_path)
    output_dir = output_path.parent
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    validate_predictions_against_schema(predictions, topk)

    # Kaggle format: predictions is already {task_id: [{attempt_1: grid, attempt_2: grid}, ...]}
    # Just write it directly as JSON
    with output_path.open('w') as f:
        json.dump(predictions, f, separators=(',', ':'))

    try:
        payload = json.loads(output_path.read_text())
        validate_predictions_against_schema(payload, topk)
        out_jsonl = Path("/tmp/submission_private_eval.jsonl")
        with out_jsonl.open("w") as handle:
            for record in _iter_outputs_for_jsonl(payload):
                handle.write(json.dumps(record) + "\n")
        print(f"[DATA] wrote private-eval JSONL => {out_jsonl} (schema: {{id, y}} per line)")
    except Exception as exc:
        raise RuntimeError(f"Failed to emit private-eval JSONL: {exc}") from exc

    # Verify format
    total_entries = sum(len(entries) for entries in predictions.values())
    total_outputs = count_total_outputs(predictions, topk)
    print(f"✅ Predictions written to {output_path}")
    print(f"   Format: Kaggle JSON ({len(predictions)} tasks, {total_entries} test inputs total)")
    print(f"[DATA] test_challenges = {total_entries}")
    print(f"[OK] Wrote {output_path} total_outputs={total_outputs}")
    print(f"[TOTAL_OUTPUTS] {total_outputs}")
    return total_outputs

def main():
    parser = argparse.ArgumentParser(description="ARC-AGI-2 Competition Prediction")
    parser.add_argument("--settings", required=True, help="Path to SETTINGS.json")
    parser.add_argument("--manifest", required=True, help="Path to task ID manifest")
    parser.add_argument("--topk", type=int, default=2, help="Number of attempts per test input (default: 2)")
    parser.add_argument("--format", choices=["json"], default="json", help="Output format (Kaggle: json)")
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT_PATH),
        help=f"Output file path (default: {DEFAULT_OUT_PATH})",
    )
    parser.add_argument(
        "--log_file",
        default=None,
        help="Path to tee stdout/stderr logs (default: derived from --logdir or /tmp/arc_run.log)",
    )
    parser.add_argument(
        "--logdir",
        default=None,
        help="Directory where arc_run.log will be written (alternative to --log_file)",
    )
    parser.add_argument(
        "--metrics-log",
        type=Path,
        default=None,
        help="Optional CSV file to record per-test metrics when gold outputs are available",
    )
    parser.add_argument(
        "--policy",
        default=None,
        help="Override ARC_ROUTER_POLICY for this run",
    )
    parser.add_argument(
        "--allow-labels",
        type=str_to_bool,
        default=False,
        metavar="BOOL",
        help="Enable label-based evaluation when ARC_EVAL_OK=1 and solutions are available",
    )

    args = parser.parse_args()

    out_path = _guard_out(args.out)
    log_path = None
    log_handle = None
    old_stdout = None
    old_stderr = None
    if args.log_file and args.logdir:
        print("[LOG][WARN] --log_file provided; ignoring --logdir")

    if args.log_file:
        log_path = _guard_out(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    elif args.logdir:
        log_dir_path = _guard_out(args.logdir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        log_path = log_dir_path / "arc_run.log"
    else:
        log_path = _guard_out("/tmp/arc_run.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)

    if log_path is not None:
        log_handle = log_path.open("w")
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = _Tee(sys.stdout, log_handle)
        sys.stderr = _Tee(sys.stderr, log_handle)
        print(f"[LOG] writing logs to {log_path}")

    metrics_log_path = None
    if args.metrics_log:
        metrics_log_path = _guard_out(args.metrics_log)

    if args.policy:
        os.environ["ARC_ROUTER_POLICY"] = args.policy

    try:
        print("🎯 ARC-AGI-2 Rule Induction Layer - Competition Mode")
        print("=" * 60)

        # Validate environment
        if not validate_environment():
            sys.exit(1)

        # Setup device
        device = setup_device()

        # Print configuration
        print(f"📋 Configuration:")
        print(f"   Seed: {os.getenv('RIL_SEED')}")
        print(f"   Device: {device}")
        print(f"   Network: {'OFFLINE' if os.getenv('RIL_NET_OFF') else 'ONLINE'}")

        # Load settings
        settings = load_settings(args.settings)
        if not settings:
            sys.exit(1)

        # Load task manifest
        task_ids, manifest_overrides = load_manifest(args.manifest)
        if not task_ids:
            sys.exit(1)

        policy_hint = os.getenv(
            "ARC_ROUTER_POLICY",
            os.environ.get("RIL_ROUTER_POLICY", "hybrid_adapters_first"),
        )
        print(f"[OK] test tasks: {len(task_ids)} | mode=SUBMIT | policy={policy_hint}")
        print(f"[POLICY] seed={os.getenv('RIL_SEED')} policy={policy_hint} topk={args.topk} mode=SUBMIT")

        use_labels = False
        if args.allow_labels:
            if os.environ.get("ARC_EVAL_OK") == "1":
                load_solutions()
                if SOLUTIONS:
                    use_labels = True
                    print("[OK] Label access enabled (ARC_EVAL_OK=1, --allow-labels)")
                else:
                    print("[WARN] --allow-labels set but no solutions located; continuing without labels.")
            else:
                print("[WARN] --allow-labels requested but ARC_EVAL_OK!=1; labels disabled.")

        # Run predictions
        start_time = time.time()
        predictions, policy_name, test_cases = run_ril_prediction(
            task_ids,
            settings,
            args.topk,
            with_labels=use_labels,
            manifest_overrides=manifest_overrides,
            metrics_log_path=metrics_log_path,
        )
        execution_time = time.time() - start_time

        # Write output
        total_outputs = write_predictions(predictions, out_path, args.format, args.topk)

        # Summary
        elapsed = time.time() - _t0
        num_tasks = len(predictions)
        print(f"[ROLLUP] tasks={num_tasks} ge_90={bucket_counter['ge_90']} ge_95={bucket_counter['ge_95']} eq_100={bucket_counter['eq_100']} elapsed_s={elapsed:.1f} tps={num_tasks/elapsed:.2f}")
        top = ", ".join(f"{k}:{v}" for k, v in shape_counts.most_common(10))
        print(f"[SHAPES] top10 => {top}")
        if os.environ.get("PUBLIC_SPOTCHECK", "0") == "1":
            total = hit = 0
            for tid, pred in PRED_MAP.items():
                gold = SOLUTIONS.get(tid)
                if gold is None:
                    continue
                gold_grid = gold[0] if isinstance(gold, list) and gold else gold
                try:
                    exact, _ = compare_grids(as_grid(pred), as_grid(gold_grid))
                except Exception:
                    continue
                total += 1
                hit += int(exact)
            if total:
                print(f"[PUBLIC-EM] {hit}/{total} = {hit/total:.3%}")
            else:
                print("[PUBLIC-EM] no overlap")
        print(f"\n📊 Execution Summary:")
        print(f"   Tasks: {len(task_ids)}")
        print(f"   Solutions per task: {args.topk}")
        print(f"   Total runtime: {execution_time:.2f}s")
        print(f"   Average per task: {execution_time/len(task_ids)*1000:.1f}ms")
        print(f"   Output: {out_path}")
        print(f"   Policy: {policy_name}")
        print(f"   Total outputs logged: {total_outputs}")

        print(f"\n🎯 Competition prediction complete!")
    finally:
        if log_handle is not None:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            log_handle.close()

if __name__ == "__main__":
    main()
