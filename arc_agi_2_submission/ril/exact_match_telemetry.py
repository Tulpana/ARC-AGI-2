"""Exact-match telemetry for tracking solver progress toward 50% accuracy.

This module provides run-level checks to prevent backsliding to 2x2 fallback
behavior and ensure the improvement plan translates to measurable gains.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

Grid = List[List[int]]


def hamming_distance(grid1: Grid, grid2: Grid) -> float:
    """Compute normalized Hamming distance (fraction of differing cells)."""
    if not grid1 or not grid2:
        return 1.0

    h1, w1 = len(grid1), len(grid1[0]) if grid1 else 0
    h2, w2 = len(grid2), len(grid2[0]) if grid2 else 0

    if h1 != h2 or w1 != w2:
        return 1.0  # Shape mismatch = maximum distance

    total_cells = h1 * w1
    if total_cells == 0:
        return 0.0

    diff_count = 0
    for r in range(h1):
        for c in range(w1):
            if grid1[r][c] != grid2[r][c]:
                diff_count += 1

    return diff_count / total_cells


def is_exact_match(predicted: Grid, ground_truth: Grid) -> bool:
    """Check if predicted grid exactly matches ground truth."""
    return hamming_distance(predicted, ground_truth) == 0.0


class ExactMatchTracker:
    """Track exact matches and near misses across a solver run."""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir or "telemetry"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        self.results: List[Dict[str, Any]] = []
        self.exact_matches = 0
        self.near_misses = 0  # hamming <= 0.20
        self.total_tasks = 0
        self.start_time = datetime.utcnow()

    def record_prediction(
        self,
        task_id: str,
        predicted: Grid,
        ground_truth: Optional[Grid] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a prediction and compute metrics."""
        self.total_tasks += 1

        if ground_truth is None:
            # No ground truth available (e.g., test set)
            self.results.append({
                "task_id": task_id,
                "has_ground_truth": False,
                "metadata": metadata or {},
            })
            return

        hamming = hamming_distance(predicted, ground_truth)
        exact = hamming == 0.0
        near_miss = 0.0 < hamming <= 0.20

        if exact:
            self.exact_matches += 1
        elif near_miss:
            self.near_misses += 1

        self.results.append({
            "task_id": task_id,
            "hamming": hamming,
            "exact_match": exact,
            "near_miss": near_miss,
            "has_ground_truth": True,
            "metadata": metadata or {},
        })

    def get_summary(self) -> Dict[str, Any]:
        """Get current summary statistics."""
        em_rate = self.exact_matches / max(1, self.total_tasks)
        near_miss_rate = self.near_misses / max(1, self.total_tasks)

        return {
            "exact_matches": self.exact_matches,
            "near_misses": self.near_misses,
            "total_tasks": self.total_tasks,
            "exact_match_rate": em_rate,
            "near_miss_rate": near_miss_rate,
            "combined_potential": em_rate + near_miss_rate,
            "start_time": self.start_time.isoformat(),
            "elapsed_seconds": (datetime.utcnow() - self.start_time).total_seconds(),
        }

    def save_checkpoint(self, filename: Optional[str] = None):
        """Save current results to disk."""
        if filename is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"telemetry_{timestamp}.json"

        output_path = Path(self.output_dir) / filename

        payload = {
            "summary": self.get_summary(),
            "results": self.results,
        }

        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2)

        print(f"[TELEMETRY] Saved checkpoint to {output_path}")
        return output_path

    def check_progress_gate(self, min_exact_match_rate: float = 0.01) -> bool:
        """Check if solver meets minimum accuracy threshold.

        Returns True if passing, False if backsliding detected.
        """
        summary = self.get_summary()
        em_rate = summary["exact_match_rate"]

        passing = em_rate >= min_exact_match_rate

        if not passing:
            print(f"[TELEMETRY WARNING] Exact match rate {em_rate:.4f} below threshold {min_exact_match_rate:.4f}")
            print(f"[TELEMETRY WARNING] Potential backsliding detected - check recent changes")
        else:
            print(f"[TELEMETRY OK] Exact match rate {em_rate:.4f} >= {min_exact_match_rate:.4f}")

        return passing

    def print_report(self):
        """Print formatted progress report."""
        summary = self.get_summary()

        print("=" * 60)
        print("EXACT-MATCH TELEMETRY REPORT")
        print("=" * 60)
        print(f"Exact Matches:     {summary['exact_matches']:4d} / {summary['total_tasks']:4d} ({summary['exact_match_rate']:.2%})")
        print(f"Near Misses:       {summary['near_misses']:4d} / {summary['total_tasks']:4d} ({summary['near_miss_rate']:.2%})")
        print(f"Combined Potential: {summary['combined_potential']:.2%}")
        print(f"Elapsed Time:      {summary['elapsed_seconds']:.1f}s")
        print("=" * 60)

        # Check against 50% target
        if summary['exact_match_rate'] >= 0.50:
            print("*** TARGET ACHIEVED: >= 50% exact match rate!")
        elif summary['combined_potential'] >= 0.50:
            print("[HIGH POTENTIAL] Near misses + exact matches >= 50%")
            print("   -> Focus on routing and gate calibration to convert near misses")
        else:
            print("[IN PROGRESS] Continue adding transformation learners")

        print()


# Global tracker instance for convenience
_global_tracker: Optional[ExactMatchTracker] = None


def get_global_tracker() -> ExactMatchTracker:
    """Get or create global tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = ExactMatchTracker()
    return _global_tracker


def record_task_result(
    task_id: str,
    predicted: Grid,
    ground_truth: Optional[Grid] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """Convenience function to record to global tracker."""
    tracker = get_global_tracker()
    tracker.record_prediction(task_id, predicted, ground_truth, metadata)
