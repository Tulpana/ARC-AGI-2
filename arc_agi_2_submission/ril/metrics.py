"""Minimal structured metrics helpers for the RIL pipeline."""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from typing import Dict, List, Optional


class Metrics:
    """Collect counters and numeric samples and emit JSON to stderr."""

    def __init__(self) -> None:
        self.counters: Counter[str] = Counter()
        self.samples: Dict[str, List[float]] = {}
        self._t0 = time.perf_counter()

    def inc(self, key: str, n: int = 1) -> None:
        if not key:
            return
        self.counters[key] += int(n)

    def observe(self, key: str, value: Optional[float]) -> None:
        if not key or value is None:
            return
        try:
            val = float(value)
        except Exception:
            return
        if not (val == val):  # guard against NaN
            return
        self.samples.setdefault(key, []).append(val)

    def dump(self, label: str = "METRICS") -> None:
        elapsed = max(0.0, time.perf_counter() - self._t0)
        payload = {
            "elapsed_s": round(elapsed, 6),
            "counters": dict(self.counters),
            "samples": {
                name: {
                    "count": len(values),
                    "mean": (sum(values) / len(values)) if values else 0.0,
                }
                for name, values in self.samples.items()
            },
        }
        sys.stderr.write(f"[{label}] " + json.dumps(payload, sort_keys=True) + "\n")
        sys.stderr.flush()


def log_pipeline_error(metrics: Optional["Metrics"], exc: BaseException) -> None:
    """Increment the pipeline.error counter and echo the exception."""

    if metrics is not None:
        metrics.inc("pipeline.error")
    sys.stderr.write("[ERROR] " + repr(exc) + "\n")
    sys.stderr.flush()


__all__ = ["Metrics", "log_pipeline_error"]
