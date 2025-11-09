"""Runtime shims for :mod:`ril.solver` API differences.

This repository occasionally runs against older ARC release bundles that do not
expose the helper methods expected by the latest ``predict_competition`` entry
point.  The helper below normalises the interface at runtime so that we can run
predictors built against different solver revisions without crashing.

The shims intentionally stay *very* small: they only provide the bits the
entrypoint relies on (``maybe_emit_scorecard`` and ``_make_candidate``).  Newer
releases already expose these helpers, so calling :func:`attach_solver_shims`
is effectively a no-op in the common case.

LABEL LEAKAGE PREVENTION:
This module also provides filesystem firebreak utilities that prevent accidental
access to ground truth labels during hypothesis generation.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import MethodType
from typing import Any, Callable, IO


def _make_candidate_wrapper(solver: Any) -> Callable[..., Any]:
    """Return a compatibility wrapper for candidate construction."""

    fallbacks: list[Callable[..., Any]] = []
    for attr in ("_make_candidate", "make_candidate", "build_candidate"):
        candidate_fn = getattr(solver, attr, None)
        if callable(candidate_fn):
            fallbacks.append(candidate_fn)

    def _fallback_payload(
        program: Any,
        *,
        score: float = 0.0,
        meta: Any | None = None,
        cand_type: Any = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "program": program,
            "score": score,
            "meta": meta or {},
            "cand_type": cand_type,
        }
        if extra:
            payload.update(extra)
        return payload

    def _compat(
        self,
        program: Any,
        score: float = 0.0,
        meta: Any | None = None,
        *,
        cand_type: Any = None,
        **kwargs: Any,
    ) -> Any:
        for target in fallbacks:
            try:
                return target(program, score=score, meta=meta, cand_type=cand_type, **kwargs)
            except TypeError:
                try:
                    return target(program, score=score, meta=meta)
                except TypeError:
                    continue

        return _fallback_payload(program, score=score, meta=meta, cand_type=cand_type, extra=kwargs or None)

    return MethodType(_compat, solver)


def attach_solver_shims(solver: Any) -> None:
    """Ensure ``solver`` exposes the helper methods ``predict_competition`` expects.

    Older solver builds shipped without ``maybe_emit_scorecard`` (a no-op hook)
    and ``_make_candidate`` (a thin wrapper over the candidate-construction
    helpers).  The Kaggle driver calls these helpers defensively now, but we
    still attach shims here so that internal solver code which references
    ``self._make_candidate`` also behaves as expected.
    """

    if not hasattr(solver, "maybe_emit_scorecard"):
        solver.maybe_emit_scorecard = MethodType(lambda self, final=False: None, solver)

    if not hasattr(solver, "_make_candidate"):
        solver._make_candidate = _make_candidate_wrapper(solver)  # type: ignore[attr-defined]


# ================================ LABEL LEAKAGE PREVENTION ================================

# Forbidden patterns that indicate ground truth access
FORBIDDEN_PATTERNS = (
    "/solutions/",
    "solutions.json",
    "_gt.json",
    "evaluation_solutions",
    "training_solutions",
    "_solutions.json",
)

# Allowed directories for reading during solver execution
ALLOWED_READ_DIRS = (
    "./",
    "./ril/",
    "./arc-agi-2-entry_v1.2_kaggle2025_fixed_20251002T210957Z/",
    "/kaggle/input/arc-agi-2-public-dataset/arc-agi_test_challenges.json",
    "/kaggle/input/arc-agi-2-public-dataset/arc-agi_evaluation_challenges.json",
    "/kaggle/input/arc-agi-2-public-dataset/arc-agi_training_challenges.json",
)


def is_forbidden_path(path: str | Path) -> bool:
    """Check if a path contains forbidden patterns indicating ground truth access."""
    normalized = os.path.normpath(str(path)).replace("\\", "/")
    return any(pattern in normalized for pattern in FORBIDDEN_PATTERNS)


def safe_open(path: str | Path, *args: Any, **kwargs: Any) -> IO:
    """
    Safe file open that prevents access to ground truth labels.

    Raises:
        RuntimeError: If attempting to access a forbidden path containing ground truth.
    """
    normalized = os.path.normpath(str(path))

    if is_forbidden_path(normalized):
        raise RuntimeError(
            f"[LABEL LEAKAGE PREVENTION] Forbidden path access blocked: {normalized}\n"
            f"Ground truth files must not be accessed during hypothesis generation.\n"
            f"Use scripts/private_eval.py for post-hoc evaluation only."
        )

    return open(path, *args, **kwargs)


def verify_no_gt_access_in_logs(log_path: str | Path) -> None:
    """
    Verify that log files don't contain evidence of ground truth access.

    Args:
        log_path: Path to log file to verify

    Raises:
        RuntimeError: If forbidden tokens are found in logs
    """
    forbidden_tokens = [
        "gt_hamming",
        "gt_palette",
        "using_ground_truth",
        "loaded_gt",
        "ground_truth_access",
        "eval_solutions",
    ]

    if not Path(log_path).exists():
        return

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    found_tokens = [token for token in forbidden_tokens if token in content]

    if found_tokens:
        raise RuntimeError(
            f"[LABEL LEAKAGE PREVENTION] Forbidden tokens found in logs: {found_tokens}\n"
            f"Log file: {log_path}\n"
            f"This indicates potential ground truth access during hypothesis generation."
        )

