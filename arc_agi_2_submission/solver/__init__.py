"""Solver public API surface."""

from .generalization import (
    Example,
    LocalRule,
    RuleSet,
    adaptive_palette_rescue,
    apply_rule_set,
    compute_allowed_and_birth_palettes,
    dimensionless_dilate,
    dimensionless_erode,
    dimensionless_kernel_size,
    dimensionless_ring,
    entropy_scaled_gating,
    extract_local_rules,
    finalize_candidate,
    palette_completeness,
    stageA_threshold,
    validate_candidate,
)

__all__ = [
    "Example",
    "LocalRule",
    "RuleSet",
    "adaptive_palette_rescue",
    "apply_rule_set",
    "compute_allowed_and_birth_palettes",
    "dimensionless_dilate",
    "dimensionless_erode",
    "dimensionless_kernel_size",
    "dimensionless_ring",
    "entropy_scaled_gating",
    "extract_local_rules",
    "finalize_candidate",
    "palette_completeness",
    "stageA_threshold",
    "validate_candidate",
]
