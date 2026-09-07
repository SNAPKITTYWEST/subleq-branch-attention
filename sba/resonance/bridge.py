"""
bridge.py — Connects NARM/drain-pipeline kernel output to the Resonance UMO.

Mapping (drain Invariant → UMO parameters):

  trust τ     = complexity_ratio   = total_complexity / (tensor_count * max_complexity_per_tensor)
                clamped to [0, 1]; high retained complexity → high trust

  entropy ε   = normalized_entropy = total_entropy / (tensor_count * max_entropy_per_tensor)
                clamped to [0, 1]; must stay below 0.21 for the field to remain coherent

  resonance ρ = retention_ratio    = retained_count / original_count
                clamped to (0, 1]; high retention → high resonance frequency
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .umo import Umo

# Scale constants mirror the Rust drain_pipeline (SCALE = 1000)
SCALE = 1_000

# Reference maximums for normalization (tuned to typical 120B-param model ranges)
_MAX_COMPLEXITY_PER_TENSOR = 2_000 * SCALE   # param_count=2000 * SCALE
_MAX_ENTROPY_PER_TENSOR    = 2 * SCALE       # simplified entropy: p*2


@dataclass
class DrainInvariants:
    """Aggregated invariants produced by the drain pipeline after convergence."""
    total_complexity:  int   # Σ C(T_i)  fixed-point * SCALE
    total_entropy:     int   # Σ H(T_i)  fixed-point * SCALE
    total_liability:   int   # Σ L(T_i)  fixed-point * SCALE
    tensor_count:      int   # |Θ_res|
    original_count:    int   # |Θ| before drain

    @classmethod
    def from_dict(cls, d: dict) -> "DrainInvariants":
        return cls(**d)


def invariants_to_umo(inv: DrainInvariants) -> Umo:
    """
    Map drain pipeline invariants to Resonance UMO parameters.

    τ (trust)     = average complexity ratio, clamped [0, 1]
    ε (entropy)   = average entropy ratio,    clamped [0, 1]
    ρ (resonance) = retention fraction,       clamped (0, 1]
    """
    if inv.tensor_count == 0:
        return Umo(trust=0.0, entropy=1.0, resonance=0.0)

    max_c = _MAX_COMPLEXITY_PER_TENSOR * inv.original_count
    max_e = _MAX_ENTROPY_PER_TENSOR    * inv.original_count

    trust     = min(1.0, inv.total_complexity / max(max_c, 1))
    entropy   = min(1.0, inv.total_entropy    / max(max_e, 1))
    retention = inv.tensor_count / max(inv.original_count, 1)
    resonance = max(0.01, min(1.0, retention))

    return Umo(trust=trust, entropy=entropy, resonance=resonance)


def kernel_resonance_words(
    inv: DrainInvariants,
    steps: int = 8,
    width: int = 16,
    dt: float   = 0.3,
) -> list[str]:
    """
    Run the drain invariants through the UMO and return resonance word lines.

    Each element of the returned list is one time-tick of the waveform —
    a "resonance word" in the SnapKitty Resonance ISA sense.
    """
    umo = invariants_to_umo(inv)
    return umo.stream(steps=steps, width=width, dt=dt)


# ---------------------------------------------------------------------------
# Synthetic demo (runs without the compiled Rust binary)
# ---------------------------------------------------------------------------

def _demo_invariants_from_kernel_params(
    tensor_count:    int = 500,
    original_count:  int = 1000,
    complexity_frac: float = 0.72,
    entropy_frac:    float = 0.08,
) -> DrainInvariants:
    """
    Build synthetic DrainInvariants from high-level fractions.
    Useful for testing the bridge without running the full Rust pipeline.
    """
    max_c = _MAX_COMPLEXITY_PER_TENSOR * original_count
    max_e = _MAX_ENTROPY_PER_TENSOR    * original_count
    return DrainInvariants(
        total_complexity = int(max_c * complexity_frac),
        total_entropy    = int(max_e * entropy_frac),
        total_liability  = 0,
        tensor_count     = tensor_count,
        original_count   = original_count,
    )
