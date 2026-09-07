"""
words.py — Resonance word producer.

A "resonance word" is one waveform tick rendered as a glyph string:
    ☉◉◇◉▣◇◉☉◉◉◇◉▣◇◉☉

produce_words() is the top-level entry point: takes kernel invariants
and returns a complete resonance word sequence + status banner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .umo import Umo, glyph
from .bridge import DrainInvariants, invariants_to_umo, _demo_invariants_from_kernel_params


@dataclass
class ResonanceWord:
    """One time-tick of the UMO oscillator rendered as a glyph word."""
    tick:    int
    glyphs:  str    # e.g. "☉◉◇◉▣◇◉☉"
    phi_max: float  # max |Φ(t)| across the tick
    label:   str    # "COHERENT" | "MARGINAL" | "COLLAPSED"

    def __str__(self) -> str:
        return f"t={self.tick:<2} [{self.label:<9}] {self.glyphs}  |Φ|max={self.phi_max:.3f}"


def _classify(phi_max: float, entropy: float) -> str:
    if entropy >= 0.21:
        return "COLLAPSED"
    if phi_max > 0.5:
        return "COHERENT"
    return "MARGINAL"


def produce_words(
    inv:    Optional[DrainInvariants] = None,
    *,
    steps:  int   = 10,
    width:  int   = 16,
    dt:     float = 0.3,
    # Override: use synthetic demo data if inv is None
    tensor_count:    int   = 500,
    original_count:  int   = 1000,
    complexity_frac: float = 0.72,
    entropy_frac:    float = 0.08,
) -> tuple[Umo, list[ResonanceWord]]:
    """
    Produce resonance words from drain pipeline invariants.

    Parameters
    ----------
    inv : DrainInvariants, optional
        Real invariants from the Rust drain pipeline. If None, synthetic
        demo values are generated from the keyword arguments.
    steps, width, dt : waveform parameters passed to the UMO.

    Returns
    -------
    (umo, words) where words is a list of ResonanceWord objects.
    """
    if inv is None:
        inv = _demo_invariants_from_kernel_params(
            tensor_count    = tensor_count,
            original_count  = original_count,
            complexity_frac = complexity_frac,
            entropy_frac    = entropy_frac,
        )

    umo   = invariants_to_umo(inv)
    words: list[ResonanceWord] = []

    for tick in range(steps):
        t0      = float(tick)
        samples = [umo.signal(t0 + i * dt) for i in range(width)]
        phi_max = max(abs(s) for s in samples)
        line    = "".join(glyph(s) for s in samples)
        label   = _classify(phi_max, umo.entropy)
        words.append(ResonanceWord(tick=tick, glyphs=line,
                                   phi_max=phi_max, label=label))

    return umo, words


def render(
    inv:    Optional[DrainInvariants] = None,
    **kwargs,
) -> str:
    """
    Full text render of the resonance word sequence including header and seal.
    """
    umo, words = produce_words(inv, **kwargs)

    lines = [
        "⟦ Ω ⟧ RESONANCE WORD MACHINE — Kernel Output",
        "─" * 55,
        f"  τ trust     : {umo.trust:.4f}",
        f"  ε entropy   : {umo.entropy:.4f}",
        f"  ρ resonance : {umo.resonance:.4f}",
        f"  coherent    : {'YES ☉' if umo.coherent() else 'NO ⛔'}",
        "─" * 55,
        "  Φ(t) = sin(τ·t) × cos(ρ·t) × (1 − ε)",
        "",
    ]

    for w in words:
        lines.append(f"  {w}")

    lines += [
        "",
        "─" * 55,
        f"  {umo.sealed_output()}",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, io
    # Force UTF-8 on Windows consoles that default to cp1252
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    cfrac = float(sys.argv[1]) if len(sys.argv) > 1 else 0.72
    efrac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.08

    print(render(complexity_frac=cfrac, entropy_frac=efrac, steps=12, width=20))
