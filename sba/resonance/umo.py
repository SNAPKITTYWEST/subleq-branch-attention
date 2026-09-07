"""
umo.py — Python port of the SnapKitty Resonance ISA Unicode Mantra Oscillator.

Φ(t) = sin(τ·t) × cos(ρ·t) × (1 − ε)

trust τ    drives amplitude stability
resonance ρ drives frequency structure
entropy ε  damps and eventually collapses the signal

Coherence gate: ε < 0.21 and τ > 0.5
"""

from __future__ import annotations
import math
from dataclasses import dataclass


# Glyph table (mirrors crates/umo/src/lib.rs)
_GLYPHS = [
    (0.9, "☉"),
    (0.7, "◉"),
    (0.5, "◇"),
    (0.3, "▣"),
    (0.1, "▒"),
    (0.0, "⛔"),
]


def glyph(value: float) -> str:
    v = abs(value)
    for threshold, ch in _GLYPHS:
        if v > threshold:
            return ch
    return "⛔"


@dataclass
class Umo:
    trust:     float  # τ
    entropy:   float  # ε
    resonance: float  # ρ

    def signal(self, t: float) -> float:
        """Φ(t) = sin(τ·t) × cos(ρ·t) × (1 − ε)"""
        return (math.sin(self.trust * t)
                * math.cos(self.resonance * t)
                * (1.0 - self.entropy))

    def waveform(self, width: int = 12, dt: float = 0.4) -> str:
        return "".join(glyph(self.signal(i * dt)) for i in range(width))

    def stream(self, steps: int = 6, width: int = 12, dt: float = 0.4) -> list[str]:
        lines = []
        for t in range(steps):
            line = "".join(
                glyph(self.signal(t + i * dt)) for i in range(width)
            )
            lines.append(f"t={t:<2} : {line}")
        return lines

    def coherent(self) -> bool:
        return self.entropy < 0.21 and self.trust > 0.5

    def sealed_output(self) -> str:
        if self.coherent():
            return "☉◉◇◉☉◉◇◉☉ → SYSTEM COHERENT → DEED SEALED"
        return "▒░⛔⚠⛔░▒ → RESONANCE FAILURE → EXECUTION BLOCKED"
