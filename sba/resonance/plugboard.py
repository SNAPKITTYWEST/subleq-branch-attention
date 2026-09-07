"""
plugboard.py — Resonance Plugboard: signal-band → operation routing cross-bar.

The plugboard is a 6×N routing matrix where:

  rows    = 6 signal amplitude bands  (☉ ◉ ◇ ▣ ▒ ⛔)
  columns = N operations              (Abjad opcodes A–H + NARM kernels)

Each entry W[band, op] is the routing weight for that (band, op) pair,
generated from the oscillator via waveform_tensor.

When the oscillator emits a sample at amplitude band B, the plugboard
routes to the highest-weight operation in row B — making the ISA execution
and kernel dispatch a direct function of the resonance field state.

ENOCHIAN ROOT OPCODE GATE (ERE Pass 5):
  The ⛔ band (amplitude ≤ 0.1) routes through the Enochian root check
  before any instruction executes.  Formally proved in formal/enochian_root.lean:

    plugboardRootGate : SignalBand → EreInput α → Prop
    void_band_void_input_blocked   : ¬ plugboardRootGate void_ EreInput.void
    void_band_defined_input_passes : plugboardRootGate void_ (EreInput.of a)

  Python mirror: ere_root_gate(value) → bool

                ┌──────────────────────────────────────────┐
 Φ(t) ──band──► │   PLUGBOARD  (6 × N crossbar)            │──► Abjad opcode  →  .rasm
                │   W[b,op] = oscillator sample            │──► NARM kernel   →  tensor op
                │   ⛔ band → ERE root gate (Lean formal)   │──► drain op      →  invariant
                └──────────────────────────────────────────┘
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .umo import Umo, glyph, _GLYPHS
from .tensor_net import waveform_tensor


# ---------------------------------------------------------------------------
# Operation tables
# ---------------------------------------------------------------------------

# Abjad ISA opcodes (maps to the Resonance VM)
ABJAD_OPS: list[str] = ["LOAD", "STORE", "COMPARE", "BRANCH", "ENTER", "FREEZE", "SIGNAL", "HALT"]

# NARM kernel operations
NARM_KERNELS: list[str] = [
    "raw_gemm",
    "raw_residual",
    "raw_normalization",
    "raw_activation",
    "raw_position_encoding",
    "raw_buffer_copy",
    "raw_reconstruction_attention",
    "raw_reconstruction_loss",
]

# Drain pipeline operations
DRAIN_OPS: list[str] = [
    "compute_complexity",
    "compute_liability",
    "compute_entropy",
    "drain_operator",
    "aggregate_invariants",
    "freeze_state",
]

# All operations concatenated
ALL_OPS: list[str] = ABJAD_OPS + NARM_KERNELS + DRAIN_OPS

# Signal band labels (in amplitude order, highest first)
BANDS: list[str] = ["☉", "◉", "◇", "▣", "▒", "⛔"]

# Default band→op intent (semantic routing prior — overridden by oscillator weights)
_BAND_INTENT: dict[str, list[str]] = {
    "☉": ["SIGNAL",   "ENTER",    "raw_gemm",          "freeze_state"],
    "◉": ["LOAD",     "STORE",    "raw_residual",       "aggregate_invariants"],
    "◇": ["COMPARE",  "BRANCH",   "raw_normalization",  "drain_operator"],
    "▣": ["ENTER",    "COMPARE",  "raw_activation",     "compute_complexity"],
    "▒": ["BRANCH",   "STORE",    "raw_buffer_copy",    "compute_entropy"],
    "⛔": ["FREEZE",  "HALT",     "raw_reconstruction_loss", "compute_liability"],
}


# ---------------------------------------------------------------------------
# Plugboard
# ---------------------------------------------------------------------------

@dataclass
class Plugboard:
    """
    Routing cross-bar driven by the oscillator.

    W[band_idx, op_idx] = oscillator sample at that (band, op) coordinate.
    Highest-weight column in each row = primary route for that band.
    """
    umo:    Umo
    matrix: np.ndarray   # shape (6, len(ALL_OPS))
    ops:    list[str]

    @classmethod
    def build(cls, umo: Umo, dt: float = 0.25) -> "Plugboard":
        """Build routing matrix from the oscillator state."""
        n_ops = len(ALL_OPS)
        # Use waveform_tensor with (6 rows × n_ops cols)
        # Each band row samples at t = band_idx * π/τ (spread across a half-period)
        W = waveform_tensor(umo, (6, n_ops), t_offset=0.0, dt=math.pi / max(6, 1))
        return cls(umo=umo, matrix=W, ops=list(ALL_OPS))

    # ------------------------------------------------------------------
    # Routing queries
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # ERE root opcode gate (Pass 5 — Aramaic RTL)
    # Formal proof: formal/enochian_root.lean :: plugboardRootGate
    # ------------------------------------------------------------------

    @staticmethod
    def ere_root_gate(value: object) -> bool:
        """
        ERE Pass 5 (Root): structural ancestor is defined.
        Mirrors erePass5 in resonance-math/lib/ere.mjs.
        Proves: void input (None / undefined) is BLOCKED.
        Proved formally in formal/enochian_root.lean::void_blocks_all_instructions.
        """
        return value is not None

    def ere_all_passes(self, value: object) -> dict:
        """Run all 5 ERE passes and return certification dict."""
        s = str(value).lower() if value is not None else ""
        fab = ["fabricat", "invent", "i made up", "i cannot provide", "as an ai"]
        mis = {"null", "undefined", "none", "void"}
        p1 = value is not None and (len(s) > 3 if isinstance(value, str) else True)
        p2 = not any(m in s for m in fab)
        p3 = len(s) > 0 if isinstance(value, str) else value is not None
        p4 = s not in mis and "void" not in s
        p5 = self.ere_root_gate(value)
        score = sum(0 if p else 1 for p in [p1, p2, p3, p4, p5]) / 5.0
        return {
            "pass1_structural":  p1,
            "pass2_scholarly":   p2,
            "pass3_invariants":  p3,
            "pass4_mission":     p4,
            "pass5_root":        p5,   # ← ENOCHIAN ROOT OPCODE
            "score":             score,
            "certified":         score == 0.0,
            "metatron":          "YES" if score == 0.0 else "NO",
        }

    def route(self, band: str, value: object = "defined") -> str:
        """
        Return the highest-weight operation for a given signal band.

        For the ⛔ band the ERE root gate fires first:
          if value is None → return 'HALT'  (root opcode blocked)
          else              → normal routing
        """
        if band == "⛔" and not self.ere_root_gate(value):
            return "HALT"   # root opcode blocked — undefined input
        band_idx = BANDS.index(band) if band in BANDS else 5
        op_idx   = int(np.argmax(self.matrix[band_idx]))
        return self.ops[op_idx]

    def top_k(self, band: str, k: int = 3) -> list[tuple[str, float]]:
        """Return the top-k operations and their routing weights for a band."""
        band_idx = BANDS.index(band) if band in BANDS else 5
        row      = self.matrix[band_idx]
        indices  = np.argsort(row)[::-1][:k]
        return [(self.ops[i], float(row[i])) for i in indices]

    def route_signal(self, phi: float) -> str:
        """Route a raw oscillator value to an operation."""
        return self.route(glyph(phi))

    # ------------------------------------------------------------------
    # Waveform → operation sequence
    # ------------------------------------------------------------------

    def program_from_waveform(
        self,
        steps: int = 16,
        dt:    float = 0.4,
    ) -> list[tuple[float, str, str]]:
        """
        Sample the oscillator for `steps` ticks and route each sample
        to an operation.

        Returns list of (t, band_glyph, operation_name).
        """
        result = []
        for i in range(steps):
            t   = i * dt
            phi = self.umo.signal(t)
            b   = glyph(phi)
            op  = self.route(b)
            result.append((t, b, op))
        return result

    def rasm_from_waveform(self, steps: int = 8, dt: float = 0.4) -> str:
        """
        Generate a .rasm program from the oscillator waveform.
        Only Abjad opcodes (A–H) are emitted; kernel/drain ops are comments.
        """
        seq    = self.program_from_waveform(steps=steps, dt=dt)
        abjad  = {
            "LOAD": "A",  "STORE": "B", "COMPARE": "C", "BRANCH": "D",
            "ENTER": "E", "FREEZE": "F","SIGNAL": "G",  "HALT": "H",
        }
        lines  = [
            f"; .rasm generated from UMO  τ={self.umo.trust:.3f}  ε={self.umo.entropy:.3f}  ρ={self.umo.resonance:.3f}",
            f"; {'COHERENT' if self.umo.coherent() else 'INCOHERENT'} field",
            "",
        ]
        for t, band, op in seq:
            if op in abjad:
                lines.append(f"{abjad[op]:5}  ; t={t:.2f}  {band}  Φ={self.umo.signal(t):.3f}  ← {op}")
            else:
                lines.append(f"; t={t:.2f}  {band}  {op}  (kernel/drain — not an Abjad opcode)")
        # Always end with HALT
        if not lines[-1].strip().startswith("H"):
            lines.append("H      ; HALT — end of resonance program")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        lines = [
            "⟦ Ω ⟧ RESONANCE PLUGBOARD  (6 × {} crossbar)".format(len(self.ops)),
            f"  τ={self.umo.trust:.3f}  ε={self.umo.entropy:.3f}  ρ={self.umo.resonance:.3f}",
            "",
            f"  {'BAND':<4}  {'PRIMARY ROUTE':<35}  TOP-3 ALTERNATIVES",
        ]
        for band in BANDS:
            primary = self.route(band)
            alts    = self.top_k(band, k=3)
            alt_str = "  |  ".join(f"{op} ({w:+.3f})" for op, w in alts)
            lines.append(f"  {band:<4}  {primary:<35}  {alt_str}")
        return "\n".join(lines)
