"""
fabric.py — Sovereign Compute Fabric.

Wires all resonance components into one pipeline:

  DrainInvariants (kernel output)
      ↓  bridge.py
  Umo (τ, ε, ρ)
      ↓
  ├── tensor_net.py → weight tensors → ResonanceNet → forward pass
  ├── plugboard.py  → routing crossbar → .rasm program
  └── sentence.py   → coherent sentence

The fabric is the plugboard BETWEEN:
  - the NARM drain kernels (hardware)
  - the Resonance ISA (symbolic)
  - the neural network (learned)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .umo import Umo
from .bridge import DrainInvariants, invariants_to_umo, _demo_invariants_from_kernel_params
from .tensor_net import ResonanceNet, waveform_tensor
from .plugboard import Plugboard
from .sentence import produce_sentence, ResonanceSentence


@dataclass
class FabricOutput:
    umo:        Umo
    net:        ResonanceNet
    plugboard:  Plugboard
    rasm:       str
    sentence:   ResonanceSentence
    # Sample forward pass: feed the oscillator waveform through the net
    net_output: np.ndarray


def run_fabric(
    inv:         Optional[DrainInvariants] = None,
    layer_sizes: list[int] = [16, 32, 16, 8],
    steps:       int       = 12,
    dt:          float     = 0.4,
    *,
    complexity_frac: float = 0.72,
    entropy_frac:    float = 0.08,
) -> FabricOutput:
    """
    Full sovereign compute fabric run.

    Parameters
    ----------
    inv          : real drain invariants; synthetic demo if None
    layer_sizes  : ResonanceNet architecture [in, h1, h2, ..., out]
    steps        : oscillator steps for .rasm generation
    dt           : time step
    """
    if inv is None:
        inv = _demo_invariants_from_kernel_params(
            complexity_frac=complexity_frac,
            entropy_frac=entropy_frac,
        )

    umo = invariants_to_umo(inv)

    # 1. Build neural network from waveform tensors
    net = ResonanceNet.build(umo, layer_sizes, dt=0.15)

    # 2. Build plugboard crossbar
    pb = Plugboard.build(umo)

    # 3. Generate .rasm program from waveform
    rasm = pb.rasm_from_waveform(steps=steps, dt=dt)

    # 4. Generate coherent sentence
    sentence = produce_sentence(inv, complexity_frac=complexity_frac,
                                entropy_frac=entropy_frac)

    # 5. Sample forward pass: feed oscillator row-vector through net
    #    Input = one row of the waveform tensor (shape [layer_sizes[0]])
    x_in    = waveform_tensor(umo, (1, layer_sizes[0]), t_offset=6.3, dt=0.1)
    net_out = net.forward(x_in)   # shape (1, layer_sizes[-1])

    return FabricOutput(
        umo        = umo,
        net        = net,
        plugboard  = pb,
        rasm       = rasm,
        sentence   = sentence,
        net_output = net_out,
    )


def render_fabric(
    inv: Optional[DrainInvariants] = None,
    **kwargs,
) -> str:
    out = run_fabric(inv, **kwargs)
    umo = out.umo

    sep = "─" * 60
    lines = [
        "⟦ Ω ⟧ SOVEREIGN COMPUTE FABRIC",
        sep,
        f"  τ={umo.trust:.4f}  ε={umo.entropy:.4f}  ρ={umo.resonance:.4f}"
        f"  {'COHERENT ☉' if umo.coherent() else 'INCOHERENT ⛔'}",
        sep,
        "",
        "── TENSOR NETWORK ──────────────────────────────────────",
        out.net.summary(),
        "",
        "── PLUGBOARD ───────────────────────────────────────────",
        out.plugboard.summary(),
        "",
        "── RESONANCE ISA PROGRAM (.rasm) ───────────────────────",
        out.rasm,
        "",
        "── NET FORWARD PASS (waveform input → output) ──────────",
        f"  input  shape : {(1, out.net.layers[0].in_dim)}",
        f"  output shape : {out.net_output.shape}",
        f"  output       : {np.round(out.net_output[0], 4).tolist()}",
        "",
        "── COHERENT SENTENCE ───────────────────────────────────",
        f"  {out.sentence.sentence}",
        "",
        sep,
        f"  {umo.sealed_output()}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import sys, io
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    cfrac = float(sys.argv[1]) if len(sys.argv) > 1 else 0.72
    efrac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.08
    print(render_fabric(complexity_frac=cfrac, entropy_frac=efrac))
