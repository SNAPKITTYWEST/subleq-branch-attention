"""
sentence.py — Resonance Word Oracle + Coherent Sentence Generator.

The oscillator signal Φ(t) = sin(τ·t) × cos(ρ·t) × (1−ε) is sampled at
grammatically-assigned time points. Each sample determines:

  amplitude band  → word class (which column of the oracle)
  fractional part → word index (which specific word)

Grammar templates produce subject–verb–object sentences that remain
coherent when ε < 0.21 (the same gate the VM enforces).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .umo import Umo, glyph
from .bridge import DrainInvariants, invariants_to_umo, _demo_invariants_from_kernel_params


# ---------------------------------------------------------------------------
# Word oracle — vocabulary indexed by [role][amplitude_band][word_index]
# Vocabulary is drawn from the resonance ISA domain: trust, entropy, WORM,
# sovereignty, crystalline structure, phase, lattice.
# ---------------------------------------------------------------------------

_ORACLE: dict[str, dict[str, list[str]]] = {
    "subject": {
        "☉": ["the sovereign field",   "the sealed deed",       "the trust kernel",
               "the crystalline state", "the luminous proof"],
        "◉": ["the resonance lattice",  "the coherent signal",   "the worm chain",
               "the oscillator",        "the entropy gate"],
        "◇": ["the phase manifold",     "the waveform",          "the periodic orbit",
               "the propagation front", "the cycle"],
        "▣": ["the threshold boundary", "the shadow register",   "the marginal trace",
               "the partial state",     "the fading lattice"],
        "▒": ["entropy",                "the noise floor",       "scattered light",
               "the dim signal",        "residual drift"],
        "⛔": ["the void",              "the null state",         "collapsed resonance",
               "silence",               "the zero field"],
    },
    "verb": {
        "☉": ["seals",          "crystallizes into", "anchors",      "locks",        "preserves"],
        "◉": ["resonates with", "harmonizes with",   "stabilizes",   "holds",        "bounds"],
        "◇": ["drifts through", "folds into",        "encodes",      "traces",       "maps to"],
        "▣": ["approaches",     "tests",             "marks",        "grazes",       "thresholds"],
        "▒": ["scatters into",  "dims toward",       "degrades to",  "weakens into", "disperses through"],
        "⛔": ["collapses into", "nullifies",         "silences",     "erases",       "terminates"],
    },
    "object": {
        "☉": ["the immutable ledger",  "the sovereign proof",  "the trust chain",
               "the worm seal",        "the eternal deed"],
        "◉": ["the coherent field",    "the harmonic lattice", "the resonance",
               "the stable waveform",  "the bounded state"],
        "◇": ["the phase space",       "the periodic manifold","the trajectory",
               "the folded cycle",     "the orbital map"],
        "▣": ["the threshold",         "the boundary layer",   "the marginal orbit",
               "the partial trace",    "the edge condition"],
        "▒": ["the noise manifold",    "the fading signal",    "the scatter field",
               "the degraded path",    "the dim lattice"],
        "⛔": ["all coherence",         "the structure",        "the field",
               "the waveform",         "the resonance itself"],
    },
    "modifier": {
        "☉": ["with absolute trust",      "under the sovereign seal",  "in crystalline order",
               "through the eternal deed", "beyond entropy"],
        "◉": ["with harmonic precision",  "across the resonant field",  "within stable orbits",
               "through coherent phases",  "along the lattice"],
        "◇": ["through phase folds",      "along periodic cycles",      "across the manifold",
               "through the waveform",    "within the trajectory"],
        "▣": ["at the threshold",         "along the boundary",         "near the margin",
               "at the edge condition",   "approaching collapse"],
        "▒": ["through scattered noise",  "dimly",                      "with fading amplitude",
               "through the noise floor", "weakly"],
        "⛔": ["in silence",              "into the void",               "without resonance",
               "toward null",             "in total collapse"],
    },
}

# Grammar templates: list of (role, t_offset) tuples.
# t_offsets are chosen to sample Φ near its peak windows
# (t ≈ 6.0–7.0 and t ≈ 11.3–11.6 for τ=0.72, ρ=0.50)
# so the oracle draws from the high-amplitude ☉/◉ vocabulary.
_TEMPLATES = [
    # SUBJ(peak)  VERB(peak)  OBJ(peak)
    [("subject", 6.3),  ("verb", 6.6),  ("object", 6.9)],
    # SUBJ(peak)  VERB(peak)  OBJ(peak)   MOD(second peak)
    [("subject", 6.3),  ("verb", 6.6),  ("object", 6.9),  ("modifier", 11.4)],
    # MOD(edge)   SUBJ(peak)  VERB(peak)  OBJ(second peak)
    [("modifier", 5.8), ("subject", 6.2), ("verb", 6.5), ("object", 11.3)],
    # SUBJ(peak)  VERB(mid)   OBJ(peak)
    [("subject", 6.4),  ("verb", 6.8),  ("object", 7.1)],
]


# ---------------------------------------------------------------------------
# Word selection
# ---------------------------------------------------------------------------

def _band(value: float) -> str:
    v = abs(value)
    for threshold, ch in [(0.9, "☉"), (0.7, "◉"), (0.5, "◇"),
                           (0.3, "▣"), (0.1, "▒"), (0.0, "⛔")]:
        if v > threshold:
            return ch
    return "⛔"


def _pick(umo: Umo, role: str, t: float) -> str:
    phi = umo.signal(t)
    band = _band(phi)
    words = _ORACLE[role][band]
    # Use fractional phase position to select word within the band
    frac = abs(phi) % 0.1 / 0.1          # 0..1 within the amplitude band
    idx  = int(frac * len(words)) % len(words)
    return words[idx]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class ResonanceSentence:
    words:    list[tuple[str, str]]   # [(role, word), ...]
    sentence: str
    umo:      Umo
    glyphs:   str                     # waveform snapshot

    def __str__(self) -> str:
        return self.sentence


def produce_sentence(
    inv:              Optional[DrainInvariants] = None,
    template_index:   int   = 1,
    *,
    complexity_frac:  float = 0.72,
    entropy_frac:     float = 0.08,
) -> ResonanceSentence:
    """
    Generate one coherent sentence from the oscillator state.

    Parameters
    ----------
    inv : DrainInvariants, optional
        Real kernel invariants. If None, uses synthetic demo values.
    template_index : int
        Which grammar template to use (0–3).
    """
    if inv is None:
        inv = _demo_invariants_from_kernel_params(
            complexity_frac=complexity_frac,
            entropy_frac=entropy_frac,
        )

    umo = invariants_to_umo(inv)

    if not umo.coherent():
        return ResonanceSentence(
            words    = [("status", "incoherent")],
            sentence = "⛔ entropy too high — resonance collapsed — no sentence produced",
            umo      = umo,
            glyphs   = umo.waveform(12, 0.4),
        )

    template = _TEMPLATES[template_index % len(_TEMPLATES)]
    parts    = [(role, _pick(umo, role, t)) for role, t in template]
    sentence = " ".join(word for _, word in parts).strip()
    # Capitalise first letter
    sentence = sentence[0].upper() + sentence[1:] + "."

    glyphs = umo.waveform(len(sentence.split()) * 3, 0.4)

    return ResonanceSentence(
        words    = parts,
        sentence = sentence,
        umo      = umo,
        glyphs   = glyphs,
    )


def produce_all_sentences(
    inv: Optional[DrainInvariants] = None,
    **kwargs,
) -> list[ResonanceSentence]:
    """Generate one sentence per grammar template."""
    return [produce_sentence(inv, i, **kwargs) for i in range(len(_TEMPLATES))]


def render_sentence(
    inv: Optional[DrainInvariants] = None,
    **kwargs,
) -> str:
    sentences = produce_all_sentences(inv, **kwargs)
    umo = sentences[0].umo

    lines = [
        "⟦ Ω ⟧ RESONANCE WORD MACHINE — Coherent Sentence Output",
        "─" * 58,
        f"  τ={umo.trust:.3f}  ε={umo.entropy:.3f}  ρ={umo.resonance:.3f}"
        f"  {'COHERENT ☉' if umo.coherent() else 'INCOHERENT ⛔'}",
        "─" * 58,
        "",
    ]

    for i, s in enumerate(sentences):
        lines.append(f"  template {i}  {s.glyphs}")
        lines.append(f"  → {s.sentence}")
        lines.append("")

    lines += ["─" * 58, f"  {umo.sealed_output()}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, io
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    cfrac = float(sys.argv[1]) if len(sys.argv) > 1 else 0.72
    efrac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.08
    print(render_sentence(complexity_frac=cfrac, entropy_frac=efrac))
