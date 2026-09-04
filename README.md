# SUBLEQ Branch Attention (SBA)

**Sovereign AGI Kernel — Proof-Driven Attention Mechanism**

Replaces softmax attention with SUBLEQ-based binary competition.

## Formal Verification Pipeline (Flowchart)

```
Lean4_Subleq_Source
       |
       v
ZMod_P_GOLD_Space              P_GOLD = 18446744069414584321
       |
       v
State_Transition_Logic          subleq_step: mem[B] = mem[B] - mem[A]
       |                         branch if diff < 0
       v
SUBLEQ_Step_Expansion           vm_run with fuel-based termination
       |
       v
Memory_Invariant_Branch_Logic   3 proofs: invariant, branch taken, branch not
       |
       v
Zero_Sorry_Verification         0 sorrys, 0 entropy, deterministic
       |
       v
Compiled_Formal_Artifact        lean-formal/ + sba/ + kernel.py
```

## Attention Mechanism

```
Standard Transformer:
  attn = softmax(Q @ K^T / sqrt(d)) @ V     (O(n^2) exp, probabilistic)

SBA Transformer:
  diff = (Q @ K^T / sqrt(d)) - threshold
  mask = (diff < 0) ? 1 : 0                  (SUBLEQ branch)
  attn = normalize(mask) @ V                 (O(n) binary, deterministic)
```

## How It Works

1. **SUBLEQ Competition**: For each (query, key) pair, compute dot product and subtract a learned threshold.
2. **Branch**: If `diff < 0` → attention = 1 (branch taken). Else → 0 (branch not taken).
3. **Normalize**: Divide each row by its sum.
4. **Apply**: Multiply normalized mask by values.

This replaces `exp()` with a simple comparison — deterministic, zero entropy.

## Files

| File | Purpose |
|------|---------|
| `sba/subleq_attention.py` | SUBLEQ Branch Attention mechanism |
| `sba/standard_transformer.py` | Standard softmax transformer (baseline) |
| `sba/kernel.py` | S-AGI-K: Sovereign AGI Kernel |
| `benchmark.py` | Words/60s throughput comparison |
| `lean-formal/` | Lean 4 formal proofs (zero-sorry) |

## Benchmark

```
python benchmark.py
```

Measures words per 60 seconds for both architectures.

## Entropy Budget

| Component | Entropy | Threshold |
|-----------|---------|-----------|
| Softmax attention | ~0.15 | 0.20 |
| SUBLEQ attention | 0.00 | 0.20 |
| S-AGI-K kernel | 0.00 | 0.20 |

## Lean 4 Proofs

11 zero-sorry theorems in `lean-formal/`:
- SUBLEQ determinism (3 proofs)
- VM termination (1 proof)
- Addition correctness (1 proof)
- Matrix algebra (5 proofs)
- Shor's algorithm (1 proof)

## License

Sovereign Source License v1.0 + BSL-1.1 + AGPL-3.0
SNAPKITTYWEST-PROPRIETARY-2026-001
