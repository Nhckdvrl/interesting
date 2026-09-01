# 02 — Representation Gauge — STATUS

Last update: 2026-09-01

## Current evidence

### E0 — exact functional equivalence (DONE, `src/e0_exactness.py`, `results/e0_exactness.json`)

Two exact gauge families implemented in `common/gauge.py` for Qwen3/Llama-style
backbones, both verified on Qwen3-0.6B-Base over 8 held-out GSM8K batches:

* **G1 value/output gauge** — per KV head `W_V ← C_g W_V`, `W_O ← W_O C_gᵀ`.
  Exact for Qwen3 (no value-norm, no attention bias). Needs no model surgery.
* **G2 residual-stream gauge** — QuaRot-style: fold every RMSNorm γ into the
  following linear layers, untie the LM head, then rotate the whole residual
  stream by `R ∈ O(d_model)`. RoPE and Qwen3's per-head q/k-norm live on the
  head_dim axis and are untouched.

Measured logit error vs. the untransformed model (relative L2 over unmasked
positions) and the resulting eval loss:

| precision | V/O random | V/O Hadamard | γ-fold+untie | residual random | residual Hadamard |
|---|---|---|---|---|---|
| fp32 weights + fp32 compute | 2.8e-6 | 2.7e-6 | 2.7e-6 | 5.2e-6 | 4.2e-6 |
| fp32 weights + bf16 autocast | 2.0e-2 | 2.1e-2 | 2.5e-2 | 3.2e-2 | 2.9e-2 |
| bf16 weights + bf16 compute | 2.9e-2 | 2.9e-2 | 3.1e-2 | 6.0e-2 | 5.4e-2 |

Eval loss (base = 0.842330 in fp32):

| precision | base | V/O rand | V/O Had | γ-fold | res. rand | res. Had | spread |
|---|---|---|---|---|---|---|---|
| fp32 | 0.842330 | 0.842330 | 0.842330 | 0.842330 | 0.842330 | 0.842330 | **7e-7** |
| fp32+bf16 autocast | 0.840706 | 0.841008 | 0.840623 | 0.841337 | 0.841668 | 0.843011 | **2.4e-3** |
| bf16 | 0.842471 | 0.841655 | 0.844761 | 0.843792 | 0.838490 | 0.836841 | **7.9e-3** |

## Interpretation

1. **The transformations are correct.** In fp32 the gauge-transformed backbones
   agree with the original to 7 significant figures of eval loss. E0 passes;
   downstream fine-tuning comparisons are licensed.

2. **A methodological result that must gate every later experiment.** In bf16 —
   the precision essentially all PEFT work uses — *functionally identical*
   backbones already differ by **7.9e-3 nats** of eval loss before a single
   gradient step. γ-folding alone, a pure algebraic identity, moves the loss by
   1.3e-3. A residual Hadamard rotation makes the *pretrained* model look
   5.6e-3 nats **better** for free. Any gauge-dependence of fine-tuning smaller
   than ≈1e-2 nats measured in bf16 is unattributable.
   → **All topic-02 training runs use fp32 weights and fp32 compute.**
   This mirrors QuaRot/SpinQuant's observation one precision level up: the
   "exact" symmetry is only exact above bf16.

## Supported / falsified

* Supported: exact V/O and residual gauges exist and are implementable for Qwen3.
* Not yet tested: any claim about fine-tuning dynamics.

## Unresolved

* Whether vanilla LoRA + SGD is empirically gauge-equivariant on a real model
  (the positive control, E1).
* Whether NoRA's `N(AR) ≠ N(A)R` produces a measurable trajectory difference.
* Whether AdamW's own non-covariance dominates the NoRA-specific effect.

## Theory added beyond the README

For an input-side gauge `x' = Rx`, `W' = W Rᵀ`, the full-weight gradient obeys
`G' = G Rᵀ`. Coupling the adapter as `A' = A Rᵀ`, `B' = B` gives under plain SGD
`∇_{B'} = ∇_B` and `∇_{A'} = (∇_A) Rᵀ`, so the coupling is preserved **for all
time**, not just at the first step: vanilla LoRA + SGD is *exactly* gauge
equivariant, and the two runs are the same function at every step. This makes
E1 a bit-level positive control rather than an approximate one.

## Next experiment

`src/run_gauge_pair.py` — the 2×2 panel
`{SGD, AdamW} × {kaiming, NoRA-init}` × `{orig, coupled_algo, coupled_oracle}`
on the V/O gauge, fp32, Qwen3-0.6B-Base, GSM8K.

* `SGD × kaiming × coupled_*` must reproduce the original trajectory exactly
  (positive control; any deviation is an implementation bug).
* `SGD × nora × coupled_algo` vs `coupled_oracle` isolates the normalisation
  non-commutation with zero optimizer confound.
* The AdamW rows give the optimizer-only gauge-dependence floor that any
  NoRA-specific claim must exceed.
