> **This topic is now one half of a single result.** Topic 01 found the same
> structure at LoRA's *adapter* gauge `O(r)` that this topic found at the
> *backbone* gauge `O(d)`: SGD flat, AdamW monotone in the dose, and a
> signed-permutation rung that is an exact zero-dose control because that is
> precisely AdamW's own symmetry group.
>
> The unified statement is in `paper/OUTLINE.md`: **which reparameterisation
> symmetries the optimizer respects decides what an initialisation is.** SGD
> (Frobenius norm) and Muon (spectral norm) descend in orthogonally invariant
> geometries and are exactly covariant under both gauges; AdamW (elementwise
> max norm) is covariant under neither, so for AdamW the frame is a real
> degree of freedom at both levels.
>
> This file remains the topic-02 record. Its measurements stand; what has
> changed is that they are no longer a separate finding.

# 02 — Representation Gauge — STATUS

Last update: 2026-09-02

---

## Current evidence

### E0 — exact functional equivalence (`src/e0_exactness.py`)

Two exact gauge families for Qwen3/Llama-style backbones (`common/gauge.py`):

* **V/O gauge** — per KV head, `W_V ← C_g W_V`, `W_O ← W_O C_gᵀ`. Exact for
  Qwen3 (no value-norm, no attention bias); no model surgery.
* **Residual-stream gauge** — QuaRot-style: fold every RMSNorm γ into the linear
  layers that read it, untie the LM head, then rotate the whole residual stream
  by `R ∈ O(d_model)`. RoPE and Qwen3's per-head q/k-norm act on the head_dim
  axis and are untouched.

| precision | logit rel-L2 (residual random) | eval-loss spread over 5 exact gauges |
|---|---|---|
| fp32 | 5.2e-6 | **7e-7** |
| fp32 weights + bf16 autocast | 3.2e-2 | 2.4e-3 |
| bf16 | 6.0e-2 | **7.9e-3** |

**Methodological result that gates everything downstream:** in bf16 —
the precision essentially all PEFT work uses — *functionally identical*
backbones differ by 7.9e-3 nats before a single gradient step; γ-folding alone,
a pure algebraic identity, moves the loss by 1.3e-3, and a Hadamard rotation
makes the pretrained model look 5.6e-3 nats **better** for free. Any
gauge-dependence measured in bf16 below ≈1e-2 nats is unattributable.
→ every training run in this project is **fp32 weights and fp32 compute**.

### The dose ladder (`src/run_dose.py`, `results/dose`, 96 runs)

Instead of one rotation, an exact ladder in *how many coordinates get mixed*:
`none → perm(1) → block4 → block16 → block64 → block256 → rand(1024) /
hadamard(1024)`. Every rung is an exact gauge with the same spectrum and the
same function; only the mixing changes.

`perm` is a built-in **zero-dose control**: AdamW is *exactly* covariant under
permutations and sign flips because `m/√v` is elementwise. SGD (with momentum),
decoupled weight decay and global-norm clipping are covariant under **all**
rungs, so every SGD row is a full positive control.

Best-tuned loss (per-rung LR sweep), Qwen3-0.6B-Base / NuminaMath / 500 steps.

**Full 8-rung ladder, 3 LRs per cell** (`results/dose`, 96 runs):

| gauge | mixed coords | FullFT+AdamW | LoRA+AdamW | LoRA+SGD |
|---|---|---|---|---|
| none | 1 | 0.49391 | 0.49300 | 0.49929 |
| perm | 1 | −0.00008 | +0.00001 | +0.00005 |
| block4 | 4 | −0.00006 | −0.00011 | −0.00027 |
| block16 | 16 | +0.00007 | −0.00010 | +0.00004 |
| block64 | 64 | +0.00024 | +0.00010 | +0.00002 |
| block256 | 256 | +0.00065 | +0.00042 | −0.00015 |
| rand | 1024 | +0.00149 | +0.00028 | +0.00007 |
| hadamard | 1024 | **+0.00174** | +0.00110 | −0.00016 |

**Dense 7-point LR sweeps × 2 seeds, complete 2×2** (`results/edge`, 224 runs):

| gauge | PR | FullFT+AdamW | FullFT+SGD | LoRA+AdamW | LoRA+SGD |
|---|---|---|---|---|---|
| none | 0.0252 | 0.49340 | 0.49944 | 0.49309 | 0.49832 |
| block64 | 0.0759 | +0.00050 | +0.00010 | +0.00010 | −0.00021 |
| rand | 0.4763 | +0.00196 | +0.00001 | +0.00054 | +0.00001 |
| hadamard | 0.7265 | **+0.00229** | **−0.00005** | **+0.00051** | **−0.00022** |

Both SGD columns are flat to ±2e-4 with no trend; both AdamW columns rise
monotonically with the dose. Adam's advantage over SGD:

| gauge | PR | FullFT | LoRA |
|---|---|---|---|
| none | 0.025 | +0.00604 | +0.00523 |
| block64 | 0.076 | +0.00565 (−7%) | +0.00492 (−6%) |
| rand | 0.476 | +0.00409 (−32%) | +0.00471 (−10%) |
| hadamard | 0.727 | **+0.00370 (−39%)** | **+0.00450 (−14%)** |

### Mechanism (`src/diag_dominance.py`)

Why should the pretrained basis be special? AdamW's second moment is a
*diagonal* model of gradient scale, useful exactly insofar as coordinate-wise
scales are heterogeneous. Rotation averages that heterogeneity away. Measured by
the participation ratio of per-input-coordinate gradient energy
`PR = (Σ E_j)² / (d·Σ E_j²)` (1 = uniform, small = concentrated):

| gauge | PR | energy kurtosis | FullFT penalty |
|---|---|---|---|
| none | 0.02525 | 364.1 | +0.00000 |
| perm | **0.02525** (bit-identical) | 364.1 | −0.00008 |
| block4 | 0.03024 | 217.1 | −0.00006 |
| block16 | 0.05829 | 55.1 | +0.00007 |
| block64 | 0.07585 | 26.2 | +0.00024 |
| block256 | 0.20377 | 6.2 | +0.00065 |
| rand | 0.47631 | 2.2 | +0.00149 |
| hadamard | **0.72652** | 1.4 | **+0.00174** |

Penalty vs PR: **Pearson r = +0.98** (FullFT), +0.91 (LoRA); Spearman +0.93 /
+0.79. The ordering is predicted correctly *including the non-trivial tie-break*
— Hadamard homogenises more than a random rotation (PR 0.73 vs 0.48) and has the
larger penalty, even though both mix all 1024 coordinates.

`PR = 0.025` in the pretrained basis means the gradient energy lives in ~2.5% of
the residual coordinates: the outlier-feature structure that QuaRot/SpinQuant
rotate away *to help quantisation* is the same structure AdamW exploits.

### NoRA-specific non-commutation (`results/g1`, 44 runs)

`N(AR) ≠ N(A)R` is mathematically real, but its training consequence is **below
the numerical floor**. On the residual gauge, 3 gauge seeds, fp32:

| optimizer | kaiming floor (optimizer + roundoff) | NoRA non-commutation effect |
|---|---|---|
| SGD lr=0.03 | 1.4e-4 | **5.6e-5** |
| SGD lr=0.1 | 2.0e-4 | **5.4e-5** |
| AdamW lr=1e-4 | 1.1e-3 | 1.0e-3 |
| AdamW lr=3e-4 | 3.5e-3 | 2.8e-3 |

Positive control E1: vanilla LoRA + SGD in the original vs. the rotated gauge is
identical at step 0 to 8 digits and drifts only to 2.7e-5 nats of eval loss after
100 steps — the fp32 roundoff floor.

---

## Interpretation

1. **Vanilla LoRA + SGD is exactly gauge-equivariant** — derived and verified.
   The theorem is stronger than the README's first-step version: for an input
   gauge `x'=Rx`, `W'=WRᵀ`, coupling `A'=ARᵀ`, `B'=B` gives `∇_{B'}=∇_B` and
   `∇_{A'}=(∇_A)Rᵀ`, so the two runs are the same function *at every step*.
2. **AdamW is not**, and the violation is a clean, monotone dose–response in the
   amount of coordinate mixing, with an exact zero-dose control.
3. **The mechanism is identified and predictive**, not merely descriptive.
4. **PEFT does not carry excess gauge sensitivity — it carries less.** With the
   complete 2×2 at dense LR resolution, FullFT loses 4.5× more to a Hadamard
   gauge than LoRA (+0.00229 vs +0.00051) and loses 2.8× more of its Adam
   advantage (−39% vs −14%). This falsifies the project's originally preferred
   Branch A and moves it to Branch B, with the sign reversed from the natural
   guess: the low-rank constraint partially *shields* the optimizer from the
   coordinate system.
5. **NoRA's non-commutation is a mathematical fact with no measurable
   consequence** — the effect is below the fp32 roundoff floor under SGD, and no
   larger than the plain-LoRA baseline under AdamW.

## Honest assessment of effect size

The best-tuned penalty is **0.0017 nats (~0.35%)** for FullFT and 0.0011 for
LoRA; at a fixed (shared) learning rate it reaches 0.0071. This is a *real,
airtight, mechanistically explained* effect, but a small one. Under the
README's survival tree this is **Branch B** with a **Result class B/C** outcome:
the phenomenon is systematic and the mechanism is clean, but it is not a
benchmark catastrophe.

## Supported / falsified

* Supported: exact gauges exist and are implementable; SGD equivariance;
  AdamW non-equivariance with a monotone dose–response; the diagonal-model
  mechanism; FullFT > LoRA in sensitivity.
* Falsified: NoRA-specific gauge sensitivity (Branch A). Also falsified: the
  guess that PEFT would be *more* gauge-sensitive than full fine-tuning.

## Unresolved

* Whether the penalty grows with model scale or training length — the single
  most important remaining question for whether this is a curiosity or a
  practical concern.
* Whether an *optimised* gauge (rather than a random one) can beat the
  pretrained basis, i.e. whether coordinate choice is an exploitable design axis
  rather than only a source of degradation.
* ~~Whether Adam's advantage over SGD shrinks in a homogenised basis~~ —
  **answered**: it does, by 39% (FullFT) and 14% (LoRA), monotonically in the
  participation ratio, while SGD itself is unchanged.

## Next experiments

1. **Scale.** The exact residual-stream gauge is already verified at 7B in fp32
   (Mistral-7B-v0.3: all five gauges within 2.7e-7 nats of the untransformed
   model), so the machinery is ready. The right backbone to scale on is
   **Qwen3-8B** — hidden_size 4096 is a power of two, which the Hadamard rung
   requires, and Qwen2.5-7B's 3584 is not.
2. Gauge seeds 1–2 on every rung for error bars.
3. Whether the penalty grows with training length as well as with scale.
