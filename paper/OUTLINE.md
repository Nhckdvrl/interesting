# Paper outline (ICLR form)

Locked after four reframings. Every section names the experiment that carries
it and the file the numbers come from. Nothing here is aspirational.

---

## Title

**LoRA Initializers Carry a Frame Their Papers Do Not Report**

(working alternative: *What an Initialization Is Depends on the Optimizer*)

## The one-sentence claim

AdamW's behaviour depends on information that provably is not there — the
`O(r)` gauge of LoRA's factorisation, which leaves the function, the loss and
the gradient identical — and every published LoRA initialiser sets that
coordinate by accident, in a way that determines how much of its reported
advantage survives a change of optimizer.

**Why this is not the NeurIPS 2025 rotation result.** That paper rotates a
generic parameter space, which genuinely changes the loss landscape in
coordinates. Ours is an *exact redundancy*: `(A, B)` and `(QA, BQ^T)` are the
same function with the same loss and the same gradient. Adam responds to a
coordinate carrying zero function-space information. That is a different, and
stranger, statement.

---

## 1. Introduction

LoRA writes `ΔW = s B A`. The factorisation is not unique: `(SA, B S^{-1})`
gives the same function for any invertible `S`. Four recent works —
LoRA-RITE (ICLR'25), Balanced LoRA, FedRot-LoRA, Riemannion (ICLR'26) — treat
this ambiguity as a defect to remove. **None asks what it is worth, or which
part of it a given optimizer can see.**

Boxed question, NoRA-style:

> *Which part of LoRA's reparameterisation group does your optimizer see, and
> what does the initialisation literature's unreported choice of it cost?*

## 2. The group splits, and so do the optimizers

`GL(r)` factors by polar decomposition into a rotation part `O(r)` and a
scaling part. Nine optimizers, 25 steps, float64, each at its own tuned
learning rate, each normalised by its own signed-permutation noise floor
(`src/hierarchy.py`, `paper/hierarchy.txt`):

| | `O(r)`/floor | verdict |
|---|---|---|
| SGD, SGD+momentum, Muon, matrix-preconditioned Adam | 0.3 – 2.3 | blind |
| AdamW, Lion, RMSprop, Adagrad, Adadelta | 8e11 – 2e13 | sees it |

**Twelve orders of magnitude.** The line is not adaptivity and not the norm: it
is whether the preconditioner is *diagonal in the coordinates the gauge acts
on*. Lion's position was predicted before measurement and confirmed.

*Correction to concurrent work.* Riemannion motivates itself by asserting
per-factor Muon is not invariant to "scalings **or rotations**". The rotation
half is wrong by six orders of magnitude, and it is exactly the half that
matters.

## 3. The frame is real in training, not only in analysis

One gauge orbit of the vanilla draw; every logged preconditioner statistic
equal to 1.5e-8; only the frame moves (`results/frame`, `src/analyze_frame.py`).

| optimizer | tuned spread over the orbit |
|---|---|
| SGD | 0.00001 |
| matrix-preconditioned Adam | 0.00022 |
| Muon | 0.00026 |
| Lion | 0.00068 |
| **AdamW** | **0.00222** (11x the 2e-4 reproducibility floor) |

Monotone at every learning rate; three seeds. **Structural control:** the
effect is exactly zero at `r = 1`, where `O(1)` *is* AdamW's own symmetry
group, and grows through `r = 4, 16, 64, 128` (`src/analyze_rank.py`).

## 4. Every published initialiser has a frame fingerprint

Measured with **no training**, one probe pass per model, on N model families
(`src/second_order.py`, `src/analyze_audit.py`):

* the zoo spans the coordinate 3–4.3x;
* the separation is **perfect on every model** — every frame-based method
  (Kaiming, NoRA, NoRA-unit, ETF, BiMI) above every data-aware one
  (PiSSA-minor, OLoRA, EVA, PiSSA, gradsub, LoRA-One);
* the ordering is stable across architectures, so the coordinate is a property
  of the **method**, not of the backbone — one number per method, reportable;
* the whole NoRA family sits at vanilla's value, which is exactly why our
  Stage-1 audit found those five conditions mutually indistinguishable to
  1.2x the noise floor.

## 5. It is causal, not definitional

The audit alone is partly definitional: `Λ₁` measures concentration and
eigenvector-based methods concentrate by construction. So we move the frame
while holding the method fixed (`results/rot`, 96 runs). Rotating to the
gradient-metric eigenframe helps or is neutral **6/6**; rotating away hurts or
is neutral **6/6**. `BA`, `P` and all nine gauge invariants are preserved to
1e-15, so SGD cannot tell the conditions apart at all.

## 6. What it costs: a third of the margin is optimizer-specific

The same six initialisers under AdamW and under a frame-blind optimizer:

* the **ranking survives** (Kendall tau = +0.87; the one discordant pair
  separated by 0.00001 nats). We say so plainly.
* the **margins do not**: Kaiming's advantage over gradsub, EVA and PiSSA
  falls 31%, 29% and 37%.

So a LoRA initialisation result obtained with AdamW carries a component that
does not transfer, and no paper reports enough to say which component.

## 7. Limits, stated

* Effects are 0.002–0.006 nats. Large against the 2e-4 floor and against the
  gaps between published methods (2.5x the median adjacent gap), small in
  absolute terms, and **they do not move GSM8K exact-match** (0.505 vs 0.500 at
  n = 200, inside binomial noise). We report that rather than chasing it.
* The prescription "rotate to the eigenframe" holds at 0.6B, 3B, 1000 steps and
  at 8B *once the probe is large enough*, and its direction is optimizer-specific
  (Lion prefers a different frame). It is an operating range, not a method.
* SFT only. The RL and pretraining regimes are absent by choice, not oversight.
* Two of our own pre-registered predictions were falsified and one reported
  reading was over-called and corrected; all four prediction files are in the
  repository unedited with outcomes appended.

## 8. What we are not claiming

Not a better initialiser. Not that Adam's basis-dependence is new — it is not.
Not that the ranking of published methods is wrong. The contribution is a
**measurement of a subfield's methodology**: a coordinate everyone varies, no
one reports, that is free to measure and that determines a third of the numbers
being compared.

---

## Evidence inventory

| section | source | runs |
|---|---|---|
| 2 | `src/hierarchy.py` | analytic, 9 optimizers |
| 3 | `results/frame` | 130+ |
| 3 (rank) | `results/rank` | 80+ |
| 4 | `results/second_order_*.json` | 0 (training-free), N models |
| 5 | `results/rot` | 96 |
| 6 | `results/rot` (Muon arm) | 44 |
| 7 | `results/q8b`, `q8bprobe`, `frame_bf16`, `acc`, `long` | 100+ |

Total trained runs in the project: **2348**.
