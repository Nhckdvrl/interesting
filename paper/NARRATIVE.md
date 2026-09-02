# Main line: the intrinsic state space of low-rank adaptation

**Working title.** *What Are the Effective Degrees of Freedom of LoRA
Initialisation?*

This is the current-state document. `01-.../STATUS.md` and `02-.../STATUS.md`
hold the per-topic records; **where they disagree with this file, this file is
right** — Stage 1 conclusions that Stage 2 has since falsified are listed in §6.

---

## 1. The question

Low-rank adapters are initialised in an enormous parameter space, and a fast
growing literature proposes increasingly different ways to choose that
initialisation — random matrices, normalised frames, block-identity structures,
and weight-, activation-, gradient- and curvature-aware subspaces. It is not
clear whether these are genuinely distinct ways to begin adaptation, or
different coordinate descriptions of a much smaller set of effective states.

We do not compare methods. We **construct** initialisations at exactly specified
locations in a candidate state space, map the response of a real transformer to
that space, and then use the published initializers — which never touch the
fitting — as an out-of-distribution test.

## 2. The object

Three `r × r` matrices carry everything:

    M_0 = A Aᵀ            parameter metric
    M_x = A Σ Aᵀ          data / function metric
    M_g = A C_g Aᵀ        gradient metric

Under the adapter-factor gauge `A → QA` all three conjugate by `Q`; under an
exact backbone gauge (`A → ARᵀ`, `Σ → RΣRᵀ`, `C_g → RC_gRᵀ`) all three are
**invariant**. The scientific object is the simultaneous-conjugacy class of the
triple `(M_0, M_x, M_g)`, and the coordinates below are its low-order
invariants:

    S = tr M_x      D = (tr M_x)²/tr(M_x²)      W = tr M_0 / tr M_x
    R_g = tr M_g / tr M_x

If a residual ever survives these, the next place to look is a higher-order
invariant of the same triple — a generalised spectrum — not another hand-picked
feature.


For a layer with input second moment `Σ = E[xxᵀ]`, the state is the rank-space
Gram in the data metric,

    M_x = A Σ Aᵀ  ∈ R^{r×r},        A ∈ R^{r×d} the down-projection.

It is the right object because it respects both exact symmetries of the problem
at once: under the adapter-factor gauge `A → QA` it conjugates,
`M_x → Q M_x Qᵀ`, and under an exact backbone representation gauge
(`x → Rx`, `A → ARᵀ`, `Σ → RΣRᵀ`) it is **unchanged**. So `spec(M_x)` does not
depend on either coordinate system. `diag(AᵀA)` — the quantity the mother paper
normalises — has no such status; it is a description in a particular chart.

Candidate coordinates, all measurable at initialisation from a few batches:

| | definition | meaning |
|---|---|---|
| `S` | `tr M_x` | data-space scale |
| `D` | `(tr M_x)²/‖M_x‖_F²` | intrinsic spectral dimension, in [1, r] |
| `R_g` | `tr(A C_g Aᵀ)/tr(A Σ Aᵀ)` | first-order descent per unit scale |
| `W`, `ω` | `tr(AAᵀ)/tr(A Σ Aᵀ)`, normalised by `W_iso = d/tr Σ` | **parameter** metric ÷ **data** metric |

`R_g` is the quantity that actually enters `⟨G, GP⟩`: writing `A Σ^{1/2} =
Λ^{1/2}Vᵀ`, `tr(A C_g Aᵀ) = tr(Λ Vᵀ T V)` with `T = Σ^{-1/2}C_gΣ^{-1/2}` — the
row-space alignment **weighted by the intrinsic spectrum**.

`W` has a direct reading: with adapter energy `a_i²` on the eigendirections of
`Σ`, `W = 1 / Σ_i (a_i²/Σa²)λ_i`, the reciprocal of the adapter-energy-weighted
activation variance. Large `W` means the adapter must put a lot of parameter
norm into low-variance directions to achieve a given function-space effect.
Since AdamW's step lives in the parameter metric and the function lives in the
data metric, `W` measures a mismatch between the two — which is also what links
this topic to the representation-gauge result.

## 3. What the atlas has established

**Wave 1** — 18 designed points spanning 300× in `S`, 8× in `D` and 230× in an
alignment statistic, 7 learning rates each (`results/atlas`).

* **`S` is the learning-rate coordinate.** Leave-one-out on the atlas alone:
  `log η*` from `S` alone reaches R² = 0.74, and no other candidate improves it.
* **The tuned loss is nearly flat across the whole space** — 0.0059 nats total,
  against a 2.7e-4 measurement null.
* **Master curve.** Rescaling `η → η·S^0.41` (first-order theory says ½)
  tightens the spread of the optimum from 1.63× to **1.23×**, which is the
  resolution of the learning-rate grid itself.

**The out-of-distribution test** — 13 published initializers, located in the
same coordinates without training and never used to fit (`results/ood`,
`results/intrinsic_table.json`):

* **Learning rate: predicted.** Median error **1.37×** against a grid spaced
  1.7×. A law derived only from synthetic constructions predicts where an unseen
  published initializer's optimum sits, across a 300× range of `S`.
* **Tuned loss: systematically mispredicted.** 12/13 residuals positive, mean
  +0.005 nats.

**The failure diagnosed itself.** Wave 1 occupies `W/W_vanilla ∈ [27, 42]`;
every published initializer sits in `[0.018, 1.0]` — disjoint. The residual
correlates with `log W` at **r = −0.79**. The atlas had a coordinate it did not
span, and the OOD test found it.

### Wave 3 — the exact causal ladder in `W` (`results/atlas`, 42 runs)

`M_x = Λ` identically, so `S` and `D` cannot move; a Stiefel-manifold solve
drives `W` to target while holding the spectrum-weighted alignment `R_g` at the
vanilla value. Every rung is matched to 4 decimal places:

| W/W₀ | tr P | S_rel | D | R_g/R_g₀ | L* | lr* |
|---|---|---|---|---|---|---|
| 0.113 | 0.604 | 1.0000 | 6.385 | 1.000 | 0.44410 | 2e-4 |
| 0.304 | 1.621 | 1.0000 | 6.385 | 1.000 | 0.44268 | 2e-4 |
| **1.001** | **5.337** | 1.0000 | 6.385 | 1.000 | **0.44180** | 2e-4 |
| 3.000 | 16.000 | 1.0000 | 6.385 | 1.000 | 0.44223 | 2e-4 |
| 10.008 | 53.369 | 0.9999 | 6.385 | 1.000 | 0.44248 | 3e-4 |
| 30.042 | 159.981 | 0.9986 | 6.384 | 1.001 | 0.44333 | 2e-4 |

Three results at once.

1. **`W` is causally real.** 0.0023 nats across the ladder, 8.5× the
   measurement null, with everything else pinned by construction. It is the
   fourth coordinate, now isolated rather than inferred from a residual.
2. **`W` does not move the learning rate.** `tr P` spans **265×** across the
   ladder while `lr*` stays at 2–3e-4. This is the sharpest available proof
   that `tr P` is not the learning-rate coordinate — `S` is — and it separates
   `W` from `S` as an independent axis rather than a reparameterisation.
3. **The response is non-monotone, with a broad optimum near the isotropic
   baseline.** Extreme mismatch in either direction costs 0.0023 nats; the
   interior is flat to within the seed spread — the differences between
   `ω = 1, 3, 10` (0.44180 / 0.44223 / 0.44248) are 4e-4 to 7e-4, the same size
   as the null, and a quadratic in `log ω` puts the continuous minimum nearer
   `ω ≈ 2.3` than 1. The defensible statement is therefore **"extreme
   parameter-to-data metric mismatch hurts, and the isotropic baseline lies
   inside the broad optimum"**, not "the optimum is exactly the vanilla value".
   Paired seeds at the central rungs are needed before saying more.

   The natural, seed-free reference point is not one random draw but the
   isotropic baseline: for `A_ij` i.i.d. with variance `σ²`,
   `E tr(AAᵀ) = rdσ²` and `E tr(AΣAᵀ) = rσ² tr Σ`, so

       W_iso = d / tr Σ,        ω := W / W_iso = W · tr Σ / d.

   A kaiming draw sits at `ω = 1` *in expectation*, which is why `W/W₀` and `ω`
   coincide here — but `ω` is the quantity to report, since it is computable
   before training, independent of the seed, and comparable across models and
   tasks. Note also that `ω = 1` fixes only this scalar ratio; it does **not**
   mean `A` is isotropic, since many strongly anisotropic `A` share the same `ω`.

The reconstruction at `W/W₀ = 1` gives 0.44180 against the vanilla draw's
0.44276 — a gap of 1e-3, inside that draw's own seed spread of 2.3e-3. So
`(S, D, R_g, W)` reproduce a vanilla initialisation to within its own noise,
from a construction with a completely different `A`.

### The OOD closure test: `W` helps, but does not close the gap

Refitting the law on **synthetic points only**, now with `(D, ω)` and the
theoretical learning-rate exponent, and re-predicting the held-out published
initializers (`src/ood_closure.py`):

| held-out family | systematic offset | sign | lr* median ratio |
|---|---|---|---|
| wave-1 law, all 13 | +0.00500 | 12/13 positive | 1.37× (fitted features) |
| **with `ω`, B₀=0 (9)** | **+0.00283** | 8/8 positive | 1.59× (theory exponent, no fitting) |
| with `ω`, B₀≠0 (4) | +0.00378 | 3/4 positive | 1.28× |

So `ω` removes **44%** of the systematic offset — it was part of the missing
piece — but the residual is still one-sided, so `(S, D, ω)` is **not sufficient**
and we do not claim closure.

Two further honest readings:

* **The theoretical exponent is not quite right.** `log lr* = a − ½ log S` gives
  rms 0.225 on the atlas against 0.155 for a fitted `b = −0.36`. The first-order
  derivation assumes Adam's normalised `ΔB` is isotropic in rank coordinates;
  the 0.14 discrepancy is where that assumption fails, and is worth explaining
  rather than fitting away.
* **The sign of the residual is a lead.** Every atlas construction beats the
  published initializer at its own coordinates. Atlas row spaces are structured
  (eigenbases of `Σ` and `T`); published ones are random. Wave 1 already
  contained the same signal internally — a Haar row space scored 0.44342 against
  0.43926 for a windowed one at identical `(S, D, ρ)`. The next coordinate is
  therefore likely to be about **row-space delocalisation**, i.e. a higher-order
  invariant of the `(M_0, M_x, M_g)` triple, not another hand-picked feature.

## 4. What is running

* **Wave 2** — a discovery sweep of `W` via `A = Ã Σ^{-q}`. Honest about what it
  is: it fixes `S` but lets `D` and the row space drift, because
  `M_x = Ã Σ^{1-2q} Ãᵀ`. Not a causal test.
* **Wave 4 — is `W` a dynamical timescale?** `‖ΔA‖_F/‖A‖_F ~ η_A√(rd)/√(SW)`,
  so the persistence timescale of the initial scaffold is `τ_A ~ √(SW)/η_A`. If
  the tuned optimum corresponds to a roughly fixed `τ_A`, then at fixed `S`,
  **`ω* ∝ η_A²`**: halving `η_A/η_B` should move the optimum of the `ω` ladder
  from 1 to 0.25 and doubling it to 4. The existing rungs 0.3 / 1 / 3 sit almost
  exactly on that prediction. Includes `η_A = 0` (frozen `A`), the sharpest
  control: if the `ω` response flattens, `ω` acts through A-remodelling.
  This is also the point of contact with *LoRA Without Regret*, whose two scalar
  invariants `α·init_A·LR_B` and `init_A/LR_A` are the **isotropic special case**
  of `η_B√S` and `√(SW)/η_A` — our versions are the data-metric generalisation
  to arbitrary anisotropic initializers.
* **`R_g` recomputed offline** for every cached atlas point. With the corrected,
  spectrum-weighted statistic included as a candidate, leave-one-out model
  selection on the atlas still prefers `S + D`; adding `R_g` makes prediction
  worse out of sample. So the alignment null **survives the correction** — it
  was not an artefact of sweeping the unweighted statistic.

## 5. Scale: a training-free prediction, measured through 8B

`paper/scaling_predictor.py`, forward+backward only, Qwen3 0.6B→8B:

| model | d | gradient-energy PR | Σ participation | `tr(PΣ)` top-r / random |
|---|---|---|---|---|
| 0.6B | 1024 | 0.0290 | 0.00684 | 20.8× |
| 1.7B | 2048 | 0.0273 | 0.00312 | 45.2× |
| 4B | 2560 | 0.0164 | 0.00210 | 48.0× |
| 8B | 4096 | **0.0056** | **0.00126** | **87.8×** |

Both mechanisms' headroom **grows with scale**: the pretrained gradient
coordinates concentrate 5.2× further, and the data-metric amplification a
data-aware subspace can buy grows 4.2×. This is a prediction made before any
large-model training, and it is what makes a sparse 8B run informative rather
than a repeat.

## 6. Stage-1 statements that Stage 2 has falsified

| earlier claim | status |
|---|---|
| "the second and only other channel is effective rank" | **false** — `W` is a further coordinate, now causally isolated, and `D`'s effect is weaker than Stage 1 suggested |
| "method identity adds nothing after conditioning on three statistics" | **false** — the OOD test mispredicts tuned loss by +0.005 nats systematically |
| "task alignment is not an independent axis" | **now established** — the statistic swept in wave 1 was unweighted, but the corrected spectrum-weighted `R_g` has since been recomputed for every atlas point and still adds nothing on top of `(S, D)` out of sample |
| "`tr P` is not causal" | **corrected** — `tr P` is not the learning-rate coordinate (265× at constant `lr*`), but `tr P / tr(PΣ)` **is** an independent causal coordinate |
| "the gauge effect's headroom is roughly scale-invariant" | **false** — based on 0.6B/1.7B only; 4B and 8B show PR falling 5.2× |

## 7. Current confidence

    S    confirmed strong coordinate; sets the learning rate, and predicts it
         for unseen published initializers to 1.37x
    D    confirmed, weaker; enters the tuned loss but not the learning rate
    R_g  correct spectrum-weighted statistic now defined and tested; does not
         add predictive value on top of (S, D) in the atlas
    W    CONFIRMED by an exactly matched ladder: 8.5x the null, non-monotone,
         optimal at the vanilla value, and orthogonal to the learning rate
    B0≠0 leaves the equivalence class; not yet folded into the state

## 8. Positioning

| work | what it owns | what remains ours |
|---|---|---|
| **NoRA** (mother paper) | `P = α²AᵀA`; `diag(P)=I` normalisation; BIMI; NoRA-init | that `diag(P)` is a chart-dependent description, and what the chart-free coordinates are |
| *The Loss Does Not See the Basis, but Adam Does* (2608.05136) | optimizer equivariance under the **factor** gauge, on matrix sensing and small transformers | the LoRA state space, its coordinates, and prediction of unseen initializers at LLM scale |
| *Understanding Adam Requires Better Rotation Dependent Assumptions* (NeurIPS 2025) | parameter-space rotations degrade Adam in pretraining | rotations that preserve the network function exactly; a dose ladder with a permutation zero-dose control; a quantitative predictor |
| *Learning Rate Matters* (2602.04998) | 9 LoRA variants collapse after LR tuning (GLUE) | the coordinate that *predicts* the required LR shift, tested out of sample |
| *LoRA-S* (ICLR 2026), *Balanced LoRA* (ICML 2026) | quotient/Stiefel geometry of the adapter factors; balancedness | the **data-metric** state and its response surface, not a new optimizer |
| *GIDA / RaLoRA* (ICLR 2026) | matching LoRA rank to gradient intrinsic dimension | at **fixed nominal rank**, the initialisation-induced spectrum shape as an independent coordinate |

## 9. Compute plan

    0.6B   discovery microscope: the atlas, all controls, all ladders
    1.7B   law selection: a handful of designed points to check the law is not
           a finite-width accident
    8B     prospective falsification only: measure (S, D, R_g, W) on a couple of
           batches, predict lr* and the ordering, then train the predicted
           points and a neighbour.  Model scale is an out-of-distribution axis,
           not another grid dimension.
