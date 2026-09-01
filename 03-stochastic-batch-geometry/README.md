# 03 — When Low Rank Changes the Noise: Stochastic Geometry and Critical Batch Size of LoRA

**Status:** REGISTERED — Priority A3  
**Planning score:** 89/100  
**Mother paper:** Normalized Low-Rank Adaptation (Kang et al., 2026)  
**Primary target:** ICML / NeurIPS  
**Central compute requirement:** SFT-only core, 0.5B–3B models, logical batch sweeps via accumulation, ≤4 GPUs

## One-sentence paper hook

> **LoRA can become worse than full fine-tuning as batch size grows, even when increasing rank does not fix the gap; NoRA reveals a mechanism by which the same hidden preconditioner \(P=A^\top A\) filters not only the mean gradient but also minibatch noise.**

Potential stronger headline:

> **The critical batch size of LoRA is governed by a state-dependent stochastic geometry induced by its bilinear factorization, not by rank alone.**

This topic starts from an **already reported anomaly**, so it has lower phenomenon risk than a speculative new benchmark effect. The open problem is mechanistic: what exactly about \(BA\) changes the value of gradient noise?

---

## 1. Existing phenomenon

Thinking Machines Lab's *LoRA Without Regret* reports that, in some SFT settings:

- LoRA pays a larger loss penalty than FullFT as batch size grows;
- the gap grows at large batch sizes;
- increasing LoRA rank does not remove the gap;
- the authors conjecture that it is a property of the product-of-matrices parameterization \(BA\), which has different optimization dynamics from directly optimizing \(W\).

The 2026 paper *Beware of the Batch Size* independently elevates batch size to a first-order LoRA hyperparameter and shows that tuning it can change conclusions about LoRA variants.

These works establish the empirical problem but do not yet provide a full dynamic theory of why a bilinear low-rank parameterization changes the useful-batch regime.

This is exactly where NoRA's hidden-preconditioner view becomes useful.

---

## 2. NoRA-derived mechanism

Let a minibatch full-weight gradient be

\[
\widehat G=G+\xi,
\]

where \(G\) is the population/large-batch gradient and \(\xi\) is minibatch noise.

At \(B\approx0\), NoRA's early-time derivation gives, up to the chosen scaling convention,

\[
\Delta W_{\rm eff}\approx -\eta\widehat G P,
\qquad P=A^\top A.
\]

Therefore the same \(P\) acts on:

- **drift:** \(GP\);
- **diffusion/noise:** \(\xi P\).

If \(\Sigma_G=\operatorname{Cov}(\operatorname{vec}\xi)\), then the early effective update-noise covariance is

\[
\Sigma_{\Delta W}\propto
(P^\top\otimes I)\,\Sigma_G\,(P\otimes I).
\]

So changing \(A\) changes not just the mean step size. It reshapes the stochastic process seen by the merged weight.

### Later-time dynamics are even more distinctive

For separate SGD rates \(\eta_A,\eta_B\), with scaling \(s\),

\[
\delta B=-\eta_B s\widehat G A^\top,
\qquad
\delta A=-\eta_A s B^\top\widehat G.
\]

The product update contains approximately

\[
\Delta(BA)
\approx
-\eta_B s\widehat G A^\top A
-\eta_A s BB^\top\widehat G
+\delta B\,\delta A.
\]

Thus, after \(B\) grows, gradient noise is filtered from **both sides** by state-dependent low-rank factors, plus a correlated second-order term.

The resulting stochastic geometry is qualitatively different from additive SGD noise in direct FullFT.

This gives a concrete interpretation of the large-batch anomaly:

> increasing batch size removes stochasticity whose direction and scale have already been transformed by the low-rank parameterization; the useful noise regime may therefore differ from FullFT even at high rank.

This is a hypothesis, not a conclusion. The experiments below are designed to distinguish it from deterministic conditioning and ordinary hyperparameter mismatch.

---

## 3. Nearest work and what remains open

### Thinking Machines — *LoRA Without Regret*

Owns the empirical large-batch anomaly and explicitly points to the product parameterization. We must explain and predict the anomaly, not rediscover it.

### *Beware of the Batch Size* (2026)

Owns the message that batch-size tuning can reverse LoRA evaluation conclusions and studies empirical dependencies on rank, dataset size and model size. We need a causal stochastic/dynamic mechanism and a predictive quantity, not another sweep.

### Flora — ICML 2024

Interprets LoRA as a random gradient projection/compressor and resamples projections. This occupies generic “LoRA is a random projection” territory.

### *Low-Rank Adaptation Secretly Imitates Differentially Private SGD*

Shows that low-rank random projection can induce a noise-like effect and quantifies its variance versus rank. This is an important collision boundary. Our target is different:

- **minibatch stochasticity** rather than privacy as the primary object;
- the **critical batch / useful noise regime**;
- dynamic trainable \(A,B\), not only a frozen/random-projection interpretation;
- NoRA's explicit \(P\)-controlled drift-versus-diffusion geometry;
- causal noise add/remove experiments.

### *On the Convergence of Stochastic Low-Rank Adaptation* (2026)

Studies stochastic oracle complexity and variance-reduced algorithms. Do not claim to be the first stochastic LoRA theory. Our object is the **shape and usefulness of minibatch noise under bilinear adaptation**, especially the observed batch-size phase change.

### General gradient-noise-scale literature

Gradient noise scale has long been used to predict useful batch sizes in ordinary SGD. The opportunity is to ask whether the relevant scale must be defined in the **effective LoRA geometry**, analogously to recent work showing that non-Euclidean optimizers require optimizer-aware noise scales.

---

## 4. Main research questions

1. Can the LoRA-vs-FullFT large-batch gap be reproduced after per-condition LR tuning?
2. Is the gap caused by missing **stochasticity**, or by deterministic conditioning of the bilinear parameterization?
3. Does NoRA shift the critical batch size by reshaping \(P\)?
4. Is the right predictor a LoRA-aware gradient noise scale measured after projection by \(P\)?
5. Why can the batch pathology be relatively insensitive to rank even though random-projection noise often scales with rank?
6. At what time does the simple early \(\xi P\) picture fail as \(B\), \(A\), and optimizer states evolve?
7. Can a cheap early diagnostic predict which logical batch size will be safe for a new LoRA run?

---

## 5. Pre-registered causal decomposition

The project separates three possibilities rather than assuming “noise is good.”

### H1 — Stochasticity mechanism

Small-batch LoRA benefits from a particular projected noise covariance. Large batch removes it, and calibrated noise injection into a large-batch run restores a substantial fraction of the gap.

### H2 — Deterministic drift / curvature mechanism

Large batch exposes an unfavorable deterministic optimization geometry of \(BA\). Matching effective drift, step scale, or curvature-aware LR rescues performance; adding noise does little.

### H3 — Dynamic-factor mechanism

Neither early drift nor early diffusion alone explains the gap. It emerges when \(A\) starts moving, \(BB^\top\) becomes important, or optimizer moments couple to the factorization.

All three are scientifically meaningful if one can be causally identified and used to predict the batch regime.

---

## 6. Decisive experiments

### E1 — reproduce the pathology fairly

For one 0.5B–1.5B model and one SFT dataset, sweep logical batch:

\[
B\in\{8,32,128,512\}
\]

or the largest feasible range using gradient accumulation.

Compare:

- FullFT if memory permits on the small model;
- vanilla LoRA;
- NoRA-init / NoRA.

For each main condition, perform a modest LR sweep. The goal is to verify a **best-tuned large-batch gap**, not a fixed-LR artifact.

### E2 — rank independence check

At only two or three batch sizes, vary rank over a broad but cheap range, e.g.

\[
r\in\{8,32,128\}.
\]

Confirm whether the large-batch penalty is indeed weakly rank-dependent in our stack. Do not spend a full grid until this is true.

### E3 — measure drift and diffusion directly

At selected checkpoints, split a logical batch into many microbatches and estimate:

- mean gradient \(G\);
- empirical minibatch covariance/projections;
- \(\|GP\|\);
- \(\mathbb E\|\xi P\|^2\);
- anisotropy / leading eigenvalues of projected noise;
- corresponding FullFT quantities.

The first desired diagnostic is a **projected noise scale** that can be estimated without storing a full parameter covariance.

Possible scalar approximations:

\[
\mathcal S_P
=
\frac{\mathbb E\|\xi P\|_F^2}
{\|GP\|_F^2+\epsilon}
\]

and layerwise variants. More principled spectral/non-Euclidean forms should be added only if needed.

### E4 — large batch + recovered noise

This is the strongest causal experiment.

Estimate the distribution/covariance of microbatch gradient differences from the small-batch regime, then add calibrated zero-mean perturbations to a large-batch LoRA run.

Compare:

1. large batch baseline;
2. large batch + **isotropic** noise matched only in norm;
3. large batch + **projected/structured** noise approximating the small-batch LoRA covariance;
4. small batch reference.

If only structured noise rescues training, the geometry claim becomes much stronger than generic “SGD noise regularizes.”

### E5 — remove/reduce noise at small batch

Complement the rescue experiment by suppressing stochasticity while keeping the number of examples/updates carefully controlled, e.g. deterministic batch reuse, averaging microbatch gradients before the adapter projection, or synthetic noise cancellation approximations.

The exact implementation must avoid changing the optimization budget in a confounded way.

### E6 — frozen-A control

Use LoRA-FA / frozen \(A\) for a subset of experiments.

Then \(P\) remains fixed, providing a clean test of the early theory over a longer horizon. Compare with jointly trained \(A,B\):

- if the batch effect already appears with frozen \(A\), fixed projected noise is sufficient;
- if it appears only when \(A\) moves, dynamic factor geometry is necessary.

### E7 — NoRA as a causal knob

NoRA changes \(P\)'s coordinate gain structure. Test whether this changes:

- projected drift;
- projected noise scale;
- critical batch size;
- sensitivity to injected structured noise.

This is the direct connection back to the mother paper. Do not claim NoRA is superior unless LR and update magnitude are matched.

---

## 7. Desired theory / predictive law

A top-conference version should end with more than “noise injection helps.”

### Goal A — LoRA-aware critical batch predictor

Find an early measurable quantity \(\mathcal S_{\rm LoRA}\) such that the useful batch threshold approximately follows

\[
B_{\rm crit}\approx F(\mathcal S_{\rm LoRA},\eta_A/\eta_B,\text{dataset scale},\text{curvature}).
\]

The final expression need not be universal, but it should predict held-out ranks, datasets or model sizes better than ordinary unprojected GNS.

### Goal B — explain weak rank dependence

The Thinking Machines result is particularly interesting because the gap is reportedly not repaired by rank. A satisfying theory should explain why simple random-projection variance \(\propto1/r\) is not the whole story.

Possible mechanisms to test:

- \(1/r\) LoRA scaling cancels first-order rank effects;
- learned factor dynamics create a rank-insensitive timescale;
- batch interacts with curvature in the effective low-rank manifold rather than only with projection variance.

### Goal C — early-to-late transition

Identify a measurable transition from

\[
\widehat G P_0
\]

dominated dynamics to two-sided/dynamic factor filtering. This can connect the topic to NoRA-init without becoming the crowded “early critical period” story.

---

## 8. Compute ladder

### Gate 0 — synthetic bilinear model

**Cost:** CPU / 1 GPU.

Show analytically and numerically that minibatch covariance is transformed by \(P\), and validate noise-rescue methodology in a controlled problem.

### Gate 1 — reproduce large-batch gap

**Cost:** 1 GPU, 0.5B–1.5B model.

A sparse grid of batch × {FullFT, LoRA, NoRA-init} with LR tuning. If the phenomenon does not reproduce, try one second dataset before killing.

### Gate 2 — causal noise panel

**Cost:** 2–4 GPUs, same or ~1.5B model.

Microbatch covariance estimation + structured noise rescue + frozen-A control. This is the make-or-break scientific stage.

### Gate 3 — breadth

**Cost:** ≤4 GPUs, up to ~3B on 2–3 tasks; optional one 7B/8B validation.

Do not scale until a predictor is already formulated.

No RL is required. FullFT is needed only on the smaller model where it fits.

---

## 9. Survival tree

### Branch A — structured noise rescues large batch

Preferred story: LoRA changes the geometry of useful SGD noise. Derive/test a projected GNS and critical-batch predictor.

### Branch B — isotropic noise rescues equally well

The geometry claim weakens. Continue only if LoRA has a quantitatively different optimal noise scale from FullFT and the bilinear parameterization predicts it. Otherwise this collapses to generic SGD regularization and should be killed.

### Branch C — matched drift/LR rescues; noise does not

Shift to deterministic bilinear conditioning. The NoRA connection becomes \(P\)-dependent curvature/step stability. Continue only if this yields a predictive batch/LR law beyond *Learning Rate Matters* and *Beware of the Batch Size*.

### Branch D — frozen-A has no pathology, joint A/B does

This is strong evidence for dynamic factor interaction. Analyze the transition driven by \(BB^\top\), \(A\)-movement and the cross term. This is distinct from fixed random-projection work.

### Branch E — pathology disappears after fair LR tuning

Potentially important reconciliation, but only proceed if we can explain why previous studies observed it and derive a reliable joint LR/batch rule. A one-line “tune LR” result is already occupied and is not enough.

### Branch F — no reproducible gap on two settings

Kill the topic. Do not manufacture a new batch benchmark.

---

## 10. Kill criteria

Kill A3 if:

1. the tuned large-batch LoRA-vs-FullFT gap cannot be reproduced in two feasible settings;
2. ordinary LR/batch tuning fully explains it with no new predictive mechanism;
3. noise intervention effects are indistinguishable from generic FullFT SGD behavior;
4. the only surviving explanation duplicates Flora, DP-SGD-like random-projection noise, or generic stochastic-LoRA convergence theory;
5. a convincing effect requires expensive RL or large pretraining.

---

## 11. Closest-work novelty boundary

| Work | Already owns | A3 must add |
|---|---|---|
| Normalized NoRA | \(P=A^\top A\) early preconditioner / diagonal gain | show that \(P\) controls both drift and minibatch diffusion and connect it causally to batch regime |
| Thinking Machines *LoRA Without Regret* | large-batch LoRA anomaly, rank-independent observation | mechanism, causal intervention, predictor |
| *Beware of the Batch Size* | batch tuning changes LoRA conclusions | stochastic/dynamic explanation rather than another hyperparameter audit |
| Flora, ICML 2024 | LoRA ≈ random gradient compressor/projection | dynamic minibatch-noise geometry and critical batch |
| *LoRA Secretly Imitates DP-SGD* | random low-rank projection induces noise-like gradients | real minibatch stochasticity, trainable-factor dynamics, structured noise rescue |
| *Stochastic Low-Rank Adaptation* (2026) | oracle complexity / variance-reduced stochastic algorithms | useful-noise phase, bilinear diffusion geometry, critical batch |
| general GNS literature | predicts useful batch for ordinary optimization | LoRA-aware projected/non-Euclidean noise scale |

---

## 12. First 7-day plan

1. Reproduce one small-model LoRA-vs-FullFT batch curve with per-condition LR sweep.
2. Add NoRA-init to the same sparse grid.
3. Log per-microbatch gradients for 100–200 early updates.
4. Compute cheap \(\|GP\|\) and \(\mathbb E\|\xi P\|^2\) statistics.
5. Freeze \(A\) in one control to see whether the pathology persists.
6. Prototype large-batch + Gaussian norm-matched noise.
7. If there is any rescue, replace isotropic noise with an empirical low-rank approximation of the projected microbatch covariance.

---

## 13. Possible paper titles

- **When Low Rank Changes the Noise: Stochastic Geometry of LoRA**
- **Why Does LoRA Fear Large Batches?**
- **The Critical Batch Size of Bilinear Adaptation**
- **Drift, Diffusion, and the Hidden Preconditioner of LoRA**

## References

- Normalized NoRA: https://arxiv.org/abs/2608.31036
- Thinking Machines Lab, LoRA Without Regret: https://thinkingmachines.ai/blog/lora/
- Beware of the Batch Size: https://arxiv.org/abs/2602.09492
- Learning Rate Matters: https://arxiv.org/abs/2602.04998
- Flora: https://proceedings.mlr.press/v235/hao24a.html
- Low-Rank Adaptation Secretly Imitates DP-SGD: https://openreview.net/pdf?id=vsLkyuo6M5
- On the Convergence of Stochastic Low-Rank Adaptation: https://arxiv.org/abs/2607.21975
