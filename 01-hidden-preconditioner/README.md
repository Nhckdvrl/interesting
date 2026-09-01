# 01 — Dissecting the Hidden Preconditioner of Low-Rank Adaptation

**Status:** REGISTERED — Priority A1  
**Planning score:** 93/100  
**Mother paper:** Normalized Low-Rank Adaptation (Kang et al., 2026)  
**Primary target:** ICML / NeurIPS; ICLR also plausible  
**Central compute requirement:** ≤4 GPUs; no RL or pretraining required for the core paper

## One-sentence paper hook

> **Recent LoRA methods disagree about whether initialization works because of update magnitude, coordinate balance, subspace geometry, or task alignment; Normalized NoRA exposes a single hidden preconditioner \(P=A^\top A\) in which these explanations can be causally separated.**

A stronger eventual headline, if supported:

> **After matching learning rate and update magnitude, LoRA initialization methods fall into a small number of hidden-preconditioner equivalence classes whose statistics predict optimization better than method identity.**

This project is not “NoRA + another normalization.” It asks what NoRA actually discovered.

---

## 1. Mother proposition

With

\[
\Delta W=\alpha BA,\qquad B_0=0,
\]

the earliest update to the merged weight is

\[
\Delta W_1=-\eta G P,\qquad P=\alpha^2A_0^\top A_0.
\]

Normalized NoRA decomposes this object informally into:

- diagonal \(P_{jj}\): coordinate-wise own-update gain;
- off-diagonal \(P_{ij}\): cross-coordinate interference / crosstalk.

NoRA enforces equal column norms of \(A\), so the diagonal is controlled. Its BIMI experiment changes crosstalk substantially while preserving the same diagonal structure and obtains similar performance, motivating the claim that diagonal gain is decisive. NoRA-init further shows that a one-time initialization intervention captures most of the benefit.

But several strong neighboring papers provide different explanations for why LoRA initialization matters:

- **LoRAM / Primacy of Magnitude (NeurIPS 2025):** update magnitude is the fundamental driver; spectral initialization mostly amplifies magnitude.
- **Learning Rate Matters (2026):** many apparent LoRA-variant gains vanish after method-specific LR tuning; Hessian curvature shifts the optimal LR.
- **LoRA-One (ICML 2025 Oral):** alignment to a one-step full-gradient singular subspace is the important object.
- **EVA (NeurIPS 2025), LoRA-DA (ICML 2026), TLoRA (ACL 2026):** data / activation / task-relevant subspace selection matters.
- **Towards Understanding the Dynamics of LoRA (ICML 2026):** frame/subspace geometry can maximize preserved gradient information; ETF structure improves over Gaussian initialization.

These are not merely competing methods. They are **competing causal explanations of the same early low-rank optimization state**.

The opportunity is to turn NoRA's \(P\) into the common scientific language in which those explanations can be independently manipulated.

---

## 2. Research question

### Main question

**Which properties of the hidden preconditioner \(P=A^\top A\) causally control LoRA's early optimization and downstream performance once learning rate, batch size, and update magnitude are matched?**

### Subquestions

1. Is NoRA's equalized diagonal important beyond its effect on global update magnitude?
2. How much does off-diagonal crosstalk matter after trace and diagonal statistics are matched?
3. Does task-gradient alignment dominate coordinate balance when the task is sufficiently anisotropic?
4. Are apparently different initialization methods equivalent whenever they induce the same relevant \(P\)-statistics?
5. For how many steps does \(P_0\) predict the trajectory before learned changes in \(A\), \(B\), Adam moments, or curvature dominate?
6. Can we predict the best initialization family from a cheap statistic measured on one or a few batches?

---

## 3. Candidate scientific objects

Do **not** decide in advance that every statistic below matters. The pilot must eliminate irrelevant ones.

For \(P\succeq0\):

### 3.1 Global gain / magnitude

\[
m(P)=\frac{1}{k}\operatorname{tr}(P).
\]

This controls average initial gain and is the closest bridge to LoRAM and LR scaling.

### 3.2 Diagonal imbalance

\[
d(P)=\frac{\operatorname{Var}(\operatorname{diag}P)}{m(P)^2+\epsilon}.
\]

This isolates NoRA's principal claimed contribution: equalizing coordinate-wise own-update gains.

### 3.3 Crosstalk

Candidate normalized statistic:

\[
c(P)=\frac{\|P-\operatorname{Diag}(P)\|_F}{\|P\|_F+\epsilon}.
\]

Alternative metrics: coherence, maximum off-diagonal magnitude, spectral mass outside a matched diagonal surrogate.

### 3.4 Spectrum / effective rank

\[
r_{\rm eff}(P)=\frac{(\operatorname{tr}P)^2}{\operatorname{tr}(P^2)}.
\]

This connects low-rank geometry to frame-based work without turning the paper into another ETF initializer.

### 3.5 Task-gradient alignment

For a batch-level input-side gradient covariance, e.g.

\[
C_g=G^\top G,
\]

measure quantities such as

\[
a(P,C_g)=\frac{\operatorname{tr}(PC_g)}{\operatorname{tr}(P)\operatorname{tr}(C_g)+\epsilon},
\]

principal angles, or captured gradient energy

\[
\rho(P,G)=\frac{\|GP\|_F^2}{\|G\|_F^2}.
\]

This provides the clean connection to LoRA-One / data-aware initialization.

### 3.6 Curvature-relative geometry

Only promote this if the simple statistics fail. Locally,

\[
L(W-\eta GP)\approx L(W)-\eta\langle G,GP\rangle + \frac{\eta^2}{2}\langle GP,H[GP]\rangle.
\]

Therefore a useful \(P\) is not defined by its diagonal alone; it may depend on how its preferred directions interact with gradient and curvature. A Kronecker / low-rank Hessian approximation may give a tractable diagnostic, but this is **not** the starting point.

---

## 4. Core theoretical agenda

### T1 — First-step \(P\)-equivalence

Prove the exact/first-order statement:

> For the same optimizer setting at \(B_0=0\), any two initial factorizations whose \(A_0^\top A_0\) are identical induce the same first merged-weight update under SGD.

This turns “method identity” into an unnecessary label at step 1.

### T2 — Early-time deviation from \(P_0\)

Characterize when two \(P_0\)-equivalent initializations cease to be equivalent because \(A\) begins to move. Since \(\nabla_A L\) vanishes at \(B=0\), there is a natural two-timescale expansion. The goal is not a decorative Taylor series: derive a measurable timescale that predicts when initialization ceases to dominate.

### T3 — Local objective decomposition

For a local quadratic model, derive which features of \(P\) control:

- first-order descent;
- update norm;
- second-order stability;
- gradient-energy preservation.

This should explicitly show when a trace/magnitude explanation is sufficient and when directional alignment is necessary.

### T4 — Feasible matched interventions

The hardest technical piece is constructing \(A\) matrices that vary one \(P\)-property while matching others. The paper is stronger if the feasible region can be characterized rather than treated as ad-hoc random search.

Useful tools may include unit-norm frames, eigenvalue/diagonal constraints, Schur-Horn-style constructions, and controlled rotations. Avoid making ETF optimality itself the contribution because ICML 2026 already occupies that result family.

---

## 5. Decisive causal experiments

### Experiment family A — Is NoRA just magnitude?

Compare at initialization:

1. vanilla LoRA;
2. Normalized NoRA-init;
3. **mean-scale-matched LoRA:** rescale random \(A\) so \(\mathbb E\|a_j\|^2\) or \(\operatorname{tr}P\) matches NoRA, but retain column-norm variation;
4. update-norm-matched LoRA: tune \(\alpha\), LR, or both so initial \(\|\Delta W\|_F\) matches;
5. exact-diagonal-matched constructions with different off-diagonal geometry (BIMI-like / random unit-column / other frames).

Run each under:

- inherited recipe, for reproduction only;
- method-specific LR sweep;
- small local batch-size sweep.

**Interpretation:**
- if 2 ≈ 3 after LR tuning, NoRA is primarily magnitude/step-scale;
- if 2 > 3 despite matched trace/update norm, diagonal balance has a distinct causal effect;
- if methods separate only when task alignment changes, the diagonal story is incomplete.

### Experiment family B — Does crosstalk matter?

Construct initializations with approximately matched:

- trace;
- diagonal mean/variance;
- initial merged-update norm;

but substantially different off-diagonal structure.

Measure early loss decrease, gradient alignment, and final tuned performance. NoRA's BIMI result suggests weak sensitivity; this experiment asks how far that statement generalizes.

### Experiment family C — Task alignment versus coordinate balance

Use two tasks with very different gradient covariance spectra, or synthetically rotate / concentrate the supervised signal. Compare random normalized, task-aligned, ETF-like and magnitude-matched initializations at equalized step scale.

The desired object is not “method X wins.” It is a **predictive condition** for when alignment beats isotropic/balanced initialization.

### Experiment family D — Method identity versus \(P\)-statistics

Collect a panel of existing initializers, but use them only as natural samples in \(P\)-space. Fit/predict optimization behavior from pre-registered \(P\)-statistics, then test on held-out methods or ranks.

A strong result would be that method labels add little after conditioning on a small set of causal \(P\)-features.

---

## 6. Evaluation protocol

### Models

Pilot:
- Qwen2.5/3 0.5B–1.5B class or Llama 1B class.

Main:
- one ~1.5B model;
- one ~3B model.

Optional external validity:
- one 7B/8B run for only the decisive comparison.

### Tasks

Prefer deterministic, cheap SFT evaluation:
- GSM8K / math subset;
- code or instruction-following subset with exact or standard open evaluation;
- one non-reasoning task to avoid a math-only mechanism.

No API judge is required.

### Optimization measurements

Log at high frequency early in training:
- train/validation loss;
- \(\|\Delta W\|\), \(\|G\|\), \(\|GP\|\);
- \(\operatorname{tr}P\), diagonal variance, crosstalk, spectrum;
- movement \(\|A_t-A_0\|\), \(\|B_t\|\);
- gradient/preconditioner alignment;
- largest Hessian eigenvalue or cheap approximation only for shortlisted runs.

Endpoint benchmark scores are secondary to the causal trajectory.

---

## 7. Compute ladder

### Gate 0 — synthetic linear / MLP model

**Cost:** CPU or <1 GPU-day.

Validate matching constructions and T1–T3. Kill any statistic that does not behave causally in a controlled model.

### Gate 1 — 0.5B–1.5B pilot

**Cost:** 1 GPU, short SFT runs.

Run the magnitude/diagonal matched panel with 3–5 LRs, one logical batch size. Purpose: determine whether NoRA's gap survives fair step-scale matching.

### Gate 2 — causal panel

**Cost:** 2–4 GPUs, 1.5B–3B, 2–3 tasks.

Only carry forward the 2–4 most informative intervention families. Add local batch sweep and 3 seeds for decisive cells.

### Gate 3 — final validation

**Cost:** ≤4 GPUs, one 7B/8B model, a minimal comparison.

Not required unless the smaller models establish a clean rule.

No central RL run. No pretraining replication.

---

## 8. Survival tree — pre-registered before experiments

### Branch A — magnitude explains almost everything

Then the paper is **not** “NoRA fails.” Promote only if we can show a general equivalence between normalization, initialization scale, \(\alpha\), LR and early \(P\)-trace that predicts optimal settings across methods/ranks. This would reconcile NoRA with LoRAM and the LR audit.

If it is merely “tune LR and NoRA is tied,” kill the project as insufficiently novel.

### Branch B — diagonal balance survives magnitude/LR matching

This is the cleanest confirmation of NoRA's unique mechanism. Expand to characterize when diagonal variance hurts and derive a predictive threshold or local stability result.

### Branch C — crosstalk/spectrum matters substantially

Then NoRA's BIMI conclusion is too narrow. The project becomes a causal map of diagonal versus off-diagonal geometry, with the ICML 2026 ETF paper as the strongest boundary. We must explain more than “ETF is good.”

### Branch D — task alignment dominates

Then NoRA's coordinate balance is a task-agnostic prior that is good only in certain regimes. The paper becomes a phase diagram connecting NoRA to LoRA-One/EVA/LoRA-DA: when is task-agnostic conditioning safer than data-aware alignment?

### Branch E — none of the initial \(P\)-statistics predict beyond a few steps

Measure the break time and identify whether \(A\)-movement, Adam moments, or curvature causes the loss of predictiveness. If there is no reproducible mechanism, kill this project rather than invent a dynamic NoRA method.

---

## 9. Kill criteria

Kill A1 if, after fair LR/update-scale controls:

1. all apparent differences shrink to ordinary optimizer tuning with no predictive law;
2. no \(P\)-statistic or early-time quantity predicts behavior out of sample;
3. the only surviving idea is already a special case of LoRAM, ETF-LoRA, LoRA-One, EVA/LoRA-DA, or LoFT;
4. explaining the phenomenon requires full-scale pretraining/RL to be visible.

Do **not** kill simply because NoRA itself loses its headline gain; that outcome can be scientifically strong if a more general causal rule replaces it.

---

## 10. Closest-work novelty boundary

| Work | What it already owns | What A1 must add |
|---|---|---|
| Normalized NoRA (2026) | diagonal gain normalization; BIMI; init-only effect | causal separation of the whole hidden-preconditioner object under matched confounds |
| LoRAM, NeurIPS 2025 | magnitude as fundamental driver; spectral gains via amplification | test magnitude against diagonal/spectrum/alignment in one controlled object, not another magnitude method |
| LR Matters (2026) | method gaps shrink after LR tuning; Hessian explanation | treat fair tuning as a required control and recover a mechanistic quantity that survives it |
| LoRA-One, ICML 2025 | one-gradient singular subspace alignment | characterize when task alignment matters relative to task-agnostic preconditioning |
| EVA / LoRA-DA / TLoRA | data/activation/task-aware subspaces | not propose another data-aware initializer; explain their regime within \(P\)-space |
| Dynamics/ETF, ICML 2026 | subspace/frame geometry and ETF initialization | not optimize frame coherence; causally decompose multiple properties and predict regime changes |
| LoFT, ICLR 2026 | projected optimizer moments mimic FullFT | optimizer-state effects are a later branch/control, not the claimed novelty |

---

## 11. First 7-day plan

1. Implement a tiny `PInspector` that extracts all pre-registered \(P\)-statistics layerwise and globally.
2. Reproduce vanilla LoRA vs Normalized NoRA-init on one 0.5B–1.5B SFT task.
3. Add trace/update-norm-matched random LoRA.
4. Sweep 4–5 LRs for those three conditions.
5. Add two same-diagonal/different-crosstalk constructions.
6. Plot early loss descent against trace, diagonal variance, crosstalk and captured gradient energy.
7. Decide whether A1 enters Gate 2. Do not add more methods before this decision.

---

## 12. Possible paper titles

- **Dissecting the Hidden Preconditioner of Low-Rank Adaptation**
- **What Actually Matters in LoRA's Hidden Preconditioner?**
- **Scale, Balance, Crosstalk, or Alignment? A Causal Anatomy of LoRA Initialization**
- **Beyond Normalization: Equivalence Classes in Low-Rank Adaptation**

## References

- Normalized NoRA: https://arxiv.org/abs/2608.31036
- LoRAM / Primacy of Magnitude: https://papers.nips.cc/paper_files/paper/2025/hash/0010665e949927b74faf6e3ada6d7f72-Abstract-Conference.html
- Learning Rate Matters: https://arxiv.org/abs/2602.04998
- LoRA-One: https://proceedings.mlr.press/v267/zhang25ax.html
- EVA: https://papers.nips.cc/paper_files/paper/2025/hash/41d33bd41fd44bd9dba0e092047cf213-Abstract-Conference.html
- LoRA-DA: https://arxiv.org/abs/2510.24561
- TLoRA: https://aclanthology.org/2026.acl-long.1348/
- LoFT: https://proceedings.iclr.cc/paper_files/paper/2026/hash/7428310c0f97f1c6bb2ef1be99c1ec2a-Abstract-Conference.html
