# 02 — The Coordinates Are Arbitrary: Representation-Gauge Dependence in Low-Rank Adaptation

**Status:** REGISTERED — Priority A2  
**Planning score:** 91/100  
**Mother paper:** Normalized Low-Rank Adaptation (Kang et al., 2026)  
**Primary target:** NeurIPS / ICLR / ICML  
**Central compute requirement:** mostly exact transformations + 0.5B–3B SFT; ≤4 GPUs

## One-sentence paper hook

> **Can two Transformers that compute exactly the same function fine-tune differently only because we renamed their hidden coordinates? Normalized NoRA makes this question unavoidable because its key operation normalizes coordinate columns of \(A\).**

Potential stronger headline if the pilot succeeds:

> **Functionally identical Transformers, related by an exact attention gauge transformation, follow different adaptation trajectories under coordinate-dependent PEFT regularization.**

The novelty is **not** discovering Transformer symmetries. Recent work already characterizes them. The novelty target is the **interaction between exact backbone representation gauges and PEFT training dynamics**, with Normalized NoRA supplying a concrete, theoretically non-equivariant operation.

---

## 1. Why NoRA creates this question

NoRA interprets

\[
P=A^\top A
\]

as an input-side preconditioner. Its diagonal is interpreted as a set of coordinate-wise own-update gains, and it enforces equal column norms of \(A\).

But hidden coordinates inside a Transformer are often not intrinsic. Function-preserving changes of basis can reparameterize an internal representation without changing the network function.

For a simple linear module with column-vector convention,

\[
y=Wx,
\]

consider an orthogonal input-basis change

\[
x'=R^\top x,\qquad R^\top R=I,
\]

and transform the weight as

\[
W'=WR.
\]

Then

\[
W'x'=WRR^\top x=Wx.
\]

For the same adapter function,

\[
\Delta W=BA
\]

must transform as

\[
A'=AR,\qquad B'=B,
\]

so its hidden preconditioner transforms covariantly:

\[
P'=A'^\top A'=R^\top P R.
\]

### The key NoRA non-commutation

Let \(\mathcal N(A)\) normalize the columns of \(A\). In general,

\[
\boxed{\mathcal N(AR)\neq \mathcal N(A)R.}
\]

Therefore the NoRA operation does **not** commute with an arbitrary orthogonal change of input basis.

This produces a very clean conceptual challenge:

> NoRA calls a flat diagonal of \(P\) “balanced,” but the diagonal of a matrix is basis-dependent.

Indeed, orthogonal conjugation changes \(\operatorname{diag}(P)\) while preserving the eigenvalues of \(P\). By Schur-Horn, a constant diagonal equal to \(\operatorname{tr}(P)/k\) is compatible with any PSD spectrum after a suitable orthogonal basis change. Thus “equal coordinate gains” is not by itself an intrinsic property of the low-rank operator.

The research question is whether this coordinate dependence is merely a harmless mathematical description, or whether it creates measurable optimization dependence in real PEFT.

---

## 2. Exact Transformer symmetries already exist — and that is an asset

This project should **reuse**, not rediscover, known function-preserving transformations.

### QuaRot / SpinQuant

QuaRot (NeurIPS 2024) and SpinQuant (ICLR 2025) exploit Transformer rotations that preserve the full-precision function but alter quantization behavior. SpinQuant reports that different random function-preserving rotations can cause very different quantized downstream performance.

These papers establish an important experimental pattern:

> exact functional equivalence can be used as a causal intervention on parameter/representation coordinates.

They study quantization, not fine-tuning dynamics.

### Complete Transformer gauge characterization

A 2025 NeurIPS symmetry workshop paper characterizes the gauge group of canonical Transformers. For multi-head attention it identifies per-head query/key and value/output gauge freedoms; with RoPE, the query/key group is restricted but the value/output \(GL(d_v)\) gauge remains.

The value/output gauge is especially useful here because it gives a local exact transformation such as, up to matrix-convention transpose choices,

\[
W_V\rightarrow W_V C,\qquad W_O\rightarrow C^{-1}W_O,
\]

that preserves the attention function. We can restrict \(C\) to the **orthogonal subgroup** for the cleanest optimization-equivariance test.

### Why this is not LoRA-RITE

LoRA-RITE (ICLR 2025) studies the **internal adapter-factor gauge**:

\[
BA=(BQ)(Q^{-1}A).
\]

It asks why an optimizer should care how the *same adapter matrix* is factorized.

This project studies a different symmetry:

\[
\text{backbone hidden representation}\rightarrow\text{equivalent hidden representation},
\]

with the pretrained Transformer function itself unchanged.

These are distinct gauge groups and distinct failure modes.

---

## 3. A clean theoretical contrast: vanilla LoRA versus NoRA under SGD

This is the strongest reason the project is not just an empirical curiosity.

Under an orthogonal input basis transform \(R\), couple vanilla LoRA initializations by

\[
A'_0=A_0R,\qquad B'_0=B_0.
\]

If the full gradient transforms as \(G'=GR\), then for plain SGD:

\[
\nabla_B L'=G'A'^\top=GRR^\top A^\top=GA^\top,
\]

and

\[
\nabla_A L'=B^\top G'=B^\top GR=(\nabla_A L)R.
\]

Thus vanilla LoRA has a natural **orthogonal equivariance** under this coupled transformation: the transformed trajectory can remain the representation transform of the original trajectory.

Normalized NoRA breaks this coupling because

\[
\mathcal N(A_0R)\neq \mathcal N(A_0)R
\]

generically.

So there exists a setting where:

- the backbone functions are exactly identical;
- vanilla LoRA + SGD has a clear equivariant coupling;
- NoRA's column normalization generically destroys it.

That is a much sharper statement than “rotation changes performance.”

### AdamW caveat — required control

Coordinate-wise adaptive optimizers themselves are generally not invariant to arbitrary rotations. Therefore realistic AdamW experiments must include:

- FullFT + AdamW;
- vanilla LoRA + AdamW;
- NoRA + AdamW;
- and, for theory isolation, SGD versions.

The target claim is **additional PEFT / NoRA gauge dependence beyond optimizer baseline**, not the false claim that only NoRA can ever be representation-dependent.

---

## 4. Main research questions

1. Under exact function-preserving Transformer gauges, how much do PEFT training trajectories vary?
2. Does Normalized NoRA exhibit stronger basis dependence than vanilla LoRA in the SGD-controlled setting predicted by theory?
3. Can an arbitrary gauge alter the ranking between LoRA initializations or their optimal learning rates?
4. Is final accuracy stable even when early optimization, forgetting, or hyperparameter sensitivity is gauge-dependent?
5. Can the gauge be chosen adversarially to expose a large effect while keeping the pretrained function identical to machine precision?
6. Which PEFT quantities are intrinsic — spectrum/subspace/function-space effects — and which are artifacts of a selected coordinate chart?
7. Is there a simple gauge-covariant replacement or evaluation protocol that removes the pathology without becoming a new method zoo?

---

## 5. Decisive experiments

### E0 — exactness check

Before any fine-tuning claim, transform selected attention heads/layers using an exact value/output orthogonal gauge.

Verify on held-out inputs:

\[
\max_x\|f_\theta(x)-f_{\theta_R}(x)\|
\]

is at floating-point noise level and logits/loss agree within a pre-registered tolerance.

If exact equivalence is not achieved, no gauge experiment is valid.

### E1 — coupled vanilla-LoRA SGD equivariance

Take a base adapter initialization \((A_0,B_0)\). On the gauge-transformed backbone, use the mathematically transformed initialization rather than an independent random seed.

Train both with plain SGD on the same ordered minibatches.

Measure trajectory correspondence after mapping updates back to the original gauge.

This is the positive control: if the implementation is correct, vanilla LoRA should approximately obey the derived orthogonal coupling.

### E2 — NoRA non-equivariance

Repeat with:

- NoRA-init;
- continuous NoRA;
- vanilla LoRA.

For NoRA-init compare:

1. **algorithmic application in each coordinate system:** normalize \(A_0\) and normalize \(A_0R\) separately;
2. **equivariance oracle:** normalize once and then transform the already-normalized matrix.

The difference between these isolates the normalization non-commutation directly.

### E3 — random gauge distribution

Use 8–16 exact random orthogonal gauges on a small model/task. Report distributions, not cherry-picked examples:

- early loss slope;
- time-to-threshold;
- final loss/accuracy;
- optimal LR from a small sweep;
- forgetting/retention on one cheap general benchmark;
- method-ranking stability.

Primary statistic:

\[
\operatorname{Var}_R[\text{metric}].
\]

Also report FullFT variation as the optimizer-dependent floor.

### E4 — adversarial gauge

If random rotations produce small effects, do **not** immediately kill the project. Optimize/search for an exact gauge that maximizes a pre-training-free structural distortion such as:

- discrepancy between \(\mathcal N(AR)\) and \(\mathcal N(A)R\);
- variance of the induced NoRA coordinate gains;
- predicted early-step difference.

Then test whether the theoretically large distortion transfers to training dynamics.

This branch is scientifically legitimate because the gauge is selected without looking at downstream test accuracy and the pretrained function remains exact.

### E5 — realistic AdamW panel

Only after SGD isolates the mechanism, repeat a small panel with AdamW. Compare against FullFT and vanilla LoRA to separate:

- optimizer coordinate dependence;
- low-rank factor dependence;
- NoRA-specific coordinate normalization dependence.

---

## 6. Strongest possible results

### Result class A — final outcomes vary substantially

Headline:

> Functionally identical Transformers can produce materially different fine-tuned models under coordinate-dependent PEFT.

This is immediately broad. The paper should then propose gauge robustness as a new evaluation axis and derive a minimal covariant remedy.

### Result class B — final score is stable, but trajectory / optimal LR / forgetting varies

This is still useful if the effect is systematic:

> endpoint benchmarks conceal a gauge-dependent optimization path.

The paper then connects to hyperparameter transfer and stability rather than claiming benchmark catastrophe.

### Result class C — random gauges are benign but adversarial gauges are not

This gives a robustness result:

> NoRA is typically stable but has no representation-intrinsic guarantee; exact gauges can expose predictable failure modes.

A theorem plus a constructive counterexample can still be strong, but the real-model effect must be non-negligible.

### Result class D — NoRA behaves no worse than baselines under all exact gauges

Then the theory is mathematically true but practically weak. Unless we can prove a nontrivial average-case robustness theorem or identify a stronger symmetry class, **kill** rather than force a method.

---

## 7. Possible remedy space — intentionally secondary

Do not start by inventing “GaugeNoRA.” First establish the phenomenon.

If a remedy is needed, desirable properties are:

\[
\mathcal A(WR, D)=\mathcal A(W,D)R
\]

for the appropriate exact gauge/covariant action.

Possibilities to investigate only after the phenomenon is real:

- global/Frobenius scale normalization, which commutes with orthogonal right rotations but does not equalize coordinates;
- an intrinsic metric that transforms covariantly with the representation;
- normalize a function-space or gauge-invariant quantity rather than raw columns;
- canonicalize the gauge before adaptation.

Avoid simply using activation covariance/Fisher whitening as the novelty; EVA, LoRA-DA, TLoRA and natural-gradient literature make that space crowded.

---

## 8. Compute ladder

### Gate 0 — linear/module theorem + unit tests

**Cost:** CPU.

Numerically verify vanilla-LoRA SGD equivariance and NoRA non-commutation on a single matrix.

### Gate 1 — exact one-layer / few-layer gauge pilot

**Cost:** 1 GPU; 0.5B–1.5B model.

5 gauges × {LoRA, NoRA-init} × short SGD runs, one task. This is enough to determine whether the real implementation reflects the theorem.

### Gate 2 — full causal panel

**Cost:** 2–4 GPUs; 1.5B–3B models.

8–16 gauges, selected methods, SGD + AdamW, one LR local sweep. Parallelize gauges, not larger models.

### Gate 3 — external validity

**Cost:** ≤4 GPUs; one 7B/8B model and 3–5 gauges.

Only if Gate 2 produces a clear effect.

No pretraining. No RL. No API judge.

---

## 9. Survival tree

### Branch A — NoRA-specific SGD gauge sensitivity is large

Proceed immediately. This is the preferred paper path.

### Branch B — all AdamW methods are gauge-sensitive, NoRA not uniquely so

Shift from “NoRA flaw” to **adaptation under backbone gauges**. The novel object becomes excess gauge sensitivity of PEFT versus FullFT and how optimizer/factor/normalizer pieces contribute. Keep only if the decomposition is clean.

### Branch C — only hyperparameter optimum changes

Connect to *Learning Rate Matters*: same function, different representation, different preferred LR. This can be strong if we derive a gauge-dependent curvature/preconditioning predictor.

### Branch D — only adversarial gauges matter

Treat as a robustness/counterexample paper only if the gauge can be selected from model-internal quantities and causes a meaningful effect without test-set search.

### Branch E — effects are numerically tiny everywhere

Kill. Mathematical non-equivariance without practical consequence is not enough.

---

## 10. Closest-work novelty boundary

| Work | Already established | This project must establish |
|---|---|---|
| Normalized NoRA (2026) | column normalization / equal coordinate gains | whether that coordinate-level principle is representation-intrinsic and what happens under exact basis changes |
| LoRA-RITE, ICLR 2025 | invariance to **adapter factor** rescaling/rotation | invariance/equivariance to **backbone representation** gauges |
| QuaRot, NeurIPS 2024 | exact computational rotations for quantization | use exact functional equivalence as a fine-tuning causal intervention |
| SpinQuant, ICLR 2025 | function-preserving rotations change quantized behavior | adaptation dynamics, not quantization |
| Transformer gauge characterization, 2025 workshop | identifies/maximizes Transformer gauge groups | empirical/theoretical behavior of PEFT under those gauges |
| Reparameterization invariance in approximate Bayesian inference, NeurIPS 2024 | identical functions should not receive parameterization-dependent approximate posteriors | analogous principle for adaptation dynamics, with a distinct PEFT mechanism |
| Riemannian / manifold LoRA work | factor-space geometry or low-rank manifold optimization | exact backbone-gauge experiment and NoRA-specific non-commutation |

### Novelty statement we are allowed to make only after another audit

> We found no direct prior work in the current search that systematically evaluates LoRA/NoRA training equivariance under exact **backbone representation gauges**.

Do **not** claim “first work on Transformer gauge symmetry” or “first work on invariant LoRA.” Both would be false.

---

## 11. First 7-day plan

1. Implement exact orthogonal value/output attention gauge for one open Transformer family.
2. Add a numerical function-equivalence test suite.
3. Implement coupled adapter transformation between gauges.
4. Verify vanilla LoRA + SGD trajectory equivariance on a tiny model.
5. Verify \(\mathcal N(AR)\neq\mathcal N(A)R\) and NoRA-init trajectory divergence.
6. Run 5 random gauges on one 0.5B–1.5B SFT task.
7. Decide whether effect size justifies Gate 2 before touching AdamW or 7B models.

---

## 12. Possible paper titles

- **The Coordinates Are Arbitrary: Representation-Equivariant Low-Rank Adaptation**
- **Same Model, Different Fine-Tuning: Representation Gauge Dependence in PEFT**
- **When Equivalent Transformers Adapt Differently**
- **Hidden Coordinates, Visible Consequences: Gauge Dependence in Low-Rank Adaptation**

## References

- Normalized NoRA: https://arxiv.org/abs/2608.31036
- LoRA-RITE: https://proceedings.iclr.cc/paper_files/paper/2025/hash/bcbc0f660d2dde42f9d1d0ecb14a6f9a-Abstract-Conference.html
- QuaRot: https://proceedings.neurips.cc/paper_files/paper/2024/hash/b5b939436789f76f08b9d0da5e81af7c-Abstract-Conference.html
- SpinQuant: https://proceedings.iclr.cc/paper_files/paper/2025/hash/e5b1c0d4866f72393c522c8a00eed4eb-Abstract-Conference.html
- Complete Transformer gauge characterization: https://neurips.cc/virtual/2025/136893
- Reparameterization invariance in approximate Bayesian inference: https://proceedings.neurips.cc/paper_files/paper/2024/hash/0f934dd2030f5740cde0aa2697a105a9-Abstract-Conference.html
