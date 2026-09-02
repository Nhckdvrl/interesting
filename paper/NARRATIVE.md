# Combined narrative: 01 + 02

**Working title.** *The Coordinates Do the Work: Identifiability Limits of
Low-Rank Adaptation Initialization*

**Status.** Draft narrative. Every number below is measured in this repository;
provenance is given per claim. Open items are marked ▢.

---

## 1. The anomaly

Normalized LoRA (NoRA, Kang et al. 2026 — the mother paper) reports that simply
normalizing the columns of the down-projection `A` improves SFT by **+5.44
average points** (37.93 → 43.37 on their GSM8K/MATH suite), fixes a pretraining
collapse, and helps RL — with no extra parameters. Its stated mechanism: LoRA
performs full fine-tuning under an implicit **input-side preconditioner**

    ΔW = −η G P,      P = α² AᵀA,

and full fine-tuning corresponds to `P = I`; normalizing `A`'s columns sets
`diag(P) = I`, "aligning the adapter's gradient norm with full finetuning,
independent of rank r."

That derivation is correct and the object it exposes is real. But `diag(P) = I`
is only the *diagonal* of `P = I`. This paper asks what the rest of `P` does,
and finds that the answer is sharply constrained by two exact symmetries.

## 2. Two exact symmetries

For a LoRA adapter `ΔW = s·B·A` with `B₀ = 0`:

**(S1) Adapter-factor gauge.** `(A, B) → (QA, BQᵀ)` for `Q ∈ O(r)` leaves `ΔW`
and `P₀` unchanged. Conversely, if `rank(A) = r`, then `A₁ᵀA₁ = A₂ᵀA₂` **iff**
`A₂ = QA₁` for some `Q ∈ O(r)` (polar uniqueness). So the `O(r)` orbit is
*exactly* the level set of `P₀`.

**(S2) Backbone representation gauge.** For a residual-stream rotation
`x' = Rx`, `W' = WRᵀ` the network computes the *same function*; coupling the
adapter by `A' = ARᵀ`, `B' = B` represents the same adapter.

**Theorem (verified).** Under plain SGD both couplings are preserved for **all
time**, not just at step 1: `∇_{B'} = ∇_B` and `∇_{A'} = (∇_A)Rᵀ`. Hence

> under SGD, the entire trajectory of the merged update depends on `A₀` only
> through `P₀`. "Which initialization method" is **unidentifiable** given `P₀`.

AdamW is equivariant under permutations and sign flips only, because `m/√v` is
elementwise. It therefore *creates* a difference between members of the same
equivalence class — a difference that carries, by construction, **zero
preconditioner information**.

**Measured on Qwen3-0.6B, fp32, 100 steps, identical minibatch order**
(`01/results/sgdnull32`, `01/results/adamnull32`):

| optimizer | final eval spread across identical-`P₀` initializations |
|---|---|
| SGD | **1.5e-6 nats** |
| AdamW | **2.7e-4 nats** |

Both are bit-identical at step 0. **2.7e-4 nats is the yardstick**: no claimed
initialization effect below it can be attributed to the preconditioner.

## 3. What survives the yardstick — the audit

13 initializers re-implemented (`common/initializers.py`) and run as samples in
`P`-space, per-method LR sweeps (7–9 rates), 3 seeds for the reference, 3 gauge
draws for the null, three matching conventions, fp32 (so that every condition
starts from a *bit-identical function*: `base_eval_loss = 0.82531` for all).

**Qwen3-0.6B / GSM8K, matched `tr(PΣ)`, best-tuned loss** (`01/results/lit`):

| | best loss |
|---|---|
| vanilla LoRA (kaiming) | 0.44270 |
| `left_gauge` (identical `P₀`) — **the null** | 0.44292 |
| NoRA (column normalization) | 0.44277 |
| NoRA, literal unit columns | 0.44260 |
| BIMI (NoRA's block-identity init) | ▢ 7B panel |
| ETF / tight frame | 0.44283 |
| exactly flat spectrum + flat diagonal | 0.44282 |
| **cluster** | **mean 0.44277, sd 1.0e-4, range 3.3e-4** |

The six data-agnostic `B₀ = 0` initializers span **3.3e-4 nats** — *inside* the
2.7e-4 null. NoRA's diagonal balancing, ETF/frame optimization, unit column
norms, exact diagonal flattening at matched spectrum, and a perfectly flat
spectrum are **mutually unidentifiable**.

At the published scale, after per-method LR tuning, no initializer beats vanilla
LoRA on this task (all in 0.4495–0.4535 vs 0.4428).

## 4. What does matter: two channels, both measurable at initialization

**(C1) Scale in the data metric.** `P` acts on data, so the size of the function
change is set not by `tr P` but by `tr(P Σ_x)`. At matched `tr P`,
EVA carries **54×**, the gradient subspace **42×**, LoRA-One **89×** the
activation-weighted trace of a vanilla draw. This channel is *exactly equivalent
to a learning-rate change*: matching `tr(PΣ)` instead of `tr P` moves them onto
the vanilla curve (EVA 0.45104 → 0.44618, gradsub 0.45175 → 0.44503).
Across all `B₀=0` methods, `loss vs log tr(PΣ)`: **r = +0.973**.

**(C2) Effective rank, in the metric that acts.** At matched scale the only
remaining feature is `r_eff(P) = (tr P)²/‖P‖_F²`, or its data-metric version
`r_eff^Σ = (tr AΣAᵀ)²/‖AΣAᵀ‖_F²`. It has an exact first-order law:

    cos(G, GP) = √( r_eff(P) / d_in )

verified on real Qwen3 gradients to **2.5% CV over an 8.6× range of r_eff**.
On a ladder with *exactly matched trace, exactly flat diagonal, identical rank
and identical crosstalk magnitude*, sweeping `r_eff` 16 → 1.86 gives a monotone
penalty at every learning rate, `loss vs 1/√r_eff`: **r = +0.947** (raw) and
**+0.996** in the `tr(PΣ)`-matched frame.

**(C3) `B₀ ≠ 0` is the only structural escape.** PiSSA, OLoRA and LoRA-One leave
the `P₀` class because `∇_A = s BᵀG ≠ 0` at step one. Across all 13 methods,
`‖B₀‖ > 0` predicts the residual at r = +0.85.

**Why NoRA targets the inert coordinate.** NoRA's goal — make `P` look like `I`
— is right. But `P = I` has `diag(P) = I` *and* `r_eff = d_in`. Column
normalization achieves the first exactly and cannot touch the second: at rank r,
`r_eff(P) ≤ r ≪ d_in`. The audit says the first is inert and the second is what
carries signal. NoRA's own BIMI control is explained by the same fact: BIMI has
a flat spectrum and a flat diagonal, i.e. it is the *same point* as
random+normalization in `(spectrum, diagonal)` coordinates — only the crosstalk
*pattern* differs, and Gate 0 shows the crosstalk *magnitude* is not a free
parameter at all (`c(P) ∈ [√(1−r/d_in), 1]` = [0.991, 1] at r=16).

## 5. The other gauge: the coordinates of the backbone

(S2) is a *function-preserving* intervention, verified in fp32 to **5e-6
relative logit error at 0.6B and 2.7e-7 nats at 7B** (Mistral-7B-v0.3). We build
a **dose ladder** in how many coordinates a rotation mixes —
`none → perm(1) → block4 → block16 → block64 → block256 → rand/hadamard(1024)` —
where `perm` is an exact **zero-dose control** (AdamW is covariant under
permutations).

**Qwen3-0.6B / NuminaMath, 7-point LR sweeps, 2 seeds** (`02/results/edge`):

| gauge | mixed coords | FullFT+AdamW | FullFT+SGD |
|---|---|---|---|
| none | 1 | 0.49340 | 0.49944 |
| block64 | 64 | +0.00050 | +0.00010 |
| rand | 1024 | +0.00196 | +0.00001 |
| hadamard | 1024 | **+0.00229** | **−0.00005** |

**SGD is exactly flat** across the whole ladder — the positive control. AdamW is
monotone in the dose.

**Mechanism, with a quantitative predictor.** AdamW's second moment is a
*diagonal* model of gradient scale, useful only insofar as coordinate-wise
scales are heterogeneous. Measuring the participation ratio of per-coordinate
gradient energy `PR = (ΣE_j)²/(d·ΣE_j²)`: the pretrained basis has
**PR = 0.025** (energy in 2.5% of coordinates — the outlier-feature structure),
a permutation leaves it *bit-identical*, and the ladder raises it to 0.73.
Penalty vs PR: **Pearson r = +0.98** (FullFT), and the ordering is predicted
correctly *including the non-trivial tie-break* — Hadamard homogenizes more than
a random rotation (0.73 vs 0.48) and costs more, though both mix all 1024 coords.

**Consequence.** Adam's advantage over SGD is partly a property of the
coordinate system rather than of the model:

| gauge | PR | Adam's edge over SGD (FullFT) |
|---|---|---|
| none | 0.025 | +0.00604 |
| block64 | 0.076 | +0.00565 (−6%) |
| rand | 0.476 | +0.00409 (−32%) |
| hadamard | 0.727 | **+0.00370 (−39%)** |

LoRA is markedly *less* gauge-sensitive than full fine-tuning, so PEFT does not
carry excess coordinate dependence — it partially shields it.

## 6. Positioning against the closest work

| work | what it owns | what remains ours |
|---|---|---|
| **NoRA** (mother paper, 2026) | `P = α²AᵀA`; `diag(P)=I` normalization; BIMI; NoRA-init | that `diag(P)` is the inert coordinate of `P`, and which coordinates are not |
| **"The Loss Does Not See the Basis, but Adam Does"** (2608.05136) | classification of optimizers by equivariance under the *factor* gauge `(U,V)→(UQ,VQ)`; matrix sensing, 2-layer transformers, hyperspectral | the **LoRA identifiability statement** (`P₀` is the *complete* invariant, so the design space of `B₀=0` initializers is `{P ⪰ 0, rank ≤ r}`), the audit of the real initialization literature at LLM scale, and the two channels |
| **"Understanding Adam Requires Better Rotation Dependent Assumptions"** (NeurIPS 2025) | parameter-space rotations degrade Adam in pretraining (GPT-2, ViT) | rotations that preserve the *network function exactly* and keep every weight matrix a weight matrix; a dose ladder with a permutation zero-dose control; a quantitative predictor (PR, r=+0.98); the fine-tuning/PEFT setting; the Adam-vs-SGD edge decomposition |
| **"Learning Rate Matters: Vanilla LoRA May Suffice"** (2602.04998) | 9 LoRA variants collapse to within 1–2% after LR tuning (GLUE) | the *mechanism* that predicts the required LR shift (`tr(PΣ)`), and a **provable** null that says the residual is unmeasurable in principle rather than merely small |
| **LoRA-DA / EVA / LoRA-One** | data-aware subspace selection | their gain is a `tr(PΣ)` amplification; after matching it, what remains is `r_eff^Σ` |

## 7. Experimental scope (target = ICML)

| axis | mother paper (NoRA) | this work |
|---|---|---|
| models | 342M pretrain, 3B SFT, 1.5B RL | 0.6B (Qwen3) + **7B (Mistral-7B-v0.3)** ▢ |
| tasks | MetaMath, CodeFeedback | GSM8K, NuminaMath, Dolly, **MetaMath** ▢ |
| metrics | benchmark accuracy | held-out loss **and** GSM8K exact-match ▢ |
| initializers | 9 baselines | **13** incl. BIMI, plus 6 exact synthetic controls |
| LR protocol | tuned per method | swept per method (5–9 rates), reported in full |
| seeds | 1 (seed 42) | 3 for the reference and the null |
| null condition | — | `left_gauge`, provably content-free |

## 8. Open items

▢ 7B panels running: `01/results/m7b` (75 runs, MetaMath, r=32, GSM8K accuracy),
  `02/results/dose7b` (24 runs, fp32 gauge ladder).
▢ BIMI in the audit at 0.6B (implemented, not yet swept).
▢ A code task (CodeFeedback → HumanEval) for a non-math, non-chat third domain.
▢ FullFT at 7B for the gauge ladder (needs 2-GPU sharding).
▢ The prescription test: does vanilla LoRA at `lr·√(tr(PΣ) ratio)` reproduce a
  data-aware initializer's entire loss curve, or only its optimum?
