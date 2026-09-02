> **Superseded in part.** This file is the Stage-1 record (the audit of
> published initializers). Stage 2 found what Stage 1 was missing, and it is
> not another invariant: it is the **gauge frame**. See `paper/NARRATIVE.md`
> for the current main line and `paper/POSITION.md` for how it sits against
> LoRA-RITE, Balanced LoRA and FedRot-LoRA.
>
> Two Stage-1 conclusions are falsified. *"Effective rank is the second and
> only other channel"* and *"method identity adds nothing after conditioning
> on three statistics"* are not supported: an out-of-distribution test on the
> same 13 initializers mispredicts their tuned loss systematically, and the
> residual points past every invariant.
>
> The reason the Stage-1 audit found the NoRA family mutually
> indistinguishable is now understood rather than merely reported. Those
> conditions differ in statistics of `P` but sit at the **same frame** as the
> vanilla draw (`Lambda_1` 0.328-0.366 against Kaiming's 0.366, and 0.356-0.366
> excluding BiMI), and the frame
> is the coordinate AdamW actually responds to. The audit's null result was
> correct and its interpretation was incomplete.
>
> Superseding measurements, all in `results/`:
>
> | claim | evidence |
> |---|---|
> | SGD cannot see the gauge at all | `results/frame`, spread 0.00001 nats over a 2.9x `Lambda_1` ladder |
> | AdamW can, monotonically at every learning rate | same panel, tuned spread 0.00222 nats = 8.2x the null |
> | the published zoo varies the frame 4.3x, unreported | `results/second_order.json`, no training |
> | rotating vanilla Kaiming to the gradient-metric eigenframe helps for free | 0.44551 -> 0.44329, `B A` and `P` preserved to 1e-15 |

# 01 — Hidden Preconditioner — STATUS

Last update: 2026-09-02
Model: Qwen3-0.6B-Base · task: GSM8K SFT (300 steps, AdamW, r=16, α=32, all 7
linear module types, 196 adapters) unless stated · fp32 weights and compute for
the audit; bf16 for the earlier matched panels.

---

## Summary of what is established

| channel of `P = s²A₀ᵀA₀` | causal? | evidence |
|---|---|---|
| **scale in the data metric**, `tr(PΣ_x)` | **yes — dominant** | `r = +0.973` between best-tuned loss and `log tr(PΣ)` across all B₀=0 methods |
| **effective rank**, `r_eff(P)` (and its data-metric version) | **yes — second channel** | `r = +0.978` within the data-agnostic family; verified first-order law |
| **diagonal balance** (NoRA's mechanism) | **no** | inside the provably content-free null |
| **crosstalk magnitude** | not a free parameter | killed analytically at Gate 0 |
| **frame / ETF geometry, unit norms** | **no** | inside the null |
| **`B₀ ≠ 0`** | yes — leaves the equivalence class entirely | the only methods that separate after full matching |
| **`tr P` (parameter-space scale)** | **no**, not on its own | rank 4/16/64 vary `tr P` 16× and lie on one loss(LR) curve |

---

## Gate 0 — theory (`src/gate0_feasible_region.py`, CPU)

**F1. The feasible region of `P` is constructively parameterised.** Givens
rotations on the *columns* of `A` (Bendel–Mickey, `O(d_in·r)`) realise any
Schur–Horn-feasible (spectrum λ, diagonal d) exactly: spectrum error ≤2e-16,
relative diagonal error ≤5e-9 up to `r=64, d_in=2048`. Every matched control in
this project is therefore exact, not approximate.

**F2. Crosstalk magnitude is not an independent knob — killed at Gate 0.**
`tr P = Σλ = Σd`, `‖P‖_F² = Σλ²`, `‖diag P‖² = Σd²`, so
`c(P)² = 1 − Σd²/Σλ²` is a *deterministic function of (λ,d)*. Six draws at
matched (λ,d) gave `c(P) = 0.993156961279` to 12 digits while their `P` differed
by `‖P₀−P₁‖/‖P₀‖ = 1.40`. With a flat diagonal, `c(P) ∈ [√(1−r/d_in), 1]` =
[0.9910, 1] at r=16, d_in=896. The pre-registered statistic §3.3 is *pinned* in
the realistic regime; family B can only be a crosstalk-**pattern** experiment.

**F3–F6. T1 is far stronger than first-order.** For `rank(A)=r`,
`A₁ᵀA₁ = A₂ᵀA₂ ⟺ A₂ = QA₁`, `Q ∈ O(r)` (polar uniqueness; verified
constructively, `‖A₂−Q̂A₁‖ = 1.4e-15`). Under plain SGD with `B₀=0` the pair
`(A,B) → (QA, BQᵀ)` is preserved **for all time**: 30 steps, relative trajectory
difference ≤3.2e-15 at every LR, versus 0.7–1.1 for a different `P₀`.

> **Under SGD, "which initialisation method" is unidentifiable given `P₀` — not
> just at step 1, ever.** AdamW breaks it (`m/√v` is elementwise), which turns a
> random left-`O(r)` rotation of `A₀` into a control that changes **no**
> P-statistic yet perturbs training: a provably content-free null, far tighter
> than a seed change.

---

## The measurement floor, verified end to end on the real model

`left_gauge` (A₀ → QA₀, Q ∈ O(r)) is bit-identical in `P₀`, `tr P`, `r_eff`,
`diag`, crosstalk, `tr(PΣ)`, `tr(PC_g)` and `cos(G,GP)` — confirmed numerically
(`cos_sgd = 0.10813` for both kaiming and left_gauge, to 5 digits) — while
`cos_adam` differs (0.0624 vs 0.0614), exactly as the theory requires.

Run on Qwen3-0.6B-Base in **fp32**, same ordered minibatches, 100 steps
(`results/sgdnull32`, `results/adamnull32`):

| optimizer | step 0 | final eval spread across identical-`P₀` inits |
|---|---|---|
| SGD, lr 0.1 | bit-identical | **1.5e-6 nats** |
| AdamW, lr 2e-4 | bit-identical | **2.7e-4 nats** |

A factor of **180**. This is the theorem verified on a real transformer: under
SGD the whole trajectory depends on `A₀` only through `P₀`; under AdamW there is
a real channel of ≈2.7e-4 nats that carries **zero** preconditioner information.
**2.7e-4 nats is therefore the correct yardstick for any claimed LoRA
initialisation effect in the standard AdamW setting**, and the six data-agnostic
`B₀=0` initializers in the audit span 6.9e-4 — the same order.

(In bf16 the same check drifts to ~1e-3 by step 15 from rounding alone, which is
why the audit is run in fp32 and why the bf16 panels' nulls are inflated.)

---

## Gate 1a/1b — matched panels (`results/g1`, `g1b*`, 115 runs)

Best-tuned loss at exactly matched `tr P = 21.33` (r=16, α=32, 8 LRs):

| condition | r_eff(P) | diag imbalance | best loss |
|---|---|---|---|
| kaiming (vanilla LoRA) | 15.80 | 5.0e-2 | **0.44236** |
| `flatspec_flatdiag` | 16.00 | 3e-21 | 0.44326 |
| `kaimingspec_flatdiag` (trace **and** spectrum matched) | 15.80 | 3e-21 | 0.44329 |
| `nora` (trace-matched) | 15.81 | 8e-32 | 0.44348 |
| `geomspec_flatdiag0.5` | **3.00** | 1e-20 | **0.44522** |
| `nora_unit` (literal unit columns, tr P ×274) | 15.81 | 4e-32 | 0.44773 |

`nora_unit`'s apparent gain at a shared LR is a one-decade shift of the LR axis
(‖ΔW‖ ×10); at its own best LR it is *worse* than vanilla LoRA.

`tr P` is **not** a universal collapse coordinate: r=4/16/64 at α=32 vary `tr P`
16× yet lie on one loss(LR) curve (rms 1.1e-3), whereas α=8 vs α=128 at r=16 do
not (rms 1.3e-2 / 1.8e-2; α=128 gives the single best run in the panel, 0.43891).

---

## Gate 1c — the r_eff dose–response (`results/g1c`, 72 runs)

Ladder of 11 initialisations with **exactly matched trace, exactly flat
diagonal, identical nominal rank r=16, identical crosstalk magnitude**, varying
only the spectrum so that `r_eff(P)` sweeps 16.00 → 1.86:

* the penalty is monotone and present at **every one of 4 learning rates**;
* best-tuned loss vs `1/√r_eff`: **Pearson r = +0.947**, slope +0.0073 nats;
* it persists at 3× training length (900 steps: +0.0044, larger than at 300).

### The first-order law behind it (`src/grad_capture.py`)

For SGD, `ΔW₁ = −ηGP`, so the loss decrease per unit update norm is
`cos(G,GP)`. For a `P` whose eigenvectors are unrelated to the gradient,

    cos(G, GP) = √( r_eff(P) / d_in ).

Measured on real Qwen3 gradients over the whole ladder, `cos_sgd/√r_eff` =
0.0281 ± 0.0007 — **2.5% CV across an 8.6× range of r_eff**.

Adam's first step is `−lr·s·sign(GAᵀ)A`, *not* `−ηGP`; the measured
`cos_adam` falls only 2.05× while `cos_sgd` falls 2.67× over the same ladder,
i.e. Adam partially compensates, which is why the trained penalty is 0.003 and
not the ~0.01 a naive SGD reading would predict.

---

## The literature audit (`results/lit`, 228 runs)

Every published initializer re-implemented (`common/initializers.py`) and run as
a sample in P-space against the vanilla reference (3 seeds), the `left_gauge`
null (3 gauges), and exact synthetic controls, over 5–7 learning rates, under
three matching conventions. **Correctness gate: in fp32 every condition starts
from a bit-identical function — `base_eval_loss = 0.82531` for all, including
PiSSA/OLoRA/LoRA-One whose base-weight subtraction is exact only above bf16.**

### Matched `tr P` — best-tuned loss

| condition | group | r_eff | r_eff^Σ | tr(PΣ) | ‖B₀‖ | best |
|---|---|---|---|---|---|---|
| kaiming | agnostic | 15.79 | 6.51 | 1.00 | 0 | 0.44317 |
| left_gauge (null) | agnostic | 15.79 | 6.51 | 1.00 | 0 | 0.44270 |
| nora | agnostic | 15.80 | 6.48 | 1.00 | 0 | 0.44284 |
| nora_unit | agnostic | 15.80 | 6.48 | 1.00 | 0 | 0.44249 |
| etf | agnostic | 16.00 | 6.67 | 1.02 | 0 | 0.44298 |
| flatspec_flatdiag | agnostic | 16.00 | 6.30 | 1.05 | 0 | 0.44277 |
| geomspec_flatdiag0.5 | agnostic | 3.00 | 2.38 | 1.16 | 0 | 0.44550 |
| eva | data-aware | 16.00 | 4.01 | **54.5** | 0 | 0.45104 |
| gradsub (LoRA-One subspace) | data-aware | 16.00 | 3.26 | **42.0** | 0 | 0.45175 |
| pissa | B₀≠0 | 15.51 | 3.16 | 4.6 | 10.3 | 0.45098 |
| pissa_minor | B₀≠0 | 14.11 | 3.99 | 5.7 | 0.7 | 0.44999 |
| olora | B₀≠0 | 11.30 | 2.51 | 12.5 | 6.7 | 0.45054 |
| lora_one | B₀≠0 | 5.59 | 1.75 | **88.9** | 0.2 | 0.45669 |

**The six data-agnostic B₀=0 methods span 0.44249–0.44317 — a range of 0.00069,
entirely inside the `left_gauge` null.** NoRA, ETF/frame optimisation, unit
column norms, exact diagonal flattening at matched spectrum and a perfectly flat
spectrum are all mutually indistinguishable from vanilla LoRA after LR tuning.

Within that family the one statistic that predicts is the effective rank:
`loss vs 1/√r_eff`, **r = +0.978** (n=7); `loss vs 1/cos_adam`, r = +0.973.

### Matched `tr(PΣ)` — the data metric

At matched `tr P`, EVA carries **54.5×** the activation-weighted trace and
**72.8×** the gradient-weighted trace of a vanilla draw; the gradient subspace
carries 42×/103×; LoRA-One 89×/449×. Matching `tr(PΣ)` instead moves them most
of the way onto the vanilla curve:

| condition | best (matched tr P) | best (matched tr PΣ) |
|---|---|---|
| eva | 0.45104 | 0.44618 |
| gradsub | 0.45175 | 0.44503 |
| pissa_minor | 0.44999 | 0.44671 |
| vanilla cluster | 0.4425–0.4432 | 0.4426–0.4429 |

The residual (+0.002 for gradsub) is of the size predicted by their reduced
**data-metric effective rank** `r_eff^Σ = (tr AΣAᵀ)²/‖AΣAᵀ‖_F²` (3.26 vs 6.51):
`0.0073·(1/√3.26 − 1/√6.51) = 0.0012` predicted, 0.0023 observed.

Across all B₀=0 methods, `loss vs log tr(PΣ)`: **r = +0.973**.

---

## Interpretation

1. **NoRA's mechanism does not survive.** Equalising `diag P` — by NoRA's own
   operator, at exactly matched spectrum, or with a perfectly flat spectrum — is
   inside a provably content-free null. The literal unit-norm version's gain is
   a learning-rate shift.
2. **`P` acts on data, so the causal scale is `tr(PΣ_x)`, not `tr P`.** This is
   the loophole every data-aware initializer exploits: at matched `tr P` they are
   40–110× amplifications of the data-metric scale. It reconciles NoRA (a
   parameter-space normaliser), LoRAM (magnitude), and EVA/LoRA-DA/LoRA-One
   (data-aware subspaces) inside one object.
3. **The second and only other channel is effective rank**, with an exact
   first-order law verified on real gradients, and it must be measured in the
   metric that acts (`r_eff^Σ`), not in parameter space.
4. **`B₀ = 0` is the boundary of the equivalence class.** Only PiSSA, OLoRA and
   LoRA-One escape it, because `∇_A = sBᵀG ≠ 0` at step 1.
5. **Method identity adds nothing after conditioning on (data-metric scale,
   effective rank, `B₀≠0`).**

This is Branch A ∪ Branch C of the pre-registered survival tree, with a
predictive law rather than a debunk — the condition the README set for Branch A
to be promotable.

---

## Falsified

* NoRA's diagonal-balance mechanism (Branch B) — inside the null.
* Crosstalk magnitude as a causal knob — impossible by construction.
* `tr P` / `‖ΔW‖` as a universal collapse coordinate — fails across α.
* The isotropic SGD prediction that low `r_eff` should cost ~2.3× descent
  efficiency in trained loss — Adam attenuates it to ~0.003 nats.

## Unresolved / in flight

* Four cells (eva, gradsub, pissa, olora, lora_one at published or `tr P` scale)
  have their LR optimum **at the edge of the grid**; an extension down to 3e-6
  and up to 2e-3 is running (`results/lit`, +186 runs). Their "best-tuned" values
  above are therefore upper bounds on their performance.
* Whether the SGD r_eff effect is genuinely smaller than the AdamW one: the SGD
  ladder (`results/g1d`, `g1e_long`, 133 runs) gives slope +0.0007 at 300 steps
  and +0.0049 at 900 steps against AdamW's +0.0073, but the SGD LR grid is coarse
  and divergence-limited. **Not resolved at present precision.**
* Breadth: a second task (dolly instruction-following) is running
  (`results/lit_dolly`, 168 runs); a second model size is not yet done.

## Next experiments

1. Finish the LR-tail extension so every method's optimum is interior.
2. Breadth: dolly (running) and Qwen3-1.7B-Base for the decisive rows.
3. A direct test of the prescription: does vanilla LoRA at
   `lr × √(tr(PΣ)_method/tr(PΣ)_vanilla)` reproduce the data-aware
   initializer's whole loss curve?
