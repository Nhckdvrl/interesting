# 01 — Hidden Preconditioner — STATUS

Last update: 2026-09-02

## Gate 0 — theory / synthetic (`src/gate0_feasible_region.py`, CPU)

**F1. The feasible region of P is constructively parameterised.** A Bendel–Mickey
sweep of Givens rotations on the *columns* of `A` (cost `O(d_in·r)`) realises any
Schur–Horn-feasible pair (spectrum λ, diagonal d) exactly: spectrum error ≤2e-16,
relative diagonal error ≤5e-9 at every size tested up to `r=64, d_in=2048`. This
is the tool that makes every matched control in this project exact rather than
approximate.

**F2. Crosstalk magnitude is not an independent knob — killed at Gate 0.** Since
`tr P = Σλ = Σd`, `‖P‖_F² = Σλ²`, `‖diag P‖² = Σd²`,

    c(P)² = ‖P − Diag P‖_F² / ‖P‖_F² = 1 − Σd²/Σλ²

is a *deterministic function of (λ, d)*. Six independent draws with matched
(λ, d) gave `c(P) = 0.993156961279` to 12 digits while their `P` matrices differ
by `‖P₀−P₁‖/‖P₀‖ = 1.40`. Moreover, with a flat diagonal `Σλ² ≥ (tr P)²/r`, so

    c(P) ∈ [ √(1 − r/d_in), 1 ]   — [0.9910, 1] for r=16, d_in=896.

The pre-registered statistic §3.3 is therefore *pinned* in the realistic
`r ≪ d_in` regime and cannot be a causal knob. Experiment family B can only be a
crosstalk-**pattern** experiment; magnitude changes must be routed through the
spectrum. (NoRA's BIMI experiment was, necessarily, also a pattern experiment.)

**F3–F5. T1 is much stronger than first-order.** If `rank(A)=r`, then
`A₁ᵀA₁ = A₂ᵀA₂ ⟺ A₂ = QA₁` with `Q ∈ O(r)` (polar uniqueness; verified
constructively, `‖A₂ − Q̂A₁‖ = 1.4e-15`). Under plain SGD with `B₀=0` the pair
`(A,B) → (QA, BQᵀ)` is preserved **for all time**, so the merged trajectory
`s·B_t A_t` is identical:

    30 steps, relative trajectory difference ≤ 3.2e-15 at every LR tested,
    versus 0.7–1.1 for a genuinely different P₀.

→ **Under SGD, "which initialisation method" is unidentifiable given P₀.** Not
just at step 1 — ever.

**F6. AdamW breaks it, giving a zero-content null.** `m/√v` is elementwise and
not equivariant under `A → QA`. Two initialisations with *identical* P₀ diverge
by 29–36% relative in `ΔW` after 30 Adam steps, versus 130–139% for genuinely
different P₀. So a random left-`O(r)` rotation of `A₀` is a control that changes
**no** pre-registered P-statistic yet perturbs Adam training — a far tighter null
than a seed change.

## Gate 1a — matched panel (`results/g1`, 52 runs, Qwen3-0.6B-Base / GSM8K / 300 steps / AdamW / r=16 α=32, 4 LRs × 3 seeds)

| condition | best eval loss | tr P | diag imbalance | effect vs kaiming |
|---|---|---|---|---|
| kaiming (vanilla LoRA) | **0.44302** @3e-4 | 5.33 | 5.0e-2 | — |
| `left_gauge` (P₀ *identical*) | 0.44364 | 5.33 | 5.0e-2 | **the null**: sd 1e-4…2.6e-3 |
| `nora` (trace-matched) | 0.44374 | 5.33 | 8e-32 | −0.5σ … +2.4σ, sign unstable |
| `kaimingspec_flatdiag` (trace **and** spectrum matched) | 0.44455 | 5.33 | 3e-21 | −0.3σ … +5.0σ |
| `nora_unit` (literal unit columns) | 0.45061 @3e-5 | 1462.9 | 4e-32 | worse after LR tuning |

`nora_unit` has 274× the trace and 10× the realised `‖ΔW‖`; its apparent gain at a
shared LR is entirely a one-decade shift of the LR axis, and at its own best LR
it is *worse* than vanilla LoRA.

## Gate 1b — dense LR sweep + spectrum/rank/α interventions (`results/g1b*`, 63 runs)

Best-tuned loss, all at exactly matched `tr P = 21.33` (r=16, α=32):

| condition | r_eff(P) | diag imbalance | best loss |
|---|---|---|---|
| kaiming | 15.80 | 5.0e-2 | **0.44236** |
| `flatspec_flatdiag` | 16.00 | 3e-21 | 0.44326 |
| `kaimingspec_flatdiag` | 15.80 | 3e-21 | 0.44329 |
| `nora` | 15.81 | 8e-32 | 0.44348 |
| `geomspec_flatdiag0.5` | **3.00** | 1e-20 | **0.44522** |

`geomspec_flatdiag0.5` is worse at **every one of the 8 learning rates**
(+0.002…+0.003 nats in the stable region), and the penalty survives matching on
the realised `‖ΔW‖` (+0.0035 at matched update norm). Every other statistic —
trace, diagonal, crosstalk magnitude, nominal rank — is held fixed by
construction.

## Interpretation

1. **Diagonal balance, NoRA's claimed mechanism, is inert.** Once `tr P` is
   matched, flattening `diag P` — by NoRA's own operator, or exactly at matched
   spectrum, or with a perfectly flat spectrum — moves the tuned loss by at most
   ~3× the gauge null, and if anything makes it slightly *worse*.
2. **The causal content of P is in its spectrum, not its diagonal.** Collapsing
   `r_eff(P)` from 16 → 3 at matched trace and flat diagonal is a consistent,
   monotone, ~10σ penalty. This is a *positive* result and it identifies a
   different feature of the same object NoRA discovered.
3. **`tr P` is not a universal collapse coordinate.** `r=4/16/64` at α=32 vary
   `tr P` by 16× yet lie on one loss(LR) curve (rms 1.1e-3), whereas α=8 vs
   α=128 at r=16 do not (rms 1.3e-2 / 1.8e-2, and α=128 gives the single best
   run in the panel, 0.43891). So the effective step scale under Adam is not
   `tr P`, and "magnitude" needs a sharper definition than LoRAM's.

## Unresolved

* The `√r_eff` first-order prediction (see `src/grad_capture.py`) is an **SGD**
  statement: under Adam the first step is `−lr·s·sign(GAᵀ)A`, not `−ηGP`. The
  measured Adam penalty (ratio 0.93 in early descent efficiency) is far smaller
  than the isotropic SGD prediction (0.44). Whether the r_eff mechanism is
  quantitatively SGD-theoretic and merely *attenuated* by Adam is being tested.
* Whether the r_eff penalty persists at 3× training length.
* Why α and r are not interchangeable at fixed `s = α/r`.

## Next experiments

* `panel_g1c` — r_eff dose–response ladder (16 → 1.86) at matched trace and flat
  diagonal, 4 LRs, 3 seeds at 4 rungs, plus a 900-step persistence check.
* `grad_capture` — the cheap init-time predictor: `cos(G, GP)` vs
  `cos(G, sign(GAᵀ)A)`, testing `cos_sgd = √(r_eff/d_in)` on real gradients.
* r_eff ladder under **SGD**, where the theory is exact.
