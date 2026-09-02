# Pre-registered predictions: the gauge-frame ladder

Written and committed **before** the panel was launched.
Commit this file, then run. Nothing below is edited afterwards.

## The claim under test

LoRA's B-gradient is `grad_B = s G A^T`. Under the adapter gauge
`A -> QA, B -> BQ^T` (Q orthogonal) it transforms by right multiplication,
`grad_B -> grad_B Q^T`.

* **SGD is exactly covariant.** `dB = -eta s G A^T -> dB Q^T` and
  `dA = -eta s B^T G -> Q dA`, so the whole trajectory maps by the gauge, for
  all time, with or without momentum, weight decay, or global-norm clipping
  (all of which depend only on Frobenius norms, which are invariant). With
  `B_0 = 0` the two runs are the same run in different coordinates.
* **AdamW is not.** Its second-moment estimate is elementwise, so it is
  steepest descent in the elementwise `l_inf` geometry, whose dual norm is the
  elementwise `l1` norm. `||grad_B||_1` is invariant only under signed
  permutations of the rows of A, not under all of O(r).

The dimensionless ratio of the two first-order descent rates,

    Lambda_1 = ||G A^T||_1^2 / (d_out r ||G A^T||_F^2)   in (0, 1],

is therefore a coordinate AdamW can see and SGD provably cannot. Schur-Horn
pins its range at fixed invariants: its row-partition factor
`E_g = (sum_j sqrt((M_g)_jj))^2 / (r tr M_g)` runs from the eigenbasis frame
(minimum) to the flat-diagonal frame (`E_g = 1`).

## The intervention

`frame<t>` sets `A = Q(t) A_0` with `A_0` the vanilla Kaiming draw and `Q(t)`
the ladder from the eigenbasis of `M_g` (t = 0) to a flat `M_g` diagonal
(t = 1). Verified on the real model: every logged preconditioner statistic --
`tr_P`, `frob_P`, `crosstalk`, `eff_rank`, `spec_max/min`, `diag_imbalance`,
`colnorm_cv`, and crucially `tr_P_act` (= S) and `tr_P_grad` (= tr M_g) --
agrees with `kaiming` to <= 1.5e-8 relative, i.e. float32 round-off.

## Predictions

1. **SGD is flat to the null.** The spread of tuned loss across the six frame
   conditions is at most the measurement null, 2.7e-4 nats, and should be
   nearer the SGD null of 1.5e-6. This is not a fitted expectation: the runs
   are the same trajectory in rotated coordinates. Any spread above ~1e-3 nats
   falsifies the covariance argument and this whole line of reasoning.
2. **AdamW moves, monotonically in Lambda_1**, with lower loss at higher
   Lambda_1: `frame1 < frame0.75 < frame0.5 < frame0.25 < frame0`.
   `kaiming` lands near `frame1`, because a random frame already spreads the
   gradient energy (measured `E_g = 0.853` for kaiming, against 0.195 for
   LoRA-One).
3. **The AdamW spread exceeds every other single-axis effect measured so far**
   -- larger than the omega ladder (0.0027-0.0041 nats) and far larger than the
   whole NoRA-family cluster (3.3e-4). The 5-step probe gave 0.032 nats at a
   single learning rate; after per-condition LR tuning over 300 steps I expect
   this to shrink but to stay above 0.005 nats.
4. **The null is explained, not just measured.** `left_gauge` (a random point
   on the same orbit) sits at `Lambda_1 = 0.3556` against kaiming's 0.3661.
   The AdamW ladder's local slope near `t = 1`, converted to nats per unit
   `Lambda_1`, should be within a factor ~3 of the 0.0257 nats implied by the
   2.7e-4 nat left-gauge null.

## What would falsify this

* SGD spread > 1e-3 nats (covariance is wrong, or the implementation is).
* AdamW spread <= the null (the frame carries no information after all).
* AdamW spread present but non-monotone in Lambda_1 and uncorrelated with it
  (the frame matters, but Lambda_1 is the wrong summary of it).
