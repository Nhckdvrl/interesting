# Pre-registered: which functional of the frame is AdamW responding to?

Written while `results/rot` was at 7/72 runs and before `results/frame`'s
`framex` conditions had been launched. Not edited afterwards.

## The two candidates

The frame ladder established that AdamW responds to the gauge frame
(0.00222 nats tuned spread, monotone at every learning rate) and SGD does not
(0.00001). It also **falsified** the direction predicted by the first-order
`l_inf` argument: concentration wins, so higher `Lambda_1` is worse, not better.

Two functionals survive, and a one-parameter ladder cannot separate them
because along it everything moves together.

**H1 -- descent rate.** `Lambda_1 = ||G A^T||_1^2 / (d_out r ||G A^T||_F^2)`
with the empirical sign: lower is better.

**H2 -- diagonal preconditioner.** AdamW normalises elementwise, so it is a
*diagonal* preconditioner in whatever basis it is handed. The adapter-side
curvature is `M_x = A Sigma A^T`, so AdamW's approximation is exact when `M_x`
is diagonal. The functional is the off-diagonal mass
`Off_x = ||offdiag M_x||_F^2 / ||M_x||_F^2`, and lower is better.

## The design that separates them

A second ladder in the *activation* metric. `Sigma` and `C_g` have different
eigenbases, so:

| frame | `Lambda_1` | `Off_x` |
|---|---|---|
| `frame0` (eigenbasis of `M_g`) | **0.144** (lowest) | 0.380 |
| `frame0.5` | 0.309 | 0.481 |
| `frame1` (flat `M_g` diagonal) | 0.422 | 0.515 |
| `framex0` (eigenbasis of `M_x`) | 0.303 (middling) | **0.000** (exact) |
| `framex1` (flat `M_x` diagonal) | 0.368 | 0.593 |
| `kaiming` | 0.366 | 0.532 |

**H1 predicts** `framex0` lands middling, between `frame0.25` and `frame0.5`
(about 0.4446-0.4448 at lr = 3e-4).
**H2 predicts** `framex0` is the **best frame of all**, beating `frame0`'s
0.44329, because its off-diagonal curvature mass is exactly zero.

## The zoo rotation separates them again, on EVA

EVA is activation PCA, so its rows are eigenvectors of `Sigma` and
`M_x = A Sigma A^T` is diagonal by construction: measured `Off_x = 0.000`.
**EVA is already sitting at H2's optimum without its authors saying so.**
Rotating it to `@frame0` lowers `Lambda_1` (0.285 -> 0.123) but raises `Off_x`
(0.000 -> 0.486).

* **H1 predicts `eva@frame0` beats `eva`.**
* **H2 predicts `eva@frame0` is worse than `eva`, and that `eva` is close to
  the best condition in the whole panel.**

Both hypotheses agree that `@frame1` hurts everything, so that arm is not
discriminating; it is a consistency check.

For `lora_one` and `gradsub`, `@frame0` leaves `Off_x` unchanged to three
decimals (0.370 -> 0.370, 0.570 -> 0.570) while lowering `Lambda_1` by 1.8x and
1.4x. H2 therefore predicts those two rotations are near-nulls; H1 predicts
they help.

## What would falsify both

`framex0` no better than `kaiming` and `eva@frame0` indistinguishable from
`eva`, i.e. the frame response is real but neither functional captures it. In
that case the honest statement is that the frame is a coordinate we can
measure the importance of but not yet summarise, and the paper says so.
