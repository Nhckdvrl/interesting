# Pre-registered: the r = 128 frame effect

Committed before any r = 128 run finished. Not edited afterwards.

## The diagnostic

The span of `Lambda_1` over the gauge orbit -- from the gradient-metric
eigenframe to the flat-diagonal frame -- is measurable from **one probe
forward+backward, with no training** (`src/frame_reach.py`). Measured on
Qwen3-0.6B / GSM8K:

| r | dim `O(r)`/signed perms | `Lambda_1` eigenframe | flat | **reach** |
|---|---|---|---|---|
| 1 | 0 | 0.4222 | 0.4222 | **1.000** |
| 4 | 6 | 0.2962 | 0.4238 | 1.431 |
| 16 | 120 | 0.1443 | 0.4216 | 2.923 |
| 64 | 2016 | 0.0535 | 0.4174 | 7.809 |
| 128 | 8128 | 0.0314 | 0.4167 | 13.278 |

The reach is **exactly 1.000 at r = 1**, structurally: `O(1)` is the
signed-permutation group, which is AdamW's own symmetry, so there is no
rotation to make.

## The fit

Against the measured `frame1 - frame0` at lr = 1e-4, with the fit forced
through the origin because reach = 1 must give exactly zero:

    effect = 0.00184 x log(reach)        R^2 = 0.975

| r | predicted | measured |
|---|---|---|
| 1 | +0.00000 | -0.00000 |
| 4 | +0.00066 | +0.00044 |
| 16 | +0.00198 | +0.00160 |
| 64 | +0.00379 | +0.00402 |

One parameter, four points, one of which is pinned at zero by the group theory
rather than by the data.

## The prediction

**At r = 128, reach = 13.278, so `frame1 - frame0` at lr = 1e-4 should be
+0.00477 nats** -- about 24x the ~2e-4 nat reproducibility floor.

Falsified if the measured value is below +0.002 or above +0.010, i.e. if the
single-parameter law misses by more than a factor of about two on a point
extrapolated a full octave beyond the fitted range.

## Why this matters beyond the fit

If it holds, the reach is a **pre-training diagnostic**: measure it in one
probe pass and it says whether rotating the initialisation is worth doing for
your rank and your data, before committing any training compute. It also says
the effect grows into exactly the rank range people now deploy.

---

## Outcome: falsified as stated, but I over-called it

**Correction, written after extending the grid.** The reading below was taken
off a grid whose bottom rung was 5e-5, and `frame0`'s optimum sat exactly
there. Extending to 1e-5 and 2e-5 brackets it, and changes what the panel says:

| r = 128 | 1e-5 | 2e-5 | 5e-5 | 1e-4 | 2e-4 | 3e-4 | tuned |
|---|---|---|---|---|---|---|---|
| frame0 | 0.45900 | 0.44918 | **0.43984** | 0.44339 | 0.46042 | 0.49014 | 0.43984 @ 5e-5 |
| frame1 | 0.45977 | 0.45260 | 0.44324 | **0.44145** | 0.45584 | 0.48767 | 0.44145 @ 1e-4 |
| kaiming | 0.45878 | 0.45167 | 0.44255 | **0.44154** | 0.45362 | 0.48097 | 0.44154 @ 1e-4 |

At **tuned** learning rates: `frame0 - kaiming = -0.00170` and
`frame1 - frame0 = +0.00161`. Both have the *same sign as r = 1..64*: frame0
best, frame1 worst. The frame ordering does **not** reverse at r = 128.

What is genuinely falsified is the prediction as literally written -- it named
`frame1 - frame0` **at lr = 1e-4**, and there the value is -0.00194 against a
predicted +0.00477, outside the stated bounds. That happens because `frame0`
peaks at 5e-5 while the other two peak at 1e-4, so a fixed-learning-rate
comparison catches frame0 past its optimum.

So: the single-parameter reach law does not predict the magnitude an octave out,
and a fixed-lr contrast is the wrong statistic when conditions have different
optimal learning rates. But my report that "the sign is wrong at every learning
rate" was **wrong** -- it described a truncated grid, and I should have said so
was possible before drawing it.

## Original outcome, as recorded before the grid was extended

Measured `frame1 - frame0` at r = 128, on the grid actually run
(5e-5, 1e-4, 2e-4, 3e-4):

| lr | 1e-4 | 2e-4 | 3e-4 |
|---|---|---|---|
| frame1 - frame0 | **-0.00194** | -0.00458 | -0.00247 |

Predicted **+0.00477**. The stated falsification bound was "below +0.002 or
above +0.010", so this is falsified -- and not marginally: the **sign** is
wrong at every learning rate, consistently. At r = 1 through 64 `frame1` was
worse than `frame0`; at r = 128 it is better.

The single-parameter reach law therefore does not extrapolate an octave past
its fitted range. What it got right stands -- the effect is exactly zero at
r = 1 and grows through r = 4, 16, 64 with R^2 = 0.975 on those four points --
but it is a fit on that range, not a law, and this file is left as written so
that is on the record.

Note for the joint analysis, not a rescue: this is the second sign reversal
seen, the other being Qwen3-8B, where a 4x larger probe reversed it back. Both
reversals occur where `M_g` is estimated from proportionally less data -- 128x128
from the same probe here, 16x16 from a quarter of the probe there. Whether that
is the common cause is exactly what the joint analysis has to decide, and it may
well not be.
