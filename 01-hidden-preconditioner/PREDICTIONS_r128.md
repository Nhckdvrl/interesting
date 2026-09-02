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
