# Pre-registered predictions for Qwen3-8B

Committed **before** any 8B training run. Not edited afterwards.

## What is being carried up

At 0.6B, on one gauge orbit of the vanilla Kaiming draw -- every gauge
invariant identical, only the frame moving:

| optimizer | norm it descends in | orthogonally invariant? | measured spread |
|---|---|---|---|
| SGD | Frobenius | yes | **0.00001** nats |
| Muon | spectral | yes | **0.00026** nats (1.0x null) |
| AdamW | elementwise max | **no** | **0.00222** nats (8.2x null) |

and the effect is structurally zero at rank 1, growing with the dimension of
`O(r)` modulo signed permutations: 0.00000, 0.00044, 0.00160, 0.00402 at
r = 1, 4, 16, 64 (lr = 1e-4).

## The scale prediction is that it does NOT grow

Measured training-free on the pretrained models, the span of `Lambda_1` over
the gauge orbit of a fixed random `A` (r = 32) is:

| model | `d` | eigenframe | flat | max | reach |
|---|---|---|---|---|---|
| Qwen3-0.6B | 1024 | 0.051 | 0.215 | 0.231 | **4.57x** |
| Qwen3-1.7B | 2048 | 0.050 | 0.210 | 0.228 | **4.57x** |

The dose available to the intervention is **flat in scale** to three digits.
That is a real prediction and it cuts against the convenient story: unlike the
outlier-feature statistics in `scaling_predictor.json`, which sharpen with
scale, the frame's headroom does not. So:

**P1.** The AdamW frame spread at 8B is **comparable to 0.6B, not larger** --
between 0.001 and 0.005 nats, and specifically NOT an order of magnitude
above 0.00222. If it grows past 0.01 nats the flat-reach prediction is wrong
and the scaling predictor needs rethinking; if it collapses below the null the
phenomenon is small-model-only and the paper must say so.

**P2.** `frame0` beats `kaiming` at 8B, by at least the measurement null. This
is the practical claim and it is the one that has to survive scale.

**P3.** SGD at 8B is flat to within 1e-3 nats across the same three frames.
This is not a fitted expectation -- the trajectories are gauge-equivalent by
derivation -- so a spread above 1e-3 means an implementation fault, not a
scientific finding, and would invalidate the corresponding 0.6B panel too.

**P4.** The ordering at 8B follows `Off_g` (off-diagonal mass of
`M_g = A C_g A^T`), which is the within-orbit predictor at 0.6B (r = +0.859),
with the minimum at `frame0` where `Off_g = 0` exactly. `Off_g` was arrived at
**post-hoc** at 0.6B -- the pre-registered candidate named the wrong matrix,
`M_x` instead of `M_g`, and was falsified -- so 8B is its first genuinely
out-of-sample test.

## Panel

Qwen3-8B-Base, GSM8K SFT, r = 16, alpha = 32, 300 steps, fp32 master weights,
all seven linear module types, identical to the 0.6B protocol.

* AdamW x {kaiming, frame0, frame1} x {1e-4, 2e-4, 3e-4, 5e-4} = 12 runs
* SGD x {kaiming, frame0, frame1} x {0.03, 0.1} = 6 runs

18 runs. Arch-pinned, because the cross-architecture offset (4.2e-4 nats) is
larger than the effect.

## What would falsify the paper's central claim

P3 failing. Everything else is a matter of size; P3 is the derivation itself.

---

## Reading log (appended as the panel filled; predictions above unedited)

At 9 of 18 runs, `frame0` beat `kaiming` at **every shared learning rate**
(1e-4: -0.00035, 2e-4: -0.00409, 3e-4: -0.00219), the same pattern as 0.6B.
No verdict was recorded, because both conditions were still improving toward
the top of the grid and a comparison read off a truncated grid is worth nothing
-- that error already produced one reversal in the zoo panel. `analyze_8b.py`
now refuses to print P1/P2/P4 until every condition covers the same rungs with
an interior optimum.

The 8B optimum turned out to be above the grid carried over from 0.6B
(5e-4 beat 3e-4 for `kaiming`, where 0.6B peaked at 3e-4), so the grid was
extended upward to 1e-3 and 2e-3.

### fp32 rerun (the bf16 panel was precision-masked)

The bf16 8B panel gave AdamW spread 0.00256 against a bf16 SGD floor of ~9e-4
(ratio ~3) and an apparent reversal -- but bf16 raises the floor and compresses
the conditions, so it was rerun in true fp32 (amp=none). The fp32 ladder is
complete (18/18) with its own SGD covariance floor (6.3e-5) and AdamW
signed-permutation floor.

At 7e-4 -- AdamW's optimum, and the one rung where all three frames survive
(kaiming and frame1 diverge at 1e-3) -- the fp32 numbers are:

    kaiming 0.39337   frame0 0.39361   frame1 0.39461     spread 0.00123

kaiming is **best**, the reverse of the frame0-best ordering on 0.6B/OLMo/Llama.
That is a candidate scale reversal. But AdamW's own floor at 7e-4, from three
signed-permutation seeds, is {0.39212, 0.39813, 0.39254}: two cluster at 0.00042,
one lands 0.00601 out. The frame spread is 2.95x the tight floor (reversal real)
or 0.21x the full floor (washed into noise) depending entirely on whether the
third seed is a rare edge-catch -- 7e-4 sits near the divergence edge, where
AdamW's reduction-order chaos amplifies. n=3 cannot tell. Two more seeds
(signperm4/5 @ 7e-4) are running; the verdict is held until n=5 stabilises the
floor. This is the ninth read held on the floor discipline, and the first where
the held number is the difference between "we found a scale reversal" and "the
effect washes out at 8B" -- both honest scope statements, and not a narrowing of
the map, which stands on 0.6B/OLMo-1B/Llama-3B.
