# Panel B: does the gauge shift LoRA's optimal learning rate? — **No**

## The hypothesis

LoRA's optimal learning rate transfers poorly across ranks; LoRA-Muon (ICLR'26)
names it as a practical annoyance. The gauge quotient is `r(r−1)/2` dimensions,
growing quadratically in rank, carries zero function-space information, and
AdamW reads it. If the frame shifts `lr*`, and the shift grows with rank, part
of a problem the field already admits to would be gauge.

This was the one line that could have been **orthogonal** to the NeurIPS 2025
rotation result rather than a deepening of it, because it needs a factorised
parameterisation to even pose.

## The result

75 cells, five ranks, three frames, five-point grids at ~1.5× spacing so `lr*`
is located by a quadratic in log-lr rather than snapped to a rung.

| r | gauge dim | `lr*` spread across the three frames |
|---|---|---|
| **1** | **0** | **1.22×** |
| 4 | 6 | 1.03× |
| 16 | 120 | 1.13× |
| 64 | 2016 | 1.27× |
| 128 | 8128 | 1.58× |

**The `r = 1` row settles it.** At rank 1 the gauge quotient is empty — `O(1)`
is exactly AdamW's own symmetry group, `frame0` and `frame1` are the same run —
so the 1.22× measured there is the resolution of the measurement, not an effect.
Three of the four remaining ranks sit at or below it. Only `r = 128` (1.58×)
clears it, on one point.

**The hypothesis is not supported.** The frame does not measurably move `lr*`.

## Why this is recorded rather than dropped

The `r = 1` control is what makes this a conclusion instead of a failure to look
hard enough. Without it, "1.13× at r = 16, 1.58× at r = 128" reads like a small
effect growing with rank — which is exactly the story I wanted, and exactly what
I would have written.

It also means the rank-to-rank learning-rate drift belongs to μA
(*Learning Rate Scaling across LoRA Ranks*, which gives a μP-style account of it
across five domains), not to the gauge. That is the right answer and it is not
ours.

## Consequence for the paper

Panel B does not enter the main line. Under the standing rule in
`paper/OUTLINE.md` — an experiment must move a row of the requirements table —
it moved none even before it came back negative. It stays here as a recorded
negative result and as the reason the paper does not claim a connection to the
rank-transfer problem.
