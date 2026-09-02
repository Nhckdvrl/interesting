# Scope: what the claim needs, and how much work that is

The right question is not "how big are the mother paper's models". It is
**what would have to be true for a sceptic to believe the claim**, and then
which experiment settles each of those — model sizes fall out of that, they do
not drive it.

## What the claim is

> Which reparameterisation symmetries an optimizer respects decides what an
> initialisation *is*. LoRA's `O(r)` gauge frame is a real, prescribable
> coordinate for AdamW and provably vacuous for SGD, Muon and any
> matrix-preconditioned method.

## What has to be true, and what settles each

| # | must be true | what settles it | status |
|---|---|---|---|
| 1 | the derivation is exact | algebra + float64 checks | **done**, 1e-15 |
| 2 | the effect is not noise | seeds, a signed-permutation floor, monotonicity at every lr | **done** (3/3 seeds, monotone at every rung); floor in flight |
| 3 | it is optimizer *structure*, not an Adam quirk | ≥3 optimizers spanning preconditioner structures | **done** (SGD / Muon / AdamW); matrix-preconditioner arm in flight |
| 4 | it scales with the **symmetry group**, not with anything else | a **rank** sweep — this is our structural axis | **done** r = 1/4/16/64; r = 128 in flight with a committed prediction |
| 5 | it is not one model's activation geometry | ≥2 architecture families | Llama-3.2-3B in flight |
| 6 | it is not one task's gradient geometry | ≥3 tasks with different gradient structure | MetaMath, CodeFeedback, Dolly in flight |
| 7 | it survives deployment scale | one large model | Qwen3-8B, lr grid being bracketed |
| 8 | it moves something people act on | accuracy, not nats | GSM8K exact-match in flight |

**Rank, not parameter count, is the axis our effect is defined on** — the
measured effect is proportional to `log` of the frame reach, which grows with
`dim O(r)` minus the signed permutations. A 70B model at r = 16 would say less
than a 0.6B model at r = 128. That is a fact about this claim, not a convenient
excuse, and it is why the rank ladder gets the same care as the scale ladder.

## The sharpening that row 3 produced

Testing row 3 properly changed the theory. The first rule was about norm
geometry. The tighter one is about the preconditioner:

| preconditioner | example | gauge-covariant? |
|---|---|---|
| none | SGD | yes — measured 1e-5 nats |
| orthogonalised | Muon | yes — measured 2.6e-4 |
| full matrix on the `r` side | Shampoo; **LoRA-RITE** | yes — verified 9.7e-17 over a 30-step trajectory |
| **diagonal** | **AdamW** | **no** — measured 2.2e-3 |

**The frame is visible exactly when the preconditioner is diagonal.** Adam's is
the only diagonal one in that list and the only one that sees the frame.

This also explains LoRA-RITE (ICLR 2025) rather than competing with it: its
transformation-invariant preconditioner on the low-rank side is precisely the
structure that removes the coordinate Adam was responding to. We implement that
structure in isolation (`common/matprec.py`, `r x r` matrices, negligible cost)
and run the frame ladder against it.

## For calibration: what NoRA runs

Largest model **3B** (Llama-3.2-3B for SFT, DeepSeek-R1-Distill-Qwen-1.5B for
RL, custom L24-D1024 stacks for pretraining). So parameter count is not where we
are short — our 8B exceeds it. What NoRA has is **breadth**: three regimes,
three data sources, ~15 benchmarks, all accuracy. Rows 5, 6 and 8 above are
what close that, and they are chosen because the claim needs them, not because
NoRA has them.

## Deliberate omissions, stated not hidden

**RL and pretraining-from-scratch.** NoRA covers both. We do not, under the
standing constraint against large RL sweeps. The claim under test is derived
rather than fitted and does not depend on the regime, so an RL arm would show
that it transfers, not that it holds — but its absence is a limitation and is
listed as one in `NARRATIVE.md`.
