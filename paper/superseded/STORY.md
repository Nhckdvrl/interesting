# The story, rebuilt around what survived

Written after reading NoRA end to end, and after finding two ICLR 2026 papers
sitting in exactly our territory. It replaces the earlier framing, which led
with a prescription that the 8B and r=128 panels damaged.

## What changed my mind

`LoRA meets Riemannion` (ICLR 2026) optimises on the fixed-rank manifold and
says this "eliminates the parametrization ambiguity present in standard
Euclidean optimizers." It is the **fourth** recent work to treat LoRA's
factorisation ambiguity as a defect to remove:

| work | what it does with the ambiguity |
|---|---|
| LoRA-RITE (ICLR 2025) | makes the optimizer invariant to it |
| Balanced LoRA | projects onto `A^T A = B B^T` after each step |
| FedRot-LoRA | Procrustes-aligns it away as aggregation noise |
| **Riemannion (ICLR 2026)** | removes it by construction, optimising `ΔW` directly |

Four papers, one instinct: get rid of it. **Not one of them asks what it is
worth, or which part of it a given optimizer actually sees.** That is the gap,
and the crowding is evidence the question matters rather than evidence we are
too late.

## The opening: a concurrent paper is wrong about Muon, precisely

Riemannion states, to motivate its construction:

> "acting on the two factors separately makes Muon non-reparameterization-invariant:
> its per-factor orthogonalization depends on arbitrary **scalings or rotations**"

Half of that is right. The ambiguity group is `GL(r)` -- `(A, B) -> (SA, BS^{-1})`
-- and by polar decomposition it factors into a **rotation** part `O(r)` and a
**scaling** part. Muon is exactly invariant under the first and not the second.
Measured, 25 steps, float64, `B_0 != 0` so both factors move from step 0:

| optimizer | signed perms | **`O(r)` rotation** | **`GL(r)` scaling** |
|---|---|---|---|
| SGD | 2.2e-16 | **2.2e-16** | 9.8e-03 |
| Muon | 2.5e-09 | **4.4e-09** | 1.2e-02 |
| matrix-preconditioned Adam | 2.2e-16 | **2.2e-16** | 1.5e-02 |
| **AdamW** | 0.0e+00 | **1.7e-03** | 3.2e-02 |

Muon's rotation column is 3 million times smaller than its scaling column. The
claim is right about scalings and wrong about rotations, and the distinction is
not pedantic -- it is the whole paper.

## The organising structure

The optimizers form a **strict hierarchy of symmetry groups**:

```
        signed permutations   ⊂   O(r)   ⊂   GL(r)
              AdamW                SGD         LoRA-RITE
                                   Muon        Riemannion
                                   matrix-preconditioned Adam
```

Every optimizer is blind to everything inside its group and sensitive to
everything outside it. So *what an initialisation is* -- how many degrees of
freedom it actually has -- **depends on which optimizer will consume it**:

* under a fully invariant method, an initialiser is its `GL(r)` class;
* under SGD or Muon, it is its `O(r)` class -- the orbit, `P_0 = A^T A`;
* under AdamW, it is the orbit **plus the frame**, an extra `r(r-1)/2`
  coordinates that no invariant statistic can express.

AdamW is the one everybody uses.

## The punchline: the literature varies the frame without reporting it

Measured with no training on 20 published configurations (11 methods), using
`Lambda_1 = ||G A^T||_1^2 / (d_out r ||G A^T||_F^2)`:

* the zoo spans this coordinate **4.3x**;
* the separation is **perfect** -- every frame-based method (Kaiming, NoRA,
  NoRA-unit, ETF, BiMI) above every data-aware one (PiSSA-minor, OLoRA, EVA,
  PiSSA, gradsub, LoRA-One), no overlap;
* moving along it while holding `BA`, `P` and all nine gauge invariants fixed
  to 1e-15 changes tuned loss by **2.5x the median gap between adjacent methods
  in the zoo's own ranking**;
* and the whole NoRA family sits at vanilla's value, which is *why* our Stage-1
  audit found those conditions mutually indistinguishable.

So a LoRA initialisation result obtained with AdamW carries a component that
provably will not transfer to Muon, LoRA-RITE or Riemannion, and nobody reports
enough to say how much. Separating the two channels costs one probe pass.

## What each experiment is for

Every panel now serves one line, and the ones that came out against us serve it
too.

| section | experiment | what it establishes |
|---|---|---|
| 3. The hierarchy | 4 optimizers x 3 subgroups, float64 | the taxonomy, and the correction to Riemannion |
| 4. The frame is real | frame ladder, one orbit, all invariants fixed to 1.5e-8 | AdamW 0.00222 nats vs SGD 0.00001, Muon 0.00026, matprec 0.00022 |
| 4. …and structural | rank series | exactly zero at r=1, where `O(1)` *is* AdamW's own group; grows through r=64 |
| 5. The audit | 20 configurations, no training | 4.3x span, perfect separation, 2.5x the adjacent gap |
| 6. Consequences | zoo rotated both ways, 96 runs | 6/6 helped-or-neutral one way, 6/6 hurt-or-neutral the other |
| 6. …and transfer | the same zoo under Muon, 44 runs | the ranking survives (tau = +0.87) but ~30% of the gap does not |
| 7. Limits | 8B, r=128, probe and precision controls | both apparent failures are estimation quality, not scale |

## Section 6, answered: the ranking transfers, a third of the gap does not

We asked whether "which initialiser is best" depends on the optimizer, by
running the same six published initialisers under AdamW and under Muon. **The
answer is no**, and the paper says so:

```
AdamW: kaiming > bimi > gradsub > eva > pissa > lora_one
Muon : kaiming > bimi > gradsub > pissa > eva > lora_one
```

Kendall tau = **+0.87**, one discordant pair in fifteen, and that pair (EVA vs
PiSSA) is separated by 0.00001 nats under Muon -- noise. The orbit channel,
which every optimizer sees, dominates the ordering.

What does not transfer is the **size** of the gaps. The amount by which vanilla
Kaiming beats each data-aware method, under AdamW and under a frame-blind
optimizer:

| method | gap under AdamW | gap under Muon | lost |
|---|---|---|---|
| gradsub | +0.00943 | +0.00652 | **31%** |
| EVA | +0.01023 | +0.00722 | **29%** |
| PiSSA | +0.01140 | +0.00722 | **37%** |
| LoRA-One | +0.01177 | +0.01085 | 8% |

(BiMI is omitted: its AdamW gap is 0.00042, at the reproducibility floor, so
the ratio is meaningless.)

So about a third of what vanilla Kaiming appears to gain over data-aware
initialisation under AdamW is frame, and it is gone under an optimizer that
cannot see the frame. That is the transferability claim in its honest form: not
"your ranking is wrong", but "a third of your margin is optimizer-specific and
nobody reports which third".

## The audit replicates across model families

The punchline was measured on Qwen3-0.6B, where 97% of this project's runs sit.
It costs nothing to check elsewhere -- the coordinate needs no training -- and
it holds:

| model | `Lambda_1` span | separation |
|---|---|---|
| Qwen3-0.6B | 4.3x | **perfect** |
| Llama-3.2-3B | 3.0x | **perfect** |

Different architecture, different tokenizer, same structure: every data-aware
method below every frame-based one, and even the within-group order almost
unchanged (only PiSSA/EVA and Kaiming/NoRA swap).

## The prescription is a section, not the thesis

It holds at 0.6B in fp32 (-0.00222, 11x floor, 3/3 seeds) and bf16 (-0.00322),
on Llama-3.2-3B (-0.00086), at 1000 steps (-0.00098), at r = 128 (-0.00170),
and at 8B **once the probe is large enough** (+0.00202 with a 16-example probe,
-0.00280 with 64).

Both apparent failures turned out to be estimation, not scale. The 8B reversal
reverses back with a 4x probe. The r = 128 reversal was read off a grid whose
bottom rung held `frame0`'s optimum; extended to 1e-5, the ordering matches
r = 1..64. What is genuinely falsified is the reach law's *magnitude* an octave
out, and the lesson that a fixed-learning-rate contrast is the wrong statistic
when conditions have different optima -- both recorded in
`PREDICTIONS_r128.md`, prediction unedited.

Written as a thesis this is a wound. Written as **section 7, "when the frame
can be estimated well enough to act on"**, it is the paper being honest about
its own operating range -- and the probe control turns the 8B failure from an
embarrassment into the mechanism.

## What this is not

Not "we found a better LoRA initialiser". That claim died at r=128 and we do
not make it. GSM8K exact-match at n=200 is 0.505 vs 0.500, inside binomial
noise, and we say so.

It is: **initialisation research in LoRA has been measuring an
optimizer-dependent object, and the dependence is a clean group-theoretic
hierarchy that nobody has written down.**
