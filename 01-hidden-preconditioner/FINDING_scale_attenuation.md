# The frame effect's margin over the floor attenuates with model scale

Status: **live finding, recorded before any narrative edit.** Surfaced 2026-09-04
while giving the section-3 map its third training family (`llama_map`).

## What was found

Section 6 ("the classes are real") reports AdamW's frame spread against its own
reduction-order floor at 6-11x on three families. Re-measuring Llama-3.2-3B with
a cleaner probe exposed that the Llama number is not robust:

| panel | probe_bs | micro_bs | AdamW spread @ lr2e-4 | own floor | ratio |
|---|---|---|---|---|---|
| `llama_fp32` (used by section 6) | 4 | 2 | 0.00239 | 0.00040 | 6x |
| `llama_map` (this replication)   | 8 | 4 | 0.00057 | 0.00040 | 1.4x |

Same model, same learning rate, same precision (true fp32, amp none), same
r/alpha/steps/targets. The only differences are the batch sizes -- and probe_bs
sets how accurately the frame conditions are constructed, since frame0/frame1 are
built from a probe gradient. The **better-probed** panel gives the **smaller**
ratio, which means part of section 6's 0.00239 spread was probe noise inflating
the frame separation, not frame signal.

## The pattern, by scale

Ordering the own-floor ratios by model size:

    Qwen3-0.6B   11x
    OLMo-2-1B    10x
    Llama-3.2-3B  1.4x (clean) .. 6x (noisier probe)
    Qwen3-8B     marginal: condition ordering scrambles across lr
                 (frame1 best at 3e-4, kaiming at 5e-4/7e-4), own floor running

The cleanest evidence is within-family: Qwen at 0.6B is 11x, Qwen at 8B is at or
below its own noise. AdamW's reduction-order floor grows with the model
(0.00021, 0.00016, 0.00040, and larger at 8B) while the frame signal does not, so
the margin between the classes narrows with scale.

## Why this does NOT retract the map (section 3)

The map is a statement about **relative** class separation: AdamW and Lion see the
frame, SGD / Muon / matprec are blind, and the gap between them is what the map
asserts. On Llama-3B, AdamW's spread (0.00057) is still ~50x the blind optimizers'
(~1e-5) -- the decisive test is the `llama_map` blind arm (muon/sgd/matprec, still
filling in). If AdamW-spread >> blind-spread there, the map holds on the third
family regardless of AdamW's modest ratio against its *own* floor. The
attenuation is in the absolute room over the noise, not in the ordering.

## What it changes

* Section 6's Llama row should not stand on the single 6x number; the honest
  reading is 1.4-6x, near floor, and the section should say the margin attenuates
  with scale rather than implying a flat 6-11x across sizes. **Framing is the
  user's call; this file only records the measurement.**
* It turns the 8B "does the effect survive scale?" question from a yes/no into a
  quantitative one with a mechanism (the floor grows, the signal does not).

## Open, before this is fully settled

1. `llama_map` blind arm (muon/sgd/matprec) -- confirms the map holds on Llama.
2. Qwen3-8B AdamW own floor (3 signperm cells at lr 7e-4, running) -- fixes the
   8B ratio.
3. A Lambda_1-normalized spread would separate "scale attenuation" from "frame
   draw variance" cleanly; the P-stats logged (tr_P, diag_imbalance, crosstalk)
   proxy AA^T, not the gradient frame, so this needs a recompute, not a reread.

## Update 2026-09-04: the 8B floor is measured, and 8B sits at floor

AdamW's own floor at 8B (three signed-permutation seeds at the tuned lr 7e-4) is
now in hand, and the 8B verdict is: **the frame effect is at or below AdamW's own
reproducibility floor -- not separable at 8B in a matched 300-step budget.**

    metric            frame spread   AdamW own floor   ratio
    final-step        0.00123        0.00601*          0.2x
    tail-mean (last4) 0.00218        0.00175           1.2x

    * the final-step floor is inflated by one seed's snapshot: signperm2 lands at
      0.39813 vs 0.39212 / 0.39254, but its eval curve tracks the other two
      throughout (at step 275 it is the LOWEST) -- the final-step caught it on an
      up-bounce.  The tail-mean removes that; both readings put the frame spread
      at or under the floor.

The **mechanism**, now clean across all four scales: the frame spread stays
~0.001-0.002 at every size; it is the FLOOR that grows --

    OLMo-2-1B    floor 0.00016   (tuned lr 3e-4, comfortably stable) -> 10x
    Llama-3.2-3B floor 0.00040   (tuned lr 2e-4)                     -> 1.4x
    Qwen3-8B     floor 0.0018    (tuned lr 7e-4, one rung below the
                                  1e-3 that diverges)                -> ~1x

-- because on a bigger model the useful learning rate sits nearer the stability
edge, where AdamW's trajectory decoheres between gauge copies (reduction-order
chaos) and the floor rises to meet the signal. So the honest scope statement is
not "the effect vanishes at scale" but "**the effect is cleanly resolved on the
two small models (0.6B 11x, 1B 10x) and is not separable above the panel's own
floor at 3B and 8B in a matched budget**"; whether a longer run at a lower stable
lr would re-separate it at scale is untested and not claimed.

This keeps the section 3 MAP intact where it is read -- the map is the relative
gap (AdamW/Lion spread ~2e-3 vs blind spread ~5e-5, ~40x on 0.6B and OLMo), and
the ratio-to-own-floor is not the right lens for the blind optimizers anyway
(their floor is near zero, so the ratio is unstable; read the raw spread). What
scale changes is how far AdamW's own floor sits below its own signal, not the
ordering of the classes.

## Update 2026-09-04 (2): the Llama attenuation is AdamW-specific, not a class collapse

The llama_map blind arm (muon done, matprec nearly, sgd filling in) lets the map
be read on the third family by raw spread at each optimizer's stable operating
point:

    Lion     0.00422   sees, strongly (~11x the blind class)
    AdamW    0.00057   attenuated -- into the blind range
    Muon     0.00037   blind (2.4x its own floor; fp32 Newton-Schulz)
    matprec  0.00009   blind, clean

So on Llama-3B AdamW does NOT cleanly separate from Muon (0.00057 vs 0.00037,
~1.5x), where on OLMo it was 0.00175 vs 0.00004 (~40x).  But **Lion, the other
sighted optimizer, still sees strongly** (0.00422, similar to its OLMo 0.00388) --
the sighted class is carried by Lion on this family, and only AdamW's magnitude
has collapsed toward its floor.

This matters for how the map's third family is stated.  The class BOUNDARIES are
structural and proven analytically in float64 (12 orders, backbone-independent);
what a training family tests is the observable magnitude.  On Llama-3B the sighted
class is confirmed by Lion and the blind class by muon/matprec/sgd, but AdamW's
observable frame-dependence has attenuated -- an effect-SIZE statement (AdamW's
RMS normalization washes the frame out at scale where Lion's sign rule does not),
not a boundary statement.  The honest third-family claim is therefore "Lion
confirms the sighted class and the blind class is clean; AdamW's magnitude
attenuates," not "the map holds identically on three families."

Why AdamW and not Lion: AdamW divides by sqrt(v), and near the stability edge the
second-moment estimate is itself frame-scrambled and noisy, so the ratio m/sqrt(v)
loses the frame structure; Lion's sign(m) keeps it.  Testable, not yet tested.

## Update 2026-09-04 (3): correction -- the split HOLDS on Llama; Update (2) used the wrong criterion

Update (2) called AdamW "attenuated into the blind range" on Llama because it
compared AdamW (0.00057 at 2e-4) to muon (0.00037 at 3e-4).  That was the wrong
comparison: 3e-4 is near muon's OWN stability edge, where fp32 Newton-Schulz
decoheres and inflates muon's spread; it is not where muon is blind.

The clean criterion -- the same one that settles matprec on OLMo -- is whether the
frame spread VANISHES at low, stable lr.  Per-lr on Llama:

    adamw    1e-4: 0.00089   2e-4: 0.00057   3e-4: 0.00076   (holds; SEES)
    lion     3e-5: 0.00122   1e-4: 0.00422                   (holds; SEES)
    muon     1e-4: 0.00001   3e-4: 0.00037   1e-3: 0.00427   (vanishes at 1e-4; BLIND)
    matprec  1e-3: 0.00000   3e-3: 0.00010   1e-2: 0.00147   (vanishes at 1e-3; BLIND)

At the proper low lr the split is CLEAN: AdamW 0.00089 vs muon 0.00001 is ~90x,
the same order as Qwen and OLMo.  So the map's third family holds -- sighted
(adamw, lion) hold nonzero spread at low lr, blind (muon, matprec) vanish.

Two comparisons were being conflated:
  * AdamW vs the BLIND class (the map)         -> ~90x at low lr, holds cleanly;
  * AdamW vs its OWN floor (the effect size)   -> ~2x here vs 10x on OLMo, attenuated.

Both are true.  The MAP split is preserved on all three families; what attenuates
with scale is the effect SIZE (AdamW's room over its own reproducibility floor),
not the class boundary.  Update (2)'s "into the blind range" is retracted; the
attenuation is real but it is the second comparison, not the first.

(Found by cross-check against a parallel analysis that used the low-lr criterion
directly -- recording the error and its correction rather than quietly editing.)

## Update 2026-09-04: the 8B floor is in (n=5), and it settles the boundary

The "marginal, own floor running" 8B line above is now resolved. AdamW's five
signed-permutation floor seeds at 7e-4 are {0.38725, 0.39212, 0.39254, 0.39484,
0.39813} -- full range 0.01088, middle-three range 0.0027. The three frame
conditions {kaiming 0.39337, frame0 0.39361, frame1 0.39461} fall *inside* that
scatter, interleaved with the gauge-equivalent seeds; the best seed beats the best
frame by 0.006. Frame spread 0.00123 is 0.11x the full floor (0.45x the middle-3).

So at 8B the effect is **washed out**, not merely marginal, and the earlier "kaiming
best" ordering was within-floor noise rather than a scale reversal. This confirms
the mechanism this file proposed: the frame signal is flat with scale while AdamW's
reduction-order floor grows (0.00021 -> 0.00016 -> 0.00040 -> ~0.006-0.011 at 8B),
so the margin between the classes closes and the floor overtakes the signal by 8B.
The map (relative separation) is untouched; this bounds the resolvable magnitude.
