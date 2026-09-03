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
