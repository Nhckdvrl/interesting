# R07 — Forgetting-Only NoRA Paper

**Status:** REJECTED as a primary topic  
**Date:** 2026-09-01

## Tempting idea

Normalized NoRA reports better retention / less catastrophic forgetting alongside stronger adaptation. This invites a focused follow-up:

> Does balancing the hidden preconditioner preserve pretrained capabilities better than standard LoRA?

One could fine-tune on narrow tasks, then evaluate broad retained capabilities, representation drift, or layerwise function change.

## Why the idea is attractive

The stability–plasticity tradeoff is important, easy to communicate, and NoRA appears to improve both sides simultaneously. If true broadly, it could have practical value for continual or domain adaptation.

## Closest-work collision

However, forgetting and representation-drift comparisons between PEFT and FullFT are already extensively studied. PEFT-Arena and multiple LoRA-vs-FullFT representation studies examine retention, drift, rank, and adaptation behavior. Continual-learning literature also contains many regularization-based methods whose central object is catastrophic forgetting.

A NoRA-specific forgetting paper would therefore sit at the level of a **consequence** rather than a new mechanism.

## Exact rejection reason

The story

> NoRA forgets less than LoRA

is too narrow for the target venue unless it reveals a new causal principle connecting the geometry of \(P=A^\top A\) to retained function.

Benchmarking more old tasks, adding representation similarity metrics, or extending to more domains does not by itself solve the novelty problem.

## What would be required to reconsider

Reconsider only if a new mechanism appears, for example:

1. matched adaptation performance but systematically different forgetting can be causally controlled by a specific invariant of \(P\);
2. a predictive law links early preconditioner geometry to later function-space drift;
3. a surprising reversal shows that the same geometry that improves short-term optimization can either preserve or destroy old capabilities depending on task-gradient alignment, with a theory explaining the regime boundary.

Then the project should be framed around that mechanism, not “catastrophic forgetting” alone.

## Still useful inside active work

Forgetting is a valuable secondary outcome for `01-hidden-preconditioner` because it can distinguish “simply larger update” from “better-conditioned update.” Use it after adaptation quality is matched, not as the only headline metric.

Useful diagnostics include:

- base-task perplexity / broad benchmark retention;
- function-space KL on held-out generic prompts;
- representation drift;
- layerwise merged-update norm and direction;
- old-task performance at matched new-task accuracy.

Keep these as consequence-level evidence unless a new causal object emerges.