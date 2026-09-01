# R09 — Large-Scale Low-Rank Pretraining

**Status:** REJECTED for compute / execution mismatch  
**Date:** 2026-09-01

## Tempting idea

Normalized NoRA reports striking low-rank pretraining behavior: standard low-rank training can become unstable or underperform badly, while normalized initialization closes a large fraction of the gap to full-rank training. This naturally suggests a larger project:

> Can the hidden-preconditioner view explain when low-rank parameterization can replace full-rank pretraining at scale?

Possible variants include scaling laws over width/rank/tokens, stability boundaries, and compute-optimal low-rank pretraining schedules.

## Why the idea is attractive

The scientific and systems upside is large. If low-rank pretraining can reliably match full-rank training with lower optimizer-state or communication cost, the practical impact could exceed ordinary PEFT. The pretraining results in NoRA also provide a direct mother-paper connection.

## Why it is not selected here

A credible ICML/ICLR/NeurIPS paper making **pretraining-scale claims** needs more than a small toy model. It would likely require:

- several model sizes;
- several ranks;
- long token budgets;
- multiple seeds near instability boundaries;
- careful throughput/memory accounting;
- enough total compute to distinguish optimization failure from undertraining.

The local constraint is at most **4×A100 80GB or 4×RTX PRO 6000 Blackwell 96GB at one time**. Although small pretraining pilots are possible, a paper whose main claim is a scaling law or a broadly valid replacement for full-rank pretraining would require a much larger total compute budget than the repository is designed around.

## Exact rejection reason

This is a **budget rejection**, not a novelty rejection.

Do not revive by running one 300M model for a modest number of tokens and extrapolating to large-scale pretraining. That would create a weak paper whose central claim outruns the evidence.

Likewise, do not turn the project into a shallow benchmark of many tiny models merely to satisfy a “scaling” narrative.

## What would be required to reconsider

Reconsider only if one of the following becomes available:

1. substantially more compute / cluster access;
2. a theorem or diagnostic that makes the central claim testable without large-scale pretraining;
3. an externally released pretraining checkpoint suite that lets us analyze controlled low-rank/full-rank trajectories without rerunning all training;
4. a sharply local phenomenon—e.g. an early instability predictor—that can be established on small models and whose claim does not require extrapolating to foundation-model scale.

In case (4), register the local phenomenon as a new topic rather than reviving “large-scale low-rank pretraining” wholesale.

## Still useful inside active work

Small pretraining experiments can serve as **external-validity checks** for `01-hidden-preconditioner` after the SFT mechanism is established. They should test whether the same causal decomposition of \(P\) appears in a different training regime, not carry the main paper claim.