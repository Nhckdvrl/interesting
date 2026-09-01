# R05 — Adam-Aware NoRA / Optimizer-Moment Matching

**Status:** REJECTED as a primary topic  
**Date:** 2026-09-01

## Tempting idea

NoRA's clean early-time interpretation

\[
\Delta W\approx-\eta GA^\top A
\]

is most transparent under SGD. But modern LLM fine-tuning usually uses AdamW, where factor gradients are transformed through first- and second-moment states. A natural question is:

> What does NoRA actually normalize once Adam's adaptive state is included?

This suggests deriving the exact effective merged-weight dynamics under Adam and designing an optimizer-aware normalization or state correction.

## Why the idea is attractive

Scaling \(A\) changes not only the raw gradient into \(B\), but also the second-moment accumulator. Therefore a naive “column norm = effective learning rate” interpretation may cease to be exact under Adam. This is scientifically relevant because NoRA's experiments are performed in the optimization regime used by modern fine-tuning, not an idealized SGD-only setting.

## Closest-work collision

The strongest collision is **LoFT (ICLR 2026)**, which explicitly argues that low-rank training should align not only the weight update with FullFT but also the optimizer's first and second moments. The paper develops projected optimizer-state dynamics precisely in this space.

**LoRA-RITE (ICLR 2025)** also studies adaptive optimization under equivalent low-rank factorizations and uses transformation-invariant matrix preconditioning.

**LoRA Without Regret** further discusses the interactions among factor scales, \(\alpha\), separate A/B learning rates, and Adam-like optimization.

## Exact rejection reason

A paper framed as

> NoRA's SGD derivation is incomplete, so we derive Adam-NoRA and match moments

would be too close to LoFT and existing adaptive-optimizer geometry work. Even if the derivation is new in detail, the scientific object—matching or correcting optimizer-state dynamics in low-rank training—is already occupied.

The following are not sufficient for revival:

- adding a NoRA-specific Adam correction;
- rescaling \(A\) or \(B\) according to Adam's \(v_t\);
- introducing a new per-factor adaptive LR;
- showing that one optimizer works better than another without a broader mechanism.

## Still useful inside active work

Optimizer dependence is an important **secondary mechanism test** for `01-hidden-preconditioner`.

A useful panel is:

- SGD / momentum SGD;
- AdamW;
- optionally Muon or another matrix-aware optimizer if implementation is cheap.

Track:

- raw factor gradients;
- Adam-normalized factor updates;
- merged-weight update norm/direction;
- evolution of \(P_t=A_t^\top A_t\);
- optimizer-state spectra when practical.

If a NoRA effect disappears under one optimizer after matched tuning, that constrains the mechanism even though it is not itself a standalone paper.

## What would be required to reconsider

Only reconsider if there is a **large, reproducible optimizer-dependent anomaly** that existing moment-matching theory does not predict—for example, method rankings reverse across optimizers even after matched merged-weight trajectories in the early phase—and a new invariant or predictive law explains the reversal.

Without that broader phenomenon, keep Adam analysis subordinate to the active causal-anatomy project.