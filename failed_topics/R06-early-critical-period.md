# R06 — Early Critical Period of LoRA

**Status:** REJECTED as a standalone topic; retain as sub-analysis  
**Date:** 2026-09-01

## Tempting idea

Normalized NoRA reports that normalizing \(A\) **only at initialization** captures most of the benefit of continuous normalization. Because standard LoRA initializes

\[
B_0=0,
\]

we also have

\[
\nabla_A L\big|_{t=0}=0,
\]

while \(B\) moves immediately. This suggests a two-timescale picture in which the initial \(A_0\) acts as an early scaffold before \(A\) itself begins to move substantially.

The tempting project is:

> Is there a short, predictable “critical period” during which the initial low-rank geometry determines the rest of the LoRA trajectory?

One could intervene on \(A\) at different training steps and look for a transition time \(t_c\).

## Why the idea is attractive

It connects directly to one of NoRA's most surprising empirical observations: initialization-only normalization is almost enough. A real critical-period law could explain why a tiny early intervention has long-term consequences and might yield a cheap training rule.

## Closest-work collision

The standalone novelty is weakened by:

- **Stable-LoRA (ICLR 2026):** explicitly focuses on the earliest training steps and stabilizes feature learning by progressively shrinking/controlling \(A\).
- **LoRA-One (ICML 2025 Oral):** emphasizes how the first full gradient identifies a task-relevant subspace and how early alignment shapes the later low-rank trajectory.
- the NeurIPS 2024 LoRA initialization-dynamics work, which already studies strong consequences of the asymmetric zero-product initialization.

Together these works occupy much of the broad claim that **the earliest LoRA dynamics are special and initialization matters disproportionately**.

## Exact rejection reason

A paper that only shows

> changing \(A\) in the first N steps matters more than changing it later

would be too predictable and too close to existing early-dynamics results. A new schedule or early normalization window would likely read as a method tweak.

The idea is therefore rejected unless it produces a **new law**, not merely a new curve.

## What would be required to reconsider

A revival requires a predictive and transferable quantity such as

\[
t_c \approx F(r,d,\eta_A,\eta_B,\alpha,\text{optimizer},\text{gradient geometry})
\]

that successfully predicts the intervention window across models/tasks and explains why initialization-only NoRA works.

Other acceptable revival routes:

1. a sharp phase transition rather than a smooth decay of intervention effect;
2. a conserved or approximately conserved subspace quantity established during the early window;
3. evidence that the critical period is a general property of bilinear adaptation and not already explained by Stable-LoRA / LoRA-One.

## Still useful inside active work

Use intervention-time sweeps as a sub-analysis in `01-hidden-preconditioner`:

- replace or renormalize \(A_t\) at step 0, 1, 5, 20, 100, ...;
- measure when matched changes to \(P_t\) cease to alter the final trajectory;
- track subspace overlap, merged-update direction, and optimizer states.

If a strong universal transition appears, then revisit this file and perform a fresh novelty audit before promoting the topic.