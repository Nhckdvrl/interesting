# R08 — Tiny-Rank RL / Intrinsic Supervision Dimension

**Status:** DEFERRED / REJECTED for this repository's current agenda  
**Date:** 2026-09-01

## Tempting idea

Recent work suggests a striking difference between supervised fine-tuning and RL-style adaptation: RL/RLVR can sometimes obtain large gains with extremely few trainable parameters or very low-rank update trajectories. This motivates a broad question:

> What determines the number of adaptation degrees of freedom required by a supervision signal?

One could compare SFT, answer-only supervision, preference learning, process rewards, and outcome RL on the same task, and relate the minimal required rank to the effective dimension of per-example functional gradients.

## Why the idea is attractive

This has high conceptual upside. It could move beyond LoRA and ask whether supervision type itself determines an intrinsic adaptation dimension. A successful predictive law would be much broader than NoRA.

## Closest-work / timing risk

The space is moving extremely quickly. Recent 2025–2026 work already reports:

- very small or nearly rank-1 effective RL/RLVR update structure;
- strong RL gains using tiny numbers of trainable parameters;
- empirical differences between SFT and RL in required low-rank capacity;
- analyses of low-dimensional RL trajectories and policy updates.

This means a project framed only as

> RL needs lower rank than SFT

is already too late.

A viable version would need a **predictive, task-controlled law** for supervision intrinsic dimension, not another rank sweep.

## Compute reason for rejection here

A convincing causal paper would likely require several expensive axes at once:

- multiple supervision formulations;
- multiple ranks / parameter budgets;
- several RL seeds because variance is high;
- rollout generation and reward computation;
- enough tasks/models to argue the phenomenon is not math-specific.

That makes the central evidence substantially more expensive than the active NoRA-derived projects. Under the repository rule that the paper should be defensible with at most 4×A100/PRO6000 at one time, this is a poor first bet unless a very cheap diagnostic is found.

## Exact rejection reason

This topic is not rejected because the question is weak. It is rejected because:

1. the easiest empirical claim is already crowded;
2. the novelty bar has moved to a general predictive law;
3. establishing that law robustly would likely require an RL-heavy experimental campaign beyond the preferred budget/risk profile;
4. its connection to NoRA is weaker than the active topics centered directly on \(P=A^\top A\).

## What would be required to reconsider

Reconsider if a cheap diagnostic can predict required adaptation rank **before training**, for example an effective dimension of task/supervision gradients measured on a small sample:

\[
d_{\mathrm{eff}}=\frac{(\operatorname{tr} C)^2}{\operatorname{tr}(C^2)}
\]

with \(C\) some function-space or parameter-space gradient covariance.

A strong revival would show, on the **same prompts/tasks** under different supervision formulations, that this quantity predicts the minimal rank needed to recover a fixed fraction of FullFT performance. The method must work with a small pilot and not require dozens of full RL runs to discover the pattern.

## Still useful

Keep this direction in mind when interpreting NoRA under RL. If active NoRA experiments reveal that the hidden preconditioner behaves qualitatively differently under sparse scalar rewards, that may motivate a fresh, narrower entry with a stronger direct connection.