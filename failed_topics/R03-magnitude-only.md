# R03 — Magnitude-Only Explanation of NoRA

**Status:** REJECTED as a standalone topic; REQUIRED as a control  
**Date:** 2026-09-01

## Tempting idea

NoRA normalizes the columns of \(A\), which changes

\[
\operatorname{tr}(P)=\alpha^2\|A\|_F^2
\]

and therefore changes the early merged-weight update magnitude. This suggests a simple hypothesis:

> NoRA works mostly because standard LoRA initializes \(A\) at the wrong scale; column normalization merely increases the effective step size.

A straightforward project would compare NoRA with LoRA whose \(A\)-scale, \(\alpha\), \(\eta_B\), or initial merged update is magnitude-matched.

## Why the idea is attractive

This is the most important confound in NoRA. If magnitude alone explains the effect, the mechanistic interpretation of equal coordinate-wise gains becomes much weaker. The experiment is cheap and scientifically necessary.

## Closest-work collision

The standalone story is already strongly occupied by:

- **The Primacy of Magnitude in Low-Rank Adaptation / LoRAM (NeurIPS 2025):** argues that update magnitude explains much of the apparent benefit of several low-rank adaptation choices.
- **Learning Rate Matters (2026):** shows that method rankings can shrink dramatically under fair method-specific LR tuning.
- **LoRA Without Regret / Thinking Machines:** discusses parameterization relationships among initialization scale, \(\alpha\), and A/B learning rates and emphasizes fair tuning.
- **LoRA+ (ICML 2024):** already establishes the importance of asymmetric A/B learning rates.

## Exact rejection reason

A paper whose conclusion is merely

> after matching learning rate/update magnitude, NoRA does not help

would be an interesting audit of one paper but is unlikely to have sufficient width for the target venue unless it yields a broader law that unifies several existing methods.

Likewise, a paper claiming

> NoRA's real benefit is bigger update magnitude

would overlap too directly with LoRAM's central thesis.

## Mandatory role in active work

This direction is **not discarded experimentally**. It is a mandatory causal control for `01-hidden-preconditioner`.

At minimum compare:

1. vanilla LoRA;
2. NoRA / NoRA-init;
3. mean-column-norm-matched random LoRA;
4. \(\operatorname{tr}(P)\)-matched LoRA;
5. initial \(\|\Delta W\|\)-matched LoRA;
6. method-specific LR sweeps;
7. local batch-size sweeps.

The core question is whether, after matching global scale, any residual effect remains for:

- diagonal imbalance;
- off-diagonal crosstalk;
- spectrum;
- task-gradient alignment.

## What would be required to reconsider as a full project

Reconsider only if the experiments reveal a genuinely broader equivalence law, for example:

\[
(\text{init scale},\alpha,\eta_A,\eta_B,r)
\mapsto
\text{one low-dimensional control parameter}
\]

that predicts optimization across LoRA, NoRA, LoRA+, spectral initializations, and multiple optimizers better than existing magnitude/LR accounts.

A second possible revival route is a sharp regime boundary: magnitude explains NoRA in SFT but fails systematically in RL, pretraining, or adaptive optimizers for a principled reason that can be derived and causally tested.

Without such a generalization, keep this as a control, not a paper title.