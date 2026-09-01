# R02 — Activation/Fisher-Aware NoRA

**Status:** REJECTED as a primary topic  
**Date:** 2026-09-01

## Tempting idea

NoRA uses a task-agnostic coordinate normalization:

\[
\|A_{:,j}\|_2=1.
\]

A natural extension is to replace Euclidean normalization by a data-aware metric, for example an activation covariance \(\Sigma_x\), gradient covariance, Fisher matrix, or Hessian approximation. One might enforce something like

\[
A\Sigma_xA^\top\approx I
\]

or otherwise choose \(A\) so the early preconditioner is balanced in the geometry induced by real model activations.

## Why the idea is attractive

NoRA itself raises the question of whether coordinate-wise equality is the right notion of equality. If hidden dimensions have very different activation scales or task relevance, Euclidean unit norm may look arbitrary. Data-aware whitening seems like an obvious next step and could plausibly produce stronger optimization gains.

## Closest-work collision

This space is already crowded by several strong lines of work:

- **EVA / Explained Variance Adaptation (NeurIPS 2025):** uses activation variance to construct a better low-rank initialization/subspace.
- **LoRA-DA (ICML 2026):** explicitly develops data-aware low-rank adaptation/initialization ideas.
- **TLoRA (ACL 2026):** uses pretrained-weight and activation-covariance information to obtain task-aware low-rank structure.
- **LoRA-One (ICML 2025 Oral):** uses the one-step full gradient to identify task-relevant singular directions.
- broader Fisher/Hessian-aware and curvature-aware PEFT work already occupies the generic story that task geometry should replace uninformed random geometry.

## Exact rejection reason

A primary paper framed as

> NoRA uses Euclidean normalization; we use activation/Fisher normalization

is too close to existing data-aware initialization and task-subspace selection work. The method may work, but the scientific contribution would likely be judged as another choice of metric rather than a new principle.

The following are **not** enough to reopen the topic:

- trying a better covariance estimator;
- using Fisher instead of activation covariance;
- applying whitening per layer/head;
- adding online covariance updates;
- combining NoRA with EVA/TLoRA and reporting gains.

These are engineering/method variations within occupied territory.

## What would be required to reconsider

Reconsider only if the data-aware metric is a consequence of a **new invariance or causal principle** rather than a heuristic. For example:

1. derive the metric as the unique one required for backbone representation-gauge equivariance;
2. show that Euclidean NoRA provably changes under exact function-preserving basis transformations, while a covariant metric restores equivalent trajectories;
3. identify a task-independent but representation-intrinsic geometry that predicts when data-aware conditioning should beat Euclidean conditioning.

In that case the scientific object is not “activation-aware NoRA”; it would belong under the active `02-representation-gauge` project or another broader geometry principle.

## Still useful inside active work

Data-aware constructions remain useful as **derived baselines/remedies**:

- compare Euclidean NoRA against activation-covariance whitening under representation rotations;
- use Fisher/gradient covariance as competing explanations for which components of \(P\) matter;
- test whether task-aware alignment dominates diagonal balancing after update-magnitude matching.

Do not register the metric choice itself as a standalone project.