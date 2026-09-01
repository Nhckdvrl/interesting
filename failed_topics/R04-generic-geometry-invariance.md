# R04 — Generic Riemannian / Parameterization-Invariant LoRA

**Status:** REJECTED as a primary topic  
**Date:** 2026-09-01

## Tempting idea

Once NoRA is interpreted through the geometry of

\[
P=A^\top A,
\]

it is natural to ask whether the Euclidean factorization \(BA\) is the wrong optimization geometry altogether. One could optimize low-rank factors on a manifold, enforce orthogonality, or design updates invariant to equivalent factorizations.

## Why the idea is attractive

The factorization has obvious non-identifiability:

\[
BA=(BQ)(Q^{-1}A)
\]

for invertible \(Q\). If ordinary optimizers react differently to equivalent factorizations, a coordinate-free optimizer sounds principled.

## Closest-work collision

This territory is already occupied by several strong works:

- **Riemannian Preconditioned LoRA (ICML 2024):** directly treats low-rank adaptation through Riemannian/preconditioned optimization.
- **LoRA-RITE / LoRA Done RITE (ICLR 2025):** explicitly targets invariance to low-rank factor reparameterization/scaling/rotation and designs matrix-preconditioned updates.
- **StelLA (NeurIPS 2025):** optimizes low-rank subspaces on Stiefel manifolds.
- related orthogonal/manifold PEFT methods further crowd generic claims that better low-rank geometry is the missing principle.

## Exact rejection reason

The broad statement

> LoRA should be invariant to equivalent \(A,B\) factorizations

is no longer novel enough for a new primary paper. A new optimizer with a slightly different manifold or preconditioner would be method-level novelty inside an established object.

Do not revive by merely:

- changing the Riemannian metric;
- using a Cayley/QR retraction;
- adding NoRA normalization to LoRA-RITE;
- proving another factor-gauge invariance theorem;
- extending an existing invariant optimizer to a few more tasks.

## Important distinction from the active Representation Gauge topic

This rejection applies to **adapter-factor gauge**:

\[
(A,B)\to(Q^{-1}A,BQ),
\]

where the backbone representation is unchanged.

The active `02-representation-gauge` project asks a different question: if the **backbone hidden representation itself** is transformed by an exact function-preserving change of basis, should the PEFT algorithm produce an equivalent adaptation trajectory?

That symmetry acts on the pretrained representation coordinates, not merely on the internal factorization of a fixed adapter. The active topic survives only while this distinction remains clean under literature audit.

## What would be required to reconsider

Reconsider generic geometry only if a new exact symmetry is identified that existing Riemannian/LoRA-RITE formulations provably fail, and that symmetry creates a substantial empirical anomaly in functionally equivalent models.

Otherwise use Riemannian and invariant methods as baselines for the active gauge project rather than as a new topic.