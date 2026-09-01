# R01 — Off-Diagonal / Crosstalk Optimization

**Status:** REJECTED as a primary topic  
**Date:** 2026-09-01

## Tempting idea

Normalized NoRA argues that the early hidden preconditioner

\[
P=A^\top A
\]

contains useful structure. Its diagonal controls coordinate-wise own-update gain, while off-diagonal entries induce cross-coordinate crosstalk. A natural follow-up is therefore:

> Keep the NoRA diagonal normalization, then design a better \(A\) whose columns have lower coherence / smaller off-diagonal \(A^\top A\), so low-rank adaptation preserves gradients more faithfully.

Possible implementations include orthogonalized rows, equiangular frames, tight frames, coherence penalties, spectral regularizers, or explicit minimization of off-diagonal energy.

## Why the idea is attractive

It follows directly from NoRA's decomposition and seems to give a clean extension from

\[
\operatorname{diag}(P)
\]

to

\[
P-\operatorname{diag}(P).
\]

It also appears cheap to test and could plausibly improve convergence without inference overhead.

## Closest-work collision

The main collision is the ICML 2026 work *Towards Understanding the Dynamics of Low-Rank Adaptation*. That line of work analyzes the low-rank update subspace through objects such as

\[
A^\top(AA^\top)^\dagger A
\]

and explicitly studies which initial low-rank geometry best preserves full-gradient information. In particular, it derives an **Equiangular Tight Frame (ETF)** construction and argues for its optimality under the stated ignorance assumptions.

This occupies the broad scientific story:

> choose the initial low-rank frame / coherence structure so that the projected gradient loses less information.

Related manifold and orthogonality work (e.g. StelLA and other Stiefel / subspace methods) further crowds generic claims that orthogonality or reduced coherence is the missing ingredient.

## Exact rejection reason

A paper whose central claim is only

> NoRA fixes the diagonal; we additionally fix the off-diagonal

would be too incremental relative to existing frame/subspace optimization work. Even strong empirical gains would invite the reviewer response that the method is a variant of already-established optimal-frame or orthogonal-subspace initialization.

The novelty problem is not solved by:

- using a different coherence penalty;
- replacing ETF with Hadamard / random orthogonal / spherical code matrices;
- normalizing rows and columns simultaneously;
- adding a dynamic orthogonalization step;
- showing a few more LLM benchmarks.

Those change the method, not the scientific object.

## What would be required to reconsider

This topic can only be revived if a new phenomenon shows that **off-diagonal structure has a causal role not captured by existing subspace-information arguments**. Examples that might qualify:

1. two \(A\) matrices with matched rank, spectrum, trace, diagonal, and gradient-subspace overlap but different off-diagonal sign/topology produce systematically different optimization trajectories;
2. crosstalk predicts a new failure mode such as forgetting, stochastic instability, or layer interference that ETF-style gradient-preservation theory does not address;
3. a theorem identifies an invariant of crosstalk that survives basis changes and predicts behavior across optimizers/tasks.

Without such a new object, do not revive.

## Still useful inside active work

This rejected direction remains an important **control family** for `01-hidden-preconditioner`:

- match \(\operatorname{diag}(P)\) while changing off-diagonal structure;
- compare Gaussian, BIMI, ETF, random orthogonal/frame constructions;
- measure off-diagonal Frobenius energy, coherence, spectrum, and task-gradient overlap;
- use these interventions to test whether NoRA's diagonal claim survives stronger controls.

The important distinction is that these experiments serve **causal decomposition of \(P\)**, not a standalone “better frame LoRA” paper.