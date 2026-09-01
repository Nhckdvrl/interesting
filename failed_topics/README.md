# Failed / Rejected Topic Registry

**Purpose:** preserve negative search knowledge so future topic-search rounds do not repeatedly rediscover and re-register already-rejected NoRA follow-ups.

This folder contains **ideas that were considered and rejected as primary ICML / ICLR / NeurIPS projects**. Rejection does **not** necessarily mean the scientific statement is false. Most entries are rejected because the standalone story is too close to prior work, too narrow, or mismatched to the local compute budget.

## Rule for future search rounds

Before registering any new NoRA-derived topic, search this folder first. A rejected topic may be revived only if there is a **materially new scientific object, new anomaly, or new causal result** that changes the novelty boundary. Renaming the method, changing the normalization formula, swapping one covariance estimate for another, or adding more benchmarks is not sufficient.

Every rejected-topic file records:

1. the tempting original idea;
2. why it follows naturally from Normalized NoRA;
3. the closest work that occupies the space;
4. the exact reason it is not selected;
5. what would be required to reconsider it;
6. which parts remain useful as baselines or sub-analyses.

## Registry

| ID | Rejected topic | Primary rejection reason | Status |
|---|---|---|---|
| R01 | [Off-diagonal / crosstalk optimization](R01-offdiagonal-crosstalk.md) | novelty collision with ETF/frame/subspace dynamics work | **REJECTED** |
| R02 | [Activation/Fisher-aware NoRA](R02-data-aware-normalization.md) | crowded data-aware initialization/conditioning space | **REJECTED** |
| R03 | [Magnitude-only explanation of NoRA](R03-magnitude-only.md) | necessary control, but standalone story occupied by LoRAM + tuning audits | **REJECTED AS PRIMARY TOPIC** |
| R04 | [Generic Riemannian / invariant LoRA](R04-generic-geometry-invariance.md) | occupied by Riemannian LoRA, LoRA-RITE, StelLA and related geometry work | **REJECTED** |
| R05 | [Adam-aware NoRA / moment matching](R05-adam-moment-matching.md) | direct collision with LoFT / adaptive-optimizer geometry | **REJECTED** |
| R06 | [Early critical period of LoRA](R06-early-critical-period.md) | too close to Stable-LoRA and LoRA-One unless a new universal law appears | **REJECTED AS STANDALONE TOPIC** |
| R07 | [Forgetting-only NoRA paper](R07-forgetting-only.md) | consequence-level story is too narrow | **REJECTED** |
| R08 | [Tiny-rank RL / intrinsic supervision dimension](R08-tiny-rank-rl.md) | 2025–2026 literature is moving quickly; RL-heavy validation is costly | **DEFERRED / REJECTED FOR THIS REPO** |
| R09 | [Large-scale low-rank pretraining](R09-large-scale-pretraining.md) | core evidence would exceed the 4-GPU budget | **REJECTED FOR COMPUTE** |

## Relationship to active topics

Some rejected ideas remain mandatory controls inside active projects:

- **R03 magnitude-only** is a required matched control for `01-hidden-preconditioner`.
- **R04 invariance** is rejected only in its generic adapter-factor form; the active `02-representation-gauge` project studies a different symmetry: **backbone representation gauge**.
- **R05 Adam moments** can be a secondary mechanism test in `01-hidden-preconditioner`, but cannot be the paper's headline.
- **R06 early critical period** may become a sub-analysis if active experiments reveal a sharp, predictive transition.
- **R08 supervision dimension** may be revisited in a separate project only if a cheap, task-controlled diagnostic emerges that does not require a large RL campaign.

## Resurrection gate

A rejected topic can move back to the active register only if **all** of the following are documented:

- a new one-sentence anomaly not already explained by the cited closest work;
- a scientific object broader than a method tweak;
- a novelty audit showing a clear distance from the collision papers recorded here;
- at least three interpretable outcome branches;
- a decisive pilot executable with <=4 GPUs;
- explicit evidence that the revived story cannot be reduced to learning-rate, batch-size, update-magnitude, or already-known initialization effects.

If these conditions are not met, keep the idea here and use it only as a baseline, ablation, or negative-control component.