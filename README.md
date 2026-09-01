# NoRA-Derived Research Agenda

**Last literature audit:** 2026-09-01  
**Mother paper:** Jiale Kang, Ziyin Yue, Zheng Zhan, Yangyi Huang, Weiyang Liu, *Normalized Low-Rank Adaptation* (arXiv:2608.31036, 2026).  
**Scope:** research questions that genuinely continue the scientific object exposed by Normalized NoRA, with an execution budget of at most **4× A100 80GB or 4× RTX PRO 6000 Blackwell 96GB at one time**.

> **Naming note.** “NoRA” is overloaded. This repository always means **Normalized Low-Rank Adaptation (Kang et al., 2026)** unless explicitly stated otherwise. ICLR 2025 already contains a different “NoRA”, meaning LoRA with Nyström initialization, in Li et al., *On the Crucial Role of Initialization for Matrix Factorization*.

## Mother proposition

For standard LoRA

\[
\Delta W = \alpha BA,
\]

with random down-projection \(A\) and zero-initialized up-projection \(B\), the earliest merged-weight update is approximately

\[
\Delta W_{\text{early}}=-\eta G P,\qquad P=\alpha^2 A^\top A,
\]

where \(G\) is the full-weight gradient. Normalized NoRA makes every column of \(A\) unit norm, thereby controlling the diagonal of \(P\); its BIMI experiment further argues that this diagonal quantity is more important than the particular crosstalk pattern. NoRA-init shows that an intervention at initialization alone captures most of the benefit.

The repository therefore does **not** treat NoRA as “another LoRA trick.” It treats NoRA as evidence that a previously hidden object — the **input-side low-rank preconditioner \(P=A^\top A\)** and its early-time dynamics — has scientific content.

## Selected topics

| Priority | Topic | Core scientific object | Score | Why it survives a failed first experiment |
|---|---|---|---:|---|
| **A1** | [01 — Hidden Preconditioner Causal Anatomy](01-hidden-preconditioner/) | causal components / equivalence classes of \(P=A^\top A\) | **93/100** | magnitude, diagonal balance, crosstalk, task alignment, curvature and optimizer interactions are separately testable; every major outcome resolves competing claims in the literature |
| **A2** | [02 — Representation Gauge](02-representation-gauge/) | backbone representation gauge vs. coordinate-dependent adapter regularization | **91/100** | large final-score variance, trajectory-only variance, or adversarial gauge sensitivity are all meaningful; exact symmetries provide a clean causal intervention |
| **A3** | [03 — Stochastic Batch Geometry](03-stochastic-batch-geometry/) | how \(P\) filters both gradient drift and minibatch diffusion | **89/100** | noise-rescue, drift-rescue, or neither each discriminates a different mechanism for the already-reported large-batch pathology |

These are **research registrations, not claims that the hypotheses are already true**. The standards below are intentionally designed so that a topic is killed early if it degrades into a narrow method tweak.

---

# Selection standard

A topic is registered only if it passes all of the following.

## 1. Mother-proposition gate

The question must be a direct consequence of at least one object exposed by Normalized NoRA:

- the \(B_0=0\) early-time asymmetry;
- the hidden preconditioner \(P=A^\top A\);
- diagonal gain versus off-diagonal crosstalk;
- the disproportionate importance of initialization;
- the gap between a coordinate-level normalization statement and the full optimization dynamics.

“LoRA is popular, therefore try X-LoRA” fails.

## 2. Scientific-object gate

The paper must name a scientific object or principle before naming a method. Examples that pass: hidden-preconditioner invariants, backbone representation gauge, LoRA-aware stochastic diffusion. Examples that fail: a new normalization formula, a new rank scheduler, or a new acronym with only benchmark gains.

## 3. One-sentence anomaly gate

Before proposing a solution, there must be a one-sentence surprising statement that a researcher outside PEFT can understand. Strong ICML/ICLR/NeurIPS LoRA papers often use exactly this structure:

- two seemingly equivalent zero-product initializations behave differently (NeurIPS 2024 initialization paper);
- a simple LR choice changes whether a method looks superior (2026 LR audit);
- functionally identical Transformer rotations can behave very differently after a non-invariant operation (QuaRot/SpinQuant style of causal setup);
- LoRA is unusually harmed by large batches even when rank is increased (Thinking Machines).

## 4. Causal-identification gate

Every headline claim must survive matched controls. At minimum, depending on topic:

- method-specific learning-rate sweeps;
- batch-size checks;
- matched initial merged-update norm / matched \(\operatorname{tr}(P)\);
- rank and training-duration controls;
- coupled seeds / transformed initializations when testing symmetries;
- a direct intervention that changes the proposed cause while holding the nearest confound fixed.

Correlation between an \(A\)-statistic and accuracy is insufficient.

## 5. Closest-work distance gate

The candidate must remain nontrivial after comparison with the strongest nearby ICML/ICLR/NeurIPS work. We explicitly treat the following as occupied territory:

- **A/B learning-rate asymmetry:** LoRA+ (ICML 2024).
- **generic Riemannian / preconditioned LoRA:** Riemannian Preconditioned LoRA (ICML 2024).
- **random-projection interpretation:** Flora (ICML 2024).
- **FT-vs-LoRA magnitude/direction decomposition:** DoRA (ICML 2024).
- **initialization asymmetry:** *Impact of Initialization on LoRA Finetuning Dynamics* (NeurIPS 2024).
- **Nyström initialization:** ICLR 2025 “NoRA” (different method/name).
- **adapter-factor gauge invariance:** LoRA-RITE (ICLR 2025).
- **one-full-gradient / gradient-subspace initialization:** LoRA-One (ICML 2025 Oral).
- **activation-variance initialization:** EVA (NeurIPS 2025).
- **update magnitude as the primary explanatory variable:** LoRAM / *The Primacy of Magnitude* (NeurIPS 2025).
- **Stiefel / generic geometric subspace learning:** StelLA (NeurIPS 2025).
- **Adam first/second-moment matching to FullFT:** LoFT (ICLR 2026).
- **early feature-learning stabilization by shrinking \(A\):** Stable-LoRA (ICLR 2026).
- **data-aware / Fisher-like initialization:** LoRA-DA (ICML 2026), TLoRA (ACL 2026), and related activation/gradient-aware initialization work.
- **frame/coherence optimization of the initial subspace:** *Towards Understanding the Dynamics of Low-Rank Adaptation* (ICML 2026), including its ETF result.
- **hyperparameter audit alone:** *Learning Rate Matters* and *Beware of the Batch Size* (2026).

A new topic must either introduce a different object, make one of these explanations a special case, or causally reconcile multiple competing explanations.

## 6. Hyperparameter-fairness gate

The 2026 *Learning Rate Matters* audit reports that many LoRA papers use fixed or insufficiently tuned learning rates and that apparently large method gaps can shrink to roughly 1–2% after proper tuning. Therefore:

1. reproduce the fixed-recipe comparison only as a diagnostic;
2. sweep LR **per method / per main condition**;
3. perform a local batch-size sweep for headline claims;
4. include update-scale-matched controls when initialization or normalization changes \(P\);
5. only then claim a method-level advantage.

A result that exists only at one inherited LR is not a paper result.

## 7. Multi-outcome robustness gate

Each registered project must have at least **three scientifically interpretable outcome branches**. A first pilot is allowed to falsify the initial preferred mechanism without killing the entire project. The project is killed only when all branches collapse into already-known explanations or negligible effects.

This is not permission to move goalposts: branches and kill criteria must be written **before** the pilot.

## 8. Compute gate

Central claims must fit the local budget.

**Gate 0 — theory / synthetic / diagnostic:** CPU or 1 GPU.  
**Gate 1 — phenomenon pilot:** 0.5B–1.5B model, one task, 1 GPU or 2 GPUs.  
**Gate 2 — causal panel:** 1.5B–3B models, 2–3 tasks, at most 4 GPUs.  
**Gate 3 — final external-validity check:** optional 7B/8B run, at most 4 GPUs.

Not allowed as a central requirement:

- 10B-token pretraining;
- 8+ GPU RL runs;
- a scaling claim that requires dozens of 7B/14B models;
- API-judge-heavy evaluation.

Use gradient accumulation for logical batch sweeps. Spend multi-seed budget only after the mechanism survives Gate 1.

## 9. Top-conference narrative gate

The target narrative is:

> **anomaly → scientific object → mechanism/theorem → causal intervention → predictive diagnostic or minimal remedy → breadth**

not:

> method → benchmark table → ablation.

The method, if any, should be a simple consequence of the mechanism. A paper can succeed without a new method if it resolves a broad contradiction, introduces a reusable evaluation principle, or yields a predictive law.

## 10. Negative-result value gate

The project must still teach something if a presumed NoRA advantage disappears. In particular, a well-controlled result that shows a celebrated mechanism is actually explained by LR, magnitude, basis choice, or noise geometry is valuable **only if** it produces a general causal rule or prediction rather than a one-off debunk.

## 11. Reproducibility gate

Final headline experiments require:

- configs and seeds committed;
- at least 3 seeds for the smallest set of decisive comparisons;
- exact training/evaluation budgets;
- loss curves and optimization statistics, not only endpoint accuracy;
- no hidden best-of-many selection without reporting the sweep.

## 12. Scoring rule

| Dimension | Points |
|---|---:|
| Novelty / nearest-work distance | 20 |
| Direct connection to Normalized NoRA | 15 |
| Mechanistic and falsifiable core | 15 |
| Survival under multiple outcomes | 15 |
| Causal identifiability | 10 |
| ICML/ICLR/NeurIPS narrative breadth | 10 |
| ≤4-GPU compute fit | 10 |
| Execution simplicity | 5 |
| **Total** | **100** |

**Registration threshold: 80/100.** A score is a planning prior, not an acceptance prediction.

---

# What strong recent papers teach us about story width

The literature audit suggests several recurring patterns.

1. **The best story is often an equivalence that unexpectedly breaks.** The NeurIPS 2024 initialization paper begins from two zero-product initializations that look equivalent. LoRA-RITE begins from equivalent factorizations of the same adapter. SpinQuant/QuaRot exploit function-preserving Transformer rotations whose downstream numerical behavior changes after quantization.
2. **One object beats a bag of tweaks.** LoRAM centers magnitude; LoRA-One centers the one-step full-gradient subspace; LoFT centers optimizer-state alignment; NoRA centers \(A^\top A\)'s diagonal gain.
3. **A theory-derived intervention is stronger than a hand-designed variant.** LoRA-One is a clean example: a theorem identifies a subspace, then the algorithm follows almost mechanically.
4. **Endpoint accuracy is not enough.** Strong mechanistic work measures trajectories, gradient/update norms, spectra, stability, forgetting, or function-space behavior.
5. **Fair tuning is now part of novelty defense.** A new LoRA result that does not survive LR and batch tuning is especially vulnerable after the 2026 audits.
6. **Breadth should follow mechanism, not precede it.** A clean mechanism on 1B–3B models plus one 7B validation is more persuasive for our budget than a shallow benchmark zoo.

---

# Directions intentionally not registered

These may be useful baselines or sub-analyses, but they are too crowded or too expensive to be primary topics.

- “Reduce NoRA crosstalk with a better orthogonal/frame initialization” — too close to the ICML 2026 ETF dynamics paper.
- “Whiten/normalize \(A\) with activation or Fisher covariance” — too close to EVA, LoRA-DA, TLoRA and related data-aware initializations.
- “NoRA works only because it increases update magnitude” — important control, but LoRAM and the 2026 LR audit already occupy the standalone story.
- “Make LoRA Riemannian / parameterization invariant” — generic territory already occupied by Riemannian Preconditioned LoRA, LoRA-RITE, StelLA and related manifold methods.
- “Match Adam moments” — LoFT directly targets this.
- “There is an early critical period for \(A\)” — interesting, but Stable-LoRA and LoRA-One make the standalone novelty too narrow.
- “NoRA prevents forgetting” — PEFT-Arena and LoRA-vs-FullFT representation studies make forgetting-only work too narrow.
- “RL only needs tiny rank” / “intrinsic supervision dimension” — scientifically attractive, but 2025–2026 work already reports rank-1 or extremely tiny RL adapters, and a convincing RL program is more expensive than our core budget.
- large-scale low-rank pretraining — compute mismatch for this repository.

---

# Core references from the audit

- Kang et al., *Normalized Low-Rank Adaptation* (2026): https://arxiv.org/abs/2608.31036
- Hayou et al., *LoRA+: Efficient Low Rank Adaptation of Large Models* (ICML 2024): https://proceedings.mlr.press/v235/hayou24a.html
- Hao et al., *Flora: Low-Rank Adapters Are Secretly Gradient Compressors* (ICML 2024): https://proceedings.mlr.press/v235/hao24a.html
- Liu et al., *DoRA* (ICML 2024): https://proceedings.mlr.press/v235/liu24bn.html
- Hayou et al., *The Impact of Initialization on LoRA Finetuning Dynamics* (NeurIPS 2024): https://arxiv.org/abs/2406.08447
- Ashkboos et al., *QuaRot* (NeurIPS 2024): https://proceedings.neurips.cc/paper_files/paper/2024/hash/b5b939436789f76f08b9d0da5e81af7c-Abstract-Conference.html
- Li et al., *On the Crucial Role of Initialization for Matrix Factorization* (ICLR 2025; a different “NoRA”): https://proceedings.iclr.cc/paper_files/paper/2025/hash/0d67ec04032cccf4a21d04c0ae4ab268-Abstract-Conference.html
- Yen et al., *LoRA Done RITE* (ICLR 2025): https://proceedings.iclr.cc/paper_files/paper/2025/hash/bcbc0f660d2dde42f9d1d0ecb14a6f9a-Abstract-Conference.html
- Liu et al., *SpinQuant* (ICLR 2025): https://proceedings.iclr.cc/paper_files/paper/2025/hash/e5b1c0d4866f72393c522c8a00eed4eb-Abstract-Conference.html
- Zhang et al., *LoRA-One* (ICML 2025): https://proceedings.mlr.press/v267/zhang25ax.html
- Zhang et al., *The Primacy of Magnitude in Low-Rank Adaptation* (NeurIPS 2025): https://papers.nips.cc/paper_files/paper/2025/hash/0010665e949927b74faf6e3ada6d7f72-Abstract-Conference.html
- Paischer et al., *Explained Variance Adaptation* (NeurIPS 2025): https://papers.nips.cc/paper_files/paper/2025/hash/41d33bd41fd44bd9dba0e092047cf213-Abstract-Conference.html
- Li et al., *StelLA* (NeurIPS 2025): https://papers.neurips.cc/paper_files/paper/2025/hash/6cb0c6e7d50d5d65613f0456ca85e2db-Abstract-Conference.html
- Schulman / Thinking Machines Lab, *LoRA Without Regret* (2025): https://thinkingmachines.ai/blog/lora/
- Lee et al., *Learning Rate Matters: Vanilla LoRA May Suffice* (2026): https://arxiv.org/abs/2602.04998
- Lee & Lee, *Beware of the Batch Size* (2026): https://arxiv.org/abs/2602.09492
- Tastan et al., *Low-Rank Adaptation That Behaves Like Full Fine-Tuning (LoFT)* (ICLR 2026): https://proceedings.iclr.cc/paper_files/paper/2026/hash/7428310c0f97f1c6bb2ef1be99c1ec2a-Abstract-Conference.html
- Wu et al., *Stable-LoRA* (ICLR 2026): https://proceedings.iclr.cc/paper_files/paper/2026/hash/2962d47082d07cd6e28272ab471e0526-Abstract-Conference.html
- Zhang et al., *LoRA-DA* (ICML 2026): https://arxiv.org/abs/2510.24561
- Ding et al., *Towards Understanding the Dynamics of Low-Rank Adaptation* (ICML 2026): https://openreview.net/
- Lin et al., *TLoRA* (ACL 2026): https://aclanthology.org/2026.acl-long.1348/
- Wang & Wang, *Complete Characterization of Gauge Symmetries in Transformer Architectures* (NeurIPS 2025 workshop): https://neurips.cc/virtual/2025/136893
- Malekmohammadi & Farnadi, *Low-Rank Adaptation Secretly Imitates Differentially Private SGD* (ICML 2025 workshop / later journal trajectory): https://openreview.net/pdf?id=vsLkyuo6M5

## Current execution order

1. Run the **A1 hidden-preconditioner matched-control pilot** first because it directly tests whether Normalized NoRA's claimed mechanism survives magnitude/LR confounds.
2. In parallel or immediately after, run the **A2 exact-gauge invariance diagnostic**; it is cheap and has unusually clean theory.
3. Run **A3 batch/noise geometry** only after reproducing the large-batch LoRA pathology in our stack.

Do not begin with large-scale benchmark training. Every project starts with the cheapest experiment capable of falsifying its preferred explanation.
