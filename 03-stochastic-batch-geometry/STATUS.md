# 03 — Stochastic Batch Geometry — STATUS

Last update: 2026-09-02

## E1 — does the large-batch LoRA-vs-FullFT gap reproduce here? (`results/e1`, `results/e1_r128`, 60 runs)

**Setting.** Qwen3-0.6B-Base, NuminaMath-CoT single pass, **fixed example budget
of 32768** (so batch size trades directly against optimizer steps), cosine
schedule, grad-clip 1.0, fp32 master weights with bf16 matmuls for *both* FullFT
and LoRA (a matched-precision requirement: bf16 master weights would make the
two methods differ in optimizer precision, which is a confound in a study about
gradient noise). Per-cell learning-rate sweep over 5 values.

Best-tuned final eval loss:

| method | bs=8 (4096 steps) | bs=32 (1024) | bs=128 (256) | bs=512 (64) |
|---|---|---|---|---|
| FullFT | 0.48606 | **0.48506** | 0.48853 | 0.49294 |
| LoRA r=16 | 0.48824 | 0.48806 | 0.48883 | 0.49468 |
| LoRA r=128 | **0.48487** | 0.48652 | 0.48625 | 0.49330 |

Gap vs FullFT (positive = LoRA worse):

| method | bs=8 | bs=32 | bs=128 | bs=512 |
|---|---|---|---|---|
| LoRA r=16 | +0.0022 | +0.0030 | +0.0003 | +0.0017 |
| LoRA r=128 | −0.0012 | +0.0015 | −0.0023 | +0.0004 |

## Interpretation

**The reported pathology does not reproduce in this setting.** After per-cell LR
tuning the gap is ≤0.003 nats, is *not monotone* in batch size, and LoRA r=128
matches or beats FullFT at three of the four batch sizes. Everything degrades at
large batch — FullFT included (0.4851 → 0.4929) — which is the ordinary
step-count effect at fixed example budget, not a LoRA-specific one.

Scope of this negative result, stated honestly:
* one model (0.6B), one dataset, one budget (32768 examples ≈ 14M tokens);
* the largest batch reaches only 64 optimizer steps, so the large-batch cells are
  step-starved for *every* method, which compresses between-method differences;
* *LoRA Without Regret* reports the effect at larger scale and longer training.

So this is "not reproduced at 0.6B / 32k examples", **not** "the effect is false".

## Supported / falsified

* Falsified here: a *best-tuned* LoRA-vs-FullFT penalty that grows with batch
  size at this scale.
* Confirmed: the penalty is weakly rank-dependent — but only because it is
  ~absent at every rank, so E2 is uninformative as run.
* Not yet tested: H1/H2/H3 (they are only meaningful once a gap exists).

## Next experiment (decisive for the kill decision)

The README requires a second feasible setting before killing. The one designed
to maximise the chance of seeing the effect:
* **4× budget (131072 examples)** so that even bs=2048 gets 64 steps and bs=128
  gets 1024 — removing the step-starvation confound;
* batches {128, 512, 2048};
* FullFT vs LoRA r=16 vs LoRA r=128, per-cell LR sweep;
* optionally Qwen3-1.7B-Base for the decisive cells.

If the gap is again ≤0.005 nats and non-monotone, kill criterion 1 is met and A3
should be written up as a scoped non-reproduction with the LR/step-count
decomposition, not padded into a mechanism paper.
