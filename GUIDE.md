# Repository guide

Start here. `README.md` is the original topic registration and is deliberately
left unchanged; this file is the map of what has actually been built and where
the current main line is.

## The main line, in one paragraph

Three topics were registered from the Normalized LoRA (NoRA) mother paper.
Topics **01** and **02** converged on a single object and are now one line of
work; topic **03** is a scoped non-reproduction. The line is:

> LoRA's factorisation carries a `GL(r)` reparameterisation ambiguity, and by
> polar decomposition it splits into a **rotation** part `O(r)` and a
> **scaling** part. Optimizers form a strict hierarchy of what they are blind
> to, so **what an initialisation is -- how many degrees of freedom it actually
> has -- depends on which optimizer will consume it.** AdamW has the smallest
> symmetry group of any method in common use, and the extra coordinate it alone
> can see is one the initialisation literature varies by 4.3x without reporting.

Measured over 25 steps in float64 (`01-.../src/hierarchy.py`):

| optimizer | signed perms | `O(r)` rotation | `GL(r)` scaling |
|---|---|---|---|
| SGD | 2.2e-16 | **2.2e-16** | 9.8e-03 |
| Muon | 2.5e-09 | **4.4e-09** | 1.2e-02 |
| matrix-preconditioned Adam | 2.2e-16 | **2.2e-16** | 1.5e-02 |
| **AdamW** | 0.0e+00 | **1.7e-03** | 3.2e-02 |

```
   signed perms  <  O(r)  <  GL(r)
      AdamW         SGD       LoRA-RITE
                    Muon      Riemannion (ICLR 2026)
                    matrix-preconditioned Adam
```

Four recent works -- LoRA-RITE, Balanced LoRA, FedRot-LoRA and Riemannion --
all treat this ambiguity as a defect to remove. None asks what it is worth, or
which part of it a given optimizer sees. Riemannion motivates itself by saying
per-factor Muon is not invariant to "scalings **or rotations**"; the rotation
half of that is wrong by six orders of magnitude, and it is exactly the part
that matters.

## Where to read, in order

| file | what it is |
|---|---|
| **`paper/STORY.md`** | the current framing: the symmetry hierarchy, what each optimizer can see, and why the prescription is a section rather than the thesis. **Read this first.** |
| `paper/NARRATIVE.md` | the longer main-line document: claims, evidence, positioning |
| `paper/SCOPE.md` | what NoRA actually runs, and what we need to match it |
| `paper/RESULTS.md` | every headline number, generated from the run files by `paper/make_results.py` |
| `01-hidden-preconditioner/STATUS.md` | topic-01 record: what was run, what it showed, what was falsified |
| `02-representation-gauge/STATUS.md` | topic-02 record |
| `03-stochastic-batch-geometry/STATUS.md` | topic-03 record (non-reproduction, scoped) |
| `paper/POSITION.md` | what is ours and what is already known (LoRA-RITE, Balanced LoRA, FedRot-LoRA all treat the gauge as a defect to remove; we treat it as a coordinate to set) |
| `01-hidden-preconditioner/PREDICTIONS_frame.md` | predictions committed before the gauge-frame panel ran. One of them was falsified in sign, and the file is unedited |
| `01-hidden-preconditioner/PREDICTIONS_mechanism.md` | predictions committed before the mechanism panel ran; the named matrix was wrong |
| `paper/8b_predictions.md` | predictions committed before any 8B training |
| `README.md`, `0*/README.md` | the original registrations — the pre-registered questions and kill criteria |

## Reproducing the numbers

```bash
.venv/bin/python paper/make_results.py > paper/RESULTS.md
```

Everything it prints is read from the committed `results/` directories; no
training is re-run.

## Code layout

```
common/            shared library — the only place with reusable machinery
  intrinsic.py       THE scientific object: M_x, its coordinates, exact constructions
  pinit.py           A-initialisers and the Schur–Horn (spectrum, diagonal) constructions
  initializers.py    faithful re-implementations of the published initializers
  gauge.py           exact function-preserving Transformer gauges (V/O and residual)
  lora.py            minimal LoRA with full control over A0, B0 and the base weight
  train.py           the training loop and its optimisation diagnostics
  pstats.py          statistics of P = A^T A
  data.py            SFT corpora, completion masking, fixed batch order
  evaluate.py        GSM8K exact-match scoring
  cluster.py         multi-host GPU scheduler (arch-pinned)

0*/src/            per-topic runners, panels and analyses — see each src/README.md
0*/results/        raw run files, one JSON per run, grouped by panel tag
0*/plots/          figures
paper/             the write-up and the scripts that generate its numbers
failed_topics/     the pre-registered rejection registry
```

## Compute

`common/cluster.py` schedules across the fleet and **pins each panel to one GPU
architecture**: the A100 and RTX PRO 6000 nodes differ by ≈4e-4 nats on an
identical configuration, which is above the 2.7e-4 measurement null, so a panel
whose conclusions live at that resolution must not be split across them.

```bash
.venv/bin/python common/cluster.py        # list free slots
```

Two virtualenvs, because the driver versions differ: `.venv` (cu130, RTX PRO
6000) and `.venv-a100` (cu124, A100).
