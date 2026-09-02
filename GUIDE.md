# Repository guide

Start here. `README.md` is the original topic registration and is deliberately
left unchanged; this file is the map of what has actually been built and where
the current main line is.

## The main line, in one paragraph

Three topics were registered from the Normalized LoRA (NoRA) mother paper.
Topics **01** and **02** converged on a single object and are now one line of
work; topic **03** is a scoped non-reproduction. The line is:

> LoRA initialisation looks like a huge design space, but two exact symmetries
> and the data metric collapse it to a small **intrinsic state space**. Every
> published initializer is a point in it, and where the point sits — not which
> paper proposed it — predicts what training does.

The state is `M_x = A Σ_x Aᵀ`, the rank-space Gram in the data metric, with
coordinates `(S, D, ρ)` = (scale, spectral dimension, task alignment).

## Where to read, in order

| file | what it is |
|---|---|
| **`paper/NARRATIVE.md`** | the main-line document: claims, evidence, positioning against the closest work. **Read this first.** |
| `paper/RESULTS.md` | every headline number, generated from the run files by `paper/make_results.py` |
| `01-hidden-preconditioner/STATUS.md` | topic-01 record: what was run, what it showed, what was falsified |
| `02-representation-gauge/STATUS.md` | topic-02 record |
| `03-stochastic-batch-geometry/STATUS.md` | topic-03 record (non-reproduction, scoped) |
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
