# Topic 01 results index

One directory per panel; one JSON per run, named by its cell. Every JSON
carries the full `args` it was produced with, so any table in the paper can be
regenerated without re-training (`paper/make_results.py`).

## The main line — the gauge frame

| directory | runs | what it is | read with |
|---|---|---|---|
| `frame/` | 83 | the gauge-frame ladder on one orbit of the vanilla draw, under **AdamW, SGD and Muon**, plus the activation-metric ladder (`framex*`), a finer LR grid and three seeds | `src/analyze_frame.py`, `src/analyze_mechanism.py` |
| `rot/` | 96 | six published initialisers, each as published and rotated to `@frame0` / `@frame1` | `src/analyze_rot.py` |
| `rank/` | 32+ | the same ladder at r = 1, 4, 16, 64 (and 128) — the effect is structurally zero at rank 1 | `src/analyze_rank.py` |
| `q8b/` | — | Qwen3-8B, predictions pre-committed in `paper/8b_predictions.md` | `src/analyze_8b.py` |
| `dolly_frame/` | — | the ladder on a second, non-mathematical task | |
| `long/`, `acc/` | — | 1000 steps instead of 300, and GSM8K exact-match instead of nats | |
| `second_order.json` | — | all twelve frame/invariant statistics for every construction and every published initialiser, measured **without training** | `src/second_order.py` |

## Stage 1 — the audit of published initializers

| directory | runs | what it is |
|---|---|---|
| `lit/` | 414 | the literature audit on GSM8K: 13 initializers x matching conventions x LR sweep |
| `lit_dolly/` | 168 | the same audit on Dolly |
| `ood/` | 91 | the held-out set for the out-of-sample law tests |
| `adamnull32/`, `sgdnull*/` | 7 | the measurement null: a gauge move that changes nothing SGD can see |
| `intrinsic_table.json`, `pstat_table_*.json` | — | where each published initializer sits, measured without training |

## Stage 2 — the synthetic atlas

| directory | runs | what it is |
|---|---|---|
| `atlas/` | 269 | the designed `(S, D, R_g, W)` atlas, including the exactly matched `W` ladder |
| `atlas_noclip/`, `atlas_sgd/` | 95 | the same ladder without gradient clipping, and under SGD |
| `g1*/` | ~290 | earlier gate-1 panels: rank/alpha scans, long runs, LoRA+ ratios |
| `extra_coords.json`, `grad_capture.json`, `gate0_feasible_region.json` | — | offline coordinate measurements |

## Kept deliberately

`smokeA/`, `smokeM/`, `smokeW/` are single-run smoke tests of the construction
code, and the `g1b_r*` directories are four-run probes that went nowhere. They
are small, they are dated, and negative or abandoned panels are not deleted
here — see the project instructions.
