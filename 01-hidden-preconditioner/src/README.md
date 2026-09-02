# Topic 01 scripts

Three roles: **runners** train one cell, **panels** enqueue many cells across
the cluster, **analyses** read `../results/` and print or plot.

## Stage 2 — the intrinsic state space (current main line)

| script | role |
|---|---|
| `run_atlas.py` | runner: trains one point at an exactly specified `(S, D, ρ)` |
| `panel_atlas.py` | panel: the star design through the vanilla reference point |
| `panel_ood.py` | panel: the held-out published initializers, same arch and config |
| `intrinsic_table.py` | locates every published initializer in `(S, D, ρ)`, no training |
| `analyze_atlas.py` | fits the response surface on the atlas, predicts the held-out set |

## Stage 1 — the audit that motivated it

| script | role |
|---|---|
| `gate0_feasible_region.py` | the Gate-0 theorems and their numerical verification |
| `run_lit.py` | runner: one published initializer under one matching convention |
| `panel_lit.py`, `panel_lit2.py` | panels: the audit, and its breadth arm on a second task |
| `run_panel.py` | runner: the earlier matched-control conditions |
| `panel_g1*.py` | panels: matched controls, dense LR, the r_eff ladder, the SGD arm |
| `pstat_table.py` | all P-statistics for every initializer, no training |
| `grad_capture.py` | the first-order law `cos(G,GP) = √(r_eff/d_in)` on real gradients |
| `analyze_lit.py`, `family_d.py`, `predict_loo.py` | audit summaries and the leave-one-out test |
| `analyze_panel.py`, `analyze_reff.py`, `collapse.py`, `prescription.py` | the earlier analyses |
| `make_plots.py` | figures |
| `panel_7b.py` | 7B panel (parked; use Qwen3-8B, whose width is a power of two) |
