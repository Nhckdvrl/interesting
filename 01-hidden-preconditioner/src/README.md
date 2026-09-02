# Topic 01 scripts

Three roles: **runners** train one cell, **panels** enqueue many cells across
the cluster, **analyses** read `../results/` and print or plot.

## Stage 3 — the gauge frame (current main line)

The conditions are dispatched by `run_lit.py`: `frame<t>` walks the gauge orbit
of the vanilla draw in the gradient metric (`t = 0` its eigenbasis, `t = 1` a
flat diagonal, `opt`/`min` the argmax/argmin of `Λ₁`), `framex<t>` does the same
in the activation metric, `signperm<k>` is the exactly-invariant noise floor,
and `<cond>@frame<t>` applies the same rotation to any published initializer,
carrying `B` along so `BA` is preserved.

| script | role |
|---|---|
| `second_order.py` | the nine order-≤2 gauge invariants plus the frame statistics (`Λ₁`, `E_g`, `Off_g`, `Off_x`) for every construction and every published initializer — **no training** |
| `analyze_frame.py` | the frame ladder: SGD vs Muon vs AdamW on one orbit |
| `analyze_mechanism.py` | which functional of the frame the optimizer responds to, and why the candidates cannot be separated |
| `analyze_rot.py` | the zoo rotated in both directions, and how big the frame is next to what those papers compare |
| `analyze_rank.py` | the effect against `dim O(r)/signed perms`, zero at r = 1 |
| `analyze_8b.py` | the four predictions committed in `paper/8b_predictions.md` |
| `frame_closure.py` | does the frame close the out-of-distribution gap the invariant law leaves? |

## Stage 2 — the intrinsic state space

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
