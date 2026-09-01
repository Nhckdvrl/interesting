"""Topic 01, Gate 1.

The decisive question this panel answers:

  Is the *measured* effect of a LoRA initialisation method larger than the
  effect of moving inside its own P0-equivalence class?

Gate 0 (F4/F5) proved that A0 and Q A0 (Q in O(r)) have *identical* P0 -- every
pre-registered P-statistic agrees exactly -- and that under SGD they produce
identical trajectories for all time.  Under AdamW they need not.  So
`left_gauge` is an exact null condition carrying zero preconditioner content:
it is a far tighter control than a seed change, which also perturbs P0.

Conditions
  kaiming              vanilla LoRA (PEFT default init)
  left_gauge           A0 -> Q A0, Q in O(r).  IDENTICAL P0 to kaiming/seed0.
  nora                 column-normalised, rescaled to preserve tr P  (NoRA-init,
                       trace-matched: removes the pure magnitude confound)
  nora_unit            literal unit columns (tr P changes by ~170x) -- the
                       magnitude confound itself
  kaimingspec_flatdiag same tr P AND same spectrum as kaiming, flat diagonal:
                       the *pure* diagonal-balance intervention
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from common.sched import run_jobs

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PY = os.path.join(REPO, ".venv", "bin", "python")
RUN = os.path.join(REPO, "01-hidden-preconditioner", "src", "run_panel.py")

LRS = [3e-5, 1e-4, 3e-4, 1e-3]
CELLS = (
    [("kaiming", s, 0) for s in (0, 1, 2)] +
    [("left_gauge", 0, g) for g in (1, 2, 3)] +
    [("nora", s, 0) for s in (0, 1, 2)] +
    [("nora_unit", 0, 0)] +
    [("kaimingspec_flatdiag", s, 0) for s in (0, 1, 2)]
)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="g1")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    jobs = []
    for lr in LRS:
        for cond, seed, gs in CELLS:
            jobs.append(f"{PY} {RUN} --tag {a.tag} --cond {cond} --lr {lr:g} "
                        f"--seed {seed} --gauge_seed {gs} --steps {a.steps}")
    print(f"{len(jobs)} jobs")
    run_jobs(jobs, [int(g) for g in a.gpus.split(",")],
             os.path.join(REPO, "01-hidden-preconditioner", "results",
                          a.tag, "logs"), dry=a.dry)
