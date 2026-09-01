"""Topic 01, Gate 1c -- spectrum dose-response at matched trace and matched
(flat) diagonal.

Gate 1b found that every *diagonal* intervention on P is inert, but that
collapsing the SPECTRUM of P hurts systematically at every learning rate, at
exactly matched tr P and exactly flat diag P.  That isolates the effective rank

    r_eff(P) = (tr P)^2 / ||P||_F^2

as the causal feature of the hidden preconditioner -- the feature NoRA does not
control.  This panel measures the dose-response over r_eff in [1.86, 16] with
r = 16 held fixed, so the adapter's *nominal* rank never changes.
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from common import cluster

REPO = cluster.REPO
RUN = f"{REPO}/01-hidden-preconditioner/src/run_panel.py"

LADDER = ["flatspec_flatdiag"] + [f"geomspec_flatdiag{d}" for d in
                                  (0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)]
LRS = [1e-4, 2e-4, 3e-4, 5e-4]
MULTISEED = ["flatspec_flatdiag", "geomspec_flatdiag0.8",
             "geomspec_flatdiag0.6", "geomspec_flatdiag0.4"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="g1c")
    ap.add_argument("--arch", default="blackwell")
    ap.add_argument("--hosts", default=None)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    jobs = []
    for cond in LADDER:
        for lr in LRS:
            jobs.append(f"{{PY}} {RUN} --tag {a.tag} --cond {cond} --lr {lr:g} "
                        f"--seed 0 --steps {a.steps}")
    for cond in MULTISEED:
        for lr in (2e-4, 3e-4):
            for seed in (1, 2):
                jobs.append(f"{{PY}} {RUN} --tag {a.tag} --cond {cond} "
                            f"--lr {lr:g} --seed {seed} --steps {a.steps}")
    # kaiming reference at the same LRs and seeds
    for lr in LRS:
        for seed in (0, 1, 2):
            jobs.append(f"{{PY}} {RUN} --tag {a.tag} --cond kaiming --lr {lr:g} "
                        f"--seed {seed} --steps {a.steps}")
    # persistence check: does the penalty survive 3x longer training?
    for cond in ["flatspec_flatdiag", "geomspec_flatdiag0.5"]:
        for lr in (2e-4, 3e-4):
            jobs.append(f"{{PY}} {RUN} --tag {a.tag}_long --cond {cond} "
                        f"--lr {lr:g} --seed 0 --steps 900")
    print(f"{len(jobs)} jobs")
    cluster.main(jobs, a.tag,
                 f"{REPO}/01-hidden-preconditioner/results/{a.tag}/logs",
                 arch=a.arch,
                 hosts=a.hosts.split(",") if a.hosts else None, dry=a.dry)
