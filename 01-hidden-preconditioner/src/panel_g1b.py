"""Topic 01, Gate 1b -- the magnitude-collapse test.

Gate 1a showed that with tr P matched, no P-statistic intervention moved the
final loss beyond the gauge null.  1b asks the constructive question: is
tr P (equivalently the realised update scale ||dW||) the ONLY channel?

Two probes:
 (i)  dense LR sweeps for several matched-trace initialisations plus the
      trace-mismatched `nora_unit`, so we can plot final loss against the
      realised ||dW|| instead of against LR.  If the magnitude law holds, all
      conditions lie on ONE curve.
 (ii) a strong spectrum intervention at matched trace and flat diagonal:
      geomspec_flatdiag0.5 has effective rank 3.0 instead of 16.0.  Every
      pre-registered statistic except the spectrum is matched.
 (iii) rank and alpha variation, which move tr P by construction and therefore
      test whether the same single curve absorbs them.
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from common import cluster

REPO = cluster.REPO
RUN = f"{REPO}/01-hidden-preconditioner/src/run_panel.py"

DENSE = [1e-5, 3e-5, 6e-5, 1e-4, 2e-4, 3e-4, 5e-4, 1e-3]
CONDS = ["kaiming", "nora", "kaimingspec_flatdiag", "flatspec_flatdiag",
         "geomspec_flatdiag0.5"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="g1b")
    ap.add_argument("--arch", default="blackwell")
    ap.add_argument("--hosts", default=None)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    jobs = []
    # (i) + (ii): dense LR x matched-trace conditions
    for cond in CONDS:
        for lr in DENSE:
            jobs.append(f"{{PY}} {RUN} --tag {a.tag} --cond {cond} --lr {lr:g} "
                        f"--seed 0 --steps {a.steps}")
    # trace-mismatched condition, shifted LR grid
    for lr in [1e-6, 3e-6, 1e-5, 2e-5, 3e-5, 6e-5, 1e-4]:
        jobs.append(f"{{PY}} {RUN} --tag {a.tag} --cond nora_unit --lr {lr:g} "
                    f"--seed 0 --steps {a.steps}")
    # (iii) rank / alpha variation (kaiming only)
    for r, alpha in [(4, 32), (64, 32), (16, 8), (16, 128)]:
        for lr in [3e-5, 1e-4, 3e-4, 1e-3]:
            jobs.append(f"{{PY}} {RUN} --tag {a.tag}_r{r}a{alpha:g} --cond kaiming "
                        f"--lr {lr:g} --seed 0 --r {r} --alpha {alpha} "
                        f"--steps {a.steps}")
    print(f"{len(jobs)} jobs")
    cluster.main(jobs, a.tag,
                 f"{REPO}/01-hidden-preconditioner/results/{a.tag}/logs",
                 arch=a.arch,
                 hosts=a.hosts.split(",") if a.hosts else None, dry=a.dry)
