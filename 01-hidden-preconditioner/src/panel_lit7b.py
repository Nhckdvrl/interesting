"""Topic 01 at 7B -- Mistral-7B-v0.3, the decisive rows of the audit.

Kept deliberately narrow: the three claims that have to survive scale are
  (i)   the null      -- kaiming seeds vs left_gauge (identical P_0)
  (ii)  the cluster   -- NoRA / ETF / flat-spectrum / exact flat-diagonal are
                         inside it, and r_eff is the one statistic that is not
  (iii) the data metric -- EVA / gradient-subspace / PiSSA separate under
                         matched tr P and re-join under matched tr(P Sigma)
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from common import cluster

REPO = cluster.REPO
RUN = f"{REPO}/01-hidden-preconditioner/src/run_lit.py"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="lit7b")
    ap.add_argument("--model", default="mistralai/Mistral-7B-v0.3")
    ap.add_argument("--arch", default="a100")
    ap.add_argument("--hosts", default=None)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lrs", default="5e-5,1e-4,2e-4,4e-4")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    common = (f"--model {a.model} --steps {a.steps} --bs 16 --micro_bs 4 "
              f"--probe_bs 2 --probe_batches 8 --eval_batches 16 "
              f"--n_train 6000 --n_eval 256")
    jobs = []
    for lr in [float(x) for x in a.lrs.split(",")]:
        for sd in (0, 1, 2):
            jobs.append(f"{{PY}} {RUN} --tag {a.tag} --cond kaiming --lr {lr:g} "
                        f"--match trace --seed {sd} {common}")
        for gs in (1, 2):
            jobs.append(f"{{PY}} {RUN} --tag {a.tag} --cond left_gauge "
                        f"--lr {lr:g} --match trace --gauge_seed {gs} {common}")
        for c in ("nora", "etf", "flatspec_flatdiag", "geomspec_flatdiag0.5",
                  "eva", "gradsub", "pissa"):
            jobs.append(f"{{PY}} {RUN} --tag {a.tag} --cond {c} --lr {lr:g} "
                        f"--match trace {common}")
        for c in ("kaiming", "eva", "gradsub", "pissa"):
            jobs.append(f"{{PY}} {RUN} --tag {a.tag} --cond {c} --lr {lr:g} "
                        f"--match trace_act {common}")
    print(f"{len(jobs)} jobs")
    cluster.main(jobs, a.tag,
                 f"{REPO}/01-hidden-preconditioner/results/{a.tag}/logs",
                 arch=a.arch,
                 hosts=a.hosts.split(",") if a.hosts else None, dry=a.dry)
