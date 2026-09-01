"""Topic 01 -- breadth arm of the literature audit.

Same design as `panel_lit`, on a second, non-mathematical task
(databricks-dolly-15k instruction following) and optionally a second model, to
check that the P-space conclusions are not a property of GSM8K.
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from common import cluster

REPO = cluster.REPO
RUN = f"{REPO}/01-hidden-preconditioner/src/run_lit.py"
OTHERS = ["nora", "etf", "eva", "gradsub", "flatspec_flatdiag",
          "geomspec_flatdiag0.5", "pissa", "olora"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="lit_dolly")
    ap.add_argument("--task", default="dolly")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--arch", default="blackwell")
    ap.add_argument("--hosts", default=None)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lrs", default="1e-5,3e-5,1e-4,2e-4,3e-4,5e-4,1e-3")
    ap.add_argument("--n_train", type=int, default=6000)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    jobs = []
    common = (f"--task {a.task} --model {a.model} --steps {a.steps} "
              f"--n_train {a.n_train}")
    for lr in [float(x) for x in a.lrs.split(",")]:
        for match in ("trace", "trace_act"):
            for sd in ((0, 1, 2) if match == "trace" else (0,)):
                jobs.append(f"{{PY}} {RUN} --tag {a.tag} --cond kaiming "
                            f"--lr {lr:g} --match {match} --seed {sd} {common}")
            for gs in ((1, 2, 3) if match == "trace" else (1,)):
                jobs.append(f"{{PY}} {RUN} --tag {a.tag} --cond left_gauge "
                            f"--lr {lr:g} --match {match} --gauge_seed {gs} "
                            f"{common}")
            for c in OTHERS:
                jobs.append(f"{{PY}} {RUN} --tag {a.tag} --cond {c} "
                            f"--lr {lr:g} --match {match} {common}")
    print(f"{len(jobs)} jobs")
    cluster.main(jobs, a.tag,
                 f"{REPO}/01-hidden-preconditioner/results/{a.tag}/logs",
                 arch=a.arch,
                 hosts=a.hosts.split(",") if a.hosts else None, dry=a.dry)
