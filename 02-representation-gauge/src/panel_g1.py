"""Topic 02, Gate 1 panel.

Rows (per optimizer x lr), all on the SAME ordered minibatches, fp32:

  kaiming_orig            reference trajectory in the original coordinates
  kaiming_coupled_oracle  same adapter, expressed in the rotated coordinates
                          -> under SGD must be IDENTICAL (positive control);
                             under AdamW the deviation is the optimizer's own
                             gauge dependence
  nora_orig               NoRA-init applied in the original coordinates
  nora_coupled_oracle     N(A) then rotate  -> equivariance oracle
  nora_coupled_algo       rotate then N(AR) -> what a practitioner actually gets
                          on the rotated (functionally identical) backbone

  effect_NoRA = |coupled_algo - coupled_oracle|   (pure non-commutation)
  floor_opt   = |kaiming_coupled_oracle - kaiming_orig|  (optimizer + roundoff)
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from common.sched import run_jobs

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PY = os.path.join(REPO, ".venv", "bin", "python")
RUN = os.path.join(REPO, "02-representation-gauge", "src", "run_gauge_pair.py")

ROWS = [("kaiming", "orig"), ("kaiming", "coupled_oracle"),
        ("nora", "orig"), ("nora", "coupled_oracle"), ("nora", "coupled_algo")]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="g1")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--gauge", default="residual")
    ap.add_argument("--gauge_seeds", default="0")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--sgd_lrs", default="0.03,0.1")
    ap.add_argument("--adam_lrs", default="1e-4,3e-4")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    jobs = []
    for gs in a.gauge_seeds.split(","):
        for opt, lrs in (("sgd", a.sgd_lrs), ("adamw", a.adam_lrs)):
            for lr in lrs.split(","):
                for init, mode in ROWS:
                    if mode == "orig" and gs != a.gauge_seeds.split(",")[0]:
                        continue   # orig does not depend on the gauge seed
                    jobs.append(
                        f"{PY} {RUN} --tag {a.tag} --gauge {a.gauge} "
                        f"--gauge_seed {gs} --init {init} --mode {mode} "
                        f"--optimizer {opt} --lr {lr} --steps {a.steps}")
    print(f"{len(jobs)} jobs")
    run_jobs(jobs, [int(g) for g in a.gpus.split(",")],
             os.path.join(REPO, "02-representation-gauge", "results", a.tag,
                          "logs"), dry=a.dry)
