"""Topic 01, Gate 1d -- the r_eff mechanism test.

Under SGD the merged update is exactly dW = -eta G P, so the loss decrease per
unit update norm is exactly cos(G, GP), which `grad_capture` verified equals
sqrt(r_eff(P)/d_in) on real LLM gradients to 2.5% across an 8.6x range of r_eff.
Under AdamW the first step is -lr*s*sign(G A^T) A instead, and the measured
cos_adam falls with r_eff far more slowly.

Prediction: the r_eff dose-response must be substantially STEEPER under SGD than
under AdamW.  If it is not, the descent-efficiency mechanism is wrong.
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from common import cluster

REPO = cluster.REPO
RUN = f"{REPO}/01-hidden-preconditioner/src/run_panel.py"
LADDER = ["kaiming", "flatspec_flatdiag"] + \
         [f"geomspec_flatdiag{d}" for d in (0.8, 0.6, 0.5, 0.4, 0.3)]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="g1d")
    ap.add_argument("--arch", default="blackwell")
    ap.add_argument("--hosts", default="LOCAL")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lrs", default="0.03,0.1,0.3,1.0")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    jobs = []
    for cond in LADDER:
        for lr in a.lrs.split(","):
            for seed in (0, 1):
                jobs.append(f"{{PY}} {RUN} --tag {a.tag} --cond {cond} "
                            f"--lr {lr} --seed {seed} --steps {a.steps} "
                            f"--optimizer sgd --momentum 0 --warmup 0 "
                            f"--grad_clip 1e9")
    print(f"{len(jobs)} jobs")
    cluster.main(jobs, a.tag,
                 f"{REPO}/01-hidden-preconditioner/results/{a.tag}/logs",
                 arch=a.arch, hosts=a.hosts.split(","), dry=a.dry)
