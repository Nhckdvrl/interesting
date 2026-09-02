"""Topic 01 at 7B scale.

Design follows the mother paper's evaluation conventions (SFT on MetaMath,
rank 32, downstream GSM8K exact-match accuracy alongside held-out loss) but adds
the three things it does not have: a per-method learning-rate sweep, replicate
seeds, and the `left_gauge` null -- an initialisation that is bit-identical to
vanilla LoRA in every statistic of P and therefore bounds what any
initialisation effect can mean.

Conditions
  kaiming       vanilla LoRA                                (3 seeds)
  left_gauge    A0 -> Q A0, Q in O(r): identical P0          (3 gauges)  <- NULL
  nora_unit     NoRA as published: unit column norms
  nora          NoRA's operator with tr P held fixed         <- removes the
                                                                magnitude confound
  bimi          NoRA's Block Identity init (their Table 3)
  flatspec_flatdiag   flat spectrum + flat diagonal (= BIMI's (lam,d) point)
  geomspec_flatdiag0.5  r_eff(P) collapsed 16->3 at matched trace and diagonal
  eva, gradsub  data-aware subspaces
  pissa         B0 != 0, the only family that leaves the P0 class
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from common import cluster

REPO = cluster.REPO
RUN = f"{REPO}/01-hidden-preconditioner/src/run_lit.py"
LRS = [3e-5, 1e-4, 2e-4, 4e-4, 8e-4]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="m7b")
    ap.add_argument("--model", default="mistralai/Mistral-7B-v0.3")
    ap.add_argument("--task", default="metamath")
    ap.add_argument("--arch", default="a100")
    ap.add_argument("--hosts", default=None)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=64)
    ap.add_argument("--acc_n", type=int, default=200)
    ap.add_argument("--lrs", default=None)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    lrs = [float(x) for x in a.lrs.split(",")] if a.lrs else LRS
    common = (f"--model {a.model} --task {a.task} --dtype bfloat16 --amp none "
              f"--r {a.r} --alpha {a.alpha:g} --steps {a.steps} --bs 16 "
              f"--micro_bs 4 --probe_bs 2 --max_len 512 --n_train 60000 "
              f"--n_eval 512 --eval_batches 24 --acc_n {a.acc_n} --acc_bs 32 "
              f"--warmup 20 --sched cosine")
    jobs = []
    for lr in lrs:
        for sd in (0, 1, 2):
            jobs.append(f"{{PY}} {RUN} --tag {a.tag} --cond kaiming --lr {lr:g} "
                        f"--seed {sd} --match trace {common}")
        for gs in (1, 2, 3):
            jobs.append(f"{{PY}} {RUN} --tag {a.tag} --cond left_gauge "
                        f"--lr {lr:g} --gauge_seed {gs} --match trace {common}")
        for c in ("nora", "bimi", "flatspec_flatdiag", "geomspec_flatdiag0.5"):
            jobs.append(f"{{PY}} {RUN} --tag {a.tag} --cond {c} --lr {lr:g} "
                        f"--match trace {common}")
        # published scale (no trace matching) for the methods whose scale is
        # part of the method
        for c in ("nora_unit", "bimi", "eva", "gradsub", "pissa"):
            jobs.append(f"{{PY}} {RUN} --tag {a.tag} --cond {c} --lr {lr:g} "
                        f"--match none {common}")
    print(f"{len(jobs)} jobs")
    cluster.main(jobs, a.tag,
                 f"{REPO}/01-hidden-preconditioner/results/{a.tag}/logs",
                 arch=a.arch,
                 hosts=a.hosts.split(",") if a.hosts else None, dry=a.dry)
