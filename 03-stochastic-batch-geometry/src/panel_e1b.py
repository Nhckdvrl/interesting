"""Topic 03, E1b -- the second setting required before the kill decision.

E1 could not separate the methods because at a 32768-example budget the largest
batch got only 64 optimizer steps, so every method was step-starved together.
E1b quadruples the budget and pushes the batch axis further, so that even
bs=2048 gets 64 steps while bs=128 gets 1024 -- i.e. the large-batch cells are
no longer starved relative to the small-batch ones.
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from common import cluster

REPO = cluster.REPO
RUN = f"{REPO}/03-stochastic-batch-geometry/src/run_batch.py"
BATCHES = [128, 512, 2048]
FULL_LRS = [1e-5, 3e-5, 1e-4, 3e-4]
LORA_LRS = [3e-4, 1e-3, 3e-3, 1e-2]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="e1b")
    ap.add_argument("--arch", default="a100")
    ap.add_argument("--hosts", default=None)
    ap.add_argument("--budget", type=int, default=131072)
    ap.add_argument("--micro_bs", type=int, default=16)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    common = (f"--task numina --n_train 150000 --n_eval 512 --max_len 512 "
              f"--budget {a.budget} --micro_bs {a.micro_bs} --model {a.model}")
    jobs = []
    for bs in BATCHES:
        for lr in FULL_LRS:
            jobs.append(f"{{PY}} {RUN} --tag {a.tag} --method full --bs {bs} "
                        f"--lr {lr:g} --seed 0 {common}")
        for lr in LORA_LRS:
            jobs.append(f"{{PY}} {RUN} --tag {a.tag} --method lora --bs {bs} "
                        f"--lr {lr:g} --seed 0 --r 16 {common}")
            jobs.append(f"{{PY}} {RUN} --tag {a.tag}_r128 --method lora "
                        f"--bs {bs} --lr {lr:g} --seed 0 --r 128 --alpha 32 "
                        f"{common}")
    print(f"{len(jobs)} jobs")
    cluster.main(jobs, a.tag,
                 f"{REPO}/03-stochastic-batch-geometry/results/{a.tag}/logs",
                 arch=a.arch,
                 hosts=a.hosts.split(",") if a.hosts else None, dry=a.dry)
