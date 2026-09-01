"""Topic 03, E1+E2 -- does the LoRA-vs-FullFT large-batch gap reproduce here?

Fixed example budget (32768 NuminaMath-CoT examples, single pass), varying
logical batch size via gradient accumulation, so only the number of optimizer
steps and the minibatch noise level change.  Per (method, batch) LR sweep, so
the gap is a *best-tuned* gap and not a fixed-recipe artifact.

E2 is folded in: r = 16 vs r = 128 at every batch size tests the reported
rank-independence of the penalty.
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from common import cluster

REPO = cluster.REPO
RUN = f"{REPO}/03-stochastic-batch-geometry/src/run_batch.py"

BATCHES = [8, 32, 128, 512]
FULL_LRS = [3e-6, 1e-5, 3e-5, 1e-4, 3e-4]
LORA_LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="e1")
    ap.add_argument("--arch", default="a100")
    ap.add_argument("--hosts", default=None)
    ap.add_argument("--budget", type=int, default=32768)
    ap.add_argument("--micro_bs", type=int, default=16)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    common = (f"--task numina --n_train 40000 --n_eval 512 --max_len 512 "
              f"--budget {a.budget} --micro_bs {a.micro_bs} --model {a.model}")
    jobs = []
    for seed in [int(x) for x in a.seeds.split(",")]:
        for bs in BATCHES:
            for lr in FULL_LRS:
                jobs.append(f"{{PY}} {RUN} --tag {a.tag} --method full --bs {bs} "
                            f"--lr {lr:g} --seed {seed} {common}")
            for lr in LORA_LRS:
                jobs.append(f"{{PY}} {RUN} --tag {a.tag} --method lora --bs {bs} "
                            f"--lr {lr:g} --seed {seed} --r 16 {common}")
                jobs.append(f"{{PY}} {RUN} --tag {a.tag}_r128 --method lora "
                            f"--bs {bs} --lr {lr:g} --seed {seed} --r 128 "
                            f"--alpha 32 {common}")
    cluster.main(jobs, a.tag,
                 f"{REPO}/03-stochastic-batch-geometry/results/{a.tag}/logs",
                 arch=a.arch,
                 hosts=a.hosts.split(",") if a.hosts else None, dry=a.dry)
