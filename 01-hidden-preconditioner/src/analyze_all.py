"""The frame effect across every model and task we have run.

One row per (model, task, optimizer).  This is the table that decides whether
the gauge result is a statement about LoRA or a statement about Qwen3 on GSM8K,
so it refuses to report a row whose conditions do not share a bracketed
learning-rate grid -- a truncated grid already reversed one conclusion in this
project.
"""
import glob, json, os, statistics as st, sys, collections
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(REPO, "01-hidden-preconditioner", "results")
FLOOR = 2e-4          # measured reproducibility floor of a 300-step run

PANELS = [("frame", "Qwen3-0.6B", "gsm8k"), ("q8b", "Qwen3-8B", "gsm8k"),
          ("q8bprobe", "Qwen3-8B (4x probe)", "gsm8k"),
          ("llama3b", "Llama-3.2-3B", "gsm8k"),
          ("dolly_frame", "Qwen3-0.6B", "dolly"),
          ("task_metamath", "Qwen3-0.6B", "metamath"),
          ("task_codefeedback", "Qwen3-0.6B", "codefeedback"),
          ("long", "Qwen3-0.6B (1000 steps)", "gsm8k")]


def cell(tag):
    D = collections.defaultdict(dict)
    for f in glob.glob(os.path.join(RES, tag, "*.json")):
        r = json.load(open(f)); a = r["args"]
        if a.get("seed", 0) != 0:
            continue
        D[(a.get("optimizer", "adamw"), a["cond"])][a["lr"]] = \
            r["log"]["eval_loss"][-1]
    return D


def main():
    print(f"{'model':>22s} {'task':>13s} {'opt':>6s} | "
          f"{'kaiming':>9s} {'frame0':>9s} {'delta':>9s} {'x floor':>8s}  note")
    for tag, model, task in PANELS:
        D = cell(tag)
        for opt in ("adamw", "sgd", "muon"):
            k, f0 = D.get((opt, "kaiming")), D.get((opt, "frame0"))
            if not (k and f0):
                continue
            shared = sorted(set(k) & set(f0))
            if len(shared) < 3:
                note = f"only {len(shared)} shared lr"
                print(f"{model:>22s} {task:>13s} {opt:>6s} | "
                      f"{'':>9s} {'':>9s} {'':>9s} {'':>8s}  {note}")
                continue
            bk = min(k[l] for l in shared); bf = min(f0[l] for l in shared)
            edge = (min(shared, key=lambda l: k[l]) in (shared[0], shared[-1])
                    or min(shared, key=lambda l: f0[l]) in (shared[0],
                                                            shared[-1]))
            d = bf - bk
            note = "GRID EDGE -- provisional" if edge else \
                   ("frame0 better" if d < -FLOOR else
                    "kaiming better" if d > FLOOR else "within the floor")
            print(f"{model:>22s} {task:>13s} {opt:>6s} | "
                  f"{bk:9.5f} {bf:9.5f} {d:+9.5f} {d/FLOOR:+8.1f}  {note}")
    print(f"\nfloor = {FLOOR:.0e} nats, measured from two identity rotations "
          f"in results/rot.\nA row marked GRID EDGE has no tuned comparison "
          f"yet and must not be read as a result.")


if __name__ == "__main__":
    main()
