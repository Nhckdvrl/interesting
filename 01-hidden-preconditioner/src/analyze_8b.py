"""Qwen3-8B: the four pre-registered predictions in paper/8b_predictions.md.

P1  the AdamW frame spread is COMPARABLE to 0.6B (0.001-0.005 nats), not an
    order of magnitude larger -- the training-free frame reach is 4.57x at both
    0.6B and 1.7B, so the dose does not grow with scale
P2  frame0 beats kaiming by at least the measurement null
P3  SGD is flat to within 1e-3 nats.  This one is not about size: the
    trajectories are gauge-equivalent by derivation, so a failure is an
    implementation fault, not a finding
P4  the ordering follows Off_g, with the minimum at frame0 where Off_g = 0.
    Off_g was reached post-hoc at 0.6B, so this is its first out-of-sample test
"""
import glob, json, os, sys, collections
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(REPO, "01-hidden-preconditioner", "results")
NULL = 2.7e-4
SPREAD_06B = 0.00222


def main(tag="q8b"):
    D = collections.defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(RES, tag, "*.json"))):
        r = json.load(open(f)); a = r["args"]
        key = (a.get("optimizer", "adamw"), a["cond"], a.get("amp", "none"))
        D[key][a["lr"]] = r["log"]["eval_loss"][-1]
    conds = ["frame0", "frame1", "kaiming"]
    for opt, amp, lrs in (("adamw", "bf16", (1e-4, 2e-4, 3e-4, 5e-4)),
                          ("sgd", "bf16", (0.03, 0.1)),
                          ("sgd", "none", (0.03, 0.1))):
        rows = [(c, D[(opt, c, amp)]) for c in conds if D.get((opt, c, amp))]
        if not rows:
            continue
        print(f"\n{opt.upper()} (amp={amp})")
        print(f"{'cond':>9s} | " + " ".join(f"{l:>9.0e}" for l in lrs)
              + f" | {'tuned':>9s}")
        best = {}
        for c, cur in rows:
            b = min(cur.values()); best[c] = b
            lo = sorted(cur)
            edge = "*" if min(cur, key=cur.get) in (lo[0], lo[-1]) else " "
            print(f"{c:>9s} | " + " ".join(f"{cur.get(l, float('nan')):9.5f}"
                                           for l in lrs) + f" | {b:9.5f}{edge}")
        if len(best) >= 2:
            sp = max(best.values()) - min(best.values())
            print(f"  spread {sp:.5f} nats = {sp/NULL:.1f}x the 0.6B null")
            if opt == "adamw":
                ok1 = 0.001 <= sp <= 0.005
                print(f"  P1 (comparable to 0.6B's {SPREAD_06B:.5f}, in "
                      f"[0.001, 0.005]): {'PASS' if ok1 else 'FAIL'}")
                if "frame0" in best and "kaiming" in best:
                    d = best["frame0"] - best["kaiming"]
                    print(f"  P2 (frame0 beats kaiming): {d:+.5f} nats "
                          f"({d/NULL:+.1f}x null) -> "
                          f"{'PASS' if d < -NULL else 'FAIL'}")
                    arg = min(best, key=best.get)
                    print(f"  P4 (minimum at frame0, where Off_g = 0): "
                          f"argmin is {arg} -> "
                          f"{'PASS' if arg == 'frame0' else 'FAIL'}")
            else:
                print(f"  P3 (SGD flat to 1e-3): "
                      f"{'PASS' if sp < 1e-3 else 'FAIL'}"
                      + ("   NOTE: bf16 matmuls raise the floor here; the "
                         "fp32 arm is the sharp test" if amp == "bf16" else ""))
    print("\n(* = tuned learning rate at the edge of the grid)")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
