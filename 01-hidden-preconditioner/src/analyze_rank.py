"""The frame effect scales with the size of the symmetry quotient.

What the frame ladder can move is not O(r) but the quotient of O(r) by AdamW's
own symmetry group, the signed permutations: a gauge move inside that subgroup
is invisible to AdamW too.  At r = 1 the quotient is trivial -- O(1) = {+-1} IS
the signed-permutation group -- and the invariance is exact, not approximate:
with B_0 = 0, flipping A's sign flips grad_B, flips AdamW's sign-like step,
flips B, and leaves B A identical.

So the effect must be structurally ZERO at rank 1 and grow with r(r-1)/2.  No
confound of the frame ladder -- learning-rate scale, initialisation magnitude,
data order, gradient noise -- has that signature.
"""
import glob, json, os, sys, collections
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(REPO, "01-hidden-preconditioner", "results")
NULL = 2.7e-4


def main(tag="rank"):
    D = collections.defaultdict(dict)
    for f in glob.glob(os.path.join(RES, tag, "*.json")):
        r = json.load(open(f)); a = r["args"]
        D[(a["r"], a["cond"])][a["lr"]] = r["log"]["eval_loss"][-1]
    lrs = (1e-4, 2e-4, 3e-4, 5e-4)
    print(f"{'r':>4s} {'dim O(r)/signed perms':>22s} | "
          + " ".join(f"{l:>9.0e}" for l in lrs)
          + f" | {'tuned gap':>10s} {'x null':>7s}")
    for r in sorted({k[0] for k in D}):
        a0, a1 = D.get((r, "frame0")), D.get((r, "frame1"))
        if not a0 or not a1:
            continue
        gap = abs(min(a1.values()) - min(a0.values()))
        print(f"{r:4d} {r*(r-1)//2:22d} | "
              + " ".join(f"{a1.get(l, float('nan')) - a0.get(l, float('nan')):+9.5f}"
                         for l in lrs)
              + f" | {gap:10.5f} {gap/NULL:7.1f}")
    print("\ncolumns are frame1 - frame0 at each learning rate.  At r = 1 the two")
    print("runs are the same run: the rotation does not exist.")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
