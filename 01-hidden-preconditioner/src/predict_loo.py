"""Topic 01 -- leave-one-out prediction: does method identity add anything?

We fit the best-tuned loss of an initializer from three initialisation-time
statistics only,

    loss ~ a + b / sqrt(r_eff^Sigma)  +  c * log tr(P Sigma)  +  d * 1[||B0||>0]

on all methods but one, and predict the held-out method.  If the residual is of
the order of the AdamW gauge null (2.7e-4 nats), then knowing WHICH published
initializer produced A_0 adds nothing beyond those three numbers.
"""
import glob, json, math, os, statistics as st, sys
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(REPO, "01-hidden-preconditioner", "results")


def best_by_cond(tag, match):
    rows = {}
    for f in sorted(glob.glob(os.path.join(RES, tag, "*.json"))):
        r = json.load(open(f)); a = r["args"]
        if a.get("match", "trace") != match:
            continue
        rows.setdefault(a["cond"], {}).setdefault(a["lr"], []).append(
            r["log"]["final_eval_loss"])
    out = {}
    for c, d in rows.items():
        m = {lr: st.mean(v) for lr, v in d.items()}
        blr = min(m, key=m.get)
        if blr in (min(m), max(m)):        # optimum not bracketed
            continue
        out[c] = m[blr]
    return out


def design(P, c):
    p = P[c]
    return [1.0,
            1.0 / math.sqrt(p["r_eff_act"]),
            math.log(p["rel_act"]),
            1.0 if p["B0"] > 1e-6 else 0.0]


def main(tag="lit", match="trace"):
    P = json.load(open(os.path.join(RES, "pstat_table_trace.json")))
    B = best_by_cond(tag, match)
    names = [c for c in B if c in P]
    X = np.array([design(P, c) for c in names])
    y = np.array([B[c] for c in names])
    print(f"{tag} / match={match}: {len(names)} initializers with a bracketed "
          f"optimum\n")
    print(f"  {'held-out method':24s} {'actual':>9s} {'predicted':>10s} "
          f"{'error':>9s} {'in nulls':>9s}")
    NULL = 2.7e-4
    errs = []
    for i, c in enumerate(names):
        m = np.ones(len(names), bool); m[i] = False
        coef, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
        pred = float(X[i] @ coef)
        e = y[i] - pred
        errs.append(e)
        print(f"  {c:24s} {y[i]:9.5f} {pred:10.5f} {e:+9.5f} {e/NULL:+9.1f}")
    rms = float(np.sqrt(np.mean(np.square(errs))))
    print(f"\n  LOO rms error            = {rms:.5f} nats  = {rms/NULL:.1f} nulls")
    print(f"  spread of the targets    = {y.max()-y.min():.5f} nats")
    print(f"  variance explained (LOO) = "
          f"{1 - np.var(errs)/np.var(y):.3f}")
    # the same, with only an intercept (i.e. "all methods are the same")
    e0 = y - y.mean()
    print(f"  baseline (predict the mean) rms = {float(np.sqrt(np.mean(e0**2))):.5f}")
    # and with method identity but no statistics: impossible to beat LOO,
    # reported to show the design is not degenerate
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    print(f"\n  in-sample coefficients [1, 1/sqrt(r_eff^Sigma), "
          f"log tr(PSigma), 1(B0)]:")
    print("   ", " ".join(f"{c:+.5f}" for c in coef))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "lit",
         sys.argv[2] if len(sys.argv) > 2 else "trace")
