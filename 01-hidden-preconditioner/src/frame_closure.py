"""Does the Adam-visible frame coordinate close the out-of-distribution gap?

(S, D, omega) left a systematic +0.0028 nats on the B_0 = 0 published family,
8/8 of one sign.  All three are gauge invariants, and gauge invariants are
exactly the quantities SGD's first-order descent depends on.  AdamW descends in
the elementwise l_inf geometry, whose dual norm is the elementwise l1 norm of
grad_B = s G A^T, and that is invariant only under signed permutations -- so
AdamW's rate depends on the gauge FRAME, which no invariant can express.

    Lambda_1 = ||G A^T||_1^2 / (d_out r ||G A^T||_F^2)  in (0, 1]

is the exact ratio of AdamW's first-order descent rate to SGD's.  This script
adds it to the law, fits on the synthetic atlas ONLY, and predicts the held-out
published initializers.
"""
import glob, hashlib, json, math, os, statistics as st, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_atlas import ood_points
from ood_closure import refine, NONZERO_B

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(REPO, "01-hidden-preconditioner", "results")
NULL = 2.7e-4


def cache_index():
    """cache-file basename -> the (S, D) it was built at, read straight from
    the construction cache.  Joining on these rather than on a hash of the run
    args is necessary because the cache-key format gained fields (matchW) after
    the earliest atlas points were run, so old runs cannot reproduce their own
    key -- but their stored statistics still identify them exactly."""
    import torch
    from run_atlas import ACACHE
    idx = {}
    for f in sorted(glob.glob(os.path.join(ACACHE, "*.pt"))):
        try:
            c = torch.load(f, map_location="cpu")
        except Exception:
            continue
        mods = sorted(c["stats"])
        idx[os.path.basename(f)[:-3]] = (
            st.mean(c["stats"][n]["S_rel"] for n in mods),
            st.mean(c["stats"][n]["D"] for n in mods),
            st.mean(c["stats"][n]["tr_P"] / c["stats"][n]["trP_ref"]
                    for n in mods))
    return idx


def atlas_rows(tags=("atlas",)):
    SO = json.load(open(os.path.join(RES, "second_order.json")))
    IDX = cache_index()
    pts = {}
    for tag in tags:
        for f in sorted(glob.glob(os.path.join(RES, tag, "*.json"))):
            r = json.load(open(f)); a = r["args"]
            m = list(r["init_stats"].values())
            # the run reports its own statistics over ALL adapted modules while
            # the cache index averages the sampled subset, so match on the
            # closest point in log (S, D, tr P) and require a tight agreement.
            key = (st.mean(x["S_rel"] for x in m), st.mean(x["D"] for x in m),
                   st.mean(x["tr_P"] / x["trP_ref"] for x in m))
            def dist(v):
                return sum(math.log(max(v[i], 1e-12) / max(key[i], 1e-12)) ** 2
                           for i in range(3))
            k = min(IDX, key=lambda z: dist(IDX[z]))
            if dist(IDX[k]) > 0.02 ** 2 or "atlas:" + k not in SO:
                continue
            p = pts.setdefault(k, dict(
                S=key[0], D=key[1], W=key[2] / max(key[0], 1e-9),
                curve={}, **SO["atlas:" + k]))
            p["curve"].setdefault(a["lr"], []).append(r["log"]["final_eval_loss"])
    out = []
    for k, p in pts.items():
        p["curve"] = {lr: st.mean(v) for lr, v in p["curve"].items()}
        if len(p["curve"]) < 3:
            continue
        lr, L, ok = refine(p["curve"])
        if not ok:
            continue
        p["lr_star"], p["L_star"] = lr, L
        out.append(p)
    return out


def feats(p, use_lam):
    lw = math.log(max(p["W"], 1e-9))
    f = [1.0, 1.0 / math.sqrt(max(p["D"], 1e-9)), lw, lw * lw]
    if use_lam:
        f.append(math.log(max(p["Lam1"], 1e-9)))
    return f


def fit_predict(rows, ood, use_lam):
    X = np.array([feats(p, use_lam) for p in rows])
    y = np.array([p["L_star"] for p in rows])
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    tr_rms = float(np.sqrt(np.mean((X @ c - y) ** 2)))
    res = [(k, o["L_star"] - float(np.dot(feats(o, use_lam), c))) for k, o in ood]
    return c, tr_rms, res


def main():
    SO = json.load(open(os.path.join(RES, "second_order.json")))
    rows = atlas_rows()
    O = ood_points()
    ood = []
    for k, o in O.items():
        so = SO.get("lit:" + k)
        if so is None or not o["bracketed"] or o["B0"] > 0 \
                or k.split("|")[0] in NONZERO_B:
            continue
        lr, L, ok = refine(o["curve"])
        if not ok:
            continue
        ood.append((k, dict(D=o["D"], W=o["W"], L_star=L, **so)))
    print(f"fit on {len(rows)} synthetic atlas points; "
          f"test on {len(ood)} held-out published initializers (B_0 = 0)\n")
    lam = [p["Lam1"] for p in rows]
    print(f"  atlas Lambda_1 span {min(lam):.3f}..{max(lam):.3f}; "
          f"test span {min(o['Lam1'] for _, o in ood):.3f}.."
          f"{max(o['Lam1'] for _, o in ood):.3f}  "
          f"({'contained' if min(lam) <= min(o['Lam1'] for _, o in ood) and max(lam) >= max(o['Lam1'] for _, o in ood) else 'EXTRAPOLATING'})\n")
    for use_lam in (False, True):
        c, tr, res = fit_predict(rows, ood, use_lam)
        r = [v for _, v in res]
        nm = "(S, D, omega) + log Lambda_1" if use_lam else "(S, D, omega)"
        print(f"  {nm}")
        print(f"    atlas fit rms          {tr:.5f} nats")
        print(f"    held-out mean residual {st.mean(r):+.5f} nats "
              f"({st.mean(r)/NULL:+.1f}x null)")
        print(f"    held-out rms residual  "
              f"{math.sqrt(sum(v*v for v in r)/len(r)):.5f} nats")
        print(f"    sign of residual       "
              f"{sum(v > 0 for v in r)}/{len(r)} positive")
        if use_lam:
            print(f"    coefficient on log Lambda_1: {c[-1]:+.5f} "
                  f"(negative = higher Lambda_1 trains better)")
            print()
            for k, v in sorted(res, key=lambda x: x[1]):
                print(f"      {k:28s} residual {v:+.5f}")
        print()


if __name__ == "__main__":
    main()
