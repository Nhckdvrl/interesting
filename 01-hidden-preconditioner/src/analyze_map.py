"""The optimizer->class map on one training family, each optimizer against ITS OWN
floor.

Reads a panel of run_lit outputs (frame conditions kaiming/frame0/frame0.5/frame1
plus signperm floor cells across five optimizers) and, for each optimizer:

  * finds the tuned lr = the rung with the lowest min-loss across the frames;
  * reports the frame spread (max-min final eval loss) there;
  * reports that optimizer's OWN reproducibility floor = the spread across
    signed-permutation gauges (signperm1/2/3), which are exactly covariant for
    every optimizer here, measured at the signperm lr nearest the tuned point;
  * flags whether the tuned lr is INTERIOR (an edge optimum is not an optimum);
  * dumps the per-lr frame spread so stability-edge float chaos (spread ~0 at low
    lr, growing toward divergence) is distinguishable from a real, lr-stable
    frame dependence.

This is the second standing rule made executable: never a floor imported across
optimizers, never a claim read off a grid edge.

    python analyze_map.py <tag>      # e.g. olmo, llama_map, q8b_fp32
"""
import glob, json, os, sys, statistics as st

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FRAMES = ["kaiming", "frame0", "frame0.5", "frame1"]
OPTS = {"adamw": "", "lion": "_lion", "matprec": "_matprec",
        "muon": "_muon", "sgd": "_sgd"}


def loss(f, denoise=False, k=4):
    """Final eval loss (default), or the mean of the last k eval points if
    denoise=True.

    Default is the final-step snapshot, to stay consistent with the numbers in
    the paper sections.  The tail-mean is a cross-check for one specific failure:
    near a stability edge a single gauge seed's final snapshot can sit on an
    up-bounce and inflate the *floor* (at 8B, signperm2's snapshot alone swung the
    AdamW floor 0.0006 -> 0.006; the tail-mean brings it to 0.0018).  The
    condition *spreads* are common-mode in the eval noise and barely move either
    way -- so denoise is diagnostic for the floor, not a metric change.  Note the
    ratio-to-own-floor is itself unreliable when the floor is near zero (a blind
    optimizer whose gauge copies cohere to 1e-6): read the map from the raw spread
    (sees ~2e-3 vs blind ~5e-5), not the ratio, for those."""
    try:
        log = json.load(open(f))["log"]
        if denoise:
            c = log.get("eval_loss") or []
            if len(c) >= k:
                v = st.mean(c[-k:])
                return v if v == v else None
        v = log["final_eval_loss"]
        return v if v == v else None          # drop NaN
    except Exception:
        return None


def _lr(basename, suf):
    return float(basename.split("_lr")[1].split("_s0")[0])


def cells(res, cond, suf):
    out = {}
    for f in glob.glob(f"{res}/{cond}_lr*_s0{suf}.json"):
        v = loss(f)
        if v is not None:
            out[_lr(os.path.basename(f), suf)] = v
    return out


def floor(res, suf, tuned):
    sp = {}
    for i in (1, 2, 3):
        for f in glob.glob(f"{res}/signperm{i}_lr*_s0{suf}.json"):
            v = loss(f)
            if v is not None:
                sp.setdefault(_lr(os.path.basename(f), suf), []).append(v)
    if not sp:
        return None, None, 0
    cands = [lr for lr in sp if len(sp[lr]) >= 2] or list(sp)
    flr = min(cands, key=lambda lr: abs(lr - tuned))
    vals = sp[flr]
    return (max(vals) - min(vals) if len(vals) >= 2 else None), flr, len(vals)


def main(tag):
    res = os.path.join(REPO, "01-hidden-preconditioner", "results", tag)
    print(f"# optimizer -> class map on {tag}\n")
    print(f"{'opt':8s} {'tunedLR':>8s} {'spread':>9s} {'floor':>9s} "
          f"{'(lr,n)':>12s} {'ratio':>7s}  {'interior':>10s}")
    for opt, suf in OPTS.items():
        bylr = {}
        for c in FRAMES:
            for lr, v in cells(res, c, suf).items():
                bylr.setdefault(lr, {})[c] = v
        full = {lr: d for lr, d in bylr.items() if len(d) == len(FRAMES)} \
            or {lr: d for lr, d in bylr.items() if len(d) >= 3}
        if not full:
            print(f"{opt:8s}  (no full-frame lr yet)")
            continue
        tuned = min(full, key=lambda lr: min(full[lr].values()))
        d = full[tuned]
        spread = max(d.values()) - min(d.values())
        lrs = sorted(full)
        interior = "yes" if tuned not in (lrs[0], lrs[-1]) else f"EDGE"
        flr, flr_lr, n = floor(res, suf, tuned)
        fstr = f"{flr:.5f}" if flr is not None else ("n=1" if n == 1 else "none")
        lrn = f"({flr_lr:g},n{n})" if flr_lr else "-"
        ratio = f"{spread/flr:6.1f}x" if flr and flr > 0 else "   -"
        # SGD is the exactly-covariant control: its spread IS a floor, so a ratio
        # against a smaller signperm spread is a divide-by-noise artefact.
        if opt == "sgd":
            ratio = "(floor)"
        print(f"{opt:8s} {tuned:8g} {spread:9.5f} {fstr:>9s} {lrn:>12s} "
              f"{ratio:>7s}  {interior:>10s}")
        per = "   per-lr spread:"
        for lr in sorted(bylr):
            dd = bylr[lr]
            if len(dd) >= 2:
                per += f"  {lr:g}:{max(dd.values())-min(dd.values()):.4f}"
        print(per)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "olmo")
