"""Read the GroupAdam panels: the symmetry-resolution staircase and the
initializer-ranking phase diagram.

Floor discipline, as everywhere else in this project, but here the floor is
built into the design rather than bolted on: the k = 1 rotation is a sign flip,
which lies in O(1)^r and is therefore invisible to EVERY rung of the ladder.
So for each b, the k = 1 cell is that optimizer's own reproducibility floor,
measured inside the panel at the same learning rate as the cells it judges.
Nothing is imported across optimizers, models or precisions.

    staircase  prediction: |L(blockrot_k) - L(kaiming)| is at the b-rung's own
               floor iff k <= b, and above it iff k > b.

    ranking    each initialiser tuned separately (the project's per-method
               learning-rate rule), optimum required to be INTERIOR, and the
               ranking gaps judged against the same in-panel floor.
"""
import glob, json, os, sys

RES = os.path.join(os.path.dirname(__file__), "..", "results")
BS = (1, 2, 4, 8, 16)


def L(path):
    try:
        v = json.load(open(path))["log"]["final_eval_loss"]
        return v if v == v else None
    except Exception:
        return None


def staircase(tag="ga_stair", lr=3e-4):
    d = os.path.join(RES, tag)
    print(f"=== {tag}: symmetry-resolution staircase ===")
    print("cell = |L(blockrot_k) - L(kaiming)| ; floor = the k=1 cell "
          "(a sign flip, invisible to every rung)\n")
    rows, ok, missing = {}, True, 0
    for b in BS:
        base = L(os.path.join(d, f"kaiming_lr{lr:g}_s0_groupadam{b}.json"))
        if base is None:
            missing += 1
            continue
        devs = {}
        for k in BS:
            v = L(os.path.join(d, f"blockrot{k}_lr{lr:g}_s0_groupadam{b}.json"))
            devs[k] = None if v is None else abs(v - base)
        rows[b] = (base, devs)
    if not rows:
        print("no complete rungs yet"); return None
    print("      " + "".join(f"{('k=' + str(k)):>12s}" for k in BS))
    for b, (base, devs) in sorted(rows.items()):
        line = f" b={b:<3d}"
        for k in BS:
            line += f"{'--':>12s}" if devs[k] is None else f"{devs[k]:>12.2e}"
        print(line)
    print("\nagainst each rung's own floor (the k=1 cell):")
    print("      " + "".join(f"{('k=' + str(k)):>12s}" for k in BS))
    for b, (base, devs) in sorted(rows.items()):
        fl = devs.get(1)
        line = f" b={b:<3d}"
        for k in BS:
            if devs[k] is None or fl is None:
                line += f"{'--':>12s}"; continue
            # "at floor" = within 2x the k=1 cell (which is pure float noise)
            at_floor = devs[k] <= max(2 * fl, 1e-12)
            want = (k <= b)
            good = (at_floor == want)
            ok &= good
            mark = ("0" if at_floor else "X") + ("" if good else "!")
            line += f"{mark:>12s}"
        print(line)
    print(f"\nfloors (k=1 cell per rung): " +
          "  ".join(f"b={b}:{devs.get(1):.2e}" for b, (_, devs) in
                    sorted(rows.items()) if devs.get(1) is not None))
    print(f"staircase {'HOLDS on the real model' if ok else 'VIOLATED'}"
          f"  (0 = at floor, X = visible, ! = contradicts theorem)")
    if missing:
        print(f"({missing} rungs still missing their kaiming baseline)")
    return ok


def ranking(tag, inits=("gradsub", "eva", "pissa"),
            match={"gradsub": "trace", "eva": "trace", "pissa": "none"},
            bs=(1, 4, 16), lrs=(1e-4, 2e-4, 3e-4, 5e-4)):
    d = os.path.join(RES, tag)
    print(f"\n=== {tag}: initializer ranking vs symmetry resolution ===")
    print("each initialiser tuned separately; optimum must be INTERIOR\n")
    for b in bs:
        cells, edge = {}, []
        for m in inits:
            suf = "" if match[m] == "trace" else f"_m{match[m]}"
            byl = {}
            for lr in lrs:
                v = L(os.path.join(d, f"{m}_lr{lr:g}_s0{suf}_groupadam{b}.json"))
                if v is not None:
                    byl[lr] = v
            if not byl:
                continue
            best = min(byl, key=byl.get)
            slr = sorted(byl)
            if len(byl) >= 2 and best in (slr[0], slr[-1]):
                edge.append(f"{m}@{best:g}")
            cells[m] = (best, byl[best], len(byl))
        if not cells:
            print(f" b={b:<3d}  (no cells yet)"); continue
        order = sorted(cells, key=lambda m: cells[m][1])
        line = f" b={b:<3d}  " + "  ".join(
            f"{m}={cells[m][1]:.5f}@{cells[m][0]:g}" for m in order)
        print(line)
        print(f"        ranking: {' < '.join(order)}"
              + (f"   [EDGE optima: {', '.join(edge)}]" if edge else "")
              + f"   ({sum(c[2] for c in cells.values())} cells)")
    # in-panel floor: blockrot1 vs kaiming, if the second round has landed
    print("\n in-panel floor (blockrot1 vs kaiming, per b):")
    for b in bs:
        found = []
        for lr in lrs:
            base = L(os.path.join(d, f"kaiming_lr{lr:g}_s0_groupadam{b}.json"))
            rot = L(os.path.join(d, f"blockrot1_lr{lr:g}_s0_groupadam{b}.json"))
            if base is not None and rot is not None:
                found.append((lr, abs(rot - base)))
        print(f"   b={b:<3d} " + ("  ".join(f"lr{lr:g}:{v:.2e}"
                                            for lr, v in found)
                                  if found else "not measured yet"))


def collapse(tag="ga_stair", lr=3e-4):
    """Section 6's frame ladder as a function of the optimizer's symmetry group.

    frame0/frame1 are full O(r) gauge moves, so they are visible to GroupAdam_b
    for every b < r and must be EXACTLY invisible at b = r.  The floor is the
    same panel's blockrot1 cell (a sign flip, invisible to every rung) at the
    same lr, so nothing is imported.
    """
    d = os.path.join(RES, tag)
    print(f"\n=== {tag}: does the frame ladder collapse as b grows? ===")
    print("spread over {kaiming, frame0, frame1} vs each rung's own "
          "sign-flip floor\n")
    print(f"{'b':>4s} {'kaiming':>10s} {'frame0':>10s} {'frame1':>10s} "
          f"{'spread':>10s} {'floor':>10s} {'ratio':>8s}")
    for b in BS:
        vals = {}
        for c in ("kaiming", "frame0", "frame1"):
            v = L(os.path.join(d, f"{c}_lr{lr:g}_s0_groupadam{b}.json"))
            if v is not None:
                vals[c] = v
        if len(vals) < 3:
            print(f"{b:>4d} " + "  incomplete "
                  f"({len(vals)}/3 cells)"); continue
        spread = max(vals.values()) - min(vals.values())
        # floor = spread over gauge-EQUIVALENT runs: kaiming and its sign-flip
        # draws (blockrot1, which lies in O(1)^r and is invisible to every
        # rung).  Same lr, same optimizer, measured inside the panel.
        eq = [vals["kaiming"]]
        for gs in (0, 1, 2):
            suf = "" if gs == 0 else f"_g{gs}"
            v = L(os.path.join(d,
                               f"blockrot1_lr{lr:g}_s0{suf}_groupadam{b}.json"))
            if v is not None:
                eq.append(v)
        fl = (max(eq) - min(eq)) if len(eq) >= 2 else None
        ratio = f"{spread / fl:.1f}x" if fl else "--"
        print(f"{b:>4d} {vals['kaiming']:>10.5f} {vals['frame0']:>10.5f} "
              f"{vals['frame1']:>10.5f} {spread:>10.2e} "
              f"{(f'{fl:.2e}' if fl else '--'):>10s} {ratio:>8s}")
    print("\nprediction: spread stays ~2e-3 for b < 16 and collapses to the "
          "floor at b = 16,\nwhere the whole gauge group is invisible.")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "stair"):
        staircase()
    if which in ("all", "stair", "collapse"):
        collapse()
    if which in ("all", "rank"):
        for t in ("ga_rank_olmo", "ga_rank_llama"):
            if os.path.isdir(os.path.join(RES, t)):
                ranking(t)
