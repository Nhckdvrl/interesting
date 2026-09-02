"""How much room does the gauge orbit leave, as a function of rank?

The frame effect must be zero when the orbit is a point and larger when it is
large.  The span of Lambda_1 over the orbit -- from the gradient-metric
eigenframe to the flat-diagonal frame -- is measurable from one probe
forward+backward, with no training, so if it tracks the measured effect it is a
diagnostic a practitioner can run before deciding whether rotating is worth it.

The rank series is the test: measured frame1 - frame0 at lr = 1e-4 was
0.00000 / 0.00044 / 0.00160 / 0.00402 nats at r = 1 / 4 / 16 / 64.  At r = 1 the
reach is exactly 1.0 by construction, since O(1) is AdamW's own symmetry group,
so the prediction of zero there is structural rather than fitted.
"""
import argparse, hashlib, json, os, statistics as st, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from common.pinit import kaiming_A
from common.intrinsic import frame_ladder, l1_flatness, offdiag_mass
from common.data import build_sft, FixedOrderLoader
from common.train import load_model
from run_lit import collect_grads, ACT_GROUP

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--task", default="gsm8k")
    ap.add_argument("--ranks", default="1,4,16,64,128")
    ap.add_argument("--every_layer", type=int, default=4)
    ap.add_argument("--probe_batches", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(
        REPO, "01-hidden-preconditioner", "results", "frame_reach.json"))
    a = ap.parse_args()

    model, tok = load_model(a.model, dtype=torch.float32)
    tr, _ = build_sft(tok, a.task, 6000, 256, 384, seed=0)
    ld = FixedOrderLoader(tr, 16, tok.pad_token_id, seed=0)
    T = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
         "down_proj")
    allm = {n: m for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and n.split(".")[-1] in T}
    G = collect_grads(model, allm, ld, a.probe_batches)
    mods = {n: m for n, m in allm.items()
            if int(n.split("layers.")[1].split(".")[0]) % a.every_layer == 0}
    del model
    torch.cuda.empty_cache()
    print(f"{len(mods)} sampled modules\n")

    rows = {}
    print(f"{'r':>5s} {'dim quotient':>13s} {'Lam1 eig':>9s} {'Lam1 flat':>10s} "
          f"{'reach':>7s} {'Off_g flat':>11s}")
    for r in [int(x) for x in a.ranks.split(",")]:
        lo, hi, og = [], [], []
        for n, m in mods.items():
            d_in = m.weight.shape[1]
            h = int(hashlib.md5(f"{a.seed}:{n}".encode()).hexdigest()[:12], 16)
            A = kaiming_A(r, d_in, torch.Generator().manual_seed(h),
                          "cpu").double().cuda()
            Gd = G[n].cuda().double()
            GA = Gd @ A.T
            Q0, Q1 = frame_ladder(GA.T @ GA, [0.0, 1.0])
            lo.append(l1_flatness(Q0 @ A, Gd)[0])
            hi.append(l1_flatness(Q1 @ A, Gd)[0])
            A1 = Q1 @ A
            og.append(offdiag_mass((Gd @ A1.T).T @ (Gd @ A1.T)))
        e, f = st.mean(lo), st.mean(hi)
        rows[r] = dict(lam1_eig=e, lam1_flat=f, reach=f / max(e, 1e-30),
                       offg_flat=st.mean(og), dim_quotient=r * (r - 1) // 2)
        print(f"{r:5d} {r*(r-1)//2:13d} {e:9.4f} {f:10.4f} {f/max(e,1e-30):7.3f} "
              f"{st.mean(og):11.4f}")
    json.dump(rows, open(a.out, "w"), indent=2)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
