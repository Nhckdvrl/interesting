"""Topic 02 -- why is the pretrained basis privileged for Adam?

Hypothesis.  AdamW's per-coordinate second moment v ~ E[g^2] is a *diagonal*
model of the gradient scale.  It is useful exactly to the extent that the
gradient's coordinate-wise scales are heterogeneous.  A pretrained transformer
has strongly heterogeneous input coordinates (outlier features, neuron-aligned
structure); an orthogonal rotation that mixes coordinates averages that
heterogeneity away by concentration of measure, so the diagonal model becomes
less informative and Adam loses part of its advantage.  SGD does not use a
diagonal model and is exactly covariant, which is why it is flat.

Prediction: the fine-tuning penalty of a gauge should be monotone in how much
that gauge homogenises the per-coordinate gradient scale.  We measure that with
the participation ratio of the input-coordinate gradient energies

    E_j = || G e_j ||^2 ,     PR = (sum_j E_j)^2 / (d_in * sum_j E_j^2)

PR = 1 means perfectly uniform (nothing for a diagonal preconditioner to
exploit); small PR means concentrated.  A permutation leaves PR exactly
invariant -- matching the fact that AdamW is exactly covariant under it.
"""
import argparse, json, os, statistics as st, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn
from common.gauge import make_R, fold_rmsnorm_gains
from common.data import build_sft, FixedOrderLoader
from common.train import load_model

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GAUGES = ["none", "perm", "block4", "block16", "block64", "block256", "rand",
          "hadamard"]
MIX = {"none": 1, "perm": 1, "block4": 4, "block16": 16, "block64": 64,
       "block256": 256, "rand": 1024, "hadamard": 1024}
# modules whose INPUT is the residual stream (so a residual gauge acts on the
# input coordinate axis of their gradient)
IN_SIDE = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")


def pr(E):
    """participation ratio of a nonnegative vector, in (0, 1]."""
    s1 = float(E.sum()); s2 = float((E * E).sum())
    return (s1 * s1) / (len(E) * s2 + 1e-30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--task", default="numina")
    ap.add_argument("--n_batches", type=int, default=4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(
        REPO, "02-representation-gauge", "results", "diag_dominance.json"))
    a = ap.parse_args()

    model, tok = load_model(a.model, dtype=torch.float32)
    fold_rmsnorm_gains(model)
    d = model.config.hidden_size
    tr, _ = build_sft(tok, a.task, 40000, 512, 384, seed=0)
    ld = FixedOrderLoader(tr, a.bs, tok.pad_token_id, seed=0)
    mods = {n: m for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and n.split(".")[-1] in IN_SIDE}
    for p in model.parameters():
        p.requires_grad_(False)
    for m in mods.values():
        m.weight.requires_grad_(True)
    for i in range(a.n_batches):
        b = {k: v.to("cuda") for k, v in ld.get(i).items()}
        o = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
        lg = o.logits[:, :-1].float(); lb = b["labels"][:, 1:]
        msk = lb != -100
        (nn.functional.cross_entropy(lg[msk], lb[msk]) / a.n_batches).backward()
    G = {n: m.weight.grad.detach().float() for n, m in mods.items()}
    print(f"gradients for {len(G)} residual-reading modules")

    rows = {}
    g = torch.Generator().manual_seed(1000)
    for gauge in GAUGES:
        R = (torch.eye(d, dtype=torch.float64) if gauge == "none"
             else make_R(gauge, d, torch.Generator().manual_seed(1000)))
        Rf = R.float().cuda()
        prs, kurt = [], []
        for n, Gi in G.items():
            GR = Gi @ Rf                       # gradient in the rotated basis
            E = GR.pow(2).sum(0)               # energy per input coordinate
            prs.append(pr(E))
            kurt.append(float((E / E.mean()).pow(2).mean()))
        rows[gauge] = dict(mix=MIX[gauge], PR=st.mean(prs),
                           energy_kurtosis=st.mean(kurt))
        print(f"  {gauge:10s} mix={MIX[gauge]:5d}  PR={rows[gauge]['PR']:.5f}  "
              f"E-kurtosis={rows[gauge]['energy_kurtosis']:8.3f}")
    json.dump(rows, open(a.out, "w"), indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
