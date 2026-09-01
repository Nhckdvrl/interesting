"""Topic 02, E0 -- functional-equivalence check.

No gauge experiment is valid until the transformed model is numerically the
same function.  We measure, on held-out text:
   max |logit difference|, relative logit error, and the token-level NLL gap,
both in the model's native bf16 and in fp32 (to separate "the transformation is
wrong" from "bf16 is coarse").

Run: python 02-representation-gauge/src/e0_exactness.py
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from common.gauge import (random_orthogonal, hadamard, apply_vo_gauge,
                          fold_rmsnorm_gains, apply_residual_gauge)
from common.train import load_model
from common.train import eval_loss as _eval_loss


def eval_loss(m, l, n, d):
    with CTX[0]:
        return _eval_loss(m, l, n, d)
from common.data import build_sft, FixedOrderLoader

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


CTX = [torch.autocast("cuda", enabled=False)]


@torch.no_grad()
def logits_on(model, batches, device="cuda"):
    outs = []
    for b in batches:
        b = {k: v.to(device) for k, v in b.items()}
        with CTX[0]:
            o = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
        outs.append(o.logits.float().cpu())
    return outs


@torch.no_grad()
def compare(l0, l1, masks):
    num, den, mx = 0.0, 0.0, 0.0
    for a, b, m in zip(l0, l1, masks):
        m = m.bool()
        d = (a - b)[m]
        num += float(d.pow(2).sum()); den += float(a[m].pow(2).sum())
        mx = max(mx, float(d.abs().max()))
    return dict(rel_l2=(num / den) ** 0.5, max_abs=mx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--n_batches", type=int, default=8)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(
        REPO, "02-representation-gauge", "results", "e0_exactness.json"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    report = {"model": a.model}

    for dtype_name, dtype in [("bfloat16", torch.bfloat16),
                              ("float32", torch.float32),
                              ("float32_bf16autocast", torch.float32)]:
        autocast = dtype_name.endswith("autocast")
        ctx = (torch.autocast("cuda", torch.bfloat16) if autocast
               else torch.autocast("cuda", enabled=False))
        print(f"\n{'='*72}\ndtype = {dtype_name}\n{'='*72}")
        CTX[0] = ctx
        base, tok = load_model(a.model, dtype=dtype)
        _, te = build_sft(tok, "gsm8k", n_train=8, n_eval=256, max_len=384)
        tel = FixedOrderLoader(te, a.bs, tok.pad_token_id, seed=999)
        batches = [tel.get(i) for i in range(a.n_batches)]
        masks = [b["attention_mask"] for b in batches]
        L0 = logits_on(base, batches)
        nh, nkv, hd = (base.config.num_attention_heads,
                       base.config.num_key_value_heads, base.config.head_dim)
        nlayer = base.config.num_hidden_layers
        d = base.config.hidden_size
        base_loss = eval_loss(base, tel, a.n_batches, "cuda")
        del base; torch.cuda.empty_cache()
        sub = report.setdefault(dtype_name, {})
        sub["base_eval_loss"] = base_loss

        # ---- identity control: reload, do nothing -> pure nondeterminism floor
        m, _ = load_model(a.model, dtype=dtype)
        Lid = logits_on(m, batches)
        sub["identity_reload"] = compare(L0, Lid, masks)
        print(f"  identity reload (numerical floor): {sub['identity_reload']}")
        del m; torch.cuda.empty_cache()

        # ---- G1: V/O gauge, random orthogonal per KV head per layer
        g = torch.Generator().manual_seed(0)
        m, _ = load_model(a.model, dtype=dtype)
        C = [torch.stack([random_orthogonal(hd, g) for _ in range(nkv)])
             for _ in range(nlayer)]
        apply_vo_gauge(m, C)
        sub["vo_random"] = compare(L0, logits_on(m, batches), masks)
        sub["vo_random"]["eval_loss"] = eval_loss(m, tel, a.n_batches, "cuda")
        print(f"  V/O random orthogonal:  {sub['vo_random']}  "
              f"(base loss {base_loss:.6f})")
        del m; torch.cuda.empty_cache()

        # ---- G1b: V/O Hadamard (the QuaRot-style structured rotation)
        m, _ = load_model(a.model, dtype=dtype)
        H = hadamard(hd)
        apply_vo_gauge(m, [torch.stack([H] * nkv) for _ in range(nlayer)])
        sub["vo_hadamard"] = compare(L0, logits_on(m, batches), masks)
        sub["vo_hadamard"]["eval_loss"] = eval_loss(m, tel, a.n_batches, "cuda")
        print(f"  V/O Hadamard:           {sub['vo_hadamard']}")
        del m; torch.cuda.empty_cache()

        # ---- G2a: gamma-folding + untying alone (R = I)
        m, _ = load_model(a.model, dtype=dtype)
        fold_rmsnorm_gains(m)
        Lfold = logits_on(m, batches)
        sub["fold_only"] = compare(L0, Lfold, masks)
        sub["fold_only"]["eval_loss"] = eval_loss(m, tel, a.n_batches, "cuda")
        print(f"  gamma-fold + untie only:{sub['fold_only']}")
        del m; torch.cuda.empty_cache()

        # ---- G2b: residual gauge on top of folding
        for name, Rmk in [("residual_random",
                           lambda: random_orthogonal(d, torch.Generator().manual_seed(1))),
                          ("residual_hadamard", lambda: hadamard(d))]:
            m, _ = load_model(a.model, dtype=dtype)
            fold_rmsnorm_gains(m)
            apply_residual_gauge(m, Rmk())
            sub[name] = compare(Lfold, logits_on(m, batches), masks)
            sub[name]["eval_loss"] = eval_loss(m, tel, a.n_batches, "cuda")
            sub[name]["_reference"] = "folded model"
            print(f"  {name:23s} {sub[name]}")
            del m; torch.cuda.empty_cache()

    json.dump(report, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
