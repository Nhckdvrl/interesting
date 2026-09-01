"""Topic 02 -- gauge dose-response runner.

Everything here is an EXACT function-preserving reparameterisation of the same
pretrained model (verified to 7 significant figures in fp32 by e0_exactness).
The only thing that changes across conditions is which orthonormal basis the
residual stream is expressed in.

Why the ladder matters.  Under a *permutation* (or signed permutation) of the
residual stream, AdamW is exactly covariant: m/sqrt(v) is elementwise, so
permuting the coordinates permutes the update.  Under a rotation that *mixes*
coordinates it is not.  Sweeping the block size k of a block-diagonal rotation
therefore sweeps the "dose" of coordinate mixing while every rung remains an
exact gauge and every rung has the same spectrum, the same norms, and the same
function.  k = 1 is a built-in zero-dose control.

Plain SGD (with or without momentum), decoupled weight decay and global-norm
gradient clipping are all exactly covariant, so under SGD every rung must give
the same answer -- that is the positive control that separates a real optimizer
effect from an implementation or numerical artifact.
"""
import argparse, hashlib, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn
from common.gauge import (make_R, fold_rmsnorm_gains, apply_residual_gauge,
                          residual_gauge_adapter_maps)
from common.lora import apply_lora, lora_parameters, DEFAULT_TARGETS
from common.pinit import kaiming_A, normalize_columns
from common.data import build_sft, FixedOrderLoader
from common.train import load_model, train, eval_loss, set_amp

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def full_ft_params(model, targets):
    for p in model.parameters():
        p.requires_grad_(False)
    ps = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and name.split(".")[-1] in targets:
            mod.weight.requires_grad_(True)
            ps.append(mod.weight)
    return ps


def make_factory(init, seed, maps):
    store = {}

    def factory(name, r, d_in, d_out):
        h = int(hashlib.md5(f"{seed}:{name}".encode()).hexdigest()[:12], 16)
        g = torch.Generator().manual_seed(h)
        A = kaiming_A(r, d_in, g, "cpu")
        if init == "nora":
            A = normalize_columns(A)
        m = maps.get(name)
        if m is not None and m[0] == "right":
            A = A @ m[1].double()          # coupled: same adapter function
        store[name] = float(A.pow(2).sum())
        return A.float()
    factory.stats = store
    return factory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--task", default="numina")
    ap.add_argument("--method", default="lora", choices=["full", "lora", "nora"])
    ap.add_argument("--gauge", default="none")   # none|perm|block<k>|rand|hadamard
    ap.add_argument("--gauge_seed", type=int, default=0)
    ap.add_argument("--optimizer", default="adamw")
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--micro_bs", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--sched", default="cosine")
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--amp", default="none", choices=["none", "bf16"])
    ap.add_argument("--n_train", type=int, default=40000)
    ap.add_argument("--n_eval", type=int, default=512)
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--eval_batches", type=int, default=24)
    ap.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    args = ap.parse_args()

    cell = (f"{args.method}_{args.optimizer}_{args.gauge}g{args.gauge_seed}_"
            f"lr{args.lr:g}_s{args.seed}")
    outdir = os.path.join(REPO, "02-representation-gauge", "results", args.tag)
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, cell + ".json")
    if os.path.exists(outfile):
        print("skip (exists)", outfile); return

    set_amp(args.amp)
    torch.manual_seed(args.seed)
    model, tok = load_model(args.model, dtype=torch.float32)
    d = model.config.hidden_size
    # gamma-folding + untying is applied in EVERY condition (including
    # gauge=none) so the parameterisation is matched; it is itself exact.
    fold_rmsnorm_gains(model)
    maps = {}
    if args.gauge != "none":
        g = torch.Generator().manual_seed(1000 + args.gauge_seed)
        R = make_R(args.gauge, d, g)
        maps = residual_gauge_adapter_maps(model, R)
        apply_residual_gauge(model, R)

    tr, te = build_sft(tok, args.task, args.n_train, args.n_eval, args.max_len,
                       seed=0)
    trl = FixedOrderLoader(tr, args.micro_bs, tok.pad_token_id, seed=0)
    tel = FixedOrderLoader(te, 16, tok.pad_token_id, seed=999)
    targets = tuple(args.targets.split(","))

    adapters = {}
    if args.method == "full":
        params = full_ft_params(model, targets)
    else:
        fac = make_factory(args.method, args.seed, maps)
        adapters = apply_lora(model, args.r, args.alpha, fac, targets=targets)
        params = lora_parameters(adapters)
    base_eval = eval_loss(model, tel, args.eval_batches, "cuda")
    cfg = dict(steps=args.steps, accum=max(args.bs // args.micro_bs, 1),
               optimizer=args.optimizer, lr=args.lr, wd=args.wd,
               warmup=args.warmup, sched=args.sched, grad_clip=args.grad_clip,
               momentum=args.momentum)
    log = train(model, adapters, params, trl, tel, cfg, log_every=5,
                eval_every=args.eval_every, eval_batches=args.eval_batches,
                sample_layers=[n for n in adapters
                               if n.endswith("layers.13.mlp.down_proj")])
    json.dump(dict(cell=cell, args=vars(args), base_eval_loss=base_eval,
                   n_trainable=sum(p.numel() for p in params), log=log),
              open(outfile, "w"))
    print(f"[{cell}] base={base_eval:.6f} final={log['final_eval_loss']:.6f} "
          f"({log['wall_time']:.0f}s)")


if __name__ == "__main__":
    main()
