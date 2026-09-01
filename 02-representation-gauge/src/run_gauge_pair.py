"""Topic 02, E1/E2/E5 -- coupled gauge experiment.

Trains the SAME adapter, on the SAME ordered minibatches, on two backbones that
compute *exactly the same function* and differ only by a representation gauge.

Theory (see README sec.3, re-derived for our conventions):
  gauge on the input side of a module:  x' = R x,  W' = W R^T,  so G' = G R^T.
  Coupling the adapter by  A' = A R^T,  B' = B  gives, under plain SGD,
      grad_B' = s G' A'^T = s G A^T = grad_B
      grad_A' = s B'^T G' = (grad_A) R^T
  so the coupling is preserved for all time and the two runs are the SAME
  FUNCTION at every step.  Vanilla LoRA + SGD is exactly gauge-equivariant.

  Normalised NoRA breaks this because  N(A R^T) != N(A) R^T.
  AdamW breaks it too, because m/sqrt(v) is not covariant under a rotation of
  the coordinates of A.

The 2x2 {SGD, AdamW} x {LoRA, NoRA-init} panel therefore separates
  * NoRA-specific gauge dependence (visible under SGD), from
  * the optimizer's own gauge dependence (the AdamW baseline).

The `oracle` init mode is the positive control for E2: normalise once in the
original gauge, then transform -- must recover exact SGD equivariance.
"""
import argparse, hashlib, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from common.gauge import (random_orthogonal, hadamard, apply_vo_gauge,
                          vo_gauge_adapter_maps, fold_rmsnorm_gains,
                          apply_residual_gauge, residual_gauge_adapter_maps)
from common.lora import apply_lora, lora_parameters, DEFAULT_TARGETS
from common.pinit import kaiming_A, normalize_columns
from common.pstats import p_stats
from common.data import build_sft, FixedOrderLoader
from common.train import load_model, train, eval_loss

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def build_gauge(model, kind, seed, dtype=torch.float64):
    """Returns (apply_fn, adapter_maps).  apply_fn mutates a model in place."""
    nh = model.config.num_attention_heads
    nkv = model.config.num_key_value_heads
    hd = model.config.head_dim
    nl = model.config.num_hidden_layers
    d = model.config.hidden_size
    g = torch.Generator().manual_seed(seed)
    if kind == "none":
        return (lambda m: m), {}
    if kind == "vo":
        C = [torch.stack([random_orthogonal(hd, g) for _ in range(nkv)])
             for _ in range(nl)]
        maps = vo_gauge_adapter_maps(model, C)
        return (lambda m: apply_vo_gauge(m, C)), maps
    if kind == "vo_hadamard":
        H = hadamard(hd)
        C = [torch.stack([H] * nkv) for _ in range(nl)]
        maps = vo_gauge_adapter_maps(model, C)
        return (lambda m: apply_vo_gauge(m, C)), maps
    if kind in ("residual", "residual_hadamard"):
        R = hadamard(d) if kind.endswith("hadamard") else random_orthogonal(d, g)
        maps = residual_gauge_adapter_maps(model, R)

        def f(m):
            apply_residual_gauge(m, R)
            return m
        return f, maps
    raise ValueError(kind)


def make_factory(init, seed, maps, mode):
    """mode: 'orig'   -- build A0 in the original gauge (no transform)
             'coupled'-- build A0 then apply the gauge map, THEN (for 'algo')
                         re-apply the initializer in the new coordinates.
    init: 'kaiming' | 'nora'
    For NoRA we distinguish
        coupled_algo   : N(A0 R)    (what a practitioner running NoRA on the
                                     rotated model would actually get)
        coupled_oracle : N(A0) R    (the equivariant transform of the original)
    """
    store = {}

    def factory(name, r, d_in, d_out):
        h = int(hashlib.md5(f"{seed}:{name}".encode()).hexdigest()[:12], 16)
        g = torch.Generator().manual_seed(h)
        A = kaiming_A(r, d_in, g, "cpu")
        m = maps.get(name)
        R = None
        if m is not None and m[0] == "right":
            R = m[1].double()
        if mode == "orig":
            A = normalize_columns(A) if init == "nora" else A
        elif mode == "coupled_oracle":
            if init == "nora":
                A = normalize_columns(A)
            if R is not None:
                A = A @ R
        elif mode == "coupled_algo":
            if R is not None:
                A = A @ R
            if init == "nora":
                A = normalize_columns(A)
        else:
            raise ValueError(mode)
        st = p_stats(A, s=1.0); st.pop("spec_top4", None)
        store[name] = st
        return A.float()

    factory.stats = store
    return factory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--gauge", default="vo")
    ap.add_argument("--gauge_seed", type=int, default=0)
    ap.add_argument("--init", default="kaiming", choices=["kaiming", "nora"])
    ap.add_argument("--mode", default="coupled_algo",
                    choices=["orig", "coupled_algo", "coupled_oracle"])
    ap.add_argument("--optimizer", default="sgd")
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--grad_clip", type=float, default=1e9)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--n_train", type=int, default=6000)
    ap.add_argument("--n_eval", type=int, default=256)
    ap.add_argument("--eval_every", type=int, default=20)
    ap.add_argument("--eval_batches", type=int, default=16)
    ap.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    args = ap.parse_args()

    cell = (f"{args.gauge}{args.gauge_seed}_{args.init}_{args.mode}_"
            f"{args.optimizer}_lr{args.lr:g}_s{args.seed}")
    outdir = os.path.join(REPO, "02-representation-gauge", "results", args.tag)
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, cell + ".json")
    if os.path.exists(outfile):
        print("skip (exists)", outfile); return

    dtype = dict(float32=torch.float32, bfloat16=torch.bfloat16)[args.dtype]
    torch.manual_seed(args.seed)
    model, tok = load_model(args.model, dtype=dtype)
    gauge_fn, maps = build_gauge(model, args.gauge, args.gauge_seed)
    if args.gauge.startswith("residual"):
        fold_rmsnorm_gains(model)
    if args.mode != "orig":
        gauge_fn(model)

    tr, te = build_sft(tok, "gsm8k", args.n_train, args.n_eval, 384, seed=0)
    trl = FixedOrderLoader(tr, args.bs, tok.pad_token_id, seed=0)
    tel = FixedOrderLoader(te, args.bs, tok.pad_token_id, seed=999)

    fac = make_factory(args.init, args.seed, maps if args.mode != "orig" else {},
                       args.mode)
    adapters = apply_lora(model, args.r, args.alpha, fac,
                          targets=tuple(args.targets.split(",")))
    params = lora_parameters(adapters)
    base_eval = eval_loss(model, tel, args.eval_batches, "cuda")
    cfg = dict(steps=args.steps, accum=1, optimizer=args.optimizer, lr=args.lr,
               wd=0.0, warmup=args.warmup, sched="constant",
               grad_clip=args.grad_clip)
    sample = [n for n in adapters
              if n.endswith(("layers.0.self_attn.o_proj",
                             "layers.13.self_attn.o_proj",
                             "layers.13.mlp.down_proj"))]
    log = train(model, adapters, params, trl, tel, cfg, log_every=1,
                eval_every=args.eval_every, eval_batches=args.eval_batches,
                sample_layers=sample)
    json.dump(dict(cell=cell, args=vars(args), base_eval_loss=base_eval,
                   init_pstats=fac.stats, log=log), open(outfile, "w"))
    print(f"[{cell}] base={base_eval:.6f} final={log['final_eval_loss']:.6f} "
          f"({log['wall_time']:.0f}s)")


if __name__ == "__main__":
    main()
