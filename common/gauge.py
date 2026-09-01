"""Exact function-preserving representation gauges for Llama/Qwen-style
Transformers.

Two gauge families are implemented.  Both are *exact* in real arithmetic; the
functional-equivalence test suite (02-representation-gauge/src/e0_exactness.py)
measures the actual floating-point error.

--------------------------------------------------------------------------
G1.  Value/Output gauge  (per KV head, no model surgery needed)
--------------------------------------------------------------------------
For attention head h in KV group g, the head output is linear in the value
vectors, so for any invertible C_g acting on the head_dim axis

    W_V[g-block, :]  ->  C_g W_V[g-block, :]
    W_O[:, h-block]  ->  W_O[:, h-block] C_g^{-1}   for every h in group g

leaves the attention function unchanged (attention weights act on the sequence
axis and commute with C_g).  Qwen3/Llama have no value normalisation and no
attention bias, so this is exact.  We restrict C_g to O(head_dim).

Effect on adapters:
    v_proj:  Delta W_V = s B A   ->   B <- C B   (LEFT / output side; P unchanged)
    o_proj:  Delta W_O = s B A   ->   A <- A C^{-T} (RIGHT / input side; P -> C^T P C)
The o_proj adapter is therefore the one that feels the gauge in its hidden
preconditioner -- exactly the setting where N(AR) != N(A)R.

--------------------------------------------------------------------------
G2.  Residual-stream gauge  (QuaRot-style; needs gamma-folding + untying)
--------------------------------------------------------------------------
RMSNorm(x) = gamma * x / rms(x).  The bare normaliser x -> x/rms(x) commutes
with any orthogonal R because ||Rx|| = ||x||; gamma does not.  So we first
*fold* every RMSNorm gain into the input side of the following linear layers
(exact), leaving gamma = 1, and untie the LM head from the embedding (exact).
Then for a global R in O(d_model):

    embed_tokens.weight  ->  E R^T      (rows are residual-stream vectors)
    every layer input weight  W_in  ->  W_in R      (reads residual)
    every layer output weight W_out ->  R^T W_out? -- see code for exact
    lm_head.weight       ->  lm_head R

RoPE and Qwen3's per-head q_norm/k_norm live on the head_dim axis and are
untouched.  Every adapter on q/k/v/gate/up then has A -> A R (input side,
P -> R^T P R) and every adapter on o/down has B -> R^T B.
"""

import torch
import torch.nn as nn


# ------------------------------------------------------------------ helpers

def random_orthogonal(n, generator=None, device="cpu", dtype=torch.float64):
    m = torch.randn(n, n, generator=generator, device=device, dtype=dtype)
    q, r = torch.linalg.qr(m)
    return q * torch.sign(torch.diagonal(r)).unsqueeze(0)


def hadamard(n, device="cpu", dtype=torch.float64):
    """Normalised Sylvester-Hadamard matrix (n must be a power of 2)."""
    assert n & (n - 1) == 0
    H = torch.ones(1, 1, device=device, dtype=dtype)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / (n ** 0.5)


def _layers(model):
    return model.model.layers


def _cfg(model):
    c = model.config
    return c.num_attention_heads, c.num_key_value_heads, c.head_dim, c.hidden_size


# ------------------------------------------------------------------ G1: V/O

@torch.no_grad()
def apply_vo_gauge(model, C_per_layer):
    """C_per_layer[l] : (num_kv_heads, head_dim, head_dim) orthogonal, or None."""
    nh, nkv, hd, _ = _cfg(model)
    rep = nh // nkv
    for l, layer in enumerate(_layers(model)):
        C = C_per_layer[l]
        if C is None:
            continue
        attn = layer.self_attn
        Wv = attn.v_proj.weight.data          # (nkv*hd, d_model)
        Wo = attn.o_proj.weight.data          # (d_model, nh*hd)
        dt = Wv.dtype
        for g in range(nkv):
            Cg = C[g].to(Wv.device, torch.float32)
            sl = slice(g * hd, (g + 1) * hd)
            Wv[sl] = (Cg @ Wv[sl].float()).to(dt)
            for h in range(g * rep, (g + 1) * rep):
                hs = slice(h * hd, (h + 1) * hd)
                Wo[:, hs] = (Wo[:, hs].float() @ Cg.T).to(dt)
    return model


def vo_gauge_adapter_maps(model, C_per_layer):
    """Return {module_name: ('left'|'right', M)} telling how a LoRA adapter on
    that module must be transformed to represent the SAME function in the new
    gauge:  'left'  -> B <- M B ;   'right' -> A <- A M."""
    nh, nkv, hd, d = _cfg(model)
    rep = nh // nkv
    out = {}
    for l, layer in enumerate(_layers(model)):
        C = C_per_layer[l]
        if C is None:
            continue
        big_v = torch.block_diag(*[C[g] for g in range(nkv)])          # (nkv*hd)^2
        big_o = torch.block_diag(*[C[g // rep] for g in range(nh)])    # (nh*hd)^2
        out[f"model.layers.{l}.self_attn.v_proj"] = ("left", big_v)
        # W_O -> W_O big_o^T  =>  A <- A big_o^T
        out[f"model.layers.{l}.self_attn.o_proj"] = ("right", big_o.T)
    return out


# ------------------------------------------------------------------ G2: residual

@torch.no_grad()
def fold_rmsnorm_gains(model):
    """Fold every RMSNorm gamma on the *residual stream* into the input side of
    the linear layers that read it, leaving gamma = 1.  Exact.
    Also unties the LM head so it can be transformed independently."""
    # untie
    if getattr(model.config, "tie_word_embeddings", False):
        w = model.lm_head.weight.data.clone()
        model.lm_head.weight = nn.Parameter(w)
        model.config.tie_word_embeddings = False

    def fold(norm, linears):
        gam = norm.weight.data.float().clone()
        for lin in linears:
            lin.weight.data = (lin.weight.data.float() * gam.unsqueeze(0)).to(
                lin.weight.dtype)
        norm.weight.data = torch.ones_like(norm.weight.data)

    for layer in _layers(model):
        a = layer.self_attn
        fold(layer.input_layernorm, [a.q_proj, a.k_proj, a.v_proj])
        m = layer.mlp
        fold(layer.post_attention_layernorm, [m.gate_proj, m.up_proj])
    fold(model.model.norm, [model.lm_head])
    return model


@torch.no_grad()
def apply_residual_gauge(model, R):
    """x -> R^T x on the residual stream (requires fold_rmsnorm_gains first)."""
    dev = model.model.embed_tokens.weight.device
    Rf = R.to(dev, torch.float32)

    def right(lin):     # reads residual:  W x = (W R)(R^T x)
        lin.weight.data = (lin.weight.data.float() @ Rf).to(lin.weight.dtype)

    def left(lin):      # writes residual: R^T (W y)
        lin.weight.data = (Rf.T @ lin.weight.data.float()).to(lin.weight.dtype)

    E = model.model.embed_tokens.weight
    E.data = (E.data.float() @ Rf).to(E.dtype)     # rows e -> R^T e
    for layer in _layers(model):
        a, m = layer.self_attn, layer.mlp
        for lin in (a.q_proj, a.k_proj, a.v_proj, m.gate_proj, m.up_proj):
            right(lin)
        for lin in (a.o_proj, m.down_proj):
            left(lin)
    right(model.lm_head)
    return model


def residual_gauge_adapter_maps(model, R):
    out = {}
    for l, _ in enumerate(_layers(model)):
        for nm in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                   "mlp.gate_proj", "mlp.up_proj"):
            out[f"model.layers.{l}.{nm}"] = ("right", R)
        for nm in ("self_attn.o_proj", "mlp.down_proj"):
            out[f"model.layers.{l}.{nm}"] = ("left", R.T)
    return out
