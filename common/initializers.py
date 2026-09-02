"""Faithful re-implementations of the published LoRA initializers, so they can
be placed inside the matched-control framework of topic 01.

The organising dichotomy this file exists to test:

  * **B_0 = 0 family** (vanilla LoRA, NoRA, EVA, ETF/frame, gradient-subspace
    without the residual correction, ...).  By Gate-0 theorem T1 these differ
    *only* through P_0 = s^2 A_0^T A_0, and under SGD their entire merged
    trajectory is a function of P_0 alone.  Two members with the same P_0 are
    the same run; a member is fully described by a handful of P-statistics.

  * **B_0 != 0 family** (PiSSA, OLoRA, LoRA-One).  These start the adapter at a
    nonzero dW and subtract the same amount from the frozen base weight, so the
    initial function is unchanged but B_0 != 0.  They are the only way out of
    the P_0 equivalence class, because grad_A = s B^T G is nonzero at step 1.

Every initializer here is trace-matchable: `match_trace` rescales A_0 (and
compensates B_0) so that tr P equals a reference value, which removes the
update-magnitude confound identified in Gate 1.
"""
import math
import os

import torch
import torch.nn as nn

from .pinit import kaiming_A, normalize_columns, make_A


# ---------------------------------------------------------------- helpers

def _topr_svd(M, r):
    # fp32 on whatever device M lives on: these are up to 3072x1024 and a fp64
    # CPU SVD for all 196 adapted modules would dominate the run time.
    U, S, Vh = torch.linalg.svd(M.float(), full_matrices=False)
    return U[:, :r], S[:r], Vh[:r]


def rescale_A(A, target_tr):
    """Scale A so that tr(A^T A) = target_tr."""
    cur = float(A.pow(2).sum())
    return A * math.sqrt(target_tr / max(cur, 1e-30))


# ---------------------------------------------------------------- B0 = 0 family

def init_eva(X_cov, r, d_in, trace):
    """EVA (NeurIPS 2025): rows of A_0 are the top-r principal directions of the
    layer's *input activations*.  X_cov is the (d_in, d_in) uncentred second
    moment of the inputs."""
    evals, evecs = torch.linalg.eigh(X_cov.float())
    A = evecs[:, -r:].T.contiguous()               # (r, d_in), orthonormal rows
    return rescale_A(A, trace)


def init_gradsubspace(G, r, trace):
    """LoRA-One / gradient-subspace initialisation with B_0 = 0: rows of A_0 are
    the top-r right singular vectors of the one-step full-weight gradient."""
    _, _, Vh = _topr_svd(G, r)
    return rescale_A(Vh.contiguous(), trace)


ETF_CACHE = os.path.expanduser("~/.cache/nora_repo_etf")


def init_etf(r, d_in, generator, trace, iters=60):
    """Disk-cached wrapper: the alternating projection is 60 fp64 SVDs per
    module in python and dominates the run time of an `etf` job."""
    import hashlib
    os.makedirs(ETF_CACHE, exist_ok=True)
    seed = int(torch.randint(0, 2 ** 62, (1,), generator=generator))
    key = hashlib.md5(f"etf|{r}|{d_in}|{iters}|{seed}".encode()).hexdigest()
    f = os.path.join(ETF_CACHE, key + ".pt")
    if os.path.exists(f):
        try:
            return rescale_A(torch.load(f), trace)
        except Exception:
            pass
    g2 = torch.Generator().manual_seed(seed)
    A = _init_etf_raw(r, d_in, g2, 1.0, iters)
    tmp = f + f".tmp{os.getpid()}"
    torch.save(A, tmp); os.replace(tmp, f)
    return rescale_A(A, trace)


def _init_etf_raw(r, d_in, generator, trace, iters=60):
    """Approximate equiangular tight frame in the sense used by the LoRA
    dynamics literature: the d_in *columns* of A (vectors in R^r) have equal
    norms and minimal mutual coherence.  Alternating projection between
    (unit-norm columns) and (tight frame, i.e. A A^T proportional to I)."""
    A = torch.randn(r, d_in, generator=generator, dtype=torch.float64)
    A = A.to(torch.float64)
    for _ in range(iters):
        A = normalize_columns(A, target_norm=1.0)          # equal column norms
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        A = U @ Vh * math.sqrt(d_in / r)                   # tight frame
    A = normalize_columns(A, target_norm=1.0)
    return rescale_A(A, trace)


# ---------------------------------------------------------------- B0 != 0 family

def init_pissa(W, r, s, minor=False):
    """PiSSA: adapter carries the principal (or minor) r singular triplets of W,
    and the same amount is removed from the frozen base weight.
    Returns (A0, B0) with s * B0 @ A0 = U_r S_r V_r^T."""
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    if minor:
        U, S, Vh = U[:, -r:], S[-r:], Vh[-r:]
    else:
        U, S, Vh = U[:, :r], S[:r], Vh[:r]
    sq = S.clamp_min(0).sqrt()
    A0 = (torch.diag(sq) @ Vh) / math.sqrt(s)
    B0 = (U @ torch.diag(sq)) / math.sqrt(s)
    return A0, B0


def init_olora(W, r, s):
    """OLoRA: QR of the base weight; the leading r columns of Q and rows of R
    initialise the adapter, and the product is removed from the base weight."""
    Q, R = torch.linalg.qr(W.float())
    B0 = Q[:, :r] / math.sqrt(s)
    A0 = R[:r] / math.sqrt(s)
    return A0, B0


def init_lora_one(G, r, s, W=None, b0_rel=0.01):
    """LoRA-One (ICML 2025) style: the adapter is initialised on the one-step
    full-gradient subspace with a NONZERO product, so that s*B0@A0 is aligned
    with the top-r part of -G.

    The published scaling is tied to the paper's step-size analysis; here the
    magnitude is made an explicit, sweepable knob:
        ||s B0 A0||_F = b0_rel * ||W||_F ,
    so that the *subspace* (which is what the method claims) and the *scale*
    (the confound identified in Gate 1) can be varied independently."""
    U, S, Vh = _topr_svd(-G, r)
    scale = 1.0
    if W is not None:
        cur = float(S.pow(2).sum().sqrt())
        scale = b0_rel * float(W.float().norm()) / max(cur, 1e-30)
    sq = (S * scale).clamp_min(0).sqrt()
    A0 = (torch.diag(sq) @ Vh) / math.sqrt(s)
    B0 = (U @ torch.diag(sq)) / math.sqrt(s)
    return A0, B0


# ---------------------------------------------------------------- collection

ZERO_B = ["kaiming", "gaussian", "nora", "nora_unit", "kaimingspec_flatdiag",
          "flatspec_flatdiag", "left_gauge", "etf", "eva", "gradsub"] + \
         [f"geomspec_flatdiag{d}" for d in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)]
NONZERO_B = ["pissa", "pissa_minor", "olora", "lora_one"]
NEEDS_GRAD = {"gradsub", "lora_one"}
NEEDS_ACT = {"eva"}


def init_signperm(A0, seed=0):
    """A -> Pi A with Pi a signed permutation.

    This is the honest noise floor for an AdamW frame measurement.  AdamW is
    EXACTLY covariant under signed permutations of A's rows -- they permute and
    flip the columns of grad_B, and m/sqrt(v) is elementwise -- so two runs
    related this way are the same run, and any difference between them is
    floating-point reduction order compounded over training.

    The quantity previously used as "the null", a RANDOM gauge move
    (`left_gauge`), is not that: it is a small dose of the very effect being
    measured.  It happens to land near this floor, but the two mean different
    things and only this one bounds noise.
    """
    r = A0.shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(r, generator=g)
    sgn = (torch.randint(0, 2, (r, 1), generator=g).double() * 2 - 1)
    return (A0[perm] * sgn.to(A0.device)).to(A0.dtype)


def init_frame(A0, G, t, Sigma=None):
    """Walk the exact gauge orbit of a given A_0, from one extreme of the
    Adam-visible coordinate to the other.

    A -> Q A leaves P = s^2 A^T A bit-identical and every invariant of the
    triple (A A^T, A Sigma A^T, A C_g A^T) fixed to machine precision, so this
    is a dose ladder in which the ONLY thing that changes is the frame -- the
    part of the initialisation that SGD provably cannot see and AdamW provably
    can.  t = 0 puts the gradient metric in its own eigenbasis (gradient energy
    maximally concentrated across the r adapter rows, E_g minimal); t = 1
    flattens its diagonal (E_g = 1).  Schur-Horn says these are the two
    extremes, so the ladder spans the whole reachable range.
    """
    from common.intrinsic import frame_ladder, max_l1_frame
    A = A0.double()
    Gd = G.double().to(A.device)
    if str(t) == "opt":          # the actual argmax, not the flat-diagonal proxy
        Q, _ = max_l1_frame(Gd @ A.T)
    elif str(t) == "min":        # the argmin -- where the candidates disagree
        Q, _ = max_l1_frame(Gd @ A.T, sign=-1)
    elif Sigma is not None:      # ladder in the ACTIVATION metric instead
        Q = frame_ladder(A @ Sigma.double().to(A.device) @ A.T, [float(t)])[0]
    else:
        # M_g = A G^T G A^T = (G A^T)^T (G A^T).  Forming G^T G first is
        # O(d_in^2 d_out) and dominates everything else at 8B (d = 4096, r = 16:
        # 250x more work, and in float64, where the cards are slow).
        GA = Gd @ A.T
        Q = frame_ladder(GA.T @ GA, [float(t)])[0]
    return Q.to(A.device) @ A


NEEDS_GRAD.add("frame")
