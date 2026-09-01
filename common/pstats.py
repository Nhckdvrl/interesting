"""Pre-registered statistics of the hidden preconditioner P = s^2 A^T A.

Everything is computed from A (r, d_in) *without* forming the d_in x d_in
matrix P, using the small Gram K = A A^T (r, r):

    tr P        = s^2 ||A||_F^2                    = s^2 tr K
    ||P||_F^2   = s^4 ||A A^T||_F^2                = s^4 ||K||_F^2
    diag P      = s^2 * colnorm^2(A)
    nonzero eig(P) = s^2 * eig(K)

so all statistics cost O(d_in r^2).
"""

import torch


@torch.no_grad()
def p_stats(A, s=1.0, eps=1e-12):
    """A: (r, d_in) float tensor.  s: LoRA scaling (P = s^2 A^T A)."""
    A = A.double()
    r, d_in = A.shape
    s2 = float(s) ** 2

    K = A @ A.T                                   # (r, r)
    trP = s2 * float(torch.diagonal(K).sum())
    frob2 = (s2 ** 2) * float((K * K).sum())      # ||P||_F^2 = ||K||_F^2
    dg = s2 * A.pow(2).sum(0)                     # (d_in,)  diag P
    m = trP / d_in                                # mean diagonal gain
    diag_var = float(dg.var(unbiased=False))
    lam = torch.linalg.eigvalsh(K).clamp_min(0) * s2

    diag_sq = float(dg.pow(2).sum())
    cross2 = max(frob2 - diag_sq, 0.0)            # ||P - Diag P||_F^2

    return dict(
        r=r, d_in=d_in,
        tr_P=trP,
        mean_diag=m,                              # m(P) = tr P / d_in
        diag_var=diag_var,
        diag_imbalance=diag_var / (m * m + eps),  # d(P)
        frob_P=frob2 ** 0.5,
        crosstalk=(cross2 / (frob2 + eps)) ** 0.5,   # c(P)
        eff_rank=(trP * trP) / (frob2 + eps),         # r_eff(P)
        spec_max=float(lam.max()),
        spec_min=float(lam.min()),
        spec_top4=[float(x) for x in lam.flip(0)[:4]],
        colnorm_cv=float((dg.sqrt().std(unbiased=False) / (dg.sqrt().mean() + eps))),
    )


@torch.no_grad()
def gradient_alignment(A, G, s=1.0, eps=1e-12):
    """Statistics that couple P to a full-weight gradient G (d_out, d_in).

      captured energy  rho = ||G P||_F^2 / (||G||_F^2 * (tr P / d_in)^2 * d_in?) -- we
      report the raw ratios and let the analysis normalise.
    """
    A = A.double()
    G = G.double()
    s2 = float(s) ** 2
    GA = G @ A.T                     # (d_out, r)
    GP = (GA @ A) * s2               # (d_out, d_in)
    gn2 = float(G.pow(2).sum())
    gpn2 = float(GP.pow(2).sum())
    return dict(
        norm_G=gn2 ** 0.5,
        norm_GP=gpn2 ** 0.5,
        # fraction of gradient energy retained per unit mean gain
        captured=gpn2 / (gn2 + eps),
        # cosine between the true gradient direction and the preconditioned one
        cos_G_GP=float((G * GP).sum()) / ((gn2 ** 0.5) * (gpn2 ** 0.5) + eps),
        # first-order descent rate <G, G P> (>= 0 since P is PSD)
        descent=float((G * GP).sum()),
    )
