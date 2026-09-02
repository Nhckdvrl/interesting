"""The intrinsic state of a low-rank adapter, and exact constructions in it.

Scientific object
-----------------
For a layer with input second moment Sigma = E[x x^T], define the *whitened*
adapter and the input-side intrinsic state

    Atil = A Sigma^{1/2},        M_x = Atil Atil^T = A Sigma A^T   in R^{r x r}.

`M_x` is the right object because it is simultaneously covariant under both
exact symmetries of the problem:

  * adapter-factor gauge   A -> Q A          =>  M_x -> Q M_x Q^T
  * backbone representation gauge, which sends x -> R x, A -> A R^T and
    Sigma -> R Sigma R^T                     =>  M_x is *unchanged*

so `spec(M_x)` is invariant under an arbitrary choice of BOTH coordinate
systems.  By contrast `diag(A^T A)` -- the quantity the mother paper normalises
-- is a description in a particular chart and has no such status.

Coordinates on the intrinsic state
----------------------------------
    S = tr M_x                      data-space scale
    D = (tr M_x)^2 / ||M_x||_F^2    intrinsic spectral dimension, in [1, r]
    rho                             task alignment: where the adapter's row
                                    space sits in the spectrum of the whitened
                                    gradient operator
                                        T = Sigma^{-1/2} C_g Sigma^{-1/2},
                                    with C_g = G^T G the input-side gradient
                                    covariance.  T answers "at unit activation
                                    variance, which directions carry gradient
                                    energy".

Construction
------------
Because M_x = Atil Atil^T, prescribing the spectrum of M_x is exactly
prescribing the singular values of Atil, and the row space of Atil is free.  So

    Atil = diag(sqrt(lam)) V^T ,     A = Atil Sigma^{-1/2}

hits any (S, D) exactly for any choice of orthonormal V in R^{d x r}, and V
alone controls rho.  The three coordinates are therefore *exactly* independent
by construction -- this is what makes the atlas a controlled intervention
rather than a correlational sweep.
"""

import math
import torch


# ---------------------------------------------------------------- whitening

def sym_pow(M, p, eps_rel=1e-6):
    """M^p for a PSD matrix, with eigenvalues floored at eps_rel * max."""
    M = M.double()
    ev, U = torch.linalg.eigh(M)
    floor = eps_rel * float(ev.max().clamp_min(1e-30))
    ev = ev.clamp_min(floor)
    return (U * ev.pow(p).unsqueeze(0)) @ U.T


def whiten_ops(Sigma, C_g=None, eps_rel=1e-6):
    """Return Sigma^{1/2}, Sigma^{-1/2}, and the eigendecomposition of the
    whitened gradient operator T = Sigma^{-1/2} C_g Sigma^{-1/2}."""
    S_half = sym_pow(Sigma, 0.5, eps_rel)
    S_ihalf = sym_pow(Sigma, -0.5, eps_rel)
    if C_g is None:
        return S_half, S_ihalf, None, None
    T = S_ihalf @ C_g.double() @ S_ihalf
    T = 0.5 * (T + T.T)
    tau, U = torch.linalg.eigh(T)           # ascending
    return S_half, S_ihalf, tau, U


# ---------------------------------------------------------------- spectrum

def spectrum_for(S, D, r, device="cpu", dtype=torch.float64, tol=1e-10):
    """A length-r positive spectrum with sum exactly S and effective rank
    (sum)^2/(sum of squares) exactly D.  Uses a geometric family and bisects on
    the ratio; D is monotone in the ratio, from 1 (all mass on one mode) to r
    (flat)."""
    assert 1.0 <= D <= r + 1e-9
    if D >= r - 1e-9:
        lam = torch.ones(r, device=device, dtype=dtype)
        return lam * (S / lam.sum())

    def eff(q):
        lam = torch.tensor([q ** i for i in range(r)], device=device, dtype=dtype)
        return float(lam.sum() ** 2 / lam.pow(2).sum())

    lo, hi = 1e-6, 1.0 - 1e-12          # eff(lo) -> 1, eff(hi) -> r
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if eff(mid) < D:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    q = 0.5 * (lo + hi)
    lam = torch.tensor([q ** i for i in range(r)], device=device, dtype=dtype)
    return lam * (S / lam.sum())


# ---------------------------------------------------------------- row space

def rowspace_for(rho, r, tau, U, generator=None):
    """Orthonormal V in R^{d x r} realising a target task alignment.

    rho in [0, 1] slides a window of r consecutive eigenvectors of T from the
    bottom of the spectrum (rho = 0) to the top (rho = 1); rho = None returns a
    Haar-random subspace, the 'generic' reference.  A sliding window is used
    rather than a mixture so that every rung is an exact eigen-subspace and the
    alignment statistic moves monotonically.
    """
    d = U.shape[0]
    if rho is None:
        M = torch.randn(d, r, generator=generator, device=U.device,
                        dtype=U.dtype)
        Q, R = torch.linalg.qr(M)
        return Q * torch.sign(torch.diagonal(R)).unsqueeze(0)
    k = int(round(rho * (d - r)))
    k = max(0, min(d - r, k))
    return U[:, k:k + r].contiguous()


def captured_of(V, tau, U):
    """Task alignment as *captured whitened-gradient energy*, relative to what a
    random subspace of the same dimension would capture.

    rho_rel = [ tr(V^T T V) / tr(T) ] / (r / d)

    so a Haar-random row space sits at 1 by construction, the bottom-r
    eigen-subspace near 0, and the top-r at the largest value the spectrum
    allows.  This is the coordinate that enters the first-order descent, and it
    is multiplicative like the scale S -- a linear position in the eigenvalue
    *index* is not a usable coordinate because tau is heavy-tailed.
    """
    d, r = U.shape[0], V.shape[1]
    W = U.T @ V
    cap = float((W.pow(2) * tau.unsqueeze(1)).sum())
    return cap / float(tau.sum() + 1e-30) / (r / d)


def rowspace_for_captured(target_rel, r, tau, U, iters=60):
    """Slide the r-wide window over the eigenbasis of T to hit a target
    captured-energy ratio.  Captured energy is monotone in the window position,
    so a bisection is exact up to the granularity of one eigenvector."""
    d = U.shape[0]
    tot = float(tau.sum() + 1e-30)

    def cap_at(k):
        return float(tau[k:k + r].sum()) / tot / (r / d)

    lo, hi = 0, d - r
    if target_rel <= cap_at(lo):
        return U[:, :r].contiguous()
    if target_rel >= cap_at(hi):
        return U[:, hi:].contiguous()
    for _ in range(iters):
        if hi - lo <= 1:
            break
        mid = (lo + hi) // 2
        if cap_at(mid) < target_rel:
            lo = mid
        else:
            hi = mid
    k = lo if abs(cap_at(lo) - target_rel) < abs(cap_at(hi) - target_rel) else hi
    return U[:, k:k + r].contiguous()


# ---------------------------------------------------------------- assembly

def build_A(S, D, rho, r, Sigma, C_g=None, generator=None, eps_rel=1e-6,
            cache=None):
    """A (r, d) whose intrinsic state has exactly the requested (S, D) and,
    where the spectrum allows, the requested captured-energy ratio `rho`
    (rho = None gives a Haar-random row space, which sits at rho_rel = 1).
    `cache` may hold the whitening operators for this module so they are
    computed once per run rather than once per condition."""
    if cache is None or "S_ihalf" not in cache:
        S_half, S_ihalf, tau, U = whiten_ops(Sigma, C_g, eps_rel)
        if cache is not None:
            cache.update(S_half=S_half, S_ihalf=S_ihalf, tau=tau, U=U)
    else:
        S_half, S_ihalf, tau, U = (cache["S_half"], cache["S_ihalf"],
                                   cache["tau"], cache["U"])
    dev, dt = S_ihalf.device, S_ihalf.dtype
    lam = spectrum_for(S, D, r, device=dev, dtype=dt)
    if rho is None or U is None:
        V = rowspace_for(None, r, None,
                         U if U is not None else torch.eye(Sigma.shape[0],
                                                           device=dev, dtype=dt),
                         generator)
    else:
        V = rowspace_for_captured(float(rho), r, tau, U)
    Atil = torch.diag(lam.sqrt()) @ V.T          # (r, d)
    return Atil @ S_ihalf


def intrinsic_state(A, Sigma):
    """(S, D) of an arbitrary A."""
    A = A.double(); M = A @ Sigma.double() @ A.T
    t = float(torch.diagonal(M).sum())
    return t, t * t / (float((M * M).sum()) + 1e-30)


def output_state(B, Sigma_delta):
    """The output-side intrinsic state M_delta = B^T Sigma_delta B.

    Under the factor gauge A -> QA, B -> BQ^T both M_x and M_delta conjugate by
    Q, so their spectra and any joint invariant such as tr(M_delta M_x) are
    factor-basis independent.  Under a Kronecker (K-FAC) approximation of the
    layer Fisher, F = Sigma_x kron Sigma_delta, the local function-space norm of
    the merged update is exactly

        vec(BA)^T F vec(BA) = tr[(B^T Sigma_delta B)(A Sigma_x A^T)]
                            = tr(M_delta M_x),

    which is why B_0 != 0 is a *continuous* second coordinate of the same
    object rather than a binary exception.
    """
    B = B.double(); M = B.T @ Sigma_delta.double() @ B
    t = float(torch.diagonal(M).sum())
    return t, t * t / (float((M * M).sum()) + 1e-30)
