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


def gradient_alignment(A, Sigma, C_g):
    """R_g = tr(A C_g A^T) / tr(A Sigma A^T).

    This is the quantity that actually enters the first-order descent.  Writing
    C = A Sigma^{1/2} = Lambda^{1/2} V^T and T = Sigma^{-1/2} C_g Sigma^{-1/2},

        tr(A C_g A^T) = tr(C T C^T) = tr(Lambda V^T T V),

    i.e. the row-space alignment weighted by the intrinsic spectrum Lambda.
    The unweighted tr(V^T T V) used by `captured_of` coincides with it only when
    Lambda is flat, so a null result on `captured_of` does NOT rule out task
    alignment -- it rules out the unweighted statistic.  Divided by
    tr(A Sigma A^T) = tr(Lambda), this is the descent rate per unit data-space
    scale, and is invariant to rescaling A.
    """
    A = A.double()
    num = float(((A @ C_g.double()) * A).sum())
    den = float(((A @ Sigma.double()) * A).sum())
    return num / (den + 1e-30)


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


# ------------------------------------------------- exact two-constraint V

def rowspace_constrained(Lam, Sinv, T, target_W, target_Rg, iters=400,
                         lr=0.05, generator=None, V0=None, verbose=False):
    """Find V in St(d, r) realising BOTH

        W   = tr(Lam V^T Sinv V) / tr(Lam)      parameter/data metric ratio
        R_g = tr(Lam V^T T    V) / tr(Lam)      first-order descent per unit S

    Because A = Lam^{1/2} V^T Sigma^{-1/2} has M_x = Lam identically, S and D
    are fixed exactly by Lam and are untouched by V.  So sweeping W at fixed
    (S, D, R_g) is a genuinely matched intervention, unlike the whitening-
    exponent sweep, which drags D and the row space along with it.

    St(d, r) has dr - r(r+1)/2 dimensions (about 1.6e4 at d=1024, r=16), so two
    scalar constraints are far from binding; a short Riemannian-free
    parameterisation V = qr(M) with Adam is enough.
    """
    d = Sinv.shape[0]; r = Lam.shape[0]
    dt = Sinv.dtype
    if V0 is None:
        M = torch.randn(d, r, generator=generator, device=Sinv.device, dtype=dt)
    else:
        M = V0.clone()
    M = M.detach().requires_grad_(True)
    opt = torch.optim.Adam([M], lr=lr)
    lam = Lam.diagonal() if Lam.dim() == 2 else Lam
    tl = lam.sum()
    lw, lr_g = math.log(target_W), math.log(max(target_Rg, 1e-300))
    best, bestV = float("inf"), None
    for it in range(iters):
        opt.zero_grad()
        V, _ = torch.linalg.qr(M)
        w = (lam * ((V.T @ Sinv) * V.T).sum(1)).sum() / tl
        g = (lam * ((V.T @ T) * V.T).sum(1)).sum() / tl
        loss = (torch.log(w.clamp_min(1e-300)) - lw) ** 2 + \
               (torch.log(g.clamp_min(1e-300)) - lr_g) ** 2
        f = float(loss.detach())
        if f < best:
            best, bestV = f, V.detach().clone()
        loss.backward()
        opt.step()
    V = bestV
    w = float((lam * ((V.T @ Sinv) * V.T).sum(1)).sum() / tl)
    g = float((lam * ((V.T @ T) * V.T).sum(1)).sum() / tl)
    if verbose:
        print(f"    W {w:.4g} (target {target_W:.4g}), "
              f"R_g {g:.4g} (target {target_Rg:.4g})")
    return V, w, g


def build_A_matched(S, D, target_W, target_Rg, r, Sigma, C_g, cache=None,
                    generator=None, eps_rel=1e-6, iters=400):
    """A with S and D exact by construction, and (W, R_g) driven to targets."""
    if cache is None or "S_ihalf" not in cache:
        S_half, S_ihalf, tau, U = whiten_ops(Sigma, C_g, eps_rel)
        if cache is not None:
            cache.update(S_half=S_half, S_ihalf=S_ihalf, tau=tau, U=U)
    else:
        S_half, S_ihalf, tau, U = (cache["S_half"], cache["S_ihalf"],
                                   cache["tau"], cache["U"])
    if cache is not None and "Sinv" in cache:
        Sinv, T = cache["Sinv"], cache["T"]
    else:
        Sinv = sym_pow(Sigma, -1.0, eps_rel)
        T = S_ihalf @ C_g.double() @ S_ihalf
        T = 0.5 * (T + T.T)
        if cache is not None:
            cache.update(Sinv=Sinv, T=T)
    lam = spectrum_for(S, D, r, device=S_ihalf.device, dtype=S_ihalf.dtype)
    V, w, g = rowspace_constrained(lam, Sinv, T, target_W, target_Rg,
                                   iters=iters, generator=generator)
    A = torch.diag(lam.sqrt()) @ V.T @ S_ihalf
    return A, w, g


# ---------------------------------------------------------------- assembly

def build_A(S, D, rho, r, Sigma, C_g=None, generator=None, eps_rel=1e-6,
            cache=None, wexp=0.5):
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
    # `wexp` opens the fourth axis.  A = Atil Sigma^{-q}:
    #   q = 1/2 gives the fully whitened construction, whose parameter-space
    #           norm is large because it places mass on low-variance directions;
    #   q = 0   gives A = Atil, whose row space sits in the parameter metric.
    # The ratio W = tr(A A^T) / tr(A Sigma A^T) -- the parameter metric that
    # Adam's per-coordinate step sees, divided by the data metric that the
    # function sees -- sweeps over orders of magnitude with q, and the vanilla
    # draw sits at W = 1 by definition.  S is restored exactly afterwards.
    A = Atil @ sym_pow(Sigma, -wexp, eps_rel) if abs(wexp - 0.5) > 1e-12 \
        else Atil @ S_ihalf
    cur = float(((A @ Sigma.double()) * A).sum())
    return A * (S / max(cur, 1e-30)) ** 0.5


def metric_ratio(A, Sigma):
    """W = tr(A A^T) / tr(A Sigma A^T): parameter metric over data metric."""
    A = A.double()
    return float(A.pow(2).sum()) / (float(((A @ Sigma.double()) * A).sum()) + 1e-30)


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


# ---------------------------------------------------------------------------
# Second-order invariants of the triple (M_0, M_x, M_g), and the coordinates
# that are invariant under Adam's symmetry group but not under the full gauge.
#
# The state of a LoRA initialisation is the simultaneous O(r)-conjugacy class of
#
#     M_0 = A A^T ,   M_x = A Sigma A^T ,   M_g = A C_g A^T .
#
# By classical invariant theory the ring of simultaneous O(r) invariants is
# generated by traces of words; at order <= 2 there are exactly nine:
#
#     tr M_0, tr M_x, tr M_g,                       (three scales)
#     tr M_0^2, tr M_x^2, tr M_g^2,                 (three spectral spreads)
#     tr M_0 M_x, tr M_0 M_g, tr M_x M_g.           (three co-alignments)
#
# S, D, W and R_g use only four of the nine.  `second_order` returns all of them
# in dimensionless form.
#
# Adam sees strictly more than this, and the reason is a statement about which
# symmetry group each optimizer respects.
#
# Steepest descent under a norm ||.|| takes the step that maximises the linear
# decrease inside a ball, and the decrease it achieves is the DUAL norm of the
# gradient.  For LoRA's B-gradient, grad_B = s G A^T (d_out x r):
#
#   SGD  = steepest descent in the Frobenius norm  -> decrease ~ ||grad_B||_F
#   Adam = steepest descent in the elementwise max norm (Bernstein & Newhouse,
#          "Old Optimizer, New Norm"; Adam with beta = eps = 0 is exactly
#          signSGD) -> decrease ~ ||grad_B||_1, the elementwise l1 norm.
#
# Under the adapter gauge A -> Q A, B -> B Q^T the gradient transforms by right
# multiplication, grad_B -> grad_B Q^T.  The Frobenius norm is invariant under
# right multiplication by an orthogonal matrix; the elementwise l1 norm is not,
# and is invariant only under the signed-permutation subgroup.  Hence:
#
#   SGD's first-order descent depends on A only through tr M_g, a gauge
#   invariant; AdamW's depends on the gauge frame itself.
#
# That is a derivation of the measurement null rather than a description of it:
# A -> Q A costs 1.5e-6 nats under SGD and 2.7e-4 under AdamW because the gauge
# is an exact symmetry of l2 steepest descent and only a discrete symmetry of
# l_inf steepest descent.  (The statement is exact at initialisation, and exact
# for all time when A is frozen, since a trainable A also moves by
# dA = -eta s B^T G.)
#
# The dimensionless ratio of the two descent rates is therefore the coordinate
# AdamW can see and SGD cannot:
#
#   Lambda_1 = ||grad_B||_1^2 / (d_out r ||grad_B||_F^2)   in (1/(d_out r), 1],
#
# = 1 iff every entry of G A^T has the same magnitude.  It factorises as
# Lambda_1 = E_g * Delta_out, where E_g is the equipartition of gradient energy
# across the r rows of A and Delta_out the delocalisation of each G a_j inside
# the output space; only the first is a function of the triple.
# ---------------------------------------------------------------------------


def l1_flatness(A, G):
    """(Lambda_1, E_g, Delta_out) for the B-gradient G A^T.

    Lambda_1 is the exact ratio of AdamW's first-order descent rate (squared,
    normalised) to SGD's; it needs no distributional assumption.  E_g is its
    row-partition factor, computable from M_g = A C_g A^T alone.
    """
    GA = (G.double() @ A.double().T)            # d_out x r
    m, r = GA.shape
    l1 = float(GA.abs().sum())
    f2 = float(GA.pow(2).sum())
    lam = l1 * l1 / (m * r * f2 + 1e-30)
    col2 = GA.pow(2).sum(0)                     # = diag(M_g)
    Eg = float(col2.sqrt().sum() ** 2) / (r * float(col2.sum()) + 1e-30)
    return lam, Eg, lam / (Eg + 1e-30)


def equipartition(M):
    """(sum_j sqrt(M_jj))^2 / (r tr M) in (1/r, 1].

    1 iff the diagonal is flat, 1/r iff a single row carries everything.  This
    is invariant under signed permutations of the rows of A -- Adam's symmetry
    group -- but not under the full O(r) gauge, so it is zero for SGD and
    nonzero for Adam by construction.
    """
    d = torch.diagonal(M).clamp_min(0)
    r = d.numel()
    return float(d.sqrt().sum() ** 2) / (r * float(d.sum()) + 1e-30)


def second_order(M0, Mx, Mg):
    """The nine order-<=2 gauge invariants, in dimensionless form.

    D_* = (tr M)^2 / (r tr M^2) in (0, 1] is 1 iff M is isotropic;
    Psi_ab = r tr(M_a M_b) / (tr M_a tr M_b) is 1 iff either factor is
    isotropic, > 1 when the two metrics co-align and < 1 when they anti-align.
    """
    r = M0.shape[0]
    out = {}
    for nm, M in (("0", M0), ("x", Mx), ("g", Mg)):
        t = float(torch.diagonal(M).sum())
        out[f"tr_{nm}"] = t
        out[f"D_{nm}"] = t * t / (r * float((M * M).sum()) + 1e-30)
        out[f"E_{nm}"] = equipartition(M)
    for a, b in (("0", "x"), ("0", "g"), ("x", "g")):
        Ma = {"0": M0, "x": Mx, "g": Mg}[a]
        Mb = {"0": M0, "x": Mx, "g": Mg}[b]
        out[f"Psi_{a}{b}"] = (r * float((Ma * Mb).sum())
                              / (out[f"tr_{a}"] * out[f"tr_{b}"] + 1e-30))
    return out


def triple(A, Sigma, C_g):
    """(M_0, M_x, M_g) in double precision."""
    A = A.double()
    return A @ A.T, A @ Sigma.double() @ A.T, A @ C_g.double() @ A.T


def flat_frame(M, tol=1e-9):
    """Q in O(r) with diag(Q M Q^T) flat, i.e. every entry tr(M)/r.

    Bendel-Mickey in the eigenbasis: in the (i, j) plane, with a = M_ii,
    b = M_jj, m = M_ij,

        M_ii(theta) = (a+b)/2 + R cos(2 theta - phi),
        R = hypot((a-b)/2, m),  phi = atan2(m, (a-b)/2),

    so a target is reachable iff it lies within R of (a+b)/2, which holds for
    any value in [b, a].  Taking i = argmax and j = argmin of the live diagonal
    puts the mean inside [b, a] always, and after pinning d_i to the mean and
    retiring i the surviving coordinates still have that same mean -- so the
    induction closes exactly and r-1 rotations suffice.

    (The general Schur-Horn target is reachable too -- the set of achievable
    diagonals is exactly {d : d prec eig(M)} -- but the greedy retirement above
    is only valid for the flat target: for a general target the surviving
    diagonal need not majorise the surviving target.  Only the flat case is
    implemented, because only the flat case is needed: it is the extreme point
    that maximises E_g, and the eigenbasis is the extreme point that minimises
    it.)
    """
    M = M.double()
    r = M.shape[0]
    lam, U = torch.linalg.eigh(M)
    Q = U.T.clone()                      # diag(Q M Q^T) = lam
    C = torch.diag(lam.clone())
    mu = float(lam.sum()) / r
    live = list(range(r))
    for _ in range(r - 1):
        d = torch.diagonal(C)
        i = max(live, key=lambda k: float(d[k]))
        j = min(live, key=lambda k: float(d[k]))
        if i == j:
            break
        a, b, m = float(C[i, i]), float(C[j, j]), float(C[i, j])
        R = math.hypot((a - b) / 2.0, m)
        if R < 1e-300:
            break
        phi = math.atan2(m, (a - b) / 2.0)
        th = (phi + math.acos(max(-1.0, min(1.0, (mu - (a + b) / 2.0) / R)))) / 2.0
        c, s_ = math.cos(th), math.sin(th)
        Gv = torch.eye(r, dtype=torch.float64, device=M.device)
        Gv[i, i], Gv[i, j], Gv[j, i], Gv[j, j] = c, s_, -s_, c
        C = Gv @ C @ Gv.T
        Q = Gv @ Q
        live.remove(i)
    err = float((torch.diagonal(Q @ M @ Q.T) - mu).abs().max()) / max(abs(mu), 1e-30)
    if err > tol:
        raise RuntimeError(f"flat_frame failed to converge: rel err {err:.2e}")
    return Q


def frame_ladder(Mg, ts):
    """Rotations Q(t) sweeping E_g from its minimum to its maximum at fixed
    gauge invariants.

    Q(0) is the eigenbasis of M_g (diagonal = spectrum, E_g minimal) and Q(1)
    flattens the diagonal (E_g = 1); in between, the geodesic
    Q(t) = Q_0 expm(t log(Q_0^T Q_1)) in SO(r).  Every Q(t) is orthogonal by
    construction, so every invariant of the triple is held EXACTLY fixed along
    the ladder -- to machine precision, not to a fitted tolerance -- and only
    the frame moves.  That is the cleanest possible intervention on the
    Adam-visible coordinate.
    """
    Mg = Mg.double()
    Q0 = torch.linalg.eigh(Mg)[1].T.contiguous()
    Q1 = flat_frame(Mg)
    if float(torch.linalg.det(Q0)) * float(torch.linalg.det(Q1)) < 0:
        Q0 = Q0.clone(); Q0[0] = -Q0[0]        # same component of O(r)
    Lg = torch.linalg.matrix_exp  # placeholder to keep the name local
    Rel = Q0.T @ Q1
    # real log of a rotation via its Schur form (block-diagonal 2x2 rotations)
    ev, V = torch.linalg.eig(Rel)
    L = (V @ torch.diag(torch.log(ev)) @ torch.linalg.inv(V))
    out = []
    for t in ts:
        Qt = (Q0.to(torch.complex128) @ torch.linalg.matrix_exp(t * L)).real
        U_, _, Vh_ = torch.linalg.svd(Qt)      # re-orthogonalise exactly
        out.append((U_ @ Vh_).double())
    return out


def eg_bounds(Mg):
    """(E_g at the eigenbasis frame, 1.0) -- the exact range of E_g over the
    gauge orbit at fixed invariants.

    E_g = (sum_j sqrt(d_j))^2 / (r sum_j d_j) with d = diag(Q M_g Q^T).  The map
    d -> (sum sqrt(d_j))^2 is symmetric and concave, hence Schur-concave, so
    over the Schur-Horn set {d prec eig(M_g)} it is minimised at the majorising
    extreme d = eig(M_g) and maximised at the flat diagonal, where it is 1.
    """
    lam = torch.linalg.eigvalsh(Mg.double()).clamp_min(0)
    r = lam.numel()
    return float(lam.sqrt().sum() ** 2) / (r * float(lam.sum()) + 1e-30), 1.0
