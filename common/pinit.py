"""Initializers for the LoRA down-projection A, and exact constructions that
let us prescribe the spectrum and the diagonal of the hidden preconditioner

    P = s^2 * A^T A ,      A in R^{r x d_in},  s = scaling (alpha/r or alpha/sqrt(r))

Structural facts used throughout (proved/verified in
01-hidden-preconditioner/src/gate0_feasible_region.py):

  * Let Pu = A^T A (the *unscaled* Gram, d_in x d_in, rank <= r).
    Write lam in R^r_+ for its nonzero spectrum and d = diag(Pu) in R^{d_in}_+.
    Feasibility of the pair (lam, d) is exactly Schur-Horn majorization
    lam (zero-padded to d_in) >= d.  A flat d is majorized by *every* lam with
    the same sum, so "equal coordinate gains" is compatible with any spectrum.
  * tr Pu = sum(lam) = sum(d),  ||Pu||_F^2 = sum(lam^2),  ||diag Pu||^2 = sum(d^2).
    Hence the *magnitude* of crosstalk
        ct(Pu)^2 = ||Pu - Diag(Pu)||_F^2 / ||Pu||_F^2
                 = 1 - sum(d^2)/sum(lam^2)
    is a *deterministic function of (lam, d)*.  Only the crosstalk **pattern**
    is free (the stabilizer of the (lam,d) pair).  This is why "matched trace +
    matched diagonal + different crosstalk magnitude" is infeasible, and why
    Experiment family B must be phrased as pattern-vs-pattern (BIMI-like) while
    magnitude changes must be routed through the spectrum.

Column convention: A is (r, d_in); "columns of A" are the d_in vectors in R^r,
so diag(A^T A)_j = ||A[:, j]||^2 is NoRA's coordinate-wise own-update gain.
"""

import math
import torch


# --------------------------------------------------------------------------
# primitive random draws
# --------------------------------------------------------------------------

def _rand_orth(n, k, generator, device, dtype=torch.float64):
    """Random (n, k) matrix with orthonormal columns, Haar distributed."""
    m = torch.randn(n, k, generator=generator, device=device, dtype=dtype)
    q, r = torch.linalg.qr(m)
    # sign fix -> Haar measure
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    return q


def kaiming_A(r, d_in, generator, device, dtype=torch.float64):
    """PEFT/HF default LoRA A init: kaiming_uniform_(a=sqrt(5)) on a (r, d_in)
    tensor -> U(-b, b) with b = sqrt(6 / ((1 + 5) * fan_in)) and fan_in = d_in.
    """
    bound = math.sqrt(6.0 / (6.0 * d_in))
    a = (torch.rand(r, d_in, generator=generator, device=device, dtype=dtype) * 2 - 1) * bound
    return a


def gaussian_A(r, d_in, generator, device, dtype=torch.float64, std=None):
    if std is None:
        std = 1.0 / math.sqrt(d_in)
    return torch.randn(r, d_in, generator=generator, device=device, dtype=dtype) * std


# --------------------------------------------------------------------------
# Bendel-Mickey / Schur-Horn: prescribe diag(A^T A) while keeping the spectrum
# --------------------------------------------------------------------------

def set_gram_diagonal_(A, target_diag, tol=1e-13, max_extra=4):
    """In-place: right-multiply A by an orthogonal matrix (Givens rotations on
    *columns* of A) so that ||A[:, j]||^2 == target_diag[j] for all j.

    Preserves the spectrum of A^T A exactly (orthogonal congruence).  This is a
    constructive Schur-Horn / Bendel-Mickey algorithm: each rotation fixes one
    coordinate exactly and that coordinate is then retired, so d_in - 1
    rotations suffice.  Cost O(d_in * r).

    Reachable set of the (i,i) entry under a Givens rotation in the (i,j) plane
    is [mu - rho, mu + rho] with mu = (a+b)/2, rho = sqrt(((a-b)/2)^2 + m^2),
    which always contains [min(a,b), max(a,b)].  Picking i = argmax residual and
    j = argmin residual keeps the target inside that interval in practice; if it
    ever does not, we move as far as possible and retry.
    """
    d_in = A.shape[1]
    tgt = target_diag
    assert tgt.shape == (d_in,)
    total = float(A.pow(2).sum())
    assert abs(total - float(tgt.sum())) < 1e-8 * max(1.0, total), \
        f"trace mismatch: ||A||_F^2={total} vs sum(target)={float(tgt.sum())}"

    nrm2 = A.pow(2).sum(0)
    active = torch.ones(d_in, dtype=torch.bool, device=A.device)
    thresh = tol * max(float(tgt.abs().mean()), 1e-300)
    for _ in range((d_in + max_extra * d_in)):
        resid = torch.where(active, nrm2 - tgt, torch.zeros_like(nrm2))
        if float(resid.abs().max()) <= thresh:
            return A
        i = int(torch.argmax(resid))
        j = int(torch.argmin(resid))
        if i == j:
            return A
        a = float(nrm2[i]); b = float(nrm2[j]); m = float(torch.dot(A[:, i], A[:, j]))
        mu = 0.5 * (a + b)
        rho = math.sqrt(0.25 * (a - b) ** 2 + m * m)
        T = float(tgt[i])
        exact = (mu - rho - 1e-15 * max(1.0, abs(mu))) <= T <= (mu + rho + 1e-15 * max(1.0, abs(mu)))
        T = min(max(T, mu - rho), mu + rho)
        if abs(T - b) < 1e-300:
            t = 0.0 if abs(m) < 1e-300 else (T - a) / (2.0 * m)
        else:
            disc = max(m * m - (T - b) * (T - a), 0.0)
            t = (m + math.sqrt(disc)) / (T - b)
        c = 1.0 / math.sqrt(1.0 + t * t)
        s_ = c * t
        ci = A[:, i].clone(); cj = A[:, j].clone()
        A[:, i] = c * ci + s_ * cj
        A[:, j] = -s_ * ci + c * cj
        nrm2[i] = A[:, i].pow(2).sum()
        nrm2[j] = A[:, j].pow(2).sum()
        if exact:
            active[i] = False
            if int(active.sum()) <= 1:
                return A
        else:
            # target unreachable in this plane (happens only for a handful of
            # coordinates once residuals are at fp64 roundoff); retire anyway so
            # the loop terminates.  Residual stays <= ~1e-8 relative, which is
            # ~8 orders below any diagonal imbalance we ever measure.
            active[i] = False
            if int(active.sum()) <= 1:
                return A
    return A


def A_from_spectrum(r, d_in, spectrum, generator, device, dtype=torch.float64):
    """Random A (r, d_in) whose Gram A^T A has exactly the given nonzero
    spectrum (len r), with a Haar-random eigenbasis."""
    spectrum = torch.as_tensor(spectrum, device=device, dtype=dtype)
    assert spectrum.numel() == r
    V = _rand_orth(d_in, r, generator, device, dtype)     # (d_in, r) orthonormal
    A = torch.diag(spectrum.clamp_min(0).sqrt()) @ V.T    # (r, d_in)
    return A


def A_with_spectrum_and_diagonal(r, d_in, spectrum, diag, generator, device,
                                 dtype=torch.float64):
    """A (r, d_in) with prescribed Gram spectrum AND prescribed Gram diagonal.
    Realises the Schur-Horn feasible set constructively."""
    A = A_from_spectrum(r, d_in, spectrum, generator, device, dtype)
    diag = torch.as_tensor(diag, device=device, dtype=dtype)
    if diag.numel() == 1:
        diag = diag.expand(d_in).clone()
    # two passes: the first retires coordinates left-to-right and leaves the
    # accumulated fp roundoff on the final coordinate; the second sweeps it out.
    set_gram_diagonal_(A, diag)
    set_gram_diagonal_(A, diag)
    return A


# --------------------------------------------------------------------------
# spectrum shapes  (all normalised to a given trace)
# --------------------------------------------------------------------------

def spectrum_shape(kind, r, trace, device, dtype=torch.float64, decay=None):
    """Return a length-r positive spectrum summing to `trace`."""
    if kind == "flat":
        lam = torch.ones(r, device=device, dtype=dtype)
    elif kind == "marchenko":          # spectrum of a random Gaussian Gram
        raise ValueError("use empirical spectrum of a drawn A instead")
    elif kind == "geometric":
        assert decay is not None
        lam = torch.tensor([decay ** i for i in range(r)], device=device, dtype=dtype)
    elif kind == "linear":
        lam = torch.linspace(1.0, 0.05, r, device=device, dtype=dtype)
    else:
        raise ValueError(kind)
    return lam * (trace / lam.sum())


# --------------------------------------------------------------------------
# named initializers used by the experiments
# --------------------------------------------------------------------------

def normalize_columns(A, target_norm=None):
    """NoRA's operator N(.): make every column of A have norm `target_norm`.
    If target_norm is None, use the RMS of the existing column norms, which
    preserves ||A||_F^2 (hence tr P) exactly -- the trace-matched NoRA."""
    n = A.norm(dim=0, keepdim=True).clamp_min(1e-12)
    if target_norm is None:
        target_norm = float((A.pow(2).sum() / A.shape[1]).sqrt())
    return A / n * target_norm


def make_A(kind, r, d_in, generator, device, dtype=torch.float64, ref_A=None,
           **kw):
    """Central factory.  `ref_A` is a reference draw (usually the vanilla
    kaiming draw with the same seed) used by the matched controls so that every
    condition in a matched panel comes from the *same* random state."""
    if kind == "kaiming":
        return kaiming_A(r, d_in, generator, device, dtype)

    if kind == "gaussian":
        return gaussian_A(r, d_in, generator, device, dtype, std=kw.get("std"))

    base = ref_A if ref_A is not None else kaiming_A(r, d_in, generator, device, dtype)

    if kind == "nora":
        # column-normalise, preserving ||A||_F^2 (= tr P):  trace-matched NoRA
        return normalize_columns(base)

    if kind == "nora_unit":
        # literal unit columns (tr P = d_in * s^2); changes magnitude on purpose
        return normalize_columns(base, target_norm=1.0)

    if kind == "flatspec":
        # flat spectrum, diagonal left free -- isolates spectrum shape
        lam = spectrum_shape("flat", r, float(base.pow(2).sum()), device, dtype)
        return A_from_spectrum(r, d_in, lam, generator, device, dtype)

    if kind == "flatspec_flatdiag":
        # flat spectrum AND flat diagonal: minimal crosstalk at this rank
        tr = float(base.pow(2).sum())
        lam = spectrum_shape("flat", r, tr, device, dtype)
        return A_with_spectrum_and_diagonal(r, d_in, lam, tr / d_in, generator,
                                            device, dtype)

    if kind == "kaimingspec_flatdiag":
        # keep the *empirical* spectrum of the vanilla draw, flatten the diagonal.
        # -> matched trace, matched spectrum, matched crosstalk magnitude,
        #    zero diagonal variance.  The decisive control for "is NoRA's
        #    diagonal flattening anything more than a spectrum/trace change?"
        lam = torch.linalg.svdvals(base) ** 2
        tr = float(base.pow(2).sum())
        return A_with_spectrum_and_diagonal(r, d_in, lam, tr / d_in, generator,
                                            device, dtype)

    if kind == "left_gauge":
        # A -> Q A with Q in O(r):  P is IDENTICAL (bit-for-bit in exact
        # arithmetic).  By F4/F5 this is the complete P0-equivalence class, so
        # any training difference is pure optimizer non-covariance and carries
        # ZERO hidden-preconditioner content.  This is the reference "null"
        # against which every method effect must be compared.
        Q = _rand_orth(r, r, generator, device, dtype)
        return Q @ base

    if kind == "rotated":
        # A -> A R with R Haar orthogonal: identical spectrum & crosstalk
        # magnitude, scrambled diagonal and pattern.  Gauge control.
        R = _rand_orth(d_in, d_in, generator, device, dtype)
        return base @ R

    raise ValueError(f"unknown A-init kind: {kind}")
