"""Beyn's contour-integral solver for the nonlinear eigenproblem ``det A(omega) = 0``.

This is the control-integral technique behind the network's linear-stability
analysis.  It lives **above the @njit line** -- pure NumPy/SciPy linear algebra
over the already-assembled complex operator -- and needs nothing from the kernel
beyond ``A(omega)^{-1}`` applied to a random probe block, which the same sparse
factorization the forced-response driver uses (``scipy ... splu``) supplies.

Why a contour method?
---------------------
The perturbation operator ``A(omega)`` is an **entire**
matrix function of ``omega`` (its only ``omega``-dependence is ``i*omega*M``, the
duct phases ``e^{-i*omega*tau}`` and any source/BC transfer function -- all
holomorphic with no poles), so its eigenvalues are isolated and a bounded contour
encloses finitely many.  Beyn (2012, *An integral method for solving nonlinear
eigenvalue problems*) recovers **all** of them inside the contour at once, with
their eigenvectors, and never forms ``det A`` -- whose dynamic range over a
network operator is astronomical and meaningless.  Forming the contour moments

    A_0 = 1/(2*pi*i) oint A(z)^{-1} V_hat dz,
    A_1 = 1/(2*pi*i) oint z A(z)^{-1} V_hat dz,

with a random probe ``V_hat`` and trapezoidal quadrature (spectrally accurate on a
smooth closed contour for an analytic integrand), an SVD of ``A_0`` reveals the
eigenvalue count and a small ``k x k`` standard eigenproblem yields the
eigenvalues and eigenvectors.

Completeness certificate
------------------------
Beyn's SVD rank counts the modes its random probe *resolved*; it can silently
under-count if the probe is too narrow or a contour encloses too many (especially symmetric)
modes.  :func:`winding_count` supplies the independent truth: the argument principle counts
the eigenvalues actually *inside* a contour from the winding of ``det A``
(via :func:`lu_logdet_phase`, phases only -- the determinant's magnitude is never formed).
Comparing the two turns the conditional guarantee into a checkable one, and lets the driver
search until they agree.

See also
--------
eigenmodes : the driver that tiles a search region into contours and validates the modes.
nyquist : the real-frequency winding count, for the regime the contour method cannot enter.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Contour:
    """A closed quadrature contour in the complex ``omega`` plane.

    Attributes
    ----------
    nodes : ndarray
        Quadrature points ``z_j`` on the contour (complex), shape ``(N,)``.
    weights : ndarray
        Quadrature weights ``w_j`` for ``1/(2*pi*i) oint f dz ~ sum_j w_j f(z_j)``
        (complex), shape ``(N,)``.  Already fold in ``dz/dt`` and ``1/(2*pi*i)``.
    center : complex
        Contour center, for the point-inside test.
    rx, ry : float
        Semi-axes (real, imaginary) of the enclosing ellipse, for the inside test.
    """

    nodes: np.ndarray
    weights: np.ndarray
    center: complex
    rx: float
    ry: float

    def __repr__(self) -> str:
        """Concise summary: node count, ellipse center/semi-axes, and the frequency band covered."""
        f_lo = (self.center.real - self.rx) / (2.0 * np.pi)
        f_hi = (self.center.real + self.rx) / (2.0 * np.pi)
        return (
            f"Contour: {np.asarray(self.nodes).size} nodes, center {self.center:.4g} rad/s, "
            f"semi-axes (rx={self.rx:.4g}, ry={self.ry:.4g}), f in [{f_lo:.1f}, {f_hi:.1f}] Hz"
        )

    def inside(self, z, margin: float = 1.0) -> bool:
        """Whether ``z`` lies inside the contour's ellipse (scaled by ``margin``)."""
        d = z - self.center
        return (d.real / (self.rx * margin)) ** 2 + (d.imag / (self.ry * margin)) ** 2 < 1.0


def ellipse_contour(center, rx, ry, n_nodes=128) -> Contour:
    """Build an elliptical quadrature contour ``z(t) = center + rx cos t + i ry sin t``.

    Parameters
    ----------
    center : complex
        Ellipse center (rad/s).
    rx, ry : float
        Real- and imaginary-axis semi-axes (rad/s).  ``rx`` spans angular
        frequency, ``ry`` spans growth rate.
    n_nodes : int, optional
        Number of trapezoidal quadrature points.  Trapezoidal quadrature on a
        smooth closed contour converges *exponentially* for analytic integrands, so
        a few dozen to ~128 points reaches machine precision away from eigenvalues.

    Returns
    -------
    Contour
    """
    rx = float(rx)
    ry = float(ry)
    if rx <= 0.0 or ry <= 0.0:
        raise ValueError(f"contour semi-axes must be positive; got rx={rx}, ry={ry}")
    t = 2.0 * np.pi * np.arange(n_nodes) / n_nodes
    nodes = center + rx * np.cos(t) + 1j * ry * np.sin(t)
    dz_dt = -rx * np.sin(t) + 1j * ry * np.cos(t)
    # 1/(2*pi*i) * dz ; dz = dz/dt * dt, dt = 2*pi/N  ->  w_j = dz/dt / (i N)
    weights = dz_dt / (1j * n_nodes)
    return Contour(nodes=nodes, weights=weights, center=complex(center), rx=rx, ry=ry)


def circle_contour(center, radius, n_nodes=128) -> Contour:
    """Build a circular quadrature contour (an ellipse with ``rx = ry = radius``)."""
    return ellipse_contour(center, radius, radius, n_nodes)


def beyn(solve, n, contour: Contour, *, n_probe=None, svd_tol=1e-10, rng=None, max_probe=None):
    """Eigenvalues (and eigenvectors) of ``A(z)`` inside ``contour`` by Beyn's method.

    Solves the holomorphic nonlinear eigenproblem ``A(z) v = 0`` for every ``z``
    enclosed by ``contour``, given only the action of ``A(z)^{-1}`` on a block of
    vectors (``solve``).  Never forms ``det A``.

    Parameters
    ----------
    solve : callable
        ``solve(z, B) -> A(z)^{-1} B`` for a complex scalar ``z`` and an
        ``(n, l)`` complex block ``B``.  This is the only operator access used; for
        a sparse network operator it is a per-node ``splu(...).solve(B)``.
    n : int
        Operator dimension.
    contour : Contour
        The search contour (see :func:`ellipse_contour`).
    n_probe : int, optional
        Probe-block width ``l`` -- an upper bound on the eigenvalue count the call
        can resolve.  Defaults to ``min(n, 20)``; grown automatically (up to
        ``max_probe``) when the SVD rank saturates, which signals more eigenvalues
        than probes.
    svd_tol : float, optional
        Relative singular-value threshold for the numerical rank of the first
        moment (the count of eigenvalues inside the contour).  Default ``1e-10``.
    rng : numpy.random.Generator, optional
        Random source for the probe block (default: a fixed seed, for
        reproducibility).
    max_probe : int, optional
        Cap on the probe width during growth (default ``min(n, 4*n_probe)``).

    Returns
    -------
    eigenvalues : ndarray
        Complex ``z`` with ``A(z)`` singular, inside the contour (unfiltered by
        residual -- the caller validates/refines).  Shape ``(k,)``.
    eigenvectors : ndarray
        Right null vectors, one per column, shape ``(n, k)``.
    info : dict
        ``{"n_probe": l, "rank": k, "saturated": bool}`` -- ``saturated`` flags that
        the rank hit the probe width even at ``max_probe`` (modes may be missed;
        shrink the contour or raise ``max_probe``).

    Notes
    -----
    The integrand has poles *at* the eigenvalues, so a quadrature node must not
    coincide with one (generic for a smooth contour); ``solve`` may guard against a
    singular factorization by nudging ``z``.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    n = int(n)
    width = int(n_probe) if n_probe else min(n, 20)
    width = max(1, min(width, n))
    cap = int(max_probe) if max_probe else min(n, 4 * width)
    cap = max(width, min(cap, n))

    saturated = False
    while True:
        V = rng.standard_normal((n, width)) + 1j * rng.standard_normal((n, width))
        A0 = np.zeros((n, width), dtype=np.complex128)
        A1 = np.zeros((n, width), dtype=np.complex128)
        for z, w in zip(contour.nodes, contour.weights):
            Y = solve(complex(z), V)  # A(z)^{-1} V_hat, shape (n, width)
            A0 += w * Y
            A1 += (w * z) * Y
        U, s, Wh = np.linalg.svd(A0, full_matrices=False)
        if s.size == 0 or s[0] == 0.0:
            return (
                np.empty(0, dtype=np.complex128),
                np.empty((n, 0), dtype=np.complex128),
                {
                    "n_probe": width,
                    "rank": 0,
                    "saturated": False,
                },
            )
        k = int(np.sum(s > svd_tol * s[0]))
        if k < width or width >= cap:
            saturated = k >= width and width >= cap
            break
        width = min(2 * width, cap)  # rank hit the probe width: more modes than probes, grow and retry

    if k == 0:
        return (
            np.empty(0, dtype=np.complex128),
            np.empty((n, 0), dtype=np.complex128),
            {
                "n_probe": width,
                "rank": 0,
                "saturated": False,
            },
        )

    U0 = U[:, :k]
    s0 = s[:k]
    W0 = Wh[:k, :].conj().T  # (width, k) right singular vectors
    # B = U0^H A1 W0 Sigma0^{-1}  -- the small (k x k) operator whose spectrum is the eigenvalues
    B = (U0.conj().T @ A1 @ W0) / s0[None, :]
    lam, S = np.linalg.eig(B)
    vecs = U0 @ S  # right null vectors of A at each eigenvalue
    return lam, vecs, {"n_probe": width, "rank": k, "saturated": saturated}


def _perm_parity(perm) -> int:
    """Parity (0 even, 1 odd) of a permutation given as an index array."""
    perm = np.asarray(perm)
    n = perm.size
    seen = np.zeros(n, dtype=bool)
    cycles = 0
    for i in range(n):
        if seen[i]:
            continue
        cycles += 1
        j = i
        while not seen[j]:
            seen[j] = True
            j = int(perm[j])
    return (n - cycles) & 1


def lu_logdet_phase(lu) -> float:
    """``arg(det A)`` (radians, any representative mod ``2*pi``) from a SuperLU factorization.

    SuperLU factors ``Pr A Pc = L U`` with permutation matrices ``Pr``, ``Pc``, so
    ``det A = sign(Pr) sign(Pc) prod(diag L) prod(diag U)`` and

        arg(det A) = sum arg(diag U) + sum arg(diag L) + pi * (parity of Pr and Pc).

    Only the *phase* is taken, never the product, so this is immune to the
    astronomical magnitude range of ``det A`` over a network operator -- the very
    reason :func:`beyn` avoids forming the determinant.  ``arg`` of a product equals
    the sum of the ``arg``\\ s modulo ``2*pi``, which is all the winding count needs.

    Parameters
    ----------
    lu : scipy.sparse.linalg.SuperLU
        A factorization of ``A(z)`` (``scipy.sparse.linalg.splu`` output).

    Returns
    -------
    float
        ``arg(det A(z))`` modulo ``2*pi``.  ``nan`` if a diagonal entry overflowed
        to a non-finite value (the contour reaches into the operator's overflow
        regime; the count is then untrustworthy).
    """
    diag = np.concatenate([np.asarray(lu.U.diagonal()), np.asarray(lu.L.diagonal())])
    if not np.all(np.isfinite(diag)):
        return float("nan")
    phase = float(np.sum(np.angle(diag)))
    par = (_perm_parity(lu.perm_r) + _perm_parity(lu.perm_c)) & 1
    return phase + (np.pi if par else 0.0)


def winding_count(det_phase, contour: Contour):
    """Eigenvalue count enclosed by ``contour`` via the argument principle.

    For an **entire** matrix function ``A`` (holomorphic, no poles) the number of
    eigenvalues (zeros of ``det A``, with algebraic multiplicity) inside a
    positively-oriented contour equals the winding number of ``det A(z)`` about the
    origin,

        N = 1/(2*pi*i) oint det'(z)/det(z) dz = [total change in arg det A(z)] / (2*pi).

    This is the independent completeness certificate for :func:`beyn`: Beyn's SVD
    rank says how many modes its *probe* resolved, whereas this integer says how
    many are actually *there*.  Only the phase of ``det A`` enters (see
    :func:`lu_logdet_phase`), so it never suffers the determinant's dynamic range.
    The phase is unwrapped increment-by-increment around the closed contour, exact
    provided the contour resolves the modes -- each per-step rotation below ``pi``
    (``max_jump`` in the diagnostics; near ``pi`` means densify the contour).

    Parameters
    ----------
    det_phase : callable
        ``det_phase(z) -> float`` giving ``arg(det A(z))`` mod ``2*pi`` at a complex
        ``z`` (e.g. :func:`lu_logdet_phase` of a per-node factorization).
    contour : Contour
        Positively-oriented (counter-clockwise) search contour.

    Returns
    -------
    count : int or None
        Enclosed eigenvalue count (rounded winding number), or ``None`` if the phase
        was non-finite anywhere (count untrustworthy; see :func:`lu_logdet_phase`).
    info : dict
        ``{"winding", "round_error", "max_jump"}`` -- the raw winding number, its
        distance to the nearest integer (large means the contour under-resolves or
        grazes a mode), and the largest per-step phase increment.
    """
    angles = np.array([float(det_phase(complex(z))) for z in contour.nodes], dtype=float)
    if not np.all(np.isfinite(angles)):
        return None, {"winding": float("nan"), "round_error": float("nan"), "max_jump": float("nan")}
    incr = np.diff(np.append(angles, angles[0]))
    incr = (incr + np.pi) % (2.0 * np.pi) - np.pi  # wrap each step into (-pi, pi]
    winding = float(np.sum(incr)) / (2.0 * np.pi)
    count = int(round(winding))
    return count, {
        "winding": winding,
        "round_error": abs(winding - count),
        "max_jump": float(np.max(np.abs(incr))) if incr.size else 0.0,
    }
