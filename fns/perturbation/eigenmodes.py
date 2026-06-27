"""Linear-stability eigenmodes of the perturbation network (theory.md s12.7 (ii)).

The stability question is the nonlinear eigenproblem

    det A(omega) = 0,   omega = omega_r + i*omega_i in C,

whose roots are the network's free oscillations: ``omega_r/(2*pi)`` the modal
frequency (Hz), ``omega_i`` the growth rate (theory.md s12.7).  ``A(omega)`` is the
*same* assembled operator the forced/scattering driver uses
(:func:`operator.assemble_acoustic`), now searched for the complex frequencies that
make it singular instead of solved against a forcing -- so the stability analysis
adds **no** new operator and **no** new kernel.  Because every ``omega``-dependence
(``i*omega*M``, the duct phases, and -- when present -- a source/BC transfer
function) is summed into ``A(omega)`` already, an active element (e.g. a flame
``S(omega)``) drops into the spectrum with no change here.

The driver mirrors :func:`response.perturbation_response`: take a converged mean
state, build the frozen blocks once, and sweep the operator -- here over a
quadrature *contour* in the complex plane rather than a real-frequency line --
returning an :class:`EigenmodeResult` from which modal frequencies, growth rates,
and mode shapes are read.  The eigenvalues come from :func:`contour.beyn` (the
control-integral technique); each is then polished by a Newton step on
``A(omega) v = 0`` and validated by its residual, so spurious quadrature artifacts
are dropped.

Sign convention.  The operator's time dependence is ``e^{+i*omega*t}`` (the duct
propagation ``f_head = e^{-i*omega*tau} f_tail`` of theory.md s12.3 is the causal
delay of a downstream wave under that convention, as the forced-response tests
confirm).  A free mode then evolves as ``e^{+i*omega*t} = e^{i*omega_r*t}
e^{-omega_i*t}``, so a **passive lossy resonator decays for ``Im(omega) > 0``**.
Hence the **growth rate is ``-Im(omega)``** and a mode is *unstable* iff ``Im(omega)
< 0``.  (This corrects the loose wording of theory.md s12.7; it is pinned by the
lossy-duct test, where ``Im(omega) > 0`` must come out decaying.)
"""

import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import scipy.sparse.linalg as spla

from .operator import build_acoustic_blocks, assemble_acoustic
from .characteristics import edge_transforms, basis_block_from_state, edge_caloric
from .contour import Contour, ellipse_contour, beyn, winding_count, lu_logdet_phase
from .terminals import find_terminals
from .modeshape import build_geometry, reconstruct_field, VARIABLE_SPEC, NetworkGeometry
from ..solver.control import states_table
from ..derive import ES_C

# Below this Mach number a duct's entropy wave is treated as decoupled (stationary)
# in the stability assembly: its transit time tau_0 = L/u diverges as u -> 0, so for a
# complex omega the entropy phase e^{-i omega tau_0} would overflow.  The entropy wave
# does not convect at near-zero mean flow (theory.md s12.6) and never lies in the
# acoustic band, so dropping its phase is exact for the acoustic spectrum.
_ENTROPY_DECOUPLE_MACH = 1e-3


class EigenmodeWarning(UserWarning):
    """Diagnostic from the eigenmode search (no frequency dependence, saturated probes, ...)."""


# exp() overflows float64 at an argument ~709; cap the duct-phase exponent below it
# so a complex omega never produces inf in A(omega).  |e^{-i*omega*tau}| = e^{Im(omega)*tau}.
_EXP_LIMIT = 650.0

# Target eigenvalue count per Beyn sub-contour.  A single contour enclosing many
# (especially symmetrically placed) modes makes the contour moments rank-deficient and
# misses modes; tiling the frequency band into sub-contours of a few modes each is the
# standard robust practice and resolves them all.
_MODES_PER_SUBCONTOUR = 2

# Adaptive certification: if Beyn finds fewer modes than the argument-principle count
# (:func:`contour.winding_count`) says are inside the region, the band is re-tiled into
# this many times more sub-contours and the probe widened, then re-searched -- up to
# _MAX_REFINE_ROUNDS times.  This is what makes the driver self-correcting: the user
# never has to hand-tune n_probe / sub-contour counts to recover a missed mode.
_REFINE_GROWTH = 2
_MAX_REFINE_ROUNDS = 3


def _max_tau(blocks) -> float:
    """Largest finite delay across the network: duct transit times + source FTF lags.

    The contour clamp uses this so ``e^{-i omega tau}`` (duct phases and an ``n-tau``
    source) does not overflow at large growth/decay rates; a flame's transport lag
    counts the same as a duct's transit time.
    """
    taus = []
    for st in blocks.duct_stamps:
        taus.extend([st.tau_p, st.tau_m])
        if np.isfinite(st.tau_0):
            taus.append(st.tau_0)
    for st in getattr(blocks, "source_stamps", []):
        if np.isfinite(st.max_delay) and st.max_delay > 0.0:
            taus.append(float(st.max_delay))
    return max(taus) if taus else 0.0


def _estimate_mode_count(blocks, w_lo, w_hi) -> int:
    """Rough acoustic-mode count in ``[w_lo, w_hi]`` summed over the ducts.

    Each duct of round-trip time ``tau_+ + tau_-`` has its acoustic modes spaced
    ``2*pi/(tau_+ + tau_-)`` apart; the band holds about ``(w_hi - w_lo) * T /
    (2*pi)`` of them.
    """
    total = 0.0
    for st in blocks.duct_stamps:
        T = st.tau_p + st.tau_m
        total += (w_hi - w_lo) * T / (2.0 * np.pi)
    return int(np.ceil(total))


def _tile(c_re, c_im, rx, ry, n_sub, n_nodes):
    """``n_sub`` contiguous, slightly overlapping search ellipses tiling ``[c_re-rx, c_re+rx]``."""
    sub_rx = rx / n_sub
    return [
        ellipse_contour((c_re - rx + sub_rx * (2 * i + 1)) + 1j * c_im, sub_rx * 1.1, ry, n_nodes) for i in range(n_sub)
    ]


def _band_subcontours(freq_band, growth_band, n_nodes, blocks, n_probe):
    """Tile the search region (rad/s) into Beyn sub-contours from Hz/(1/s) bands.

    The real axis spans ``2*pi*freq_band`` and the imaginary axis the growth-rate band
    (growth ``= -Im(omega)``).  The band is split into ``ceil(estimated modes /
    _MODES_PER_SUBCONTOUR)`` overlapping elliptical sub-contours so each encloses only a
    few modes (a single contour over many symmetric modes is rank-deficient).  The
    imaginary half-axis is clamped to keep the duct phases from overflowing.

    Returns ``(subcontours, bound, n_probe, geom)``: the list of search ellipses, a
    bounding ellipse for the in-region test, the (possibly defaulted) probe width, and
    the region geometry ``(c_re, c_im, rx, ry)`` for adaptive re-tiling.
    """
    f_lo, f_hi = float(freq_band[0]), float(freq_band[1])
    if not f_hi > f_lo:
        raise ValueError(f"freq_band must be increasing and positive-width; got {freq_band}")
    w_lo, w_hi = 2.0 * np.pi * f_lo, 2.0 * np.pi * f_hi
    c_re = 0.5 * (w_lo + w_hi)
    rx = 0.5 * (w_hi - w_lo)

    est_modes = max(1, _estimate_mode_count(blocks, w_lo, w_hi))
    n_sub = max(1, int(np.ceil(est_modes / _MODES_PER_SUBCONTOUR)))
    sub_rx = rx / n_sub

    tau_max = _max_tau(blocks)
    im_limit = _EXP_LIMIT / tau_max if tau_max > 0.0 else np.inf
    if growth_band is None:
        c_im = 0.0
        ry = min(sub_rx, 0.8 * im_limit)  # default: near-circular sub-contours, overflow-clamped
    else:
        g_lo, g_hi = float(growth_band[0]), float(growth_band[1])
        if not g_hi > g_lo:
            raise ValueError(f"growth_band must be increasing; got {growth_band}")
        # growth rate = -Im(omega), so a growth-rate band maps to an Im(omega) band of opposite sign
        c_im = -0.5 * (g_lo + g_hi)
        ry = 0.5 * (g_hi - g_lo)
        if abs(c_im) + ry > im_limit:
            warnings.warn(
                f"growth_band reaches |Im(omega)|={abs(c_im) + ry:.3g} rad/s, beyond the "
                f"overflow-safe limit {im_limit:.3g} for the longest duct (tau={tau_max:.3g} s); "
                "the duct phases would overflow. Narrow growth_band or shorten/split the duct.",
                EigenmodeWarning,
                stacklevel=3,
            )
    ry = max(ry, 1e-9 * max(sub_rx, 1.0))

    subs = _tile(c_re, c_im, rx, ry, n_sub, n_nodes)
    bound = ellipse_contour(c_re + 1j * c_im, rx * 1.02, ry, 8)  # for the in-region test only
    if n_probe is None:
        per_sub = max(1, int(np.ceil(est_modes / n_sub)))
        n_probe = max(6, 2 * per_sub + 4)
    return subs, bound, n_probe, (c_re, c_im, rx, ry)


class _Factorizer:
    """Nudge-guarded sparse factorization of ``A(z)``, shared by the Beyn solves and the
    argument-principle count.

    A quadrature node must not coincide with an eigenvalue (where ``A(z)`` is exactly
    singular); if it does, the node is nudged by a negligible fraction of ``|z|`` -- far
    below the contour scale, so neither the moment integral nor the winding phase is
    affected -- and refactored.
    """

    def __init__(self, A_of):
        self._A_of = A_of

    def _factor(self, z):
        last = None
        for k in range(5):
            zz = z if k == 0 else z + (1e-7 * (k + 1)) * abs(z) * (1.0 + 1.0j)
            try:
                return spla.splu(self._A_of(zz).tocsc())
            except RuntimeError as exc:  # "Factor is exactly singular"
                last = exc
        raise last

    def solve(self, z, B):
        """``A(z)^{-1} B`` for Beyn (matches its ``solve(z, B)`` interface)."""
        return self._factor(z).solve(B)

    def det_phase(self, z):
        """``arg(det A(z))`` mod ``2*pi`` for the winding count."""
        return lu_logdet_phase(self._factor(z))


def _residual(A_of, omega, x):
    """Scaled residual ``||A(omega) x|| / max|A(omega)|`` for a unit-norm ``x``.

    Normalizing by the operator's largest entry makes the threshold dimensionless:
    a genuine null vector gives ``<< 1``, a spurious one ``O(1)``.
    """
    A = A_of(omega)
    scale = float(np.abs(A.data).max()) if A.nnz else 1.0
    return float(np.linalg.norm(A @ x)) / max(scale, 1e-300)


def _refine(A_of, omega, x, *, tol, maxit=8):
    """Polish ``(omega, x)`` by Newton on the bordered system ``[A(omega) x; x^H x - 1] = 0``.

    Newton's method for the nonlinear eigenproblem (residual-inverse-iteration form):
    each step solves ``A(omega) y = A'(omega) x`` and updates ``d_omega = -1/(x^H
    y)``, ``x <- -d_omega y`` (renormalized).  The derivative action ``A'(omega) x``
    is taken by a central difference in ``omega`` -- which keeps the polish
    *source-agnostic* (it re-evaluates the same assembled operator, so a future
    flame/storage term is differentiated automatically, never re-derived here).
    """
    x = x / np.linalg.norm(x)
    w = complex(omega)
    r = _residual(A_of, w, x)
    for _ in range(maxit):
        if r < tol:
            break
        A = A_of(w)
        try:
            lu = spla.splu(A.tocsc())
        except RuntimeError:
            break  # singular factorization (w sits on the eigenvalue): residual already tiny
        h = 1e-6 * (abs(w) + 1.0)
        Ap_x = (A_of(w + h) @ x - A_of(w - h) @ x) / (2.0 * h)  # A'(omega) x
        y = lu.solve(Ap_x)
        denom = np.vdot(x, y)  # x^H y
        if denom == 0.0:
            break
        dw = -1.0 / denom
        x_new = -dw * y
        nrm = np.linalg.norm(x_new)
        if nrm == 0.0:
            break
        x_new /= nrm
        w_new = w + dw
        r_new = _residual(A_of, w_new, x_new)
        if not np.isfinite(r_new) or r_new > r:
            break  # diverging: keep the better iterate
        w, x, r = w_new, x_new, r_new
    return w, x, r


def _dedup(omegas, modes, residuals, rtol=1e-4):
    """Merge eigenvalues that coincide to ``rtol`` (Beyn can return repeats), keeping the best residual."""
    order = sorted(range(len(omegas)), key=lambda i: residuals[i])
    kept = []
    for i in order:
        w = omegas[i]
        scale = max(abs(w), 1.0)
        if all(abs(w - omegas[j]) > rtol * scale for j in kept):
            kept.append(i)
    kept.sort(key=lambda i: omegas[i].real)
    return kept


def eigenmodes(
    prob,
    x_bar,
    freq_band=None,
    *,
    growth_band=None,
    n_nodes=128,
    n_probe=None,
    eps=None,
    eps_fb=1e-6,
    u_floor=1e-8,
    isentropic=False,
    svd_tol=1e-10,
    residual_tol=1e-6,
    refine=True,
    certify=True,
    max_refine_rounds=_MAX_REFINE_ROUNDS,
    rng=None,
    contour=None,
):
    """Free-oscillation eigenmodes of the perturbation network in a region of the complex plane.

    Finds the complex frequencies ``omega`` at which ``A(omega)`` (the assembled
    perturbation operator under each terminal's declared :class:`PerturbationBC`) is
    singular -- the network's self-sustained modes -- by Beyn's contour-integral
    method (:func:`contour.beyn`), polishing and validating each by its residual.
    The operator is identical to the one the forced/scattering driver uses, so the
    spectrum and the response are guaranteed consistent.

    Use **passive** terminal BCs (``hard_wall``/``open_end``/``anechoic``/
    ``reflection``/``impedance``, or ``inherit``); a terminal's ``driven`` forcing
    has no meaning for a free oscillation (the eigenproblem ignores ``b``).  At least
    one length-bearing duct (or another ``omega``-dependent term) must be present, else
    ``A`` has no frequency dependence and there is no spectrum.

    Parameters
    ----------
    prob : CompiledProblem
        Compiled flow network (carries the terminal BCs in ``prob.node_bc``).
    x_bar : ndarray
        Converged mean-flow state, shape ``(n_solve, E)``.
    freq_band : tuple of float
        ``(f_lo, f_hi)`` real-frequency window to search, in **Hz**.  Required unless
        an explicit ``contour`` is given.
    growth_band : tuple of float, optional
        ``(g_lo, g_hi)`` growth-rate window in 1/s (growth rate ``= -Im(omega)``;
        positive is unstable).  Default: a roughly square region about the real axis
        (clamped to keep the duct phases from overflowing).  Widen it to hunt
        strongly growing/decaying modes.
    n_nodes : int, optional
        Quadrature points on the contour (default 128).  Trapezoidal quadrature
        converges exponentially, so more points buy accuracy cheaply; each costs one
        sparse factorization.
    n_probe : int, optional
        Beyn probe-block width (upper bound on the modes resolved per call).
        Default: estimated from the ducts' mode spacing in the band, grown
        automatically if it saturates.
    eps, eps_fb, u_floor : float, optional
        Operator-assembly regularizers forwarded to ``build_acoustic_blocks``.
    isentropic : bool, optional
        Force isentropic perturbations (``rho' = p'/c^2``): the convected entropy wave is
        pinned to zero on every edge, leaving the two acoustic waves (default False).  This
        is the standard acoustic-stability assumption -- it drops entropy/convective modes
        from the spectrum and removes the near-stagnant entropy-phase overflow entirely --
        and uses the *same* solver, contour, and certificate machinery (no reconfiguration).
    svd_tol : float, optional
        Relative singular-value cutoff for the Beyn rank (mode count). Default 1e-10.
    residual_tol : float, optional
        Scaled-residual cutoff to keep a mode (drops Beyn quadrature artifacts).
        Default 1e-6.
    refine : bool, optional
        Whether to Newton-polish each eigenpair before validating (default True).
    certify : bool, optional
        Whether to certify completeness with the argument principle (default True).
        The number of modes the region truly contains is counted independently from
        the winding of ``det A`` (:func:`contour.winding_count`); if Beyn resolves
        fewer, the band is re-tiled finer and the probe widened, then re-searched,
        until the two agree or ``max_refine_rounds`` is exhausted (then a warning is
        raised).  The count is reported on :attr:`EigenmodeResult.expected` and the
        match on :attr:`EigenmodeResult.certified`.
    max_refine_rounds : int, optional
        Maximum adaptive re-tile/re-search rounds when the Beyn count falls short of
        the certificate (default 3).  Each round multiplies the sub-contour count and
        probe width by a fixed factor.
    rng : numpy.random.Generator, optional
        Random source for the Beyn probe (default: a fixed seed, reproducible).
    contour : Contour, optional
        A fully specified search contour, overriding ``freq_band``/``growth_band``
        (for total control of the region; see :func:`contour.ellipse_contour`).

    Returns
    -------
    EigenmodeResult
        The validated modes (frequencies in Hz, growth rates in 1/s, mode shapes).

    Raises
    ------
    ValueError
        If the band is degenerate.

    Notes
    -----
    ``omega``-dependent table/impedance BCs are evaluated at the *complex* contour
    frequency; a tabulated reflection (interpolated on a real grid) is not
    analytically continuable and is unsupported for stability -- use a constant or
    closed-form BC.

    The completeness certificate (``certify``) counts *algebraic* multiplicity, so a
    genuine repeated root contributes more than one to :attr:`EigenmodeResult.expected`
    while the de-duplicated mode list holds it once; such (non-generic) exact
    degeneracies therefore read as uncertified.  A large ``round_error`` or a mode on
    the region boundary likewise leaves the count ambiguous and is warned about.
    """
    K = float(prob.tf[0]) / float(prob.tf[1])
    est = states_table(prob, x_bar)
    cals = edge_caloric(prob, x_bar)
    _, L = edge_transforms(est, K, cals)
    terminals = find_terminals(prob, x_bar)
    # decouple the entropy wave on near-stagnant ducts (tau_0 -> inf would overflow at
    # complex omega); never affects the acoustic spectrum (theory.md s12.6).
    u_floor = max(u_floor, _ENTROPY_DECOUPLE_MACH * float(np.max(est[ES_C])))
    blocks = build_acoustic_blocks(prob, x_bar, eps=eps, eps_fb=eps_fb, u_floor=u_floor, isentropic=isentropic)
    n = int(blocks.J_alg.shape[0])

    if not blocks.duct_stamps and blocks.M.nnz == 0 and not blocks.has_sources:
        warnings.warn(
            "no duct (length-bearing element) and no storage: A(omega) has no frequency "
            "dependence beyond the boundary conditions, so the spectrum is empty or ill-posed.",
            EigenmodeWarning,
            stacklevel=2,
        )
    if blocks.has_sources and not all(st.analytic for st in blocks.source_stamps):
        raise ValueError(
            "the stability eigenproblem searches complex frequencies, but a dynamic source "
            "carries a transfer function that is not analytically continuable (e.g. a table "
            "interpolated on a real grid). Supply a closed-form model (e.g. n_tau) for stability, "
            "or use the forced response for a real-frequency sweep."
        )

    rng = np.random.default_rng(0) if rng is None else rng
    if contour is not None:
        subs, bound = [contour], contour
        geom = (float(contour.center.real), float(contour.center.imag), float(contour.rx), float(contour.ry))
        if n_probe is None:
            n_probe = min(n, 20)
    elif freq_band is None:
        raise ValueError("provide a freq_band (Hz) to search, or an explicit contour")
    else:
        subs, bound, n_probe, geom = _band_subcontours(freq_band, growth_band, n_nodes, blocks, n_probe)

    def A_of(omega):
        return assemble_acoustic(omega, blocks, with_boundaries=True)

    factorizer = _Factorizer(A_of)

    # Completeness certificate: count the eigenvalues actually inside the region from the
    # winding of det A (argument principle) -- independent of what Beyn's probe resolves.
    # The counting contour matches the acceptance region (bound at the validate margin), so
    # "counted inside" and "kept inside" agree; it is resolved with enough nodes that the
    # det-phase rotates well under pi per step even at the full mode count.
    expected = None
    if certify:
        rx_c, ry_c = bound.rx * 1.05, bound.ry * 1.05
        est_region = max(1, _estimate_mode_count(blocks, bound.center.real - rx_c, bound.center.real + rx_c))
        n_count = int(min(4096, max(n_nodes, 8 * est_region)))
        count_contour = ellipse_contour(bound.center, rx_c, ry_c, n_count)
        expected, cert_info = winding_count(factorizer.det_phase, count_contour)
        if expected is None:
            warnings.warn(
                "completeness uncertified: the operator overflowed on the counting contour, so the "
                "argument-principle mode count is unavailable. Narrow growth_band or split long ducts.",
                EigenmodeWarning,
                stacklevel=2,
            )
        elif cert_info["max_jump"] > 0.9 * np.pi:
            warnings.warn(
                "a mode lies very close to the search-region boundary (rapid det-phase rotation on the "
                "counting contour); its membership is ambiguous -- shift freq_band/growth_band to resolve it.",
                EigenmodeWarning,
                stacklevel=2,
            )

    def _search(subcontours, probe):
        """Beyn over each sub-contour; pool the candidate eigenpairs that fall inside it."""
        cand_w, cand_v, sat = [], [], False
        for sub in subcontours:
            lam, vecs, info = beyn(factorizer.solve, n, sub, n_probe=probe, svd_tol=svd_tol, rng=rng)
            sat = sat or info.get("saturated", False)
            for i in range(lam.size):
                if sub.inside(complex(lam[i])):
                    cand_w.append(complex(lam[i]))
                    cand_v.append(vecs[:, i])
        return cand_w, cand_v, sat

    def _validate(cand_w, cand_v):
        """Polish, residual-filter, region-clip and de-duplicate the Beyn candidates."""
        oms, mds, res = [], [], []
        for w0, v0 in zip(cand_w, cand_v):
            w, x = w0, v0 / np.linalg.norm(v0)
            if refine:
                w, x, r = _refine(A_of, w, x, tol=residual_tol)
            else:
                r = _residual(A_of, w, x)
            if r < residual_tol and bound.inside(w, margin=1.05):
                oms.append(w)
                mds.append(x)
                res.append(r)
        keep = _dedup(oms, mds, res)
        return [oms[i] for i in keep], [mds[i] for i in keep], [res[i] for i in keep]

    # Search, then adaptively re-tile finer / widen the probe until the resolved count
    # meets the certificate (or the round budget is spent).
    subcontours, probe = subs, n_probe
    omegas, modes, residuals, saturated = [], [], [], False
    for round_idx in range(max_refine_rounds + 1):
        cand_w, cand_v, saturated = _search(subcontours, probe)
        omegas, modes, residuals = _validate(cand_w, cand_v)
        if expected is None or len(omegas) >= expected or round_idx == max_refine_rounds:
            break
        probe = min(n, _REFINE_GROWTH * probe)
        c_re, c_im, rx, ry = geom
        subcontours = _tile(c_re, c_im, rx, ry, len(subcontours) * _REFINE_GROWTH, n_nodes)

    if expected is not None and len(omegas) != expected:
        warnings.warn(
            f"completeness check: the argument principle counts {expected} mode(s) in the region but "
            f"{len(omegas)} were resolved after {max_refine_rounds} refinement round(s). "
            "Widen n_probe/n_nodes, narrow the band, or check for a near-degenerate (repeated) mode.",
            EigenmodeWarning,
            stacklevel=2,
        )
    elif expected is None and saturated:
        warnings.warn(
            "Beyn probe width saturated: a sub-contour may hold more modes than were resolved. "
            "Raise n_probe, or narrow freq_band so the sub-contours enclose fewer modes.",
            EigenmodeWarning,
            stacklevel=2,
        )

    omega = np.array(omegas, dtype=np.complex128)
    mode_arr = np.array(modes, dtype=np.complex128) if modes else np.empty((0, n), np.complex128)
    resid = np.array(residuals, dtype=float)

    return EigenmodeResult(
        omega=omega,
        modes=mode_arr,
        residuals=resid,
        L=L,
        est=est,
        K=K,
        cals=cals,
        terminals=terminals,
        n_solve=int(prob.n_solve),
        n_edges=int(prob.n_edges),
        contour=bound,
        node_names=tuple(getattr(prob, "node_names", ()) or ()),
        expected=expected,
        geometry=build_geometry(prob),
    )


_CHAR_SYM = ("f", "g", "h")

# Cap on the number of mode rows shown in the plain-text repr (the HTML repr lists all of them).
_REPR_MAX_ROWS = 20


@dataclass
class EigenmodeResult:
    """Validated free-oscillation eigenmodes of the perturbation network.

    Frequencies are reported in **Hz** (``Re(omega)/(2*pi)``) and growth rates in
    1/s (``-Im(omega)``); a mode is unstable iff its growth rate is positive
    (equivalently ``Im(omega) < 0``).  Mode shapes are stored as the full nodal
    eigenvector and projected to characteristic amplitudes ``(f, g, h)`` on demand.

    Attributes
    ----------
    omega : ndarray
        Complex modal angular frequencies (rad/s), shape ``(n_modes,)``.
    modes : ndarray
        Unit-norm nodal eigenvectors (mode shapes in solution variables), shape
        ``(n_modes, n_col)``.
    residuals : ndarray
        Per-mode scaled residual ``||A(omega) v|| / max|A(omega)|``.
    L : list of ndarray
        Per-edge ``dx_to_char`` (3x3) maps at the frozen mean state.
    est : ndarray
        Frozen mean edge-state table.
    K : float
        ``cp / R`` of the mean gas.
    n_solve : int
        Solve-variable stride per edge in the nodal vector.
    n_edges : int
        Number of edges.
    contour : Contour
        The search contour the modes were found in.
    node_names : tuple
        Per-node element labels (for plot annotation).
    expected : int or None
        Eigenvalue count the region must contain by the argument principle
        (completeness certificate); ``None`` if certification was disabled or
        unavailable.  Compare against :attr:`n_modes` via :attr:`certified`.
    """

    omega: np.ndarray
    modes: np.ndarray
    residuals: np.ndarray
    L: List[np.ndarray]
    est: np.ndarray
    K: float
    n_solve: int
    n_edges: int
    contour: Optional[Contour] = None
    node_names: tuple = field(default=())
    expected: Optional[int] = None
    # per-edge caloric rows (characteristics.edge_caloric) for the reacting "network" flavor
    cals: Optional[list] = None
    # 1-port boundary terminals (terminals.find_terminals) for acoustic-power diagnostics
    terminals: Optional[list] = None
    # topology + duct lengths for spatial mode-shape reconstruction (modeshape.build_geometry)
    geometry: Optional[NetworkGeometry] = None

    def __len__(self) -> int:
        return int(self.omega.size)

    @property
    def n_modes(self) -> int:
        """Number of modes found."""
        return int(self.omega.size)

    @property
    def certified(self) -> bool:
        """Whether the resolved mode count matches the argument-principle certificate.

        ``True`` only when completeness was checked (``certify=True``) and the number
        of modes found equals :attr:`expected` -- i.e. every eigenvalue the region
        contains was recovered.  ``False`` if certification was off/unavailable or the
        counts disagree (a warning is raised in the latter case).
        """
        return self.expected is not None and self.expected == self.n_modes

    @property
    def freqs(self) -> np.ndarray:
        """Modal frequencies ``Re(omega)/(2*pi)`` in Hz."""
        return self.omega.real / (2.0 * np.pi)

    @property
    def growth_rates(self) -> np.ndarray:
        """Growth rates ``-Im(omega)`` in 1/s (positive = growing = unstable)."""
        return -self.omega.imag

    @property
    def damping_ratios(self) -> np.ndarray:
        """Modal damping ratios ``Im(omega) / |omega_r|`` (positive = decaying)."""
        wr = np.abs(self.omega.real)
        return self.omega.imag / np.where(wr > 0.0, wr, np.inf)

    @property
    def unstable(self) -> np.ndarray:
        """Boolean mask of growing (unstable) modes (``Im(omega) < 0``)."""
        return self.omega.imag < 0.0

    def _search_band(self):
        """Frequency/growth extent of the search contour.

        Returns
        -------
        tuple of float or None
            ``(f_lo, f_hi, g_lo, g_hi)`` -- the searched frequency band (Hz) and growth-rate
            band (1/s) inferred from the contour's ellipse, or ``None`` if no contour is stored.
        """
        c = self.contour
        if c is None:
            return None
        f_lo = (c.center.real - c.rx) / (2.0 * np.pi)
        f_hi = (c.center.real + c.rx) / (2.0 * np.pi)
        # growth = -Im(omega), so the imaginary extent [center.imag +/- ry] maps to growth flipped
        g_lo = -c.center.imag - c.ry
        g_hi = -c.center.imag + c.ry
        return f_lo, f_hi, g_lo, g_hi

    def _status(self):
        """``(n_unstable, certification_text)`` for the repr headers."""
        n_unst = int(np.count_nonzero(self.unstable))
        if self.expected is None:
            cert = "uncertified"
        elif self.certified:
            cert = "certified complete"
        else:
            cert = f"incomplete ({self.n_modes}/{self.expected})"
        return n_unst, cert

    def __repr__(self) -> str:
        """Compact text summary: mode count, search band, and a per-mode stability table.

        Modes are listed in order of increasing frequency (the displayed ``#`` is the original
        mode index, as accepted by :meth:`mode_shape`/:meth:`plot_mode`); an unstable mode is
        flagged with a trailing ``*``.
        """
        n = self.n_modes
        n_unst, cert = self._status()
        lines = [f"EigenmodeResult: {n} mode{'' if n == 1 else 's'}, {n_unst} unstable, {cert}"]
        band = self._search_band()
        if band is not None:
            f_lo, f_hi, g_lo, g_hi = band
            lines.append(f"  search band: f in [{f_lo:.1f}, {f_hi:.1f}] Hz, growth in [{g_lo:+.1f}, {g_hi:+.1f}] 1/s")
        if n == 0:
            return "\n".join(lines)
        lines.append("")
        lines.append(f"  {'#':>4}  {'f [Hz]':>10}  {'growth [1/s]':>13}  {'damping':>9}  {'residual':>9}")
        order = np.argsort(self.freqs)
        for i in order[:_REPR_MAX_ROWS]:
            tag = f"{int(i)}{'*' if self.unstable[i] else ''}"
            lines.append(
                f"  {tag:>4}  {self.freqs[i]:>10.3f}  {self.growth_rates[i]:>+13.3f}  "
                f"{self.damping_ratios[i]:>+9.4f}  {self.residuals[i]:>9.1e}"
            )
        if n > _REPR_MAX_ROWS:
            lines.append(f"  ... ({n - _REPR_MAX_ROWS} more)")
        lines.append("")
        lines.append("  * = unstable (growth > 0)")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """Rich HTML summary for Jupyter: header line plus a per-mode stability table.

        Unstable modes are highlighted; rows are sorted by frequency and the ``#`` column holds
        the original mode index.
        """
        n = self.n_modes
        n_unst, cert = self._status()
        cert_color = {"certified complete": "#2a8a4a", "uncertified": "#888"}.get(cert, "#c0392b")
        parts = [
            f"{n} mode{'' if n == 1 else 's'}",
            (f"<b style='color:#c0392b'>{n_unst} unstable</b>" if n_unst else f"{n_unst} unstable"),
            f"<span style='color:{cert_color}'>{cert}</span>",
        ]
        band = self._search_band()
        if band is not None:
            f_lo, f_hi, g_lo, g_hi = band
            parts.append(
                f"search f &isin; [{f_lo:.1f}, {f_hi:.1f}] Hz, growth &isin; [{g_lo:+.1f}, {g_hi:+.1f}] s<sup>-1</sup>"
            )
        header = (
            "<div style='font-family:sans-serif;margin-bottom:4px'>"
            "<b>EigenmodeResult</b> &nbsp;&middot;&nbsp; " + " &nbsp;|&nbsp; ".join(parts) + "</div>"
        )
        if n == 0:
            return header
        th = "style='text-align:right;padding:2px 8px;border-bottom:1px solid #ccc'"
        head_row = (
            f"<tr><th {th}>#</th><th {th}>f [Hz]</th><th {th}>growth [1/s]</th>"
            f"<th {th}>damping ratio</th><th {th}>residual</th><th style='padding:2px 8px'>stability</th></tr>"
        )
        body = []
        for i in np.argsort(self.freqs):
            unst = bool(self.unstable[i])
            bg = "background:#fdecea;" if unst else ""
            tag = (
                "<span style='color:#c0392b;font-weight:bold'>unstable</span>"
                if unst
                else "<span style='color:#888'>stable</span>"
            )
            td = "style='text-align:right;padding:2px 8px'"
            body.append(
                f"<tr style='{bg}'><td {td}>{int(i)}</td><td {td}>{self.freqs[i]:.3f}</td>"
                f"<td {td}>{self.growth_rates[i]:+.3f}</td><td {td}>{self.damping_ratios[i]:+.4f}</td>"
                f"<td {td}>{self.residuals[i]:.1e}</td><td style='padding:2px 8px'>{tag}</td></tr>"
            )
        table = (
            "<table style='border-collapse:collapse;font-family:monospace;font-size:0.9em'>"
            + head_row
            + "".join(body)
            + "</table>"
        )
        return header + table

    def mode_waves(self, i, edge):
        """Characteristic amplitudes ``(f, g, h)`` of mode ``i`` at ``edge``.

        Parameters
        ----------
        i : int
            Mode index.
        edge : int
            Edge id.

        Returns
        -------
        ndarray
            Complex shape ``(n_char,)`` -- the wave amplitudes of the mode shape.
        """
        ns = self.n_solve
        xe = self.modes[i, ns * edge : ns * edge + self.L[edge].shape[0]]
        return self.L[edge] @ xe

    def mode_shape(self, i, basis="char"):
        """Mode ``i`` projected onto every edge, shape ``(n_edges, n_char)``.

        Parameters
        ----------
        i : int
            Mode index.
        basis : str, optional
            Variable flavor (``characteristics.BASIS_LABELS``; e.g. ``"char"`` for
            ``(f, g, h)``, ``"primitive"`` for ``(rho', u', p')``).  Default ``"char"``.

        Returns
        -------
        ndarray
            Complex shape ``(n_edges, n_char)``.
        """
        out = np.empty((self.n_edges, self.L[0].shape[0]), dtype=np.complex128)
        for e in range(self.n_edges):
            w = self.mode_waves(i, e)
            if basis != "char":
                cal = None if self.cals is None else self.cals[e]
                w = basis_block_from_state(basis, self.est[:, e], self.K, cal) @ w
            out[e] = w
        return out

    def summary(self):
        """List of per-mode dicts: frequency (Hz), growth rate (1/s), damping ratio, residual.

        Returns
        -------
        list of dict
            One entry per mode, ordered by frequency.
        """
        return [
            {
                "freq_hz": float(self.freqs[i]),
                "growth_rate": float(self.growth_rates[i]),
                "damping_ratio": float(self.damping_ratios[i]),
                "unstable": bool(self.unstable[i]),
                "residual": float(self.residuals[i]),
            }
            for i in range(self.n_modes)
        ]

    def plot_spectrum(self, **kwargs):
        """Plot the spectrum: growth rate vs modal frequency, with the stability boundary.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        from ..plotting import plot_spectrum as _plot

        return _plot(self.freqs, self.growth_rates, residuals=self.residuals, **kwargs)

    def plot_mode(self, i, basis="char", **kwargs):
        """Plot the shape of mode ``i`` (wave magnitude and phase along the edges).

        Parameters
        ----------
        i : int
            Mode index.
        basis : str, optional
            Variable flavor (default ``"char"``).
        **kwargs
            Forwarded to :func:`fns.plotting.plot_mode_shape`.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        from ..plotting import plot_mode_shape as _plot
        from .characteristics import BASIS_LABELS

        shape = self.mode_shape(i, basis=basis)
        labels = BASIS_LABELS.get(basis)
        title = kwargs.pop("title", None) or (
            f"Mode {i}: f = {self.freqs[i]:.4g} Hz, growth = {self.growth_rates[i]:.4g} 1/s"
        )
        return _plot(shape, labels=labels, title=title, **kwargs)

    def field_along_network(self, i, *, variable="p", root=None, n_x=160):
        """Reconstruct mode ``i``'s spatial field along every root->leaf path.

        The continuous perturbation field *inside* every duct, recovered analytically
        from the mode's face wave-amplitudes (theory.md s12.3); see
        :func:`fns.perturbation.modeshape.reconstruct_field`.

        Parameters
        ----------
        i : int
            Mode index.
        variable : str, optional
            Plotted quantity (``"p"``, ``"u"``, ``"rho"``, ``"mdot"``, ``"f"``,
            ``"g"``, ``"h"``); default ``"p"``.
        root : int, optional
            Developed-length origin element (default: a mean-flow inlet).
        n_x : int, optional
            Interior samples per duct (default 160).

        Returns
        -------
        list of fns.perturbation.modeshape.PathField

        Raises
        ------
        ValueError
            If the result carries no geometry (constructed without a problem).
        """
        if self.geometry is None:
            raise ValueError("no network geometry stored; rebuild via eigenmodes() to enable spatial reconstruction")
        return reconstruct_field(
            self.geometry,
            lambda e: self.mode_waves(i, e),
            self.est,
            self.K,
            complex(self.omega[i]),
            variable=variable,
            root=root,
            n_x=n_x,
            cals=self.cals,
        )

    def animate_mode(self, i, *, variable="p", root=None, n_x=160, n_frames=48, normalize=True, **layout):
        """Animate mode ``i``'s spatial shape over one phase cycle (slider + play).

        Draws the instantaneous physical perturbation ``Re{psi(x) e^{i theta}}`` along
        the developed length, sweeping the phase ``theta`` with a play button, framed by
        the static ``+/- |psi(x)|`` envelope.  A serial network is one trace; a branched
        one shows one trace per root->leaf path, with compact elements marked where the
        field jumps.

        Parameters
        ----------
        i : int
            Mode index.
        variable : str, optional
            Plotted quantity (see :meth:`field_along_network`); default ``"p"``.
        root : int, optional
            Developed-length origin element (default: a mean-flow inlet).
        n_x : int, optional
            Interior samples per duct (default 160).
        n_frames : int, optional
            Phase frames over one cycle (default 48).
        normalize : bool, optional
            Scale the peak magnitude to 1 (default True; eigenvectors are arbitrary scale).
        **layout
            Forwarded to ``Figure.update_layout``.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        from ..plotting import animate_mode_shape as _animate

        fields = self.field_along_network(i, variable=variable, root=root, n_x=n_x)
        label = VARIABLE_SPEC[variable][2]
        title = layout.pop("title", None) or (
            f"Mode {i}: f = {self.freqs[i]:.4g} Hz, growth = {self.growth_rates[i]:.4g} 1/s"
        )
        return _animate(
            fields, var_label=label, title=title, n_frames=n_frames, normalize=normalize,
            freq_hz=float(self.freqs[i]), **layout,
        )

    def boundary_power(self, i=0):
        """Acoustic-power budget across the boundaries for mode ``i``.

        Attributes a mode's growth to the boundaries that feed or drain its acoustic
        energy (Myers flux through each terminal face).

        Parameters
        ----------
        i : int, optional
            Mode index (default 0).

        Returns
        -------
        fns.perturbation.power.BoundaryPower
            Per-terminal signed power shares; ``.net`` is the energy growth ``dE/dt``
            and ``.sign_consistent`` cross-checks it against the growth rate.
        """
        from .power import boundary_power as _bp

        return _bp(self, i, terminals=self.terminals)

    def energy_balance(self, i=0):
        """Acoustic-energy budget and energy-derived growth rate of mode ``i``.

        Forms the node-wise ledger (interior generation, boundary flux, stored duct energy) and
        returns the growth rate it implies, ``(generation + boundary_flux) / (2 E)``, beside this
        result's contour eigenvalue -- a cross-check on the eigensolver.

        Parameters
        ----------
        i : int, optional
            Mode index (default 0).

        Returns
        -------
        fns.perturbation.power.ModalEnergyBalance
        """
        from .power import modal_energy_balance as _meb

        return _meb(self, i)

    def plot_boundary_power(self, i=0, **kwargs):
        """Bar chart of each boundary's signed acoustic-power share for mode ``i``.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        return self.boundary_power(i).plot(**kwargs)
