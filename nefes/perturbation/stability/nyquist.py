"""Open-loop (Nyquist) stability of the perturbation network.

The companion to :func:`eigenmodes`.  Where the eigensolver hunts the complex
frequencies that make ``A(omega)`` singular by a *contour integral in the complex
plane*, this driver answers the same stability question -- **how many unstable
modes** -- from a sweep on the **real** frequency axis only.  That real-axis
restriction is what makes it the robust tool once a convected wave is retained (the
entropy ``h`` or a composition scalar, ``isentropic=False`` on a flowing network):

* A **measured / tabulated FTF** can be used directly.  The eigensolver searches
  *complex* frequencies, so it needs an analytically continuable flame model and
  rejects a real-grid table; the real-axis sweep evaluates the table where it is
  defined.  This is the standard industrial Nyquist test driven by a measured flame
  response.
* The count is a **robust integer**.  A flowing network with convected waves fills the
  spectrum with dense, near-marginal entropy/composition modes (period ``~ u/L``) that
  defeat the contour method's rank-revealing step and completeness certificate; the
  argument-principle winding on the real axis sidesteps them.
* No **overflow**.  The convected phase ``e^{-i omega tau_c}`` (``tau_c = L/u``) grows
  like ``e^{Im(omega) tau_c}`` and overflows float64 at the complex frequencies the
  contour samples once ``tau_c`` is long (eigenmodes.py ``_max_tau`` clamp); on the
  real axis ``|e^{-i omega tau}| = 1`` -- always bounded.

So the entropy / indirect-noise / equivalence-ratio instabilities are exactly what
Nyquist resolves cleanly.

The method is the classic feedback picture of a driven combustor.  Split the
operator into its **passive** part and the **dynamic source** ``S(omega)`` an
active element contributes (e.g. a flame's unsteady heat release, an injector's
fluctuating fuel feed):

    A(omega) = A_0(omega) + S(omega),    S(omega) = sum_k F_k(omega) a_k b_k^T,

where ``A_0`` is the network with the source switched off, ``F_k`` is term ``k``'s
transfer function, ``a_k`` its injection vector (the rows it feeds) and ``b_k`` its
sensing vector (the reference-edge fluctuation it reads).  ``S`` is **low rank**
(one rank-1 term per source term -- :mod:`nefes.perturbation.operator.stamps`).  The
matrix-determinant lemma then factors the stability determinant exactly,

    det A = det A_0 * det(I_r + M(omega)),    M(omega) = diag(F) B^T A_0^{-1} A,

so the modes are the passive resonances of ``A_0`` (stable, for a passive network)
plus the zeros of ``det(I_r + M)``.  The **return ratio** is ``L = -M`` (rank ``r``,
scalar for a single flame), and an instability is a solution of ``det(I - L) = 0``,
i.e. an eigenvalue of ``L`` reaching ``+1``.  Counting the encirclements of the
critical point by the real-frequency locus of ``L`` -- equivalently the winding of
``D = det(I_r + M) = det A / det A_0`` about the origin -- gives the number of
unstable modes (the argument principle), **with no complex-plane evaluation**.

Because the transfer functions are evaluated at *real* frequency, a **measured /
tabulated** FTF works here (unlike the eigensolver, which needs an analytically
continuable model) -- this is the standard industrial Nyquist test driven by a
measured flame response.

Assumption.  The count equals the number of unstable modes only if ``A_0`` itself
is stable (no unstable passive resonance, ``N_unstable(A_0) = 0``).  For the
passive, lossy terminations used here (a choked nozzle, a constant-mass-flow
outlet, the energy-neutral inherited reservoir -- see ``outlet_boundaries.ipynb``)
that holds; the winding then returns ``N_unstable(A) - N_unstable(A_0) =
N_unstable(A)``.  The sign/growth convention is :mod:`nefes.perturbation.eigenmodes`'
(time dependence ``e^{+i omega t}``, growth ``= -Im(omega)``, unstable modes in the
lower-half plane).

Beyond the count.  The primary output is the robust integer :attr:`NyquistResponse.n_unstable`.
The individual **unstable-mode frequencies and their growth rates** are also recoverable from
the same real-axis sweep, without any complex-plane evaluation: :meth:`NyquistResponse.crossings`
reports the onset (least-stable) frequencies where the locus skims the critical point, and
:meth:`NyquistResponse.mode_estimates` fits a rational (AAA) interpolant to ``D(omega)`` and
reads off its complex zeros -- one ``(frequency, growth rate)`` per mode -- so a measured /
tabulated FTF or a dense convected spectrum still yields the off-axis mode locations that the
contour eigensolver would otherwise provide.

See also
--------
eigenmodes : the contour eigensolver this driver complements (and whose sign convention it shares).
eigenvalue_trajectory : mode-tracking parameter sweep; the count-based analog here is
    :func:`nyquist_stability_map`.
contour.winding_count : the same argument principle, applied on a complex contour instead.
"""

import dataclasses
import warnings
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import scipy.sparse.linalg as spla

from ..operator.operator import build_acoustic_blocks, assemble_acoustic


class NyquistWarning(UserWarning):
    """Diagnostic from the Nyquist sweep (unclosed locus, coarse grid, odd winding, ...)."""


# Refine a frequency interval whose endpoints' return-ratio determinant rotate by more than this
# about the origin: the locus is under-resolved there and the winding count could miss an
# encirclement.  Below pi the count is unambiguous.
_MAX_PHASE_STEP = 0.4 * np.pi
# Cap on the total number of sweep frequencies the adaptive refinement may grow to.
_MAX_REFINE_POINTS = 4000
# Maximum bisection passes the refinement makes over the whole grid.
_REFINE_MAX_ITERS = 20
# Relative amount by which a factorization frequency is nudged off an exact passive resonance
# (where A_0 is singular) before it is factorized; on the real axis, so a real offset suffices.
_NUDGE_REL = 1e-8


@dataclass
class _SourceTermRank1:
    """One rank-1 piece ``F(omega) a b^T`` of the dynamic source ``S(omega)``."""

    a: np.ndarray  # injection vector (n,), nonzero on the residual rows the source feeds
    b: np.ndarray  # sensing vector (n,), nonzero on the reference-edge columns it reads
    transfer: object  # TransferFunction F(f), f in Hz
    label: str  # source element label (for reporting / per-term locus)


def _rank1_terms(blocks) -> List[_SourceTermRank1]:
    """Decompose the dynamic source ``S(omega)`` into rank-1 terms ``F_k a_k b_k^T``.

    Mirrors :func:`stamps.stamp_sources`: each :class:`~nefes.perturbation.operator.stamps.SourceStamp`
    contributes its ``factors`` on ``rows`` as the injection vector, and each of its
    :class:`~nefes.perturbation.operator.stamps.SourceTerm` contributes its ``coeff`` on ``cols`` as
    the sensing vector, with the term's transfer function.
    """
    n = int(blocks.n)
    names = tuple(getattr(blocks.prob, "node_names", ()) or ())
    terms = []
    for st in blocks.source_stamps:
        a = np.zeros(n, dtype=np.complex128)
        a[list(st.rows)] = st.factors
        label = names[st.node] if st.node < len(names) else f"node {st.node}"
        for t in st.terms:
            b = np.zeros(n, dtype=np.complex128)
            b[np.asarray(t.cols, dtype=np.intp)] = t.coeff
            terms.append(_SourceTermRank1(a=a.copy(), b=b, transfer=t.transfer, label=label))
    return terms


def _passive_blocks(blocks):
    """A copy of ``blocks`` with the dynamic source removed -- the operator ``A_0(omega)``.

    The mean flame (its steady jump in ``J_alg``) and the isentropic flame-edge skip are
    kept; only the active ``S(omega)`` feedback is dropped, so ``A - A_0 = S`` exactly.
    """
    return dataclasses.replace(blocks, source_stamps=[], _plans={})


def _factor_nudged(A_of, omega):
    """``splu`` of ``A_0(omega)``, nudging ``omega`` off an exact passive resonance."""
    last = None
    for k in range(5):
        zz = omega if k == 0 else omega + (_NUDGE_REL * (k + 1)) * (abs(omega) + 1.0)
        try:
            return spla.splu(A_of(zz).tocsc())
        except RuntimeError as exc:  # "Factor is exactly singular"
            last = exc
    raise last


def open_loop_response(
    prob,
    x_bar,
    freqs,
    *,
    isentropic=False,
    eps=None,
    eps_fb=1e-6,
    u_floor=1e-8,
    refine=True,
):
    """Open-loop return ratio ``L(omega)`` of the dynamic source(s) over a real-frequency sweep.

    Builds the passive operator ``A_0(omega)`` and, at each real frequency, forms the
    return-ratio matrix ``L(omega) = -diag(F) B^T A_0^{-1} A`` (rank ``r`` = number of
    source terms) and the stability determinant ``D(omega) = det(I_r - L) = det A / det
    A_0``.  These are the ingredients of the Nyquist stability test
    (:func:`nyquist_stability`).

    Parameters
    ----------
    prob : CompiledProblem
        Compiled network carrying at least one dynamic source (a flame FTF or a
        fluctuating injector); terminals carry their :class:`PerturbationBC`.
    x_bar : ndarray
        Converged mean-flow state, shape ``(n_solve, E)``.
    freqs : array_like
        Real frequencies (Hz).  Should span from ``~0`` (DC) to well above the highest
        acoustic mode so the locus closes near ``D = 1``; ``0`` is inserted if missing.
    isentropic : bool, optional
        Drop the convected entropy/composition waves (``rho' = p'/c^2`` on every edge,
        the flame edge excepted), leaving the two acoustic waves -- the pure-acoustic
        Nyquist test.  Default ``False`` keeps every supported wave (the entropy and
        composition paths that make this driver necessary).
    eps, eps_fb, u_floor : float, optional
        Operator-assembly regularizers forwarded to :func:`build_acoustic_blocks`.
    refine : bool, optional
        Adaptively insert frequencies where the determinant locus rotates too far between
        samples (default ``True``), so the winding count is unambiguous.

    Returns
    -------
    NyquistResponse
        The locus (``L``, ``D``) over the frequency grid, from which the unstable-mode
        count, crossing frequencies and stability margin are read.

    Raises
    ------
    ValueError
        If the network carries no dynamic source (use :func:`eigenmodes` for a
        purely passive / boundary-driven stability analysis).
    """
    freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
    if np.any(freqs < 0.0):
        raise ValueError("freqs must be non-negative (the negative axis is the conjugate mirror)")
    freqs = np.unique(freqs)
    if freqs[0] > 0.0:
        freqs = np.insert(freqs, 0, 0.0)  # the locus must reach DC to close across omega = 0

    blocks = build_acoustic_blocks(prob, x_bar, eps=eps, eps_fb=eps_fb, u_floor=u_floor, isentropic=isentropic)
    if not blocks.has_sources:
        raise ValueError(
            "Nyquist open-loop stability needs at least one dynamic source (a flame FTF or a "
            "fluctuating injector) to form a return ratio. For a passive/boundary-driven network "
            "use eigenmodes()."
        )
    terms = _rank1_terms(blocks)
    r = len(terms)
    A_mat = np.column_stack([t.a for t in terms])  # (n, r) injection
    B = np.column_stack([t.b for t in terms])  # (n, r) sensing
    passive = _passive_blocks(blocks)

    def A0_of(omega):
        return assemble_acoustic(omega, passive, with_boundaries=True)

    def evaluate(f):
        omega = 2.0 * np.pi * f
        lu = _factor_nudged(A0_of, omega)
        X = lu.solve(np.ascontiguousarray(A_mat))  # A_0^{-1} A, shape (n, r)
        Fvec = np.array([complex(np.asarray(t.transfer(f)).reshape(-1)[0]) for t in terms], dtype=np.complex128)
        M = Fvec[:, None] * (B.T @ X)  # diag(F) B^T A_0^{-1} A, (r, r)
        D = complex(np.linalg.det(np.eye(r) + M))
        return -M, D  # L = -M, D = det(I - L)

    fgrid = list(freqs)
    Ls, Ds = [], []
    for f in fgrid:
        L, D = evaluate(f)
        Ls.append(L)
        Ds.append(D)

    if refine:
        fgrid, Ls, Ds = _refine_locus(fgrid, Ls, Ds, evaluate)

    fgrid = np.asarray(fgrid, dtype=float)
    Larr = np.array(Ls, dtype=np.complex128)  # (n_freq, r, r)
    Darr = np.array(Ds, dtype=np.complex128)  # (n_freq,)
    L_out = Larr[:, 0, 0] if r == 1 else Larr

    return NyquistResponse(
        freqs=fgrid,
        L=L_out,
        D=Darr,
        rank=r,
        source_labels=tuple(t.label for t in terms),
        isentropic=bool(isentropic),
    )


def _refine_locus(fgrid, Ls, Ds, evaluate):
    """Bisect intervals whose ``D`` endpoints rotate too far about 0, until resolved."""
    fgrid = list(fgrid)
    Ls = list(Ls)
    Ds = list(Ds)
    for _ in range(_REFINE_MAX_ITERS):
        if len(fgrid) >= _MAX_REFINE_POINTS:
            break
        insert = []
        for i in range(len(fgrid) - 1):
            d0, d1 = Ds[i], Ds[i + 1]
            if d0 == 0.0 or d1 == 0.0:
                continue
            step = abs(np.angle(d1 / d0))
            if step > _MAX_PHASE_STEP:
                insert.append(i)
        if not insert:
            break
        for i in reversed(insert):
            fm = 0.5 * (fgrid[i] + fgrid[i + 1])
            Lm, Dm = evaluate(fm)
            fgrid.insert(i + 1, fm)
            Ls.insert(i + 1, Lm)
            Ds.insert(i + 1, Dm)
    return fgrid, Ls, Ds


def nyquist_stability(prob, x_bar, freqs, *, isentropic=False, **kwargs):
    """Number of unstable modes of the network from the real-frequency Nyquist sweep.

    Convenience wrapper over :func:`open_loop_response`: it computes the return ratio and
    returns the same :class:`NyquistResponse`, whose :attr:`~NyquistResponse.n_unstable`
    is the encirclement count.  See :func:`open_loop_response` for the parameters and the
    passive-``A_0`` assumption the count relies on.

    Returns
    -------
    NyquistResponse
        Carries :attr:`~NyquistResponse.n_unstable`, the crossing frequencies
        (:meth:`~NyquistResponse.crossings`) and the stability margin
        (:attr:`~NyquistResponse.margin`).
    """
    return open_loop_response(prob, x_bar, freqs, isentropic=isentropic, **kwargs)


@dataclass
class NyquistResponse:
    """Open-loop return ratio and stability verdict over a real-frequency sweep.

    The return ratio ``L(omega)`` and the stability determinant ``D(omega) = det(I - L) =
    det A / det A_0`` are tabulated on the (positive) frequency grid; the negative axis is
    the conjugate mirror ``D(-omega) = conj(D(omega))``.  An instability is an eigenvalue
    of ``L`` reaching the critical point ``+1`` (equivalently a zero of ``D``); the count
    is the winding of ``D`` about the origin over the closed real-axis contour.

    Attributes
    ----------
    freqs : ndarray
        Positive frequency grid (Hz), starting at ``0``, possibly adaptively refined.
    L : ndarray
        Return ratio: ``(n_freq,)`` complex for a single source term (rank 1), or
        ``(n_freq, r, r)`` the open-loop matrix for ``r`` terms.
    D : ndarray
        Stability determinant ``det(I_r - L) = det A / det A_0``, shape ``(n_freq,)``.
    rank : int
        Number of rank-1 source terms ``r``.
    source_labels : tuple
        Per-term source element labels.
    isentropic : bool
        Whether the entropy/composition waves were dropped (pure-acoustic test).
    """

    freqs: np.ndarray
    L: np.ndarray
    D: np.ndarray
    rank: int
    source_labels: tuple = ()
    isentropic: bool = False

    # ------------------------------------------------------------------ winding
    def _closed_locus(self):
        """Determinant ``D`` along the closed lower-half-plane contour (omega increasing).

        Uses ``D(-omega) = conj(D(omega))``: the real axis runs ``-omega_hi -> +omega_hi``
        as ``[conj(D)[::-1], D]``; the semicircle at infinity (where ``D -> 1``) closes it,
        represented by appending the single value ``1``.
        """
        D = self.D
        return np.concatenate([np.conj(D[::-1]), D, [1.0 + 0.0j]])

    @property
    def _winding(self) -> float:
        """Signed winding number of ``D`` about the origin over the closed contour."""
        loc = self._closed_locus()
        ang = np.angle(loc)
        incr = np.diff(np.append(ang, ang[0]))
        incr = (incr + np.pi) % (2.0 * np.pi) - np.pi
        return float(np.sum(incr)) / (2.0 * np.pi)

    @property
    def closed(self) -> bool:
        """Whether the band edge sits in a quiet region (``D(f_max) ~ 1``, ``|L| -> 0``).

        The count is taken with the band edge closed through the value at infinity
        (``D -> 1``); that is clean only when the top frequency lands between resonances
        where the return ratio is small.  A realistic flame (FTF gain roll-off) and a
        damped network reach ``|L| -> 0``, so the locus closes and the count converges; an
        idealized lossless duct with a constant-gain delay does not (infinitely many
        unstable modes), and the count is then only the tally up to ``f_max``.
        """
        return bool(abs(self.D[-1] - 1.0) < 0.2)

    @property
    def n_unstable(self) -> int:
        """Number of unstable modes with frequency in ``[0, max(freqs)]``.

        The winding of ``D`` over the (conjugate-symmetric) lower-half-plane contour counts
        every unstable zero -- each physical mode and its negative-frequency image -- so the
        positive-frequency count is half the winding.  It tallies the modes whose frequency
        lies in the swept band; with a damped / roll-off network the locus closes and this
        is the total unstable-mode count.  Assumes a passive ``A_0`` (no unstable passive
        resonance); see the module docstring.
        """
        if not self.closed:
            warnings.warn(
                f"Nyquist band edge is not quiet (|D(f_max) - 1| = {abs(self.D[-1] - 1.0):.2g}); the "
                "count is the tally of unstable modes up to f_max -- extend freqs past the last "
                "resonance (or to a trough) for the converged total.",
                NyquistWarning,
                stacklevel=2,
            )
        w = self._winding
        # the lower-half-plane contour traversed with omega increasing is negatively
        # oriented about the region (LHP on the right), so unstable zeros give a negative
        # winding; physical (positive-frequency) modes are half of |full count|.
        n2 = -w
        n = int(round(n2 / 2.0))
        if abs(n2 - 2.0 * n) > 0.25:
            warnings.warn(
                f"Nyquist winding {n2:.2f} is not an even integer; the grid may under-resolve a "
                "crossing (raise the frequency resolution) or A_0 may not be passive.",
                NyquistWarning,
                stacklevel=2,
            )
        return max(0, n)

    @property
    def stable(self) -> bool:
        """Whether the network is linearly stable (no unstable mode)."""
        return self.n_unstable == 0

    @property
    def margin(self) -> float:
        """Stability margin ``min_omega |D| = min |det(I - L)|`` (0 at a marginal mode).

        The closest the locus approaches the critical point; it shrinks to zero as a mode
        reaches onset, so it is a frequency-domain proxy for the gain/phase margin.
        """
        return float(np.min(np.abs(self.D)))

    def _quiet_count(self) -> int:
        """The unstable-mode count without raising the band-edge/parity warnings (for reprs)."""
        return max(0, int(round(-self._winding / 2.0)))

    def __repr__(self) -> str:
        """Stability verdict, swept band, source rank, and the margin to the critical point."""
        f = np.asarray(self.freqs, dtype=float)
        band = f"f in [0, {f.max():.1f}] Hz" if f.size else "empty"
        n = self._quiet_count()
        verdict = "STABLE" if n == 0 else f"UNSTABLE ({n} mode{'' if n == 1 else 's'})"
        caveat = "" if self.closed else "   [band edge not quiet: count is the tally up to f_max]"
        srcs = ", ".join(self.source_labels) if self.source_labels else f"{self.rank} source(s)"
        flavor = "isentropic (acoustic-only)" if self.isentropic else "entropy/composition included"
        try:
            n_passive = self.n_unstable_passive  # rational-fit check of the passive premise
        except Exception:
            n_passive = 0  # AAA unavailable; fall back to the encirclement reading silently
        passive_line = (
            ""
            if n_passive == 0
            else f"\n  [A_0 not passive: {n_passive} unstable passive resonance(s); absolute count = {n + n_passive}]"
        )
        return (
            f"NyquistResponse: {verdict}{caveat}\n"
            f"  {band}, rank {self.rank} [{srcs}], {flavor}\n"
            f"  stability margin min|D| = {self.margin:.3g}{passive_line}"
        )

    def _repr_html_(self) -> str:
        """Rich HTML stability card for notebooks: verdict, band, rank, and margin."""
        n = self._quiet_count()
        if n == 0:
            verdict = "<span style='color:#2a8a4a;font-weight:bold'>STABLE</span>"
        else:
            verdict = f"<span style='color:#c0392b;font-weight:bold'>UNSTABLE ({n} mode{'' if n == 1 else 's'})</span>"
        f = np.asarray(self.freqs, dtype=float)
        band = f"0 &ndash; {f.max():.1f} Hz" if f.size else "empty"
        srcs = ", ".join(self.source_labels) if self.source_labels else f"{self.rank} source(s)"
        flavor = "isentropic (acoustic-only)" if self.isentropic else "entropy/composition included"
        caveat = (
            ""
            if self.closed
            else "<div style='color:#c0392b;font-size:0.85em'>band edge not quiet: count is the tally up to f_max</div>"
        )
        td = "style='text-align:right;padding:2px 8px'"
        tdl = "style='text-align:left;padding:2px 8px'"
        rows = [
            ("verdict", verdict),
            ("swept band", band),
            ("source rank", f"{self.rank} [{srcs}]"),
            ("flavor", flavor),
            ("margin min|D|", f"{self.margin:.3g}"),
        ]
        body = "".join(f"<tr><td {tdl}>{k}</td><td {td}>{v}</td></tr>" for k, v in rows)
        table = "<table style='border-collapse:collapse;font-family:monospace;font-size:0.9em'>" + body + "</table>"
        return "<div style='font-family:sans-serif;margin-bottom:4px'><b>NyquistResponse</b></div>" + table + caveat

    def crossings(self, tol=0.25):
        """Real frequencies where the locus skims the critical point (``|D| < tol``).

        Here ``D(omega) = det(I - L) = det A / det A_0`` is the **stability determinant** (see
        the class docstring): its magnitude ``|D|`` measures how close the return-ratio locus
        passes to the critical point, and ``|D| -> 0`` marks a mode reaching the real axis.

        These are the **least-stable / onset** frequencies -- where a real-axis sample comes
        closest to the critical point.  A mode at onset (growth -> 0) sits on the real axis and
        pins ``|D| -> 0`` here, so its onset frequency is read off directly; a *strongly*
        unstable (or strongly damped) mode lies well off the real axis and need not produce a
        ``|D|`` dip, so these are not, in general, the unstable-mode frequencies (use
        :meth:`mode_estimates` or :func:`eigenmodes` for the off-axis mode locations).  The
        encirclement :attr:`n_unstable` is the robust quantity.

        Parameters
        ----------
        tol : float, optional
            ``|D|`` threshold for a near-crossing (default 0.25).

        Returns
        -------
        list of dict
            ``{"freq_hz", "abs_D", "L"}`` at each local minimum of ``|D|`` below ``tol``.

        See also
        --------
        mode_estimates : the off-axis complex mode frequencies (frequency and growth) from the sweep.
        n_unstable : the robust encirclement count these onset frequencies accompany.
        """
        absD = np.abs(self.D)
        out = []
        for i in range(len(absD)):
            lo = absD[i - 1] if i > 0 else np.inf
            hi = absD[i + 1] if i < len(absD) - 1 else np.inf
            if absD[i] < tol and absD[i] <= lo and absD[i] <= hi:
                Li = self.L[i] if self.L.ndim == 1 else self.L[i]
                out.append({"freq_hz": float(self.freqs[i]), "abs_D": float(absD[i]), "L": Li})
        return out

    # ----------------------------------------------- off-axis mode-frequency estimates
    def _aaa_fit(self):
        """Rational (AAA) interpolant of ``D(omega)`` over the real-frequency sweep.

        ``D`` is meromorphic in the frequency plane -- its **zeros** are the network modes and its
        **poles** are the passive operator ``A_0``'s resonances (``D = det A / det A_0``).  A
        barycentric rational fit of the real-axis samples continues analytically off the axis, so
        its zeros/poles recover both sets without any complex-plane evaluation (which the convected
        / tabulated regime forbids).  Cached on first use.
        """
        cached = getattr(self, "_aaa_cache", None)
        if cached is not None:
            return cached
        try:
            from scipy.interpolate import AAA
        except ImportError as exc:  # scipy < 1.15
            raise RuntimeError(
                "off-axis mode estimation needs scipy.interpolate.AAA (scipy >= 1.15); "
                "upgrade scipy, or use the encirclement count n_unstable / eigenmodes()"
            ) from exc
        f = np.asarray(self.freqs, dtype=float)
        fit = AAA(f.astype(np.complex128), np.asarray(self.D, dtype=np.complex128))
        try:
            fit.clean_up()  # drop spurious Froissart doublets (near-cancelling pole/zero pairs)
        except Exception:  # pragma: no cover - clean_up is best-effort
            pass
        object.__setattr__(self, "_aaa_cache", fit)
        return fit

    def _aaa_modes(self, points, *, margin_hz, max_growth):
        """Map AAA roots/poles (``points`` in Hz) to in-band mode dicts (freq, growth, unstable)."""
        f = np.asarray(self.freqs, dtype=float)
        fmax = float(f.max()) if f.size else 0.0
        out = []
        for z in points:
            z = complex(z)
            if not (-margin_hz <= z.real <= fmax + margin_hz):
                continue
            growth = -2.0 * np.pi * z.imag  # e^{+i omega t}: growth = -Im(omega) = -2 pi Im(f)
            if max_growth is not None and abs(growth) > max_growth:
                continue
            out.append({"freq_hz": float(z.real), "growth_rate": float(growth), "unstable": bool(growth > 0.0)})
        return sorted(out, key=lambda d: d["freq_hz"])

    def mode_estimates(self, *, margin_hz=None, max_growth=None, unstable_only=False):
        """Estimate the network's **off-axis** complex mode frequencies from the real-axis sweep.

        :meth:`crossings` reports where ``|D|`` dips -- the *onset* (least-stable) frequencies,
        which coincide with a mode only at marginal stability; a strongly growing or damped mode
        sits off the real axis and need not dip ``|D|`` at all.  This instead fits a rational
        (AAA) interpolant to ``D(omega)`` along the real axis and reads off its complex **zeros**
        -- the actual modes, each with its frequency *and growth rate* -- analytically continued
        from the sweep with no complex-plane evaluation.  It recovers the same modes
        :func:`nefes.perturbation.eigenmodes` would, but works where the eigensolver cannot run (a
        measured / tabulated FTF, a dense convected entropy/composition spectrum).

        It is an estimate from a finite, band-limited fit: trust modes well inside the swept band,
        and refine the grid or widen ``freqs`` if an estimate sits at the edge.

        Parameters
        ----------
        margin_hz : float, optional
            Keep zeros whose frequency lies within this margin of the swept band (default: 5% of
            the band).
        max_growth : float, optional
            Drop estimates with ``|growth_rate|`` above this (1/s) as spurious far-off-axis fit
            artifacts (default: ``2*pi * 0.15 * f_max``).
        unstable_only : bool, optional
            Return only the growing modes (``growth_rate > 0``).

        Returns
        -------
        list of dict
            ``{"freq_hz", "growth_rate", "unstable"}`` per mode, growth in 1/s (``> 0`` unstable,
            the ``e^{+i omega t}`` convention), sorted by frequency.
        """
        margin_hz, max_growth = self._estimate_bounds(margin_hz, max_growth)
        modes = self._aaa_modes(self._aaa_fit().roots(), margin_hz=margin_hz, max_growth=max_growth)
        return [m for m in modes if m["unstable"]] if unstable_only else modes

    def passive_resonances(self, *, margin_hz=None, max_growth=None):
        """Resonances of the **passive** operator ``A_0``, recovered as the poles of the ``D`` fit.

        Since ``D = det A / det A_0``, the poles of the AAA fit (:meth:`mode_estimates` uses its
        zeros) are exactly ``A_0``'s modes.  The Nyquist count is ``N_unstable(A) - N_unstable(A_0)``
        and equals the absolute unstable-mode number only when ``A_0`` is itself stable; this lets
        that premise be **checked** -- any passive resonance with ``growth_rate > 0`` is an unstable
        passive mode the encirclement count would otherwise hide.

        Returns
        -------
        list of dict
            ``{"freq_hz", "growth_rate", "unstable"}`` per estimated passive resonance.
        """
        margin_hz, max_growth = self._estimate_bounds(margin_hz, max_growth)
        return self._aaa_modes(self._aaa_fit().poles(), margin_hz=margin_hz, max_growth=max_growth)

    def _estimate_bounds(self, margin_hz, max_growth):
        fmax = float(np.asarray(self.freqs, dtype=float).max()) if np.size(self.freqs) else 0.0
        if margin_hz is None:
            margin_hz = 0.05 * fmax
        if max_growth is None:
            max_growth = 2.0 * np.pi * 0.15 * fmax
        return margin_hz, max_growth

    @property
    def n_unstable_passive(self) -> int:
        """Estimated number of **unstable passive resonances** ``N_unstable(A_0)`` in the band.

        Zero for the passive terminations this driver is built for; a positive value warns that the
        encirclement reading :attr:`n_unstable` is no longer the absolute count (add this).  An
        estimate from the rational fit -- see :meth:`passive_resonances`.
        """
        return sum(1 for r in self.passive_resonances() if r["unstable"])

    @property
    def n_unstable_absolute(self) -> int:
        """Absolute unstable-mode count ``N_unstable(A) = (encirclement count) + N_unstable(A_0)``."""
        return self.n_unstable + self.n_unstable_passive

    @property
    def passive_assumption_ok(self) -> bool:
        """Whether ``A_0`` looks stable (``N_unstable(A_0) = 0``), so :attr:`n_unstable` is absolute."""
        return self.n_unstable_passive == 0

    def summary(self):
        """One-line-per-field dict: unstable count, stability, margin, crossings."""
        return {
            "n_unstable": self.n_unstable,
            "stable": self.stable,
            "margin": self.margin,
            "rank": self.rank,
            "isentropic": self.isentropic,
            "crossings": self.crossings(),
        }

    # ------------------------------------------------------------------- plots
    def plot(self, **layout):
        """Nyquist diagram: the return-ratio locus with the critical point ``+1``.

        For a single source term the scalar ``L(omega)`` locus (and its conjugate mirror)
        is drawn; encircling ``+1`` signals instability.  For several terms the
        determinant locus ``D(omega)`` about the origin is drawn instead (the generalized
        Nyquist criterion), with the critical point at ``0``.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        import plotly.graph_objects as go

        if self.rank == 1:
            z = self.L
            crit, crit_name, title = 1.0 + 0.0j, "+1 (critical)", "Nyquist locus of the return ratio L(omega)"
        else:
            z = self.D
            crit, crit_name, title = 0.0 + 0.0j, "0 (critical)", "Nyquist locus of det(I - L)(omega)"
        fig = go.Figure()
        fig.add_scatter(
            x=z.real, y=z.imag, mode="lines", name="omega > 0", hovertext=[f"{f:.1f} Hz" for f in self.freqs]
        )
        fig.add_scatter(x=z.real, y=(-z.imag), mode="lines", line=dict(dash="dot"), name="omega < 0 (mirror)")
        fig.add_scatter(
            x=[crit.real],
            y=[crit.imag],
            mode="markers",
            marker=dict(size=11, symbol="x", color="#d62728"),
            name=crit_name,
        )
        verdict = "UNSTABLE" if not self.stable else "stable"
        fig.update_layout(
            title=f"{title} -- {self.n_unstable} unstable mode(s), {verdict}",
            xaxis_title="Re",
            yaxis_title="Im",
            **layout,
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1.0)
        return fig

    def plot_margin(self, **layout):
        """Plot the distance-to-instability ``|D(f)| = |det(I - L)|`` versus frequency.

        Dips toward zero mark the (near-)unstable mode frequencies.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_scatter(x=self.freqs, y=np.abs(self.D), mode="lines", name="|det(I - L)|")
        for c in self.crossings():
            fig.add_scatter(
                x=[c["freq_hz"]],
                y=[c["abs_D"]],
                mode="markers",
                marker=dict(size=9, color="#d62728"),
                name=f"{c['freq_hz']:.1f} Hz",
            )
        fig.update_layout(
            title="Stability margin |det(I - L)| over frequency",
            xaxis_title="frequency [Hz]",
            yaxis_title="|det(I - L)|",
            **layout,
        )
        return fig


# ===================================================================================================
# Parameter sweep: the Nyquist stability map (the entropy-regime analog of eigenvalue_trajectory)
# ===================================================================================================
#
# The contour eigensolver tracks where the *modes* sit, so a parameter sweep of it
# (:func:`nefes.perturbation.eigenvalue_trajectory`) draws each eigenvalue's path in the complex
# plane.  That breaks down once a convected entropy / composition wave is retained or the flame
# response is a measured table -- the very regime this Nyquist driver was built for.  There the
# right "trajectory" is not a mode path but the **stability verdict over the sweep**: how the
# unstable-mode count, the margin to the critical point, and the onset (crossing) frequencies move
# as one setup parameter is varied.  That is a bifurcation / stability-boundary diagram, and it is
# obtainable wherever the real-frequency winding is -- including the entropy regime.


@dataclass
class NyquistStabilityMap:
    """Nyquist stability verdict of a network over a one-parameter sweep.

    The robust real-frequency analog of :class:`nefes.perturbation.TrajectoryResult`: rather than
    tracking eigenvalue *positions* (which needs an analytic flame model and an off-axis-stable
    operator), it records, at each parameter value, the **integer unstable-mode count** (the
    encirclement number), the **stability margin** ``min|D|``, and the **onset frequencies** where
    the locus skims the critical point.  Stepping the count locates a stability boundary; the
    margin collapsing to zero pins where (and the crossing pins at what frequency) a mode crosses
    it.  Works in the entropy / composition / tabulated-FTF regime where the eigenvalue trajectory
    cannot be drawn.

    Attributes
    ----------
    params : ndarray
        Swept parameter values, in march order, shape ``(k,)``.
    param_name : str
        Label for the swept parameter (for reprs / plots).
    n_unstable : ndarray of int
        Unstable-mode count in ``[0, max(freqs)]`` at each parameter value, shape ``(k,)``.
    margin : ndarray
        Stability margin ``min|D| = min|det(I - L)|`` at each value (0 at marginal), shape ``(k,)``.
    closed : ndarray of bool
        Whether the locus closed (band edge quiet, ``|D(f_max)| ~ 1``) at each value; where False
        the count is a tally up to ``f_max`` rather than the converged total.
    crossings : list of list of dict
        Per parameter value, the near-critical crossings ``{"freq_hz", "abs_D"}`` -- the onset
        frequencies (see :meth:`NyquistResponse.crossings`).
    responses : list of NyquistResponse or None
        The full per-step responses when ``store_responses=True`` (for the locus at any step),
        else ``None``.
    isentropic : bool
        Whether the convected entropy/composition waves were dropped (pure-acoustic test).
    """

    params: np.ndarray
    param_name: str
    n_unstable: np.ndarray
    margin: np.ndarray
    closed: np.ndarray
    crossings: list
    responses: Optional[list] = None
    isentropic: bool = False

    @property
    def onsets(self) -> List[tuple]:
        """Bracketed parameter intervals where the unstable-mode count changes.

        Each entry ``(p_lo, p_hi, delta)`` says the count stepped by ``delta`` (signed) between
        consecutive samples ``p_lo`` and ``p_hi`` -- a stability boundary was crossed in that
        interval.  Refine the sweep there (and read :attr:`crossings`) for the onset frequency.
        """
        out = []
        n = self.n_unstable
        for k in range(1, len(n)):
            if int(n[k]) != int(n[k - 1]):
                out.append((float(self.params[k - 1]), float(self.params[k]), int(n[k]) - int(n[k - 1])))
        return out

    @property
    def all_closed(self) -> bool:
        """Whether every swept point's locus closed (so every count is a converged total)."""
        return bool(np.all(self.closed))

    def __repr__(self) -> str:
        p = np.asarray(self.params, dtype=float)
        span = "empty" if p.size == 0 else f"{self.param_name} {p[0]:.4g} -> {p[-1]:.4g} ({p.size} steps)"
        nmin, nmax = int(np.min(self.n_unstable)), int(np.max(self.n_unstable))
        lines = [
            f"NyquistStabilityMap: {span}",
            f"  unstable modes: {nmin}..{nmax}; {len(self.onsets)} bifurcation(s)",
            f"  margin min|D|: {float(np.min(self.margin)):.3g} .. {float(np.max(self.margin)):.3g}",
        ]
        for lo, hi, d in self.onsets:
            lines.append(f"  bifurcation: n {d:+d} between {self.param_name} {lo:.4g} and {hi:.4g}")
        if not self.all_closed:
            lines.append("  [some sweeps' band edge not quiet: those counts are tallies up to f_max]")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        p = np.asarray(self.params, dtype=float)
        span = "empty" if p.size == 0 else f"{self.param_name}: {p[0]:.4g} &rarr; {p[-1]:.4g} ({p.size} steps)"
        nmin, nmax = int(np.min(self.n_unstable)), int(np.max(self.n_unstable))
        head = (
            f"<b>NyquistStabilityMap</b> &mdash; unstable modes {nmin}&ndash;{nmax}, "
            f"{len(self.onsets)} bifurcation(s)<br><span style='color:#52606d'>{span}</span>"
        )
        td = "style='text-align:right;padding:2px 8px'"
        tdl = "style='text-align:left;padding:2px 8px'"
        rows = [f"<tr><th {tdl}>bifurcation</th><th {tdl}>between</th></tr>"]
        for lo, hi, d in self.onsets:
            rows.append(f"<tr><td {td}>n {d:+d}</td><td {tdl}>{self.param_name} {lo:.4g} &rarr; {hi:.4g}</td></tr>")
        if len(rows) == 1:
            rows.append(f"<tr><td {tdl} colspan='2'>no count change over the sweep</td></tr>")
        table = "<table style='border-collapse:collapse;font-size:0.9em'>" + "".join(rows) + "</table>"
        caveat = (
            ""
            if self.all_closed
            else (
                "<div style='color:#c0392b;font-size:0.85em'>"
                "some band edges not quiet: those counts are tallies up to f_max</div>"
            )
        )
        return head + "<br>" + table + caveat

    def plot(self, *, title=None, **layout):
        """Stability map: unstable-mode count, margin, and onset frequencies versus the parameter.

        Three stacked panels sharing the parameter axis: the integer unstable-mode count (a step
        plot -- each step is a stability boundary), the margin ``min|D|`` (its dips mark the
        boundaries), and the near-critical crossing frequencies (the onset frequencies, where a
        mode reaches the real axis).

        Returns
        -------
        plotly.graph_objects.Figure
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        from ...plotting.theme import COLORWAY, NEFES_TEMPLATE_NAME

        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=("unstable modes", "stability margin  min|D|", "onset (crossing) frequencies [Hz]"),
        )
        fig.add_trace(
            go.Scatter(
                x=self.params,
                y=self.n_unstable,
                mode="lines+markers",
                line=dict(color=COLORWAY[0], width=2, shape="hv"),
                marker=dict(size=5, color=COLORWAY[0]),
                name="n_unstable",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=self.params,
                y=self.margin,
                mode="lines+markers",
                line=dict(color=COLORWAY[1], width=2),
                marker=dict(size=4, color=COLORWAY[1]),
                name="min|D|",
            ),
            row=2,
            col=1,
        )
        cx, cy, cs = [], [], []
        for p, cl in zip(self.params, self.crossings):
            for c in cl:
                cx.append(p)
                cy.append(c["freq_hz"])
                cs.append(c["abs_D"])
        if cx:
            fig.add_trace(
                go.Scatter(
                    x=cx,
                    y=cy,
                    mode="markers",
                    marker=dict(size=8, color=COLORWAY[3], symbol="x"),
                    name="onset crossing",
                    customdata=cs,
                    hovertemplate=(
                        f"{self.param_name} = %{{x:.4g}}<br>f = %{{y:.4g}} Hz"
                        "<br>|D| = %{customdata:.3g}<extra></extra>"
                    ),
                ),
                row=3,
                col=1,
            )
        for lo, hi, _ in self.onsets:
            fig.add_vline(x=0.5 * (lo + hi), line_dash="dot", line_color="#9aa5b1", line_width=1.2)
        fig.update_yaxes(title_text="count", row=1, col=1)
        fig.update_yaxes(title_text="min|D|", row=2, col=1)
        fig.update_yaxes(title_text="f [Hz]", row=3, col=1)
        fig.update_xaxes(title_text=self.param_name, row=3, col=1)
        fig.update_layout(template=NEFES_TEMPLATE_NAME, title=title or f"Nyquist stability map vs {self.param_name}")
        fig.update_layout(**layout)
        return fig


def nyquist_stability_map(
    build,
    params,
    freqs,
    *,
    isentropic=False,
    eps=None,
    eps_fb=1e-6,
    u_floor=1e-8,
    refine=True,
    warm_start=True,
    crossing_tol=0.25,
    param_name="parameter",
    store_responses=False,
):
    """Sweep one setup parameter and track the Nyquist stability verdict at each value.

    The real-frequency, count-based companion to :func:`nefes.perturbation.eigenvalue_trajectory`.
    For each parameter value it solves the mean flow (warm-started from the previous step), runs
    the Nyquist open-loop test (:func:`nyquist_stability`), and records the unstable-mode count,
    the margin ``min|D|`` and the onset (crossing) frequencies.  The result is a bifurcation /
    stability-boundary diagram that is valid in the entropy / composition / tabulated-FTF regime
    where the eigenvalue trajectory cannot be drawn (and is a robust cross-check where it can).

    Parameters
    ----------
    build : callable
        ``build(p)`` returning the network at parameter value ``p`` -- an **unsolved**
        :class:`nefes.shell.Network` (solved here, warm-started when ``warm_start``) or an
        already-solved solution (exposing ``.problem``, ``.x``, ``.converged``).  The network must
        carry at least one dynamic source (a flame FTF / fluctuating injector) at every value.
    params : array_like
        Parameter values to sweep, in march order (e.g. ``np.linspace(1.0, 0.0, 41)``).
    freqs : array_like
        Real frequencies (Hz) for the open-loop sweep at each value; span ``~0`` to past the
        highest acoustic mode so the locus closes (see :func:`open_loop_response`).
    isentropic : bool, optional
        Drop the convected entropy/composition waves (pure-acoustic test), default False.
    eps, eps_fb, u_floor : float, optional
        Operator-assembly regularizers forwarded to :func:`open_loop_response`.
    refine : bool, optional
        Adaptively refine each frequency locus so the winding is unambiguous (default True).
    warm_start : bool, optional
        Seed each mean-flow solve from the previous step's converged state (default True).
    crossing_tol : float, optional
        ``|D|`` threshold for recording an onset crossing (default 0.25).
    param_name : str, optional
        Label for the swept parameter, used in reprs and plots.
    store_responses : bool, optional
        Keep the full :class:`NyquistResponse` at every step (default False) for plotting an
        individual locus; set True only if you need them (memory).

    Returns
    -------
    NyquistStabilityMap
        The count / margin / onset-frequency map over the sweep.

    Raises
    ------
    ValueError
        If fewer than two parameter values are given, or a built network carries no dynamic
        source (use :func:`nefes.perturbation.eigenvalue_trajectory` for a passive sweep).
    """
    from .trajectory import _solved_state  # lazy import: shared warm-start solve, avoids a cycle

    params = np.asarray(params, dtype=float)
    if params.size < 2:
        raise ValueError("provide at least two parameter values to trace a stability map")

    n_unstable = np.empty(params.size, dtype=int)
    margin = np.empty(params.size, dtype=float)
    closed = np.empty(params.size, dtype=bool)
    crossings: list = []
    responses: Optional[list] = [] if store_responses else None
    x_warm = None
    for k, p in enumerate(params):
        prob, x = _solved_state(build(float(p)), x_warm, warm_start)
        x_warm = x
        resp = open_loop_response(
            prob, x, freqs, isentropic=isentropic, eps=eps, eps_fb=eps_fb, u_floor=u_floor, refine=refine
        )
        n_unstable[k] = resp._quiet_count()
        margin[k] = resp.margin
        closed[k] = resp.closed
        crossings.append([{"freq_hz": c["freq_hz"], "abs_D": c["abs_D"]} for c in resp.crossings(crossing_tol)])
        if store_responses:
            responses.append(resp)
    return NyquistStabilityMap(
        params=params,
        param_name=param_name,
        n_unstable=n_unstable,
        margin=margin,
        closed=closed,
        crossings=crossings,
        responses=responses,
        isentropic=bool(isentropic),
    )
