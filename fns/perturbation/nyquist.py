"""Open-loop (Nyquist) stability of the perturbation network (theory.md s12.7).

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
active element contributes (a flame's unsteady heat release, an injector's
fluctuating fuel feed):

    A(omega) = A_0(omega) + S(omega),    S(omega) = sum_k F_k(omega) a_k b_k^T,

where ``A_0`` is the network with the source switched off, ``F_k`` is term ``k``'s
transfer function, ``a_k`` its injection vector (the rows it feeds) and ``b_k`` its
sensing vector (the reference-edge fluctuation it reads).  ``S`` is **low rank**
(one rank-1 term per source term -- :mod:`fns.perturbation.stamps`).  The
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
N_unstable(A)``.  The sign/growth convention is :mod:`fns.perturbation.eigenmodes`'
(time dependence ``e^{+i omega t}``, growth ``= -Im(omega)``, unstable modes in the
lower-half plane).
"""

import dataclasses
import warnings
from dataclasses import dataclass
from typing import List

import numpy as np
import scipy.sparse.linalg as spla

from .operator import build_acoustic_blocks, assemble_acoustic


class NyquistWarning(UserWarning):
    """Diagnostic from the Nyquist sweep (unclosed locus, coarse grid, odd winding, ...)."""


# Refine a frequency interval whose endpoints' return-ratio determinant rotate by
# more than this about the origin: the locus is under-resolved there and the winding
# count could miss an encirclement.  Below pi the count is unambiguous.
_MAX_PHASE_STEP = 0.4 * np.pi
_MAX_REFINE_POINTS = 4000


@dataclass
class _SourceTermRank1:
    """One rank-1 piece ``F(omega) a b^T`` of the dynamic source ``S(omega)``."""

    a: np.ndarray  # injection vector (n,), nonzero on the residual rows the source feeds
    b: np.ndarray  # sensing vector (n,), nonzero on the reference-edge columns it reads
    transfer: object  # TransferFunction F(f), f in Hz
    label: str  # source element label (for reporting / per-term locus)


def _rank1_terms(blocks) -> List[_SourceTermRank1]:
    """Decompose the dynamic source ``S(omega)`` into rank-1 terms ``F_k a_k b_k^T``.

    Mirrors :func:`stamps.stamp_sources`: each :class:`~fns.perturbation.stamps.SourceStamp`
    contributes its ``factors`` on ``rows`` as the injection vector, and each of its
    :class:`~fns.perturbation.stamps.SourceTerm` contributes its ``coeff`` on ``cols`` as
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
        zz = omega if k == 0 else omega + (1e-8 * (k + 1)) * (abs(omega) + 1.0)
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
    for _ in range(20):
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

    def crossings(self, tol=0.25):
        """Real frequencies where the locus skims the critical point (``|D| < tol``).

        These are the **least-stable / onset** frequencies -- where a real-axis sample comes
        closest to the critical point.  A mode at onset (growth -> 0) sits on the real axis and
        pins ``|D| -> 0`` here, so its onset frequency is read off directly; a *strongly*
        unstable (or strongly damped) mode lies well off the real axis and need not produce a
        ``|D|`` dip, so these are not, in general, the unstable-mode frequencies (use
        :func:`eigenmodes` for the off-axis mode locations).  The encirclement
        :attr:`n_unstable` is the robust quantity.

        Parameters
        ----------
        tol : float, optional
            ``|D|`` threshold for a near-crossing (default 0.25).

        Returns
        -------
        list of dict
            ``{"freq_hz", "abs_D", "L"}`` at each local minimum of ``|D|`` below ``tol``.
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
