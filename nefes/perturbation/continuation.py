"""Analytic continuation of tabulated frequency-response data.

A measured or otherwise tabulated frequency response -- a flame transfer function
``F(f)``, a velocity-modulated fuel-injection response, a boundary reflection
coefficient ``R(f)`` -- lives only on a **real** frequency grid.  The forced response
and the real-axis Nyquist driver consume it directly (:func:`open_loop_response`).  The
**stability eigenproblem**, however, searches the *complex* frequency plane (it looks
for ``det A(omega) = 0`` off the real axis), so it needs the response as a function that
can be evaluated at a complex frequency -- and a grid interpolant (linear, spline,
magnitude/phase) is not analytic, so it cannot.

This module closes that gap.  :class:`RationalFit` fits the tabulated samples with a
**barycentric rational function** (the AAA algorithm, Nakatsukasa--Sete--Trefethen 2018,
via :class:`scipy.interpolate.AAA`).  A rational function is analytic everywhere except
at its poles, so it continues off the real axis for free: the same object that
reproduces the data on the grid is also a legal ``S(omega)`` / ``R(omega)`` for the
contour eigensolver.  Because it is a :class:`~nefes.elements.dynamic_source.TransferFunction`
with :attr:`analytic` ``True``, it drops straight into
:func:`~nefes.elements.dynamic_source.heat_release_response` /
:func:`~nefes.elements.dynamic_source.mass_flow_response`, and -- being a plain
``freq -> complex`` callable -- into :meth:`~nefes.perturbation.operator.boundary_bc.PerturbationBC.reflection`.

Robustness
----------
* The fit is built in a **normalized** frequency ``s = f / f_scale`` for conditioning, so
  kilohertz grids fit as cleanly as unit ones.
* ``clean_up`` (on by default) removes the spurious near-cancelling pole/zero pairs
  ("Froissart doublets") AAA can introduce on noisy data; ``rtol`` stops the fit once the
  residual reaches a tolerance, so it does not chase measurement noise to machine
  precision (which would seed poles on the real axis).
* For a **delay-dominated** response (every flame FTF carries a transport lag), the pure
  delay ``e^{-i 2 pi f tau}`` is entire but not rational, so AAA spends poles approximating
  it.  Passing ``delay="auto"`` (or an explicit ``tau``) factors the delay out before the
  fit and re-applies it analytically on evaluation: the rational part is then slowly
  varying and fits with a handful of poles, all comfortably off the real axis.
* :meth:`RationalFit.poles_in_region` reports any pole that lands inside a stability search
  window so the fit's validity there can be checked before trusting an eigenvalue.

Frequency convention
--------------------
Everything is a function of **frequency in Hz** (project convention).  A stability search
window is given as ``(freq_band, growth_band)`` with ``growth = -Im(omega)`` (positive =
unstable); in the complex-frequency plane that maps to ``Im(f) = -growth / (2 pi)``.
"""

from __future__ import annotations

import warnings

import numpy as np

from ..elements.dynamic_source import TransferFunction


def _estimate_delay(freqs, values):
    """Estimate a pure time lag ``tau`` [s] from the slope of the unwrapped phase.

    Under the ``e^{+i omega t}`` convention a causal delay contributes the phase factor
    ``e^{-i 2 pi f tau}``, i.e. a slope ``-2 pi tau`` of the phase against angular
    frequency.  A least-squares slope of the unwrapped phase recovers it; the result is
    clamped to ``>= 0`` (a negative estimate means there is no causal bulk delay to peel
    off, so none is removed).
    """
    f = np.asarray(freqs, dtype=float)
    v = np.asarray(values, dtype=np.complex128)
    if f.size < 2:
        return 0.0
    phase = np.unwrap(np.angle(v))
    slope = np.polyfit(2.0 * np.pi * f, phase, 1)[0]
    return float(max(0.0, -slope))


class RationalFit(TransferFunction):
    """Analytically-continuable barycentric-rational fit of tabulated ``F(f)`` data.

    Fits the complex samples ``values`` at frequencies ``freqs`` [Hz] with the AAA
    algorithm and exposes the result as a :class:`~nefes.elements.dynamic_source.TransferFunction`
    that can be evaluated at a **complex** frequency -- so, unlike
    :class:`~nefes.elements.dynamic_source.Tabulated`, it is usable in the stability
    eigenproblem.  It is also a bare ``freq -> complex`` callable, so it serves equally as
    a boundary reflection coefficient
    (:meth:`~nefes.perturbation.operator.boundary_bc.PerturbationBC.reflection`).

    Parameters
    ----------
    freqs : array_like
        Tabulated frequencies [Hz] (distinct; need not be sorted).
    values : array_like
        Complex response samples at ``freqs``.
    rtol : float, optional
        Relative tolerance at which AAA stops adding support points.  ``None`` (default)
        uses the SciPy default (~machine precision); set it near the data's noise floor
        for measured input to avoid over-fitting.
    max_terms : int, optional
        Maximum number of barycentric support points (default 100).
    clean_up : bool, optional
        Remove spurious Froissart-doublet poles after the fit (default True).
    clean_up_tol : float, optional
        Residue tolerance for the clean-up pass (default ``1e-13``).
    delay : {None, "auto"} or float, optional
        Pure time lag ``tau`` [s] to factor out before fitting and re-apply analytically
        on evaluation.  ``None`` (default) fits the raw data; ``"auto"`` estimates the lag
        from the phase slope; a float pins it.  Recommended for flame FTFs.
    freq_scale : float, optional
        Normalization frequency for conditioning (default: the largest ``|freqs|``, or 1).

    Attributes
    ----------
    freqs, values : ndarray
        The original tabulated data (kept for diagnostics and plotting).
    delay : float
        The pure lag [s] factored out (0 if none).
    analytic : bool
        Always ``True`` -- the fit continues to complex frequency.
    """

    analytic = True

    def __init__(
        self,
        freqs,
        values,
        *,
        rtol=None,
        max_terms=100,
        clean_up=True,
        clean_up_tol=1e-13,
        delay=None,
        freq_scale=None,
    ):
        from scipy.interpolate import AAA

        f = np.asarray(freqs, dtype=float).ravel()
        v = np.asarray(values, dtype=np.complex128).ravel()
        if f.shape != v.shape:
            raise ValueError("freqs and values must be 1-D arrays of equal length")
        if f.size < 2:
            raise ValueError("need at least two samples to fit a rational continuation")
        if not (np.all(np.isfinite(f)) and np.all(np.isfinite(v))):
            raise ValueError("freqs and values must be finite")
        if np.unique(f).size != f.size:
            raise ValueError("freqs must be distinct")

        if delay is None:
            tau = 0.0
        elif isinstance(delay, str):
            if delay != "auto":
                raise ValueError(f"delay must be None, 'auto', or a float; got {delay!r}")
            tau = _estimate_delay(f, v)
        else:
            tau = float(delay)

        fscale = float(freq_scale) if freq_scale is not None else max(float(np.max(np.abs(f))), 1.0)
        if fscale <= 0.0:
            raise ValueError("freq_scale must be positive")

        # peel off the pure delay (entire, so it carries no poles); fit the smooth remainder
        resid = v * np.exp(2j * np.pi * f * tau)
        self._aaa = AAA(f / fscale, resid, rtol=rtol, max_terms=max_terms, clean_up=clean_up, clean_up_tol=clean_up_tol)

        self.freqs = f
        self.values = v
        self.delay = tau
        self.max_delay = abs(tau)
        self._fscale = fscale

    # -- evaluation ---------------------------------------------------------

    def __call__(self, f):
        fa = np.asarray(f, dtype=np.complex128)
        z = (fa / self._fscale).ravel()
        r = np.asarray(self._aaa(z), dtype=np.complex128).ravel()
        if self.delay:
            r = r * np.exp(-2j * np.pi * fa.ravel() * self.delay)
        return r.reshape(fa.shape)

    # -- diagnostics --------------------------------------------------------

    @property
    def n_terms(self) -> int:
        """Number of barycentric support points (the fit's degree of freedom count)."""
        return int(np.asarray(self._aaa.support_points).size)

    @property
    def support_points(self) -> np.ndarray:
        """Support (interpolation) frequencies the fit selected [Hz]."""
        return np.asarray(self._aaa.support_points) * self._fscale

    @property
    def poles(self) -> np.ndarray:
        """Poles of the rational part in the complex-frequency plane [Hz].

        The factored-out pure delay is entire and contributes none, so these are all the
        poles of the continuation.
        """
        return np.asarray(self._aaa.poles()) * self._fscale

    @property
    def zeros(self) -> np.ndarray:
        """Zeros (roots) of the rational part in the complex-frequency plane [Hz]."""
        return np.asarray(self._aaa.roots()) * self._fscale

    @property
    def residues(self) -> np.ndarray:
        """Residues of the rational part at its :attr:`poles` (delay-free part)."""
        return np.asarray(self._aaa.residues()) * self._fscale

    def errors(self):
        """AAA's per-iteration max fit error history (decreasing as support points are added)."""
        return np.asarray(self._aaa.errors)

    def max_error(self) -> float:
        """Maximum absolute fit error on the original tabulated grid, ``max|fit - data|``."""
        return float(np.max(np.abs(self(self.freqs) - self.values)))

    def rms_error(self) -> float:
        """Root-mean-square absolute fit error on the original tabulated grid."""
        d = self(self.freqs) - self.values
        return float(np.sqrt(np.mean(np.abs(d) ** 2)))

    def poles_in_region(self, freq_band, growth_band=None) -> np.ndarray:
        """Poles falling inside a stability search window (so its validity can be checked).

        A pole inside the contour the eigensolver sweeps makes the continuation unreliable
        there (the fit is trustworthy only in the strip around the real axis spanned by the
        data).  This returns the poles whose ``(freq, growth)`` lands in the window, where
        ``freq = Re(pole)`` and ``growth = -2 pi Im(pole)`` (positive = unstable), matching
        the :func:`~nefes.perturbation.eigenmodes` convention.

        Parameters
        ----------
        freq_band : tuple of float
            ``(f_lo, f_hi)`` frequency window [Hz].
        growth_band : tuple of float, optional
            ``(g_lo, g_hi)`` growth-rate window [1/s]; default unbounded.

        Returns
        -------
        ndarray
            The offending poles [Hz] (complex), empty if the window is clean.
        """
        p = self.poles
        if p.size == 0:
            return p
        fr = p.real
        gr = -2.0 * np.pi * p.imag
        inside = (fr >= freq_band[0]) & (fr <= freq_band[1])
        if growth_band is not None:
            inside &= (gr >= growth_band[0]) & (gr <= growth_band[1])
        return p[inside]

    # -- plotting -----------------------------------------------------------

    def plot_fit(self, **kwargs):
        """Overlay the continued curve on the tabulated data (see :func:`nefes.plotting.plot_fit`)."""
        from ..plotting import plot_fit

        return plot_fit(self, **kwargs)

    def plot_pole_map(self, **kwargs):
        """Pole/zero map in the frequency-growth plane (see :func:`nefes.plotting.plot_pole_map`)."""
        from ..plotting import plot_pole_map

        return plot_pole_map(self, **kwargs)

    def __repr__(self):
        lag = f", delay={self.delay * 1e3:.3g} ms" if self.delay else ""
        return (
            f"RationalFit({self.n_terms} terms, {self.freqs[0]:.4g}-{self.freqs[-1]:.4g} Hz"
            f"{lag}, max_err={self.max_error():.2e})"
        )


def rational_fit(freqs, values, **kwargs) -> RationalFit:
    """Analytically-continuable rational fit of tabulated ``F(f)`` data (see :class:`RationalFit`).

    The headline builder: wrap a measured / tabulated transfer function or reflection
    coefficient so it can be used in the **stability eigenproblem** (which evaluates at
    complex frequency), where a real-grid :func:`~nefes.elements.dynamic_source.tabulated`
    interpolant cannot go.

    Examples
    --------
    >>> ftf = rational_fit(freqs_hz, measured_F, delay="auto")          # flame FTF
    >>> ds = heat_release_response(ftf, ref_edge=1)
    >>> bc = PerturbationBC.reflection(rational_fit(freqs_hz, measured_R))  # boundary R(f)
    """
    return RationalFit(freqs, values, **kwargs)


def continuation_warning(fit, freq_band, growth_band=None, *, stacklevel=2) -> np.ndarray:
    """Warn if ``fit`` has poles inside a search window; return them (empty if clean).

    A convenience guard for stability drivers / notebooks: it calls
    :meth:`RationalFit.poles_in_region` and, when non-empty, emits a warning naming the
    offending poles so a suspicious eigenvalue can be traced back to the fit rather than
    the physics.
    """
    bad = fit.poles_in_region(freq_band, growth_band)
    if bad.size:
        warnings.warn(
            f"rational continuation has {bad.size} pole(s) inside the search window "
            f"freq_band={freq_band}, growth_band={growth_band}: {np.round(bad, 2)}. "
            "The fit is reliable only in the strip around the real axis spanned by the data; "
            "tighten the growth band, add data, or refit with delay extraction / a coarser rtol.",
            RuntimeWarning,
            stacklevel=stacklevel,
        )
    return bad
