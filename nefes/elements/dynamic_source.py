"""Dynamic source response ``S(omega)``: frequency response of a source term.

A mass source (a fuel/air injector) or a heat source carries a **dynamic** part: a
fluctuating injection / heat release whose amplitude responds to the unsteady flow
(perturbations) elsewhere in the network.  The classic example is a velocity-driven
flame transfer function (FTF): the heat release fluctuates with the acoustic velocity
``u'`` an edge upstream of the flame, with a gain and a time lag. This feedback is
what makes the perturbation operator non-self-adjoint and drives thermoacoustic
instability.

The general response is a **superposition of transfer functions**, each on its own
reference edge and quantity::

    q'(omega) / q_bar = sum_k  gain_k * F_k(omega) * ( phi'_k(omega) / phi_bar_k )

where ``q'`` is the fluctuation of the modulated source quantity (heat release for a
flame, injected mass-flow for a source), ``q_bar`` its mean, ``F_k`` a (generally
complex) transfer function of frequency, and ``phi_k`` a reference flow quantity
(``u``, ``p``, ``rho``, ``mdot`` or a composition scalar ``Z:<name>``) at a chosen
reference edge.  Most flames are modelled with a single velocity term.

This module owns only the **specification** (the descriptor + the transfer-function
objects); the mean flow ignores it entirely (a constant mean source is acoustically
passive), and the perturbation layer (:mod:`nefes.perturbation.operator.stamps`)
consumes it to stamp the ``S(omega)`` block of the operator.  Nothing here depends on
the perturbation layer.

Frequency convention
--------------------
Every transfer function is a function of **frequency in Hz** (project convention --
graphs and user input use frequency, not angular frequency).  The perturbation
assembler evaluates ``F(omega / 2 pi)``.  For a stability analysis the frequency is
**complex**, so a transfer function must be analytically continuable
(:attr:`TransferFunction.analytic`); the closed-form models are, a table interpolated
on a real grid is not (use it for the forced response, or make use of the bundled
RationalFit to fit a closed-form model to the data).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# Reference quantities a response term may read at its edge.  Composition scalars are
# named "Z:<scalar-name>" and resolved against the compiled problem's ``scalar_names``.
_QUANTITIES = ("u", "p", "rho", "mdot")

# Which source quantity a descriptor modulates, and the residual it feeds.
_TARGETS = ("Qdot", "mdot")


# ==========================================================================
# Transfer functions  F(f) : (complex) frequency [Hz] -> complex
# ==========================================================================


class TransferFunction:
    """A complex-valued function of frequency ``F(f)`` with ``f`` in Hz.

    Subclasses implement :meth:`__call__`.  Two attributes inform the drivers:

    * :attr:`analytic` -- ``True`` if ``F`` can be evaluated at a *complex* frequency
      (required for the stability eigenproblem, which searches the complex plane).
    * :attr:`max_delay` -- the longest pure time lag [s] the function carries; the
      stability driver uses it to clamp the search contour so ``e^{-i omega tau}``
      does not overflow at large growth/decay rates.
    """

    analytic: bool = True
    max_delay: float = 0.0

    def __call__(self, f):  # pragma: no cover - abstract
        raise NotImplementedError

    def plot(self, freqs, **kwargs):
        """Plot magnitude and phase versus frequency (see :func:`nefes.plotting.plot_transfer_function`)."""
        from ..plotting import plot_transfer_function

        return plot_transfer_function(self, freqs, **kwargs)


class Constant(TransferFunction):
    """A frequency-independent (generally complex) response ``F(f) = value``."""

    analytic = True

    def __init__(self, value):
        self.value = complex(value)

    def __call__(self, f):
        return self.value * np.ones_like(np.asarray(f, dtype=np.complex128))

    def __repr__(self):
        return f"Constant({self.value!r})"


class NTau(TransferFunction):
    """The ``n-tau`` flame model ``F(f) = n * exp(-i * 2 pi f * tau)``.

    The (generally complex) interaction index ``n`` times a pure time lag ``tau``
    [s].  Entire in frequency, so it is usable in the stability eigenproblem.  Under
    the ``e^{+i omega t}`` convention the factor ``exp(-i omega tau)`` is the causal
    delay of the response behind the driving fluctuation (same sign as the duct
    propagation phase).

    Parameters
    ----------
    n : float or complex
        Interaction index; the gain of the model is ``abs(n)``.
    tau : float
        Time lag [s] (``>= 0`` for a causal response).
    """

    analytic = True

    def __init__(self, n, tau):
        self.n = complex(n)
        self.tau = float(tau)
        self.max_delay = abs(float(tau))

    def __call__(self, f):
        f = np.asarray(f, dtype=np.complex128)
        return self.n * np.exp(-2j * np.pi * f * self.tau)

    def __repr__(self):
        return f"NTau(n={self.n!r}, tau={self.tau!r})"


class NTauLowpass(TransferFunction):
    """The ``n-tau`` flame with a first-order gain roll-off ``F(f) = n e^{-i 2 pi f tau} / (1 + i f / f_c)``.

    The bare :class:`NTau` model has a frequency-independent gain ``n``, which lets a
    lossless duct destabilize an unbounded comb of high-frequency modes -- unphysical.
    A real flame is a **low-pass** responder: its gain rolls off above a cutoff ``f_c``
    (the flame cannot follow forcing faster than its own response time), bounding the
    instability to a finite band.  This is the canonical model for a converged
    (Nyquist) stability count.

    Entire in the unstable (lower-half ``omega``) plane -- the low-pass pole sits at
    ``f = i f_c`` (the *stable* upper half), so it is analytically continuable for the
    eigenproblem (as long as the search region does not reach down to growth
    ``-2 pi f_c``).

    Parameters
    ----------
    n : float or complex
        Low-frequency interaction index; the zero-frequency gain is ``abs(n)``.
    tau : float
        Time lag [s] (``>= 0`` for a causal response).
    fc : float
        Roll-off cutoff frequency [Hz] (``> 0``).
    """

    analytic = True

    def __init__(self, n, tau, fc):
        self.n = complex(n)
        self.tau = float(tau)
        self.fc = float(fc)
        if self.fc <= 0.0:
            raise ValueError(f"roll-off cutoff fc must be positive; got {fc}")
        self.max_delay = abs(float(tau))

    def __call__(self, f):
        f = np.asarray(f, dtype=np.complex128)
        return self.n * np.exp(-2j * np.pi * f * self.tau) / (1.0 + 1j * f / self.fc)

    def __repr__(self):
        return f"NTauLowpass(n={self.n!r}, tau={self.tau!r}, fc={self.fc!r})"


class Tabulated(TransferFunction):
    """A measured transfer function interpolated from a table ``F(freqs) = values``.

    Real-frequency only: this grid interpolant is **not** analytically continuable, so it
    cannot be evaluated at the complex frequencies the stability eigenproblem visits
    (:attr:`analytic` is ``False``).  Use it for the forced response / scattering sweep,
    which stay on the real axis.  For **stability** analysis from the same tabulated data,
    fit it with :func:`~nefes.perturbation.continuation.rational_fit`
    (:class:`~nefes.perturbation.continuation.RationalFit`): a barycentric rational fit is
    analytic off the real axis, so it drops straight into the eigensolver.  A closed-form
    model (:class:`NTau`) fitted by hand works too.

    Magnitude and (unwrapped) phase are interpolated separately so the gain stays
    non-negative and the phase reads smoothly; outside the tabulated band the value
    is held at the nearest endpoint (``extrapolate="hold"``) or set to zero
    (``"zero"``).

    Parameters
    ----------
    freqs : array_like
        Tabulated frequencies [Hz], strictly increasing.
    values : array_like
        Complex transfer-function values at ``freqs``.
    kind : {"linear", "cubic"}, optional
        Interpolation order on the magnitude/phase curves (default ``"cubic"`` when
        SciPy is available and there are enough points, else ``"linear"``).
    extrapolate : {"hold", "zero"}, optional
        Behaviour outside ``[freqs[0], freqs[-1]]`` (default ``"hold"``).
    """

    analytic = False

    def __init__(self, freqs, values, *, kind="cubic", extrapolate="hold"):
        f = np.asarray(freqs, dtype=float)
        v = np.asarray(values, dtype=np.complex128)
        if f.ndim != 1 or v.shape != f.shape:
            raise ValueError("freqs and values must be 1-D arrays of equal length")
        if np.any(np.diff(f) <= 0.0):
            raise ValueError("freqs must be strictly increasing")
        if extrapolate not in ("hold", "zero"):
            raise ValueError("extrapolate must be 'hold' or 'zero'")
        self.freqs = f
        self.values = v
        self.extrapolate = extrapolate
        self._mag = np.abs(v)
        self._phase = np.unwrap(np.angle(v))
        self._kind = kind if (kind == "linear" or f.size >= 4) else "linear"

    def _interp1(self, curve, f):
        if self._kind == "cubic":
            from scipy.interpolate import CubicSpline

            spline = CubicSpline(self.freqs, curve, extrapolate=False)
            out = spline(f)
        else:
            out = np.interp(f, self.freqs, curve, left=np.nan, right=np.nan)
        return out

    def __call__(self, f):
        f = np.asarray(f)
        if np.iscomplexobj(f) and np.any(np.abs(f.imag) > 1e-12 * (np.abs(f.real) + 1.0)):
            raise ValueError(
                "a tabulated transfer function cannot be evaluated at a complex frequency "
                "(real-grid interpolation is not analytic). Use it for the forced response, "
                "or supply a closed-form model (e.g. n_tau) for the stability eigenproblem."
            )
        fr = np.asarray(f.real if np.iscomplexobj(f) else f, dtype=float)
        mag = self._interp1(self._mag, fr)
        ph = self._interp1(self._phase, fr)
        out = mag * np.exp(1j * ph)
        outside = np.isnan(out)
        if np.any(outside):
            if self.extrapolate == "zero":
                out = np.where(outside, 0.0, out)
            else:  # hold the nearest endpoint value
                held = np.where(fr <= self.freqs[0], self.values[0], self.values[-1])
                out = np.where(outside, held, out)
        return out

    def __repr__(self):
        return f"Tabulated(n={self.freqs.size} points, {self.freqs[0]:.4g}-{self.freqs[-1]:.4g} Hz)"


class _CallableTF(TransferFunction):
    """Wrap a bare ``omega_hz -> complex`` callable as a :class:`TransferFunction`."""

    def __init__(self, fn, *, analytic=False, max_delay=0.0):
        self._fn = fn
        self.analytic = bool(analytic)
        self.max_delay = float(max_delay)

    def __call__(self, f):
        return np.asarray(self._fn(f), dtype=np.complex128)


# -- builders --------------------------------------------------------------


def n_tau(n, tau) -> NTau:
    """The ``n-tau`` flame model ``F(f) = n * exp(-i 2 pi f tau)`` (see :class:`NTau`).

    Parameters
    ----------
    n : float or complex
        Interaction index; the gain of the model is ``abs(n)``.
    tau : float
        Time lag [s] (``>= 0`` for a causal response).

    Returns
    -------
    NTau
    """
    return NTau(n, tau)


def n_tau_lowpass(n, tau, fc) -> NTauLowpass:
    """The ``n-tau`` flame with a first-order gain roll-off (see :class:`NTauLowpass`).

    Parameters
    ----------
    n : float or complex
        Low-frequency interaction index; the zero-frequency gain is ``abs(n)``.
    tau : float
        Time lag [s] (``>= 0`` for a causal response).
    fc : float
        Roll-off cutoff frequency [Hz] (``> 0``).

    Returns
    -------
    NTauLowpass
    """
    return NTauLowpass(n, tau, fc)


def constant(value) -> Constant:
    """A frequency-independent (generally complex) response (see :class:`Constant`).

    Parameters
    ----------
    value : float or complex
        The constant response ``F(f) = value``; its gain is ``abs(value)``.

    Returns
    -------
    Constant
    """
    return Constant(value)


def tabulated(freqs, values, **kwargs) -> Tabulated:
    """A measured transfer function from a table (see :class:`Tabulated`).

    Parameters
    ----------
    freqs : array_like
        Tabulated frequencies [Hz], strictly increasing.
    values : array_like
        Complex transfer-function values at ``freqs``.
    **kwargs
        Forwarded to :class:`Tabulated` (e.g. ``kind``, ``extrapolate``).

    Returns
    -------
    Tabulated
    """
    return Tabulated(freqs, values, **kwargs)


def as_transfer(obj) -> TransferFunction:
    """Coerce ``obj`` into a :class:`TransferFunction`.

    Parameters
    ----------
    obj : TransferFunction or (n, tau) or number or callable
        An existing :class:`TransferFunction`, an ``(n, tau)`` pair (-> n-tau), a
        real/complex number (-> constant), or a bare ``f -> complex`` callable (wrapped;
        treated as *non*-analytic, so usable only for the forced response unless it is a
        :class:`TransferFunction` declaring ``analytic = True``).

    Returns
    -------
    TransferFunction
    """
    if isinstance(obj, TransferFunction):
        return obj
    if isinstance(obj, (tuple, list)) and len(obj) == 2 and all(np.isscalar(v) for v in obj):
        return NTau(obj[0], obj[1])
    if np.isscalar(obj):
        return Constant(obj)
    if callable(obj):
        return _CallableTF(obj)
    raise TypeError(
        f"cannot interpret {obj!r} as a transfer function; pass a TransferFunction, an "
        "(n, tau) pair, a number, or a callable f->complex"
    )


# ==========================================================================
# Dynamic-source descriptor
# ==========================================================================


@dataclass
class DynamicResponseTerm:
    """One transfer-function term ``gain * F(omega) * (phi'_ref / phi_bar_ref)``.

    Parameters
    ----------
    transfer : TransferFunction or (n, tau) or number or callable
        The frequency response ``F``; coerced via :func:`as_transfer`.
    ref_edge : int
        Edge id whose fluctuation drives this term (e.g. the edge just upstream of a
        flame for a velocity FTF).
    quantity : str, optional
        Reference quantity at ``ref_edge``: ``"u"`` (velocity, default), ``"p"``,
        ``"rho"``, ``"mdot"``, or ``"Z:<name>"`` for a transported composition scalar.
    gain : float, optional
        Real scalar multiplier on this term (default 1.0).  A complex weight belongs in
        ``transfer``, not here: a gain is the magnitude ``abs(F)`` of a response.
    """

    transfer: object
    ref_edge: int
    quantity: str = "u"
    gain: float = 1.0

    def __post_init__(self):
        self.transfer = as_transfer(self.transfer)
        self.ref_edge = int(self.ref_edge)
        q = self.quantity
        if q not in _QUANTITIES and not q.startswith("Z:"):
            raise ValueError(f"quantity must be one of {_QUANTITIES} or 'Z:<scalar-name>'; got {q!r}")
        # a gain is the magnitude abs(F) of a response, hence real; a complex weight is a
        # response in its own right and belongs in ``transfer``
        if isinstance(self.gain, complex):
            raise TypeError(
                f"gain must be real (a gain is abs(F)); pass a complex weight in 'transfer'; got {self.gain!r}"
            )
        self.gain = float(self.gain)


@dataclass
class DynamicSource:
    """How a source term's fluctuation responds to the unsteady flow ``S(omega)``.

    The modulated source quantity fluctuates as a sum over :class:`DynamicResponseTerm`::

        q'(omega) = q_mean * sum_k  term_k.gain * F_k(omega) * (phi'_k / phi_bar_k)

    For ``target="Qdot"`` (a flame) ``q'`` is the unsteady heat release [W], stamped
    onto the downstream edge's total-enthalpy (energy) row as ``q'/mdot``.  For
    ``target="mdot"`` (a mass source) ``q'`` is the unsteady injected mass-flow [kg/s],
    stamped onto the source element's mass row and, in proportion to the injection
    velocity ``u_inj``, its momentum row (as ``q' u_inj / A``).  ``u_inj`` is zero by
    default, so a quiescent injector perturbs mass only -- the momentum stamp appears
    only for an injector given a non-zero velocity.

    Parameters
    ----------
    terms : list of DynamicResponseTerm
        The transfer-function terms summed to form the response.
    target : {"Qdot", "mdot"}, optional
        Which source quantity is modulated (default ``"Qdot"``).
    q_mean : float, optional
        Mean of the modulated quantity used to de-normalize the fractional response:
        ``Q_bar`` [W] for ``target="Qdot"`` or the mean injected ``mdot`` [kg/s] for
        ``target="mdot"``.  ``None`` (default) auto-derives it from the converged mean
        flame/source (the flame's mean enthalpy rise times ``mdot`` for heat release,
        the element's injected ``mdot`` for a mass source); pass a value to override.
    """

    terms: List[DynamicResponseTerm] = field(default_factory=list)
    target: str = "Qdot"
    q_mean: Optional[float] = None

    def __post_init__(self):
        if self.target not in _TARGETS:
            raise ValueError(f"target must be one of {_TARGETS}; got {self.target!r}")
        self.terms = [t if isinstance(t, DynamicResponseTerm) else DynamicResponseTerm(**t) for t in self.terms]
        if not self.terms:
            raise ValueError("a DynamicSource needs at least one DynamicResponseTerm")
        if self.q_mean is not None:
            self.q_mean = float(self.q_mean)

    @property
    def analytic(self) -> bool:
        """Whether every term is analytically continuable (usable for stability)."""
        return all(t.transfer.analytic for t in self.terms)

    @property
    def max_delay(self) -> float:
        """Longest pure time delay across the terms [s] (for the stability contour clamp)."""
        return max((t.transfer.max_delay for t in self.terms), default=0.0)


# -- convenience constructors ----------------------------------------------


def heat_release_response(transfer, ref_edge, *, quantity="u", gain=1.0, q_mean=None) -> DynamicSource:
    """A single-term heat-release response (the common velocity-FTF flame).

    Equivalent to ``DynamicSource([DynamicResponseTerm(transfer, ref_edge, quantity,
    gain)], target="Qdot", q_mean=q_mean)``.

    Parameters
    ----------
    transfer : TransferFunction or (n, tau) or number or callable
        The frequency response ``F``; coerced via :func:`as_transfer`.
    ref_edge : int
        Edge whose fluctuation drives the response (e.g. the edge just upstream of the flame).
    quantity : str, optional
        Reference quantity: one of ``"u"``, ``"p"``, ``"rho"``, ``"mdot"`` or a composition
        scalar ``"Z:<name>"`` (default ``"u"``).
    gain : float, optional
        Real scalar multiplier on the term (default ``1.0``).
    q_mean : float, optional
        Mean heat release [W]; ``None`` (default) auto-derives it from the mean flame.

    Returns
    -------
    DynamicSource
    """
    return DynamicSource(terms=[DynamicResponseTerm(transfer, ref_edge, quantity, gain)], target="Qdot", q_mean=q_mean)


def n_tau_flame(n, tau, ref_edge, *, quantity="u", q_mean=None) -> DynamicSource:
    """The headline ``n-tau`` flame: heat release ``= q_mean * n e^{-i omega tau} * (phi'_ref/phi_bar)``.

    Parameters
    ----------
    n : float or complex
        Interaction index; the gain of the model is ``abs(n)``.
    tau : float
        Time lag [s].
    ref_edge : int
        Reference edge (typically the edge just upstream of the flame).
    quantity : str, optional
        Reference quantity (default ``"u"``).
    q_mean : float, optional
        Mean heat release [W]; ``None`` auto-derives it from the mean flame.
    """
    return heat_release_response(NTau(n, tau), ref_edge, quantity=quantity, q_mean=q_mean)


def mass_flow_response(transfer, ref_edge, *, quantity="u", gain=1.0, mdot_mean=None) -> DynamicSource:
    """A single-term injected-mass-flow response (e.g. a velocity-modulated fuel feed).

    Equivalent to ``DynamicSource([...], target="mdot", q_mean=mdot_mean)``.

    Parameters
    ----------
    transfer : TransferFunction or (n, tau) or number or callable
        The frequency response ``F``; coerced via :func:`as_transfer`.
    ref_edge : int
        Edge whose fluctuation drives the injected-mass modulation.
    quantity : str, optional
        Reference quantity (default ``"u"``); see :func:`heat_release_response`.
    gain : float, optional
        Real scalar multiplier on the term (default ``1.0``).
    mdot_mean : float, optional
        Mean injected mass flow [kg/s]; ``None`` (default) auto-derives it from the element.

    Returns
    -------
    DynamicSource
    """
    return DynamicSource(
        terms=[DynamicResponseTerm(transfer, ref_edge, quantity, gain)], target="mdot", q_mean=mdot_mean
    )
