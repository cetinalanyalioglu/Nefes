"""Frequency-domain complex-matrix descriptors: transfer and scattering matrices.

A great deal of the perturbation layer speaks in **complex matrices of frequency** --
a 2-port transfer matrix ``T(f)``, a scattering matrix ``S(f)``, a multiport response.
:class:`TransferMatrix` and :class:`ScatteringMatrix` wrap such a table (shape
``(n_f, N, N)``) with the operations that recur: evaluate / re-interpolate on a new grid,
**analytically continue** it off the real axis (per-entry, via an impulse-response fit or a
rational fit -- see :meth:`FreqMatrix.continue_`; do this in *scattering* form, whose entries
are finite-memory responses), convert between variable **flavors**
(``characteristics.basis_matrix``) and between the transfer and scattering forms, and plot.

These are the objects a user constructs to feed a
:func:`~nefes.elements.catalog.transfer_matrix_element` (its acoustic identity), and the
objects the identification layer (:mod:`nefes.perturbation.identify`) returns.

Frequency convention
--------------------
Everything is a function of **frequency in Hz** (project convention).  A real-grid table
interpolates only on the real axis; call :meth:`FreqMatrix.continue_` to obtain a version
that evaluates at a *complex* frequency (what the contour eigensolver needs).

Conventions
-----------
* A **transfer matrix** relates the two stations' variables along their own arrow,
  ``v_down = T @ v_up`` (``matrices``), in the flavor named by :attr:`FreqMatrix.basis`
  (``characteristics.BASIS_LABELS``; the default is ``"char"``, defined below).
* A **scattering matrix** maps the *incoming* waves to the *outgoing* ones, ordered by
  ``matrices.scattering_labels``; its basis must be diagonal in the waves (``"char"`` or
  ``"riemann"``).
* Flavor and transfer<->scattering conversions need the mean state at each face; supply
  them as :class:`PortState` in ``ports`` (identification attaches them automatically).

Wave bases
----------
Two flavors are diagonal in the characteristic waves -- each component *is* one wave -- which
is exactly what a scattering matrix (an incoming-wave -> outgoing-wave map) needs:

* ``"char"`` -- the characteristic amplitudes ``w = (f, g, h)``: ``f`` the downstream acoustic
  Riemann wave, ``g`` the upstream acoustic Riemann wave, and ``h`` the convected entropy wave,
  related to the primitives by ``p' = rho c (f + g)``, ``u' = f - g`` and ``rho' = h + p'/c^2``.
* ``"riemann"`` -- the De Domenico normalization ``(P+, P-, sigma) = (f/c, g/c, -h/rho)``, a
  per-wave rescaling of ``"char"`` (so it shares the same wave-diagonal structure).

The remaining flavors (``"primitive"``, ``"network"``, ``"pu_entropy"``, ``"pu_rho"``) mix the
waves and so are valid for a *transfer* matrix only; see ``characteristics.BASIS_LABELS`` for
the full catalogue.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .operator import matrices as mat
from .operator.characteristics import BASIS_LABELS, basis_matrix


@dataclass(frozen=True)
class PortState:
    """Mean flow at one face of a 2-port, enough to change flavor / form.

    Parameters
    ----------
    rho, c, u, p : float
        Mean density [kg/m^3], sound speed [m/s], axial velocity [m/s] (signed along the
        edge arrow) and static pressure [Pa].
    area : float, optional
        Edge area [m^2] (default 1.0; only the ``network`` flavor uses it).
    cal : tuple, optional
        The edge's caloric coupling row ``(a, u, b)`` (:func:`characteristics.caloric_row`),
        needed only for the ``network`` flavor -- it carries the gas-model-specific
        ``(dh/drho)_p`` and ``(dh/dp)_rho``.  ``None`` (the default) leaves the network flavor
        unavailable from this port.
    """

    rho: float
    c: float
    u: float
    p: float
    area: float = 1.0
    cal: Optional[tuple] = None

    def basis_block(self, basis):
        """The ``v = B @ w`` block for ``basis`` at this state (``characteristics.basis_matrix``)."""
        return basis_matrix(basis, self.rho, self.c, self.u, self.area, self.cal)


def _as_port(x) -> Optional[PortState]:
    if x is None or isinstance(x, PortState):
        return x
    if isinstance(x, (tuple, list)):
        return PortState(*x)
    raise TypeError(f"a port state must be a PortState or a (rho, c, u, p[, area[, cal]]) tuple; got {x!r}")


class FreqMatrix:
    """Base: a complex ``(n_f, N, N)`` table sampled on a real frequency grid [Hz].

    Not used directly -- see :class:`TransferMatrix` / :class:`ScatteringMatrix`.  Holds
    the grid, the data, the variable :attr:`basis`, the optional per-face
    :class:`PortState` pair, and (once :meth:`continue_` is called) a per-entry rational
    continuation used for complex-frequency evaluation.
    """

    kind = "matrix"

    def __init__(self, freqs, data, *, basis="char", ports=None):
        f = np.asarray(freqs, dtype=float).ravel()
        d = np.asarray(data, dtype=np.complex128)
        if d.ndim == 2:  # a single constant matrix -> broadcast over the grid
            d = np.broadcast_to(d, (f.size,) + d.shape).copy()
        if d.ndim != 3 or d.shape[0] != f.size or d.shape[1] != d.shape[2]:
            raise ValueError(f"data must be (n_f, N, N) with n_f = len(freqs) = {f.size}; got {d.shape}")
        if f.size < 1:
            raise ValueError("need at least one frequency sample")
        if f.size > 1 and np.any(np.diff(f) <= 0.0):
            raise ValueError("freqs must be strictly increasing")
        if basis not in BASIS_LABELS:
            raise ValueError(f"unknown basis/flavor {basis!r}; choose from {sorted(BASIS_LABELS)}")
        self.freqs = f
        self.data = d
        self.basis = basis
        p = ports if ports is not None else (None, None)
        self.ports: Tuple[Optional[PortState], Optional[PortState]] = (_as_port(p[0]), _as_port(p[1]))
        self._fits = None  # (N, N) object array of RationalFit once continued

    # -- shape ------------------------------------------------------------
    @property
    def n(self) -> int:
        """Matrix dimension ``N``."""
        return self.data.shape[1]

    @property
    def n_freqs(self) -> int:
        """Number of tabulated frequencies."""
        return self.freqs.size

    def _require_ports(self, what):
        if self.ports[0] is None or self.ports[1] is None:
            raise ValueError(
                f"{what} needs the mean state at both faces; construct with ports=(PortState, PortState) "
                "(identification attaches them automatically)"
            )
        return self.ports

    def _new(self, data, *, basis=None, ports=None):
        """A sibling of the same concrete class with new data (metadata inherited)."""
        return type(self)(
            self.freqs,
            data,
            basis=self.basis if basis is None else basis,
            ports=self.ports if ports is None else ports,
        )

    # -- evaluation -------------------------------------------------------
    def __call__(self, f):
        """Evaluate at frequency ``f`` [Hz]; shape ``(len(f), N, N)`` (or ``(N, N)`` if scalar).

        On the real axis a raw table interpolates its entries (real / imaginary parts,
        cubic where possible); at a **complex** frequency it requires a prior
        :meth:`continue_` (real-grid interpolation is not analytic).
        """
        fa = np.asarray(f, dtype=np.complex128)
        scalar = fa.ndim == 0
        fr = fa.ravel()
        if self._fits is not None:
            out = self._eval_fits(fr)
        else:
            if np.any(np.abs(fr.imag) > 1e-9 * (np.abs(fr.real) + 1.0)):
                raise ValueError(
                    "a tabulated complex matrix cannot be evaluated at a complex frequency; "
                    "call .continue_() first (rational continuation) for the stability eigenproblem"
                )
            out = self._interp_entries(fr.real)
        return out[0] if scalar else out

    def _interp_entries(self, fr):
        """Per-entry real/imag interpolation on the real grid, held outside the band."""
        N = self.n
        fr = np.asarray(fr, dtype=float)
        out = np.empty((fr.size, N, N), dtype=np.complex128)
        if self.freqs.size == 1:  # single sample -> constant
            out[:] = self.data[0]
            return out
        use_cubic = self.freqs.size >= 4
        for i in range(N):
            for j in range(N):
                out[:, i, j] = _interp1_complex(self.freqs, self.data[:, i, j], fr, use_cubic)
        return out

    def _eval_fits(self, fr):
        N = self.n
        out = np.empty((fr.size, N, N), dtype=np.complex128)
        for i in range(N):
            for j in range(N):
                out[:, i, j] = np.asarray(self._fits[i, j](fr), dtype=np.complex128).ravel()
        return out

    def resample(self, freqs):
        """A copy re-interpolated onto a new frequency grid [Hz]."""
        fr = np.asarray(freqs, dtype=float).ravel()
        return type(self)(fr, self(fr), basis=self.basis, ports=self.ports)

    def continue_(self, method="rational", **fit_kwargs):
        """A copy that evaluates at **complex** frequency via a per-entry fit.

        Continue in **scattering** form, not transfer form.  The entries of a scattering
        matrix are causal responses of outgoing waves to incoming ones, so for a compact
        element they die out after a finite time and both fit methods below describe them
        faithfully.  The entries of a *transfer* matrix are an algebraic rearrangement of
        those responses: they mix delays of both signs (a plain duct already has
        ``cosh``/``sinh`` entries that grow off the real axis in both directions) and the
        rearrangement divides by the transmission response, creating genuine poles.  So:
        :meth:`TransferMatrix.to_scattering` first, continue there, and convert back --
        the conversions preserve analyticity.

        ``method="impulse"`` fits each entry with
        :func:`~nefes.elements.dynamic_source.fit_impulse_response` (recommended for
        finite-memory entries; requires ``duration``, accepts ``dt`` and ``smoothing``):
        the continuation is a finite sum of delays with no poles anywhere.
        ``method="rational"`` (default, and the choice for entries with a genuine
        resonance) fits with :class:`~nefes.perturbation.continuation.RationalFit` (AAA);
        ``fit_kwargs`` are forwarded (e.g. ``rtol`` near the noise floor of measured
        data, or ``delay="auto"`` -- the default here -- to peel a transport lag before
        fitting).
        """
        if self.freqs.size < 2:
            raise ValueError("need at least two frequency samples to build a continuation")
        if method == "impulse":
            from ..elements.dynamic_source import fit_impulse_response

            entry_fit = lambda values: fit_impulse_response(self.freqs, values, **fit_kwargs)  # noqa: E731
        elif method == "rational":
            from .continuation import RationalFit

            fit_kwargs.setdefault("delay", "auto")
            entry_fit = lambda values: RationalFit(self.freqs, values, **fit_kwargs)  # noqa: E731
        else:
            raise ValueError(f"method must be 'impulse' or 'rational'; got {method!r}")
        N = self.n
        fits = np.empty((N, N), dtype=object)
        for i in range(N):
            for j in range(N):
                fits[i, j] = entry_fit(self.data[:, i, j])
        out = self._new(self.data)
        out._fits = fits
        return out

    @property
    def analytic(self) -> bool:
        """Whether this matrix can be evaluated at a complex frequency (was continued)."""
        return self._fits is not None

    def max_fit_error(self) -> float:
        """Largest per-entry fit error on the grid (0 before :meth:`continue_`)."""
        if self._fits is None:
            return 0.0

        def entry_error(fit):
            return fit.max_error() if hasattr(fit, "max_error") else fit.max_misfit

        return float(max(entry_error(self._fits[i, j]) for i in range(self.n) for j in range(self.n)))

    # -- flavor change ----------------------------------------------------
    def to_basis(self, basis):
        """Re-express (a *transfer* matrix) in another variable flavor (``BASIS_LABELS``)."""
        raise NotImplementedError

    def _labels(self):
        """Per-component symbols for plotting (from the flavor)."""
        return BASIS_LABELS.get(self.basis)

    def __repr__(self):
        tag = ", continued" if self.analytic else ""
        span = f"{self.freqs[0]:.4g}-{self.freqs[-1]:.4g} Hz" if self.freqs.size > 1 else f"{self.freqs[0]:.4g} Hz"
        return f"{type(self).__name__}(N={self.n}, {self.freqs.size} pts, {span}, basis={self.basis!r}{tag})"


def _interp1_complex(freqs, values, fr, use_cubic):
    """Interpolate a complex series on real/imag parts, holding the endpoints outside."""
    lo, hi = freqs[0], freqs[-1]
    frc = np.clip(fr, lo, hi)
    if use_cubic:
        from scipy.interpolate import CubicSpline

        re = CubicSpline(freqs, values.real)(frc)
        im = CubicSpline(freqs, values.imag)(frc)
    else:
        re = np.interp(frc, freqs, values.real)
        im = np.interp(frc, freqs, values.imag)
    return re + 1j * im


class TransferMatrix(FreqMatrix):
    """A 2-port **transfer matrix** ``v_down = T(f) @ v_up`` versus frequency [Hz].

    Parameters
    ----------
    freqs : array_like
        Tabulated frequencies [Hz], strictly increasing.
    data : array_like
        Complex ``(n_f, N, N)`` samples (or a single ``(N, N)`` constant matrix, broadcast).
    basis : str, optional
        Variable flavor of ``v`` (default ``"char"``, the amplitudes ``(f, g, h)``); any of
        ``characteristics.BASIS_LABELS``.
    ports : (PortState, PortState), optional
        Mean state at the upstream and downstream face; required for flavor and
        transfer<->scattering conversions.  Each port carries its own caloric row (``cal``),
        so the ``network`` flavor is reacting-correct with no gas-model constant.

    Notes
    -----
    ``N`` is the characteristic count and fixes whether the convected **scalar (entropy) wave**
    is carried: ``N = 2`` is acoustics-only, ``(f, g)``, with no scalar wave; ``N = 3`` adds the
    entropy amplitude ``h`` as the third characteristic, so a scalar wave is present exactly when
    ``N = 3``.  The 2-D case uses the classic 2x2 acoustic conventions (``matrices.tm_fg_to_sm2``
    etc.); the 3-D case the general characteristic algebra.
    """

    kind = "transfer"

    def to_basis(self, basis):
        """A copy re-expressed in flavor ``basis`` (needs :attr:`ports`)."""
        if basis == self.basis:
            return self
        if basis not in BASIS_LABELS:
            raise ValueError(f"unknown flavor {basis!r}; choose from {sorted(BASIS_LABELS)}")
        pa, pb = self._require_ports("changing flavor")
        T_char = self._to_char_data()
        if self.n == 2:
            data = _tm2_char_to_basis(T_char, basis)
        else:
            Ba = pa.basis_block(basis)
            Bb = pb.basis_block(basis)
            data = mat.tm_in_basis(T_char, Ba, Bb)
        return self._new(data, basis=basis)

    def _to_char_data(self):
        """This TM's samples re-expressed in the characteristic basis (n_f, N, N)."""
        if self.basis == "char":
            return self.data
        pa, pb = self._require_ports("changing flavor")
        if self.n == 2:
            return _tm2_basis_to_char(self.data, self.basis)
        Ba = pa.basis_block(self.basis)
        Bb = pb.basis_block(self.basis)
        # v = B w  =>  T_char = Bb^-1 T_self Ba  (inverse similarity of tm_in_basis)
        return mat.tm_in_basis(self.data, np.linalg.inv(Ba), np.linalg.inv(Bb))

    def to_scattering(self):
        """The equivalent :class:`ScatteringMatrix` (needs :attr:`ports`)."""
        pa, pb = self._require_ports("converting to a scattering matrix")
        T_char = self._to_char_data()
        if self.n == 2:
            S = mat.tm_fg_to_sm2(T_char)
        else:
            S, _in, _out = mat.tm_to_sm(T_char, pa.u, pa.c, pb.u, pb.c)
        out = ScatteringMatrix(self.freqs, S, basis="char", ports=self.ports)
        if self._fits is not None:  # keep it analytic across the conversion
            out = out.continue_()
        return out

    def plot(self, freqs=None, **kwargs):
        """Magnitude/phase grid of the entries (see ``plotting.plot_transfer_matrix``)."""
        from ..plotting.complex_matrix import plot_transfer_matrix

        fr = self.freqs if freqs is None else np.asarray(freqs, dtype=float)
        return plot_transfer_matrix(self(fr), fr, labels=self._labels(), **kwargs)


class ScatteringMatrix(FreqMatrix):
    """A 2-port **scattering matrix** ``w_out = S(f) @ w_in`` versus frequency [Hz].

    Incoming/outgoing waves are ordered by ``matrices.scattering_labels``.  The basis must
    be diagonal in the characteristics (``"char"`` or ``"riemann"``).  See
    :class:`TransferMatrix` for the constructor arguments.
    """

    kind = "scattering"

    _DIAGONAL_BASES = ("char", "riemann")

    def __init__(self, freqs, data, *, basis="char", ports=None):
        if basis not in self._DIAGONAL_BASES:
            raise ValueError(f"a scattering matrix basis must be one of {self._DIAGONAL_BASES}; got {basis!r}")
        super().__init__(freqs, data, basis=basis, ports=ports)

    def to_transfer(self):
        """The equivalent :class:`TransferMatrix` in the characteristic basis (needs :attr:`ports`)."""
        pa, pb = self._require_ports("converting to a transfer matrix")
        S = self.data if self.basis == "char" else self._scale_to_char()
        if self.n == 2:
            T = _sm2_to_tm_fg(S)
        else:
            T = mat.sm_to_tm(S, pa.u, pa.c, pb.u, pb.c)
        out = TransferMatrix(self.freqs, T, basis="char", ports=self.ports)
        if self._fits is not None:
            out = out.continue_()
        return out

    def _scale_to_char(self):
        """Undo a diagonal (riemann) rescaling of the incoming/outgoing amplitudes -> char."""
        if self.basis == "char":
            return self.data
        pa, pb = self._require_ports("changing scattering basis")
        incoming, outgoing = mat.scattering_labels(pa.u, pa.c, pb.u, pb.c, self.n)
        din = np.array([self._wave_scale(st, i, pa, pb) for (st, i) in incoming])
        dout = np.array([self._wave_scale(st, i, pa, pb) for (st, i) in outgoing])
        # S_basis = diag(dout) S_char diag(din)^-1  ->  S_char = diag(dout)^-1 S_basis diag(din)
        return (self.data * din[None, None, :]) / dout[None, :, None]

    def _wave_scale(self, station, i, pa, pb):
        port = pa if station == "a" else pb
        return port.basis_block(self.basis)[i, i]

    def plot(self, freqs=None, **kwargs):
        """Magnitude/phase grid of the entries (see ``plotting.plot_scattering_matrix``)."""
        from ..plotting.complex_matrix import plot_transfer_matrix

        fr = self.freqs if freqs is None else np.asarray(freqs, dtype=float)
        return plot_transfer_matrix(self(fr), fr, labels=self._labels(), **kwargs)


# --------------------------------------------------------------------------
# 2x2 acoustic flavor helpers (only (f,g) <-> (p'/(rho c), u') are meaningful for N=2)
# --------------------------------------------------------------------------


def _tm2_char_to_basis(T_fg, basis):
    if basis == "char":
        return T_fg
    if basis in ("primitive", "pu_rho", "pu_entropy"):  # all share the (p'/(rho c), u') acoustic block
        return mat.tm_fg_to_pu(T_fg)
    raise ValueError(f"2x2 acoustic transfer matrix supports flavors 'char' or 'primitive'/'pu_*'; got {basis!r}")


def _tm2_basis_to_char(T_basis, basis):
    if basis == "char":
        return T_basis
    if basis in ("primitive", "pu_rho", "pu_entropy"):
        return mat.tm_pu_to_fg(T_basis)
    raise ValueError(f"2x2 acoustic transfer matrix supports flavors 'char' or 'primitive'/'pu_*'; got {basis!r}")


def _sm2_to_tm_fg(S):
    """Invert the classic 2x2 (f,g) scattering matrix back to the (f,g) transfer matrix."""
    S = np.asarray(S, dtype=np.complex128)
    was2d = S.ndim == 2
    Sb = S[None] if was2d else S
    T = np.empty_like(Sb)
    # tm_fg_to_sm2:  s11=t11-t12 t21/t22, s12=t12/t22, s21=-t21/t22, s22=1/t22
    s11, s12, s21, s22 = Sb[:, 0, 0], Sb[:, 0, 1], Sb[:, 1, 0], Sb[:, 1, 1]
    t22 = 1.0 / s22
    t12 = s12 * t22
    t21 = -s21 * t22
    t11 = s11 + t12 * t21 / t22
    T[:, 0, 0], T[:, 0, 1], T[:, 1, 0], T[:, 1, 1] = t11, t12, t21, t22
    return T[0] if was2d else T
