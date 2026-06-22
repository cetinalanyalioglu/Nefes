"""Perturbation (acoustic) boundary conditions for single-port terminals.

A single-port element fixes the *mean* boundary condition (a mass flow, a total
pressure, a static pressure, or a wall).  The *perturbation* boundary condition is
extra information the mean BC cannot supply: how the terminal closes the linear
fluctuation problem.  theory.md s12.4 settles the form -- every terminal BC is a
**characteristic closure** mapping the waves that *arrive* at the boundary from the
interior to the waves the boundary must *specify* (those propagating into the
domain)::

    w_specify = A(omega) @ w_arriving + b(omega)

The mean state partitions the three characteristics ``w = (f, g, h)`` into the
``arriving`` set (speeds pointing out of the domain) and the ``specify`` set (speeds
pointing in) -- see :func:`matrices.partition`.  For a subsonic terminal that is a
duct **tail** (an inlet) the flow carries ``f`` and ``h`` in and lets ``g`` out, so
``A`` is ``2 x 1``; at a duct **head** (an outlet) only ``g`` enters while ``f`` and
``h`` leave, so ``A`` is ``1 x 2``.  ``b`` is an optional excitation forcing.

The classic scalar reflection is the **diagonal** case: ``A`` couples the to-specify
acoustic wave to the arriving acoustic wave by a single coefficient ``R``, and (at an
inlet) seats the incoming entropy independently.  Every named closure below is such a
diagonal ``A`` plus a ``b`` -- *except* ``choked_nozzle``, whose off-diagonal entry is
the entropy -> acoustic coupling (indirect/entropy noise):

===================  =========================================================
kind                 closure
===================  =========================================================
``inherit``          *no stamp* -- keep the linearized mean BC already in J_alg
``hard_wall``        ``R = +1``                   (``u' = 0``)
``open_end``         ``R = -1``                   (``p' = 0``, pressure release)
``mean_flow_open_end`` ``R = -(1 - M)/(1 + M)``   convective open end (``M`` = the
                                                  outward-normal mean Mach)
``anechoic``         ``R = 0``                    (reflection-free termination)
``reflection``       user ``R(omega)``            constant / table / callable
``impedance``        ``R = (Z - rho c)/(Z + rho c)`` from a (specific/absolute) ``Z``
``excitation``       ``R = base_R`` (default 0), with forcing ``b``
``choked_nozzle``    ``g = R f + R_s h``          compact choked outlet (Marble--Candel):
                                                  ``R = (2-(g-1)M)/(2+(g-1)M)``,
                                                  ``R_s = (c/rho) M/(2+(g-1)M)``
===================  =========================================================

Off-diagonal coupling is also available on the generic closures: ``entropy_coupling``
sets the arriving-entropy -> reflected-acoustic term ``R_s`` at an outlet, and
``acoustic_to_entropy`` sets the arriving-acoustic -> specified-entropy term at an inlet.

The impedance map uses the **outward-normal** velocity convention, so it is uniform
at an inlet and an outlet: a rigid wall ``Z -> inf`` gives ``R = +1``, a
pressure-release end ``Z -> 0`` gives ``R = -1``, and the matched impedance
``Z = rho c`` gives ``R = 0``.

Each numeric carrier (``R``, ``Z``, ``amplitude``, ``entropy_in``) may be a complex
constant, a frequency table ``(freqs_hz, values)`` interpolated in frequency, or a
callable ``freq_hz -> complex`` (Python API only; YAML/UI use constants or a table).
Frequencies here are in **Hz**, matching the public perturbation API.  This object
lives entirely *above* the @njit line -- it is evaluated on the frozen mean state at
assembly time, so no complex-step differentiation flows through it.
"""

import cmath
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

KINDS = (
    "inherit",
    "hard_wall",
    "open_end",
    "mean_flow_open_end",
    "anechoic",
    "reflection",
    "impedance",
    "excitation",
    "choked_nozzle",
)

# Characteristic indices (theory.md s9.1): f downstream, g upstream, h entropy.
_F, _G, _H = 0, 1, 2


def _eval(value, freq):
    """Evaluate a coefficient at frequency ``freq`` (Hz).

    ``value`` is a complex constant, a callable ``freq -> complex``, or a frequency
    table ``(freqs, values)`` linearly interpolated (real and imaginary parts) in
    ``freq`` and held flat outside its range.
    """
    if value is None:
        return 0.0 + 0.0j
    if callable(value):
        return complex(value(freq))
    if isinstance(value, tuple) and len(value) == 2:
        xs, ys = np.asarray(value[0], dtype=float), np.asarray(value[1], dtype=complex)
        re = np.interp(freq, xs, ys.real)
        im = np.interp(freq, xs, ys.imag)
        return complex(re, im)
    return complex(value)


def _gamma_minus_one(K):
    """``gamma - 1`` from ``K = cp/R`` (``gamma = K/(K-1)`` so ``gamma-1 = 1/(K-1)``)."""
    if K is None:
        raise ValueError("choked_nozzle needs the gas constant ratio K = cp/R")
    return 1.0 / (float(K) - 1.0)


@dataclass
class PerturbationBC:
    """Acoustic/perturbation closure for a single-port terminal (theory.md s12.4).

    Build one with the named constructors (:meth:`hard_wall`, :meth:`open_end`,
    :meth:`anechoic`, :meth:`reflection`, :meth:`impedance`, :meth:`excitation`,
    :meth:`mean_flow_open_end`); :meth:`inherit` (the default) leaves the linearized
    mean boundary row untouched.

    Attributes
    ----------
    kind : str
        One of :data:`KINDS`.
    R : complex or callable or tuple, optional
        Reflection coefficient for ``kind == "reflection"``.
    Z : complex or callable or tuple, optional
        Acoustic impedance for ``kind == "impedance"`` (specific if ``specific``).
    specific : bool
        If True, ``Z`` is normalized by the characteristic impedance ``rho c``.
    amplitude : complex or callable or tuple, optional
        Acoustic excitation forcing ``b`` for ``kind == "excitation"``.
    base_R : complex or callable or tuple, optional
        Reflection coefficient of an excitation terminal (default ``0`` -- a clean,
        reflection-free source).
    entropy_in : complex or callable or tuple, optional
        Incoming entropy-wave amplitude seated at an inflow terminal (default ``0``).
    family : str
        ``"acoustic"`` (default) or ``"entropy"`` -- which incoming wave an
        :meth:`excitation` drives.
    entropy_coupling : complex or callable or tuple, optional
        Off-diagonal ``R_s``: arriving entropy -> reflected acoustic at an **outlet**
        (entropy noise).  Default ``0``.  Ignored at an inlet (entropy is to-specify).
    acoustic_to_entropy : complex or callable or tuple, optional
        Off-diagonal: arriving acoustic -> specified entropy at an **inlet**.  Default
        ``0``.  Ignored at an outlet.
    """

    kind: str = "inherit"
    R: object = None
    Z: object = None
    specific: bool = False
    amplitude: object = None
    base_R: object = None
    entropy_in: object = None
    family: str = "acoustic"
    entropy_coupling: object = None
    acoustic_to_entropy: object = None

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown perturbation BC kind {self.kind!r}; choose from {KINDS}")

    # -- evaluation on the frozen mean state --------------------------------

    def reflection_coefficient(self, freq, rho, c, M, K=None) -> Optional[complex]:
        """Acoustic reflection coefficient ``R`` at ``freq`` (Hz) and the terminal mean state.

        Returns ``None`` for ``inherit`` (signalling "do not stamp this terminal").
        ``M`` is the **outward-normal** mean Mach number at the terminal edge; ``K =
        cp/R`` is required only for ``choked_nozzle`` (``gamma - 1 = 1/(K-1)``).
        """
        k = self.kind
        if k == "inherit":
            return None
        if k == "hard_wall":
            return 1.0 + 0.0j
        if k == "open_end":
            return -1.0 + 0.0j
        if k == "mean_flow_open_end":
            return complex(-(1.0 - M) / (1.0 + M))
        if k == "anechoic":
            return 0.0 + 0.0j
        if k == "reflection":
            return _eval(self.R, freq)
        if k == "impedance":
            z = _eval(self.Z, freq)
            if self.specific:
                z = z * (rho * c)
            zc = rho * c
            return (z - zc) / (z + zc)
        if k == "excitation":
            return _eval(self.base_R, freq)
        if k == "choked_nozzle":  # compact choked outlet: delta_M = 0 (Marble--Candel)
            gm1 = _gamma_minus_one(K)
            return complex((2.0 - gm1 * M) / (2.0 + gm1 * M))
        raise ValueError(f"unhandled perturbation BC kind {k!r}")  # pragma: no cover

    def entropy_coupling_coefficient(self, freq, rho, c, M, K=None) -> complex:
        """Off-diagonal ``R_s``: arriving entropy -> reflected acoustic at an outlet.

        Zero unless ``choked_nozzle`` (``R_s = (c/rho) M/(2+(gamma-1)M)``) or an explicit
        ``entropy_coupling`` carrier is set.  ``M`` is the (outward-normal) mean Mach.
        """
        if self.kind == "choked_nozzle":
            gm1 = _gamma_minus_one(K)
            return complex((c / rho) * M / (2.0 + gm1 * M))
        if self.entropy_coupling is not None:
            return _eval(self.entropy_coupling, freq)
        return 0.0 + 0.0j

    def acoustic_to_entropy_coefficient(self, freq) -> complex:
        """Off-diagonal: arriving acoustic -> specified entropy at an inlet (0 unless set)."""
        if self.acoustic_to_entropy is not None:
            return _eval(self.acoustic_to_entropy, freq)
        return 0.0 + 0.0j

    def closure(self, freq, rho, c, u, M, K, specify, arriving):
        """Matrix closure ``(A, b)`` for one terminal at ``freq`` (Hz) and the mean state.

        ``specify`` / ``arriving`` are the to-specify / arriving characteristic indices
        from :func:`matrices.partition`.  Returns ``A`` of shape
        ``(len(specify), len(arriving))`` and forcing ``b`` of length ``len(specify)``
        such that ``w[specify] = A @ w[arriving] + b``.  The scalar reflection sits on
        the acoustic-acoustic entry; the off-diagonals carry entropy <-> acoustic coupling.
        """
        if self.kind == "choked_nozzle" and _H not in arriving:
            raise ValueError("choked_nozzle is an outlet termination (entropy must be an arriving wave)")
        spos = {ch: i for i, ch in enumerate(specify)}
        apos = {ch: i for i, ch in enumerate(arriving)}
        if (_F in spos) == (_G in spos):  # exactly one acoustic wave must be on each side
            raise ValueError("non-subsonic / degenerate terminal: acoustic waves not split one-each-way")
        A = np.zeros((len(specify), len(arriving)), dtype=complex)
        b = np.zeros(len(specify), dtype=complex)

        ac_spec = _F if _F in spos else _G  # the to-specify acoustic wave
        ac_arr = _G if ac_spec == _F else _F  # the arriving acoustic wave
        R = self.reflection_coefficient(freq, rho, c, M, K)
        A[spos[ac_spec], apos[ac_arr]] = R
        b[spos[ac_spec]] = self.forcing(freq)

        if _H in apos:  # outlet: arriving entropy can reflect into the specified acoustic wave
            A[spos[ac_spec], apos[_H]] = self.entropy_coupling_coefficient(freq, rho, c, M, K)
        if _H in spos:  # inlet: entropy is to-specify (seated, with an optional acoustic source term)
            b[spos[_H]] = self.entropy_forcing(freq)
            A[spos[_H], apos[ac_arr]] = self.acoustic_to_entropy_coefficient(freq)
        return A, b

    def forcing(self, freq) -> complex:
        """Acoustic-row excitation forcing ``b`` at ``freq`` (Hz) (0 unless excitation)."""
        if self.kind == "excitation" and self.family == "acoustic":
            return _eval(self.amplitude, freq)
        return 0.0 + 0.0j

    def entropy_forcing(self, freq) -> complex:
        """Incoming entropy amplitude seated at an inflow terminal, at ``freq`` (Hz)."""
        b = _eval(self.entropy_in, freq) if self.entropy_in is not None else 0.0 + 0.0j
        if self.kind == "excitation" and self.family == "entropy":
            b = b + _eval(self.amplitude, freq)
        return b

    @property
    def stamps_terminal(self) -> bool:
        """True if this BC overwrites the terminal row (everything but ``inherit``)."""
        return self.kind != "inherit"

    # -- named constructors -------------------------------------------------

    @classmethod
    def inherit(cls) -> "PerturbationBC":
        """Keep the linearized mean boundary row (the default)."""
        return cls("inherit")

    @classmethod
    def hard_wall(cls, entropy_in=None) -> "PerturbationBC":
        """Rigid wall, ``u' = 0`` (``R = +1``)."""
        return cls("hard_wall", entropy_in=entropy_in)

    @classmethod
    def open_end(cls, entropy_in=None) -> "PerturbationBC":
        """Ideal pressure-release open end, ``p' = 0`` (``R = -1``)."""
        return cls("open_end", entropy_in=entropy_in)

    @classmethod
    def mean_flow_open_end(cls, entropy_in=None) -> "PerturbationBC":
        """Convective open end, ``R = -(1 - M)/(1 + M)`` (``-1`` at ``M=0``)."""
        return cls("mean_flow_open_end", entropy_in=entropy_in)

    @classmethod
    def anechoic(cls, entropy_in=None) -> "PerturbationBC":
        """Reflection-free termination (``R = 0``)."""
        return cls("anechoic", entropy_in=entropy_in)

    @classmethod
    def reflection(cls, R, entropy_coupling=None, entropy_in=None) -> "PerturbationBC":
        """Prescribed reflection coefficient ``R`` (constant, table, or callable).

        ``entropy_coupling`` optionally adds the off-diagonal ``R_s`` (arriving entropy ->
        reflected acoustic) at an outlet, for a generic entropy-noise termination.
        """
        return cls("reflection", R=R, entropy_coupling=entropy_coupling, entropy_in=entropy_in)

    @classmethod
    def choked_nozzle(cls, entropy_in=None) -> "PerturbationBC":
        """Compact (low-frequency) choked-nozzle outlet -- Marble--Candel.

        Enforces ``delta_M = 0`` at the nozzle inlet (the duct outlet edge), so the
        reflected acoustic wave couples to the arriving entropy wave::

            g = R f + R_s h,
            R   = (2 - (gamma-1) M) / (2 + (gamma-1) M),
            R_s = (c/rho) M / (2 + (gamma-1) M),

        with ``M`` the outlet mean Mach.  At ``M -> 0`` this is a hard wall (``R = +1``,
        ``R_s = 0``).  Must terminate an outlet (entropy must be an arriving wave).
        """
        return cls("choked_nozzle", entropy_in=entropy_in)

    @classmethod
    def compact_nozzle(cls, entropy_in=None) -> "PerturbationBC":
        """Alias of :meth:`choked_nozzle`."""
        return cls.choked_nozzle(entropy_in=entropy_in)

    @classmethod
    def impedance(cls, Z, specific=False, entropy_in=None) -> "PerturbationBC":
        """Acoustic impedance ``Z`` (absolute Pa.s/m, or specific if ``specific``)."""
        return cls("impedance", Z=Z, specific=specific, entropy_in=entropy_in)

    @classmethod
    def impedance_polar(cls, magnitude, phase_deg=0.0, specific=True, entropy_in=None) -> "PerturbationBC":
        """Impedance from a magnitude and phase (degrees); specific (``Z/rho c``) by default.

        This is the closure the UI exposes: ``magnitude = 1, phase = 0`` is the matched
        (anechoic) termination, and the rigid-wall limit is ``magnitude -> inf``.
        """
        Z = float(magnitude) * cmath.exp(1j * math.radians(float(phase_deg)))
        return cls("impedance", Z=Z, specific=specific, entropy_in=entropy_in)

    @classmethod
    def excitation(cls, amplitude, family="acoustic", base_R=0.0, entropy_in=None) -> "PerturbationBC":
        """Drive an incoming wave with forcing ``amplitude`` on top of ``base_R``.

        ``family`` selects which incoming wave is driven (``"acoustic"`` or
        ``"entropy"``); ``base_R`` is the terminal's own reflection (default ``0`` --
        a clean source).
        """
        if family not in ("acoustic", "entropy"):
            raise ValueError(f"excitation family must be 'acoustic' or 'entropy'; got {family!r}")
        return cls("excitation", amplitude=amplitude, base_R=base_R, family=family, entropy_in=entropy_in)
