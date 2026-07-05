"""Perturbation (acoustic) boundary conditions for single-port nodes.

A single-port element fixes the *mean* boundary condition (a mass flow, a total
pressure, a static pressure, or a wall).  The *perturbation* boundary condition is
extra information the mean BC cannot supply: how the terminal closes the linear
fluctuation problem.  Every terminal BC takes the form of a **characteristic closure**
mapping the waves that *arrive* at the boundary from the interior to the waves the
boundary must *specify* (those propagating into the domain)::

    w_specify = A(omega) @ w_arriving + b(omega)

The mean state partitions the three characteristics ``w = (f, g, h)`` into the
``arriving`` set (speeds pointing out of the domain) and the ``specify`` set (speeds
pointing in) -- see :func:`matrices.partition`.  For a subsonic terminal that is a
duct **tail** (an inlet) the flow carries ``f`` and ``h`` in and lets ``g`` out, so
``A`` is ``2 x 1``; at a duct **head** (an outlet) only ``g`` enters while ``f`` and
``h`` leave, so ``A`` is ``1 x 2``.  ``b`` is the optional forcing -- an incoming wave
injected at the boundary (see ``driven`` below).  Transported reacting scalars (composition
waves) are further convected characteristics beyond ``(f, g, h)``: the reflection closures
here act on the acoustic and entropy waves only, and a scalar wave is at most *seated* at a
genuine inflow (via ``driven``), never reflected.

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
``choked_nozzle``    ``g = R f + R_s h``          compact choked outlet (Marble--Candel):
                                                  ``R = (2-(g-1)M)/(2+(g-1)M)``,
                                                  ``R_s = (c/rho) M/(2+(g-1)M)``
===================  =========================================================

**Forcing is orthogonal to the reflection.**  Any closure may additionally *drive* one
or more incoming waves: ``driven`` is the set of wave families injected at the boundary
(``"acoustic"`` and/or ``"entropy"``), and ``amplitudes`` optionally sets their complex
amplitudes (a marked-but-unspecified family drives a *unit* wave).  Because the forcing
``b`` is stamped on the to-specify rows independently of ``A`` (see :meth:`closure`),
a drive composes with *any* reflection -- e.g. ``mean_flow_open_end(driven=("acoustic",))``
keeps the convective-neutral ``R`` and adds a unit incoming acoustic wave.  The
``"acoustic"`` family is always to-specify (exactly one acoustic wave enters); the
``"entropy"`` family is drivable only at an inflow terminal (where entropy is to-specify).

Off-diagonal coupling is also available on the generic closures: ``entropy_coupling``
sets the arriving-entropy -> reflected-acoustic term ``R_s`` at an outlet, and
``acoustic_to_entropy`` sets the arriving-acoustic -> specified-entropy term at an inlet.
This can be used to model e.g. response of a specific nozzle to incident entropy waves.

The impedance map uses the **outward-normal** velocity convention, so it is uniform
at an inlet and an outlet: a rigid wall ``Z -> inf`` gives ``R = +1``, a
pressure-release end ``Z -> 0`` gives ``R = -1``, and the matched impedance
``Z = rho c`` gives ``R = 0``.

Each numeric carrier (``R``, ``Z``, ``entropy_in``, an ``amplitudes`` value) may be a complex
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
    "choked_nozzle",
    "constant_mass_flow",
)

# Wave families that the closure can inject at a boundary; reacting-scalar waves
# (named by ``prob.scalar_names``) are not carried through the terminal closure.
DRIVABLE_FAMILIES = ("acoustic", "entropy")

# Characteristic indices: f downstream, g upstream, h entropy.
_F, _G, _H = 0, 1, 2


def _eval(value, freq):
    """Evaluate a coefficient at frequency ``freq`` (Hz).

    ``value`` is a complex constant, a callable ``freq -> complex``, or a frequency
    table ``(freqs, values)`` linearly interpolated (real and imaginary parts) in
    ``freq`` and held flat outside its range.

    A raw ``(freqs, values)`` table is a real-grid interpolant -- fine for a real-frequency
    sweep (the forced response, the Nyquist driver), but **not analytic**, so it cannot be
    evaluated at the complex frequencies the stability eigenproblem visits.  Asking for a
    complex ``freq`` raises a pointed error; wrap the table with
    :func:`~nefes.perturbation.continuation.rational_fit` to get an analytically-continuable
    coefficient usable for stability.
    """
    if value is None:
        return 0.0 + 0.0j
    if callable(value):
        return complex(value(freq))
    if isinstance(value, tuple) and len(value) == 2:
        fc = complex(freq)
        if abs(fc.imag) > 1e-12 * (abs(fc.real) + 1.0):
            raise TypeError(
                "a tabulated reflection/impedance table (freqs, values) cannot be evaluated at a "
                "complex frequency (real-grid interpolation is not analytic). Use it for the forced "
                "response / Nyquist sweep, or wrap it with nefes.perturbation.rational_fit(freqs, values) "
                "for the stability eigenproblem."
            )
        xs, ys = np.asarray(value[0], dtype=float), np.asarray(value[1], dtype=complex)
        re = np.interp(fc.real, xs, ys.real)
        im = np.interp(fc.real, xs, ys.imag)
        return complex(re, im)
    return complex(value)


def _gamma_minus_one(rho, c, p):
    """Effective ``gamma - 1`` for the Marble--Candel coefficients, from the mean state.

    The local isentropic exponent ``gamma = rho c^2 / p`` is exact for a perfect gas and is
    the value consistent with the sound speed the acoustics already use for a reacting /
    variable-composition gas, so ``gamma - 1 = rho c^2 / p - 1`` is read straight from the
    terminal's own mean state -- no gas-model constant.  Shared by
    :meth:`PerturbationBC.reflection_coefficient` and
    :meth:`PerturbationBC.entropy_coupling_coefficient`.
    """
    if p is None:
        raise ValueError("choked_nozzle needs the mean static pressure p for the effective gamma = rho c^2 / p")
    return float(rho) * float(c) * float(c) / float(p) - 1.0


@dataclass
class PerturbationBC:
    """Acoustic/perturbation closure for a single-port terminal node.

    Build one with the named constructors (:meth:`hard_wall`, :meth:`open_end`,
    :meth:`anechoic`, :meth:`reflection`, :meth:`impedance`, :meth:`mean_flow_open_end`);
    :meth:`inherit` (the default) leaves the linearized mean boundary row untouched.
    Any of these accepts ``driven``/``amplitudes`` to additionally inject incoming waves.

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
    entropy_in : complex or callable or tuple, optional
        Incoming entropy-wave amplitude seated at an inflow terminal (default ``0``).
        Shorthand for ``driven=("entropy",)`` with this amplitude; the two add if both set.
    entropy_coupling : complex or callable or tuple, optional
        Off-diagonal ``R_s``: arriving entropy -> reflected acoustic at an **outlet**
        (entropy noise).  Default ``0``.  Ignored at an inlet (entropy is to-specify).
    acoustic_to_entropy : complex or callable or tuple, optional
        Off-diagonal: arriving acoustic -> specified entropy at an **inlet**.  Default
        ``0``.  Ignored at an outlet.
    driven : tuple of str
        Wave families injected at this boundary.  ``"acoustic"`` and ``"entropy"`` (the
        :data:`DRIVABLE_FAMILIES`) plus, at a genuine inflow, any transported reacting scalar
        named in ``prob.scalar_names`` (a composition / equivalence-ratio wave) -- scalar names
        are resolved at stamp time, since the BC has no problem context yet.  Empty (the default)
        is a passive, unforced terminal.  A family listed here drives a **unit** incoming wave
        unless ``amplitudes`` overrides it.  A driven scalar convects and *does* radiate sound
        wherever the linearization is inherited (a flame, an area change, a resolved or inherited
        compact nozzle -- the full Jacobian carries composition -> acoustic).  The one place that
        coupling is dropped is a hand-written compact-nozzle closure (see
        :class:`~nefes.perturbation.response.forced.CompositionalNoiseWarning`).
    amplitudes : dict, optional
        Per-family complex amplitude (constant, table, or callable), keyed by a family in
        ``driven``.  Families in ``driven`` but absent here drive a unit wave.
    """

    kind: str = "inherit"
    R: object = None
    Z: object = None
    specific: bool = False
    entropy_in: object = None
    entropy_coupling: object = None
    acoustic_to_entropy: object = None
    driven: tuple = ()
    amplitudes: object = None

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown perturbation BC kind {self.kind!r}; choose from {KINDS}")
        self.driven = tuple(self.driven) if self.driven else ()
        for fam in self.driven:
            if not isinstance(fam, str):
                raise TypeError(f"driven families must be strings; got {fam!r}")
        # "acoustic"/"entropy" are validated here; a reacting-scalar family (any other name) is
        # resolved against prob.scalar_names at stamp time, since the BC has no problem context yet.
        if len(set(self.driven)) != len(self.driven):
            raise ValueError(f"duplicate family in driven={self.driven!r}")
        if self.amplitudes is not None:
            extra = set(self.amplitudes) - set(self.driven)
            if extra:
                raise ValueError(
                    f"amplitudes given for families {sorted(extra)} not in driven={self.driven!r}; "
                    "every amplitude key must be a driven family"
                )

    # -- evaluation on the frozen mean state --------------------------------

    def reflection_coefficient(self, freq, rho, c, M, p=None) -> Optional[complex]:
        """Acoustic reflection coefficient ``R`` at ``freq`` (Hz) and the terminal mean state.

        Returns ``None`` for ``inherit`` (signalling "do not stamp this terminal").  ``M`` is
        the **outward-normal** mean Mach number at the terminal edge.  The effective ``gamma``
        for ``choked_nozzle`` is taken from the mean state (``gamma = rho c^2 / p``), so ``p``
        (the mean static pressure) must be supplied for that kind -- correct for any thermo
        backend (see :func:`_gamma_minus_one`).
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
        if k == "choked_nozzle":  # compact choked outlet: delta_M = 0 (Marble--Candel)
            gm1 = _gamma_minus_one(rho, c, p)
            return complex((2.0 - gm1 * M) / (2.0 + gm1 * M))
        if k == "constant_mass_flow":  # mdot' = 0: g = (1+M)/(1-M) f + R_s h
            return complex((1.0 + M) / (1.0 - M))
        raise ValueError(f"unhandled perturbation BC kind {k!r}")  # pragma: no cover

    def entropy_coupling_coefficient(self, freq, rho, c, M, p=None) -> complex:
        """Off-diagonal ``R_s``: arriving entropy -> reflected acoustic at an outlet.

        Zero unless ``choked_nozzle`` (``R_s = (c/rho) M/(2+(gamma-1)M)``) or an explicit
        ``entropy_coupling`` carrier is set.  ``M`` is the (outward-normal) mean Mach; the
        effective ``gamma`` is taken from the mean state (``gamma = rho c^2 / p``).
        """
        if self.kind == "choked_nozzle":
            gm1 = _gamma_minus_one(rho, c, p)
            return complex((c / rho) * M / (2.0 + gm1 * M))
        if self.kind == "constant_mass_flow":  # mdot' = 0 -> R_s = u / (rho (1 - M)) = c M / (rho (1 - M))
            return complex((c * M) / (rho * (1.0 - M)))
        if self.entropy_coupling is not None:
            return _eval(self.entropy_coupling, freq)
        return 0.0 + 0.0j

    def acoustic_to_entropy_coefficient(self, freq) -> complex:
        """Off-diagonal: arriving acoustic -> specified entropy at an inlet (0 unless set)."""
        if self.acoustic_to_entropy is not None:
            return _eval(self.acoustic_to_entropy, freq)
        return 0.0 + 0.0j

    def closure(self, freq, rho, c, u, M, specify, arriving, p=None):
        """Matrix closure ``(A, b)`` for one terminal at ``freq`` (Hz) and the mean state.

        The closure is the linear law ``w[specify] = A @ w[arriving] + b``: the waves the
        boundary must specify (those entering the domain) as a reflection ``A`` of the waves
        arriving from the interior, plus an injected forcing ``b``.  The scalar reflection
        sits on the acoustic-acoustic entry; the off-diagonals carry entropy <-> acoustic
        coupling.

        Parameters
        ----------
        freq : float
            Frequency (Hz) at which the coefficients are evaluated.
        rho, c, u : float
            Mean density, sound speed, and axial velocity at the terminal edge.
        M : float
            Outward-normal mean Mach number at the terminal.
        specify, arriving : sequence of int
            To-specify and arriving characteristic indices from :func:`matrices.partition`.
        p : float, optional
            Mean static pressure; the ``choked_nozzle`` coefficients take the effective
            ``gamma = rho c^2 / p`` from it (required for that kind), consistent with any
            thermo backend.

        Returns
        -------
        A : ndarray
            Reflection matrix, shape ``(len(specify), len(arriving))``.
        b : ndarray
            Forcing vector, length ``len(specify)``.
        """
        if self.kind in ("choked_nozzle", "constant_mass_flow") and _H not in arriving:
            raise ValueError(f"{self.kind} is an outlet termination (entropy must be an arriving wave)")
        spos = {ch: i for i, ch in enumerate(specify)}
        apos = {ch: i for i, ch in enumerate(arriving)}
        if (_F in spos) == (_G in spos):  # exactly one acoustic wave must be on each side
            raise ValueError("non-subsonic / degenerate terminal: acoustic waves not split one-each-way")
        if "entropy" in self.driven and _H not in spos:  # entropy drivable only where it is to-specify
            raise ValueError(
                "cannot drive the 'entropy' wave at this terminal: entropy is an arriving (outgoing) "
                "characteristic here -- drive it only at an inflow terminal"
            )
        A = np.zeros((len(specify), len(arriving)), dtype=complex)
        b = np.zeros(len(specify), dtype=complex)

        ac_spec = _F if _F in spos else _G  # the to-specify acoustic wave
        ac_arr = _G if ac_spec == _F else _F  # the arriving acoustic wave
        R = self.reflection_coefficient(freq, rho, c, M, p)
        A[spos[ac_spec], apos[ac_arr]] = R
        b[spos[ac_spec]] = self.forcing(freq)

        if _H in apos:  # outlet: arriving entropy can reflect into the specified acoustic wave
            A[spos[ac_spec], apos[_H]] = self.entropy_coupling_coefficient(freq, rho, c, M, p)
        if _H in spos:  # inlet: entropy is to-specify (seated, with an optional acoustic source term)
            b[spos[_H]] = self.entropy_forcing(freq)
            A[spos[_H], apos[ac_arr]] = self.acoustic_to_entropy_coefficient(freq)
        return A, b

    def _drive_amplitude(self, family, freq) -> complex:
        """Forcing amplitude of a ``driven`` ``family`` at ``freq`` (Hz); 0 if not driven.

        A family marked in ``driven`` but absent from ``amplitudes`` drives a **unit** wave.
        """
        if family not in self.driven:
            return 0.0 + 0.0j
        amps = self.amplitudes or {}
        if family in amps:
            return _eval(amps[family], freq)
        return 1.0 + 0.0j

    def forcing(self, freq) -> complex:
        """Acoustic-row forcing ``b`` at ``freq`` (Hz): the driven acoustic wave (0 if none)."""
        return self._drive_amplitude("acoustic", freq)

    def entropy_forcing(self, freq) -> complex:
        """Entropy-row forcing at ``freq`` (Hz): the seated ``entropy_in`` plus any driven entropy wave."""
        b = _eval(self.entropy_in, freq) if self.entropy_in is not None else 0.0 + 0.0j
        return b + self._drive_amplitude("entropy", freq)

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
    def hard_wall(cls, entropy_in=None, driven=(), amplitudes=None) -> "PerturbationBC":
        """Rigid wall, ``u' = 0`` (``R = +1``)."""
        return cls("hard_wall", entropy_in=entropy_in, driven=driven, amplitudes=amplitudes)

    @classmethod
    def open_end(cls, entropy_in=None, driven=(), amplitudes=None) -> "PerturbationBC":
        """Ideal pressure-release open end, ``p' = 0`` (``R = -1``)."""
        return cls("open_end", entropy_in=entropy_in, driven=driven, amplitudes=amplitudes)

    @classmethod
    def mean_flow_open_end(cls, entropy_in=None, driven=(), amplitudes=None) -> "PerturbationBC":
        """Convective open end, ``R = -(1 - M)/(1 + M)`` (``-1`` at ``M=0``)."""
        return cls("mean_flow_open_end", entropy_in=entropy_in, driven=driven, amplitudes=amplitudes)

    @classmethod
    def anechoic(cls, entropy_in=None, driven=(), amplitudes=None) -> "PerturbationBC":
        """Reflection-free termination (``R = 0``)."""
        return cls("anechoic", entropy_in=entropy_in, driven=driven, amplitudes=amplitudes)

    @classmethod
    def reflection(cls, R, entropy_coupling=None, entropy_in=None, driven=(), amplitudes=None) -> "PerturbationBC":
        """Prescribed reflection coefficient ``R`` (constant, table, or callable).

        ``entropy_coupling`` optionally adds the off-diagonal ``R_s`` (arriving entropy ->
        reflected acoustic) at an outlet, for a generic entropy-noise termination.
        """
        return cls(
            "reflection",
            R=R,
            entropy_coupling=entropy_coupling,
            entropy_in=entropy_in,
            driven=driven,
            amplitudes=amplitudes,
        )

    @classmethod
    def choked_nozzle(cls, entropy_in=None, driven=(), amplitudes=None) -> "PerturbationBC":
        """Compact (low-frequency) choked-nozzle outlet -- Marble--Candel.

        Enforces ``delta_M = 0`` at the nozzle inlet (the duct outlet edge), so the
        reflected acoustic wave couples to the arriving entropy wave::

            g = R f + R_s h,
            R   = (2 - (gamma-1) M) / (2 + (gamma-1) M),
            R_s = (c/rho) M / (2 + (gamma-1) M),

        with ``M`` the outlet mean Mach.  At ``M -> 0`` this is a hard wall (``R = +1``,
        ``R_s = 0``).  Must terminate an outlet (entropy must be an arriving wave).
        """
        return cls("choked_nozzle", entropy_in=entropy_in, driven=driven, amplitudes=amplitudes)

    @classmethod
    def compact_nozzle(cls, entropy_in=None, driven=(), amplitudes=None) -> "PerturbationBC":
        """Alias of :meth:`choked_nozzle`."""
        return cls.choked_nozzle(entropy_in=entropy_in, driven=driven, amplitudes=amplitudes)

    @classmethod
    def constant_mass_flow(cls, entropy_in=None, driven=(), amplitudes=None) -> "PerturbationBC":
        """Constant-mass-flow outlet termination -- pins ``mdot' = 0`` acoustically.

        The linearization of a fixed-mass-flow boundary: the unsteady mass flux vanishes,
        ``A(u rho' + rho u') = 0``, which in characteristics is::

            g = R f + R_s h,
            R   = (1 + M) / (1 - M),
            R_s = u / (rho (1 - M)) = c M / (rho (1 - M)),

        with ``M`` the outlet mean Mach.  At ``M -> 0`` this is a hard wall (``R = +1``,
        ``R_s = 0``).  This is the closure a :func:`~nefes.elements.catalog.mass_flow_outlet`
        inherits automatically; use it to impose a constant-mass-flow termination on any
        other outlet (e.g. a choked upstream injector seen from downstream).  Must
        terminate an outlet (entropy must be an arriving wave).
        """
        return cls("constant_mass_flow", entropy_in=entropy_in, driven=driven, amplitudes=amplitudes)

    @classmethod
    def impedance(cls, Z, specific=False, entropy_in=None, driven=(), amplitudes=None) -> "PerturbationBC":
        """Acoustic impedance ``Z`` (absolute Pa.s/m, or specific if ``specific``)."""
        return cls("impedance", Z=Z, specific=specific, entropy_in=entropy_in, driven=driven, amplitudes=amplitudes)

    @classmethod
    def impedance_polar(
        cls, magnitude, phase_deg=0.0, specific=True, entropy_in=None, driven=(), amplitudes=None
    ) -> "PerturbationBC":
        """Impedance from a magnitude and phase (degrees); specific (``Z/rho c``) by default.

        This is the closure the UI exposes: ``magnitude = 1, phase = 0`` is the matched
        (anechoic) termination, and the rigid-wall limit is ``magnitude -> inf``.
        """
        Z = float(magnitude) * cmath.exp(1j * math.radians(float(phase_deg)))
        return cls("impedance", Z=Z, specific=specific, entropy_in=entropy_in, driven=driven, amplitudes=amplitudes)
