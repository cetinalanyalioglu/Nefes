"""Species thermodynamic data and the :class:`SpeciesLibrary` container.

A **species library** is the thermochemical *material database*: a set of chemical
species, each with an element composition, a molar mass and a NASA polynomial for its
standard-state ``cp/h/s/g`` as a function of temperature. It carries **no reactions**;
chemical equilibrium needs only the per-species thermodynamics (the element-potential
formulation). Reactions live in a :class:`thermolib.mechanism.Mechanism`, which
*associates* a species library with a reaction set (the term "mechanism" is reserved for
that combination).

Libraries load from Cantera's YAML format through a single :meth:`SpeciesLibrary.from_cantera`
that adapts to its input: a file path (or parsed ``dict``) is read directly with no Cantera
dependency, supporting the subset of the format thermolib needs, while a live
``cantera.Solution`` is extracted through Cantera itself. Per-species ``cp,h,s,g(T)`` are
evaluated complex-analytically in ``T``.

All species share one **canonical 9-term NASA representation** and are evaluated in a
single vectorized expression over ``(n_species, 9)`` coefficient arrays, with no
per-species Python loop. Temperature-interval selection branches only on ``Re(T)``
("locate on the real part"), so the whole evaluation stays complex-step differentiable.

Public: :class:`ThermoPoly`, :func:`NASA7`, :func:`NASA9`, :class:`Species`,
:class:`SpeciesLibrary`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import yaml

from .constants import P_REF
from .elements import atomic_weight

__all__ = ["ThermoPoly", "NASA7", "NASA9", "Species", "SpeciesLibrary"]


# ---------------------------------------------------------------------------
# Canonical NASA-9 polynomial (subsumes NASA-7)
# ---------------------------------------------------------------------------
#
# Per interval, coefficients are stored as ``[a1..a7, b1, b2]`` and define the
# dimensionless standard-state properties (NASA Glenn 9-term form):
#
#   cp/R = a1 T^-2 + a2 T^-1 + a3 + a4 T + a5 T^2 + a6 T^3 + a7 T^4
#   h/RT = -a1 T^-2 + a2 ln(T)/T + a3 + a4 T/2 + a5 T^2/3 + a6 T^3/4
#          + a7 T^4/5 + b1/T
#   s/R  = -a1 T^-2/2 - a2 T^-1 + a3 ln(T) + a4 T + a5 T^2/2 + a6 T^3/3
#          + a7 T^4/4 + b2
#
# A 7-term NASA polynomial is the special case a1 = a2 = 0 with
# (a3..a7, b1, b2) = (c0..c4, c5, c6); see :func:`NASA7`.


def _eval_nasa9(A, T):
    """Evaluate ``(cp/R, h/RT, s/R)`` for coefficient rows ``A`` at scalar ``T``.

    ``A`` has shape ``(..., 9)``; the result tuple matches the leading shape.
    Uses only complex-analytic operations, so complex-step ``T`` propagates.
    """
    Tinv = 1.0 / T
    Tinv2 = Tinv * Tinv
    lnT = np.log(T)
    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T

    a1, a2, a3, a4 = A[..., 0], A[..., 1], A[..., 2], A[..., 3]
    a5, a6, a7, b1, b2 = A[..., 4], A[..., 5], A[..., 6], A[..., 7], A[..., 8]

    cp_R = a1 * Tinv2 + a2 * Tinv + a3 + a4 * T + a5 * T2 + a6 * T3 + a7 * T4
    h_RT = -a1 * Tinv2 + a2 * lnT * Tinv + a3 + a4 * T / 2.0 + a5 * T2 / 3.0 + a6 * T3 / 4.0 + a7 * T4 / 5.0 + b1 * Tinv
    s_R = -a1 * Tinv2 / 2.0 - a2 * Tinv + a3 * lnT + a4 * T + a5 * T2 / 2.0 + a6 * T3 / 3.0 + a7 * T4 / 4.0 + b2
    return cp_R, h_RT, s_R


@dataclass
class ThermoPoly:
    """A piecewise NASA polynomial in canonical 9-term form.

    ``Tranges`` are the ``n_intervals + 1`` temperature breakpoints (ascending);
    ``coeffs`` has shape ``(n_intervals, 9)``.  ``kind`` records the source form
    (``"NASA7"`` or ``"NASA9"``) so serialization can round-trip it faithfully.
    """

    Tranges: np.ndarray  # (n_intervals + 1,)
    coeffs: np.ndarray  # (n_intervals, 9)
    kind: str = "NASA9"

    def __post_init__(self):
        self.Tranges = np.asarray(self.Tranges, float)
        self.coeffs = np.asarray(self.coeffs, float)

    @property
    def T_low(self):
        return float(self.Tranges[0])

    @property
    def T_high(self):
        return float(self.Tranges[-1])

    def _row(self, T):
        # Branch on the real part only -> complex-step safe.
        interior = self.Tranges[1:-1]
        k = int(np.sum(np.asarray(interior, float) <= np.real(T)))
        return self.coeffs[k]

    def cp_R(self, T):
        return _eval_nasa9(self._row(T), T)[0]

    def h_RT(self, T):
        return _eval_nasa9(self._row(T), T)[1]

    def s_R(self, T):
        return _eval_nasa9(self._row(T), T)[2]

    def g_RT(self, T):
        cp, h, s = _eval_nasa9(self._row(T), T)
        return h - s


def _nasa7_to9(c):
    """Embed 7 NASA-7 coefficients into the canonical 9-term row."""
    c = np.asarray(c, float)
    a = np.zeros(9)
    a[2:9] = c[:7]  # a3..a7 = c0..c4 ; b1 = c5 ; b2 = c6 (a1 = a2 = 0)
    return a


def _nasa9_to7(a):
    """Recover the 7 NASA-7 coefficients from a canonical 9-term row."""
    return np.asarray(a, float)[2:9].copy()


def NASA7(T_low, T_mid, T_high, coeffs_low, coeffs_high):
    """Construct a :class:`ThermoPoly` from a NASA-7 low/high coefficient pair."""
    return ThermoPoly(
        Tranges=[T_low, T_mid, T_high],
        coeffs=np.array([_nasa7_to9(coeffs_low), _nasa7_to9(coeffs_high)]),
        kind="NASA7",
    )


def NASA9(Tranges, coeffs):
    """Construct a :class:`ThermoPoly` from NASA-9 ranges and per-interval rows."""
    return ThermoPoly(Tranges=Tranges, coeffs=coeffs, kind="NASA9")


# ---------------------------------------------------------------------------
# Species
# ---------------------------------------------------------------------------
@dataclass
class Species:
    """A single chemical species: composition, thermo polynomial, molar mass."""

    name: str
    composition: dict  # element symbol -> atom count
    thermo: ThermoPoly
    molar_mass: float = 0.0  # kg/mol; computed from composition if zero
    note: str = ""  # provenance/comment (e.g. CEA reference code)
    phase: int = 0  # 0 = gas, non-zero = condensed (CEA phase flag)

    def __post_init__(self):
        if not self.molar_mass:
            self.molar_mass = sum(n * atomic_weight(el) for el, n in self.composition.items())

    # Per-species thermo (convenience; the library vectorizes over all species).
    def cp_R(self, T):
        return self.thermo.cp_R(T)

    def h_RT(self, T):
        return self.thermo.h_RT(T)

    def s_R(self, T):
        return self.thermo.s_R(T)

    def g_RT(self, T):
        return self.thermo.g_RT(T)


# ---------------------------------------------------------------------------
# SpeciesLibrary
# ---------------------------------------------------------------------------
@dataclass
class SpeciesLibrary:
    """An ordered set of species with their thermodynamic data (no reactions).

    The element matrix ``element_matrix[i, j]`` is the number of atoms of element ``i`` in
    species ``j``, exactly the constraint matrix of the element-potential equilibrium
    formulation. ``P_ref`` is the standard-state pressure the polynomials are referenced to
    (one atm for the Cantera/YAML path, one bar for the NASA Glenn ``thermo.inp`` path).
    """

    elements: list
    species: list
    P_ref: float = P_REF

    def __post_init__(self):
        self.species_index = {s.name: j for j, s in enumerate(self.species)}
        self.element_index = {e: i for i, e in enumerate(self.elements)}
        self.molar_masses = np.array([s.molar_mass for s in self.species])

        E, S = len(self.elements), len(self.species)
        a = np.zeros((E, S))
        for j, s in enumerate(self.species):
            for el, n in s.composition.items():
                a[self.element_index[el], j] = n
        self.element_matrix = a
        self.element_weights = np.array([atomic_weight(e) for e in self.elements])

        # Product mask: which species may appear as equilibrium products.  Only gaseous
        # species (CEA phase 0) are kept; condensed species are feed-only (they set elements
        # and enthalpy but their polynomials are not valid at flame temperatures).
        self.product_mask = np.array([getattr(s, "phase", 0) == 0 for s in self.species], dtype=bool)

        self._pack_thermo()

    @property
    def species_names(self):
        """Species names in library order."""
        return [s.name for s in self.species]

    # -- vectorized thermo packing --------------------------------------
    def _pack_thermo(self):
        """Pack all species coefficients into dense arrays for one-shot eval."""
        polys = [s.thermo for s in self.species]
        S = len(polys)
        n_int = np.array([p.coeffs.shape[0] for p in polys]) if S else np.array([])
        max_int = int(n_int.max()) if S else 1

        coeffs = np.zeros((S, max_int, 9))
        for j, p in enumerate(polys):
            coeffs[j, : p.coeffs.shape[0], :] = p.coeffs
        self._coeffs = coeffs

        # Interior breakpoints (per species), padded with +inf so the padded
        # slots never count toward the interval index.
        max_interior = max(0, max_int - 1)
        if max_interior:
            Tint = np.full((S, max_interior), np.inf)
            for j, p in enumerate(polys):
                interior = p.Tranges[1:-1]
                Tint[j, : interior.size] = interior
            self._Tint = Tint
        else:
            self._Tint = None
        self._arangeS = np.arange(S)

    def _rows(self, T):
        """Select the active coefficient row for every species at scalar ``T``."""
        if self._Tint is None:
            k = np.zeros(len(self.species), dtype=int)
        else:
            k = np.sum(self._Tint <= np.real(T), axis=1)
        return self._coeffs[self._arangeS, k]  # (n_species, 9)

    def nasa9_arrays(self):
        """Dense NASA-9 arrays for a compiled consumer: ``(coeffs, Tint)``.

        ``coeffs`` is ``(n_species, max_intervals, 9)`` (zero-padded for species with fewer
        intervals); ``Tint`` is ``(n_species, max_intervals - 1)`` of interior breakpoints,
        padded with ``+inf`` so an unused slot never advances the interval index. The active
        interval for species ``j`` at temperature ``T`` is ``sum(Tint[j] <= T)``, the same
        locate-on-real rule the pure-numpy path uses. Provided so a compiled (e.g. numba)
        consumer evaluates the identical thermodynamics from flat arrays without importing
        this package's Python objects.
        """
        coeffs = np.ascontiguousarray(self._coeffs, dtype=float)
        if self._Tint is None:
            Tint = np.empty((self.n_species, 0), dtype=float)
        else:
            Tint = np.ascontiguousarray(self._Tint, dtype=float)
        return coeffs, Tint

    # -- sizes -----------------------------------------------------------
    @property
    def n_species(self):
        return len(self.species)

    @property
    def n_elements(self):
        return len(self.elements)

    # -- vectorized species thermo (complex-safe) ------------------------
    def cp_R(self, T):
        return _eval_nasa9(self._rows(T), T)[0]

    def h_RT(self, T):
        return _eval_nasa9(self._rows(T), T)[1]

    def s_R(self, T):
        return _eval_nasa9(self._rows(T), T)[2]

    def g_RT(self, T):
        cp, h, s = _eval_nasa9(self._rows(T), T)
        return h - s

    # -- subsetting ------------------------------------------------------
    def subset(self, names):
        """Return a new library restricted to ``names`` (order preserved)."""
        chosen = [self.species[self.species_index[n]] for n in names]
        used = sorted({el for s in chosen for el in s.composition}, key=lambda e: self.element_index.get(e, 1 << 30))
        return SpeciesLibrary(elements=used, species=chosen, P_ref=self.P_ref)

    # -- loaders ---------------------------------------------------------
    @classmethod
    def from_cantera(cls, source):
        """Build a library from Cantera data, adapting to the input type.

        ``source`` may be a Cantera-YAML file path (``str``/``os.PathLike``), an
        already-parsed ``dict`` of such a document, or a live ``cantera.Solution``.  Paths
        and dicts are read directly with no Cantera dependency and honour the subset of the
        format thermolib needs (NASA7/NASA9 thermo, no transport); a ``cantera.Solution`` is
        extracted through Cantera itself, so passing one requires Cantera to be installed.

        The direct-parse routes never touch the runtime/equilibrium code path.
        """
        if isinstance(source, (str, os.PathLike)):
            with open(source, "r") as fh:
                source = yaml.safe_load(fh)
        if isinstance(source, dict):
            return cls.from_dict(source)
        # A live cantera.Solution: extract species and thermo through Cantera.
        species = [_species_from_cantera(source, name) for name in source.species_names]
        return cls(elements=list(source.element_names), species=species)

    @classmethod
    def from_dict(cls, doc):
        """Build a library from an already-parsed Cantera-YAML document."""
        elements, species, _ = _parse_cantera_doc(doc)
        return cls(elements=elements, species=species)

    @classmethod
    def from_cea(cls, path=None, species=None, P_ref=None):
        """Build a library from a NASA Glenn / CEA ``thermo.inp`` file.

        ``path`` defaults to the packaged ``thermo.inp`` (so the database need not be
        named). ``species`` selects a subset by name (recommended, as ``thermo.inp`` holds
        ~2000 species); ``None`` loads every gaseous record. ``P_ref`` defaults to one bar
        (the database's standard state).
        """
        from .cea import ThermoInp

        return ThermoInp(path).library(species, P_ref=P_ref)

    # -- writer ----------------------------------------------------------
    def to_cantera_dict(self):
        """Serialize to a Cantera-YAML-compatible dict (round-trips with :meth:`from_dict`)."""
        return {
            "phases": [
                {
                    "name": "gas",
                    "thermo": "ideal-gas",
                    "elements": list(self.elements),
                    "species": [s.name for s in self.species],
                }
            ],
            "species": [_species_to_dict(s) for s in self.species],
        }

    def write_cantera_yaml(self, path):
        with open(path, "w") as fh:
            yaml.safe_dump(self.to_cantera_dict(), fh, sort_keys=False)


# ---------------------------------------------------------------------------
# Shared (de)serialization helpers used by both SpeciesLibrary and Mechanism
# ---------------------------------------------------------------------------
def _parse_cantera_doc(doc):
    """Return ``(elements, species, raw_reactions)`` from a Cantera-YAML doc."""
    phase = doc.get("phases", [{}])[0] if doc.get("phases") else {}
    species_by_name = {sp["name"]: _species_from_dict(sp) for sp in doc.get("species", [])}

    names = phase.get("species") or list(species_by_name.keys())
    species = [species_by_name[n] for n in names]

    if phase.get("elements"):
        elements = list(phase["elements"])
    else:
        elements = sorted({el for s in species for el in s.composition})

    return elements, species, doc.get("reactions", [])


def _species_from_dict(sp):
    """Build a :class:`Species` from a Cantera-YAML species block."""
    th = sp["thermo"]
    model = th.get("model", "NASA7")
    ranges = th["temperature-ranges"]
    data = th["data"]

    if model == "NASA7":
        if len(ranges) == 3 and len(data) == 2:
            thermo = NASA7(ranges[0], ranges[1], ranges[2], data[0], data[1])
        elif len(ranges) == 2 and len(data) == 1:
            # Single-range NASA7: duplicate so both pieces are identical.
            thermo = NASA7(ranges[0], ranges[1], ranges[1], data[0], data[0])
        else:
            raise ValueError(f"Species {sp['name']!r}: unsupported NASA7 range layout.")
    elif model == "NASA9":
        if len(data) != len(ranges) - 1:
            raise ValueError(f"Species {sp['name']!r}: NASA9 needs len(data) == " f"len(temperature-ranges) - 1.")
        thermo = NASA9(ranges, data)
    else:
        raise ValueError(f"Species {sp['name']!r}: thermolib reads NASA7/NASA9 thermo " f"only, got {model!r}.")

    return Species(name=sp["name"], composition=dict(sp["composition"]), thermo=thermo, note=sp.get("note", ""))


def _species_to_dict(s):
    """Serialize a :class:`Species` to a Cantera-YAML species block."""
    p = s.thermo
    if p.kind == "NASA7":
        thermo = {
            "model": "NASA7",
            "temperature-ranges": [float(t) for t in p.Tranges],
            "data": [[float(c) for c in _nasa9_to7(row)] for row in p.coeffs],
        }
    else:
        thermo = {
            "model": "NASA9",
            "temperature-ranges": [float(t) for t in p.Tranges],
            "data": [[float(c) for c in row] for row in p.coeffs],
        }
    out = {"name": s.name, "composition": dict(s.composition), "thermo": thermo}
    if s.note:
        out["note"] = s.note
    return out


def _species_from_cantera(gas, name):
    csp = gas.species(name)
    coeffs = csp.thermo.coeffs
    # Cantera packs NASA7 coeffs as [Tmid, 7 high-T, 7 low-T].
    thermo = NASA7(csp.thermo.min_temp, coeffs[0], csp.thermo.max_temp, coeffs[8:15], coeffs[1:8])
    return Species(
        name=name,
        composition={k: int(round(v)) if float(v).is_integer() else v for k, v in csp.composition.items()},
        thermo=thermo,
        # Cantera molecular weights are kg/kmol -> kg/mol.
        molar_mass=gas.molecular_weights[gas.species_index(name)] * 1e-3,
    )
