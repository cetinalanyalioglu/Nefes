"""Reader for the NASA Glenn / CEA ``thermo.inp`` species database (R-A2.1).

``thermo.inp`` is the canonical NASA-9 thermodynamic database used by CEA
(McBride & Gordon).  It holds ~2000 species in a fixed-column FORTRAN format.
This module gives an *easy interface* over it: parse once, search by name, and
``select`` the handful of species you actually need into a
:class:`~thermolib.species.SpeciesLibrary` that the equilibrium/property code
consumes directly.

    from thermolib import ThermoInp, Thermo
    db   = ThermoInp("data/thermo.inp")
    db.search("H2O")                       # -> ['H2O', 'H2O2', 'H2O(cr)', ...]
    lib  = db.library(["H2", "O2", "H2O", "OH", "H", "O", "N2"])
    gas  = Thermo(lib)                      # equilibrium, properties, ...

Record layout (per species), mirroring CEA's ``thermo.inp`` reader:

* line 1 -- ``name`` then a free-text reference/comment;
* line 2 -- interval count, code, up to five ``element count`` pairs, a phase
  flag, the molar mass [g/mol] and the formation enthalpy;
* then, per interval, three lines: an interval header (``T_lo T_hi n_coef`` and
  the term exponents) followed by two coefficient lines in FORTRAN ``D``
  exponent notation.

The exponents are the standard NASA-9 set ``[-2,-1,0,1,2,3,4]``; coefficients
are stored as the canonical 9-term row ``[a1..a7, b1, b2]``.
"""

from __future__ import annotations

import os

import numpy as np

from .constants import P_REF_BAR
from .elements import normalize_element
from .species import NASA9, Species, SpeciesLibrary

__all__ = ["ThermoInp", "read_thermo_inp"]

# Column slices for line 2 element pairs (symbol at n:n+2, count at n+2:n+8).
_ELEMENT_COLS = (10, 18, 26, 34, 42)


def _f(text):
    """Parse a FORTRAN float, accepting the ``D`` exponent marker."""
    return float(text.strip().replace("D", "E").replace("d", "e"))


def _parse_record_2(line):
    n_intervals = int(line[0:2])
    composition = {}
    for n in _ELEMENT_COLS:
        sym = line[n : n + 2].strip()
        if sym and sym[0].isalpha():
            count = _f(line[n + 2 : n + 8])
            if count:
                composition[normalize_element(sym)] = int(count) if float(count).is_integer() else count
    molar_mass_g = _f(line[52:65])  # g/mol
    return n_intervals, composition, molar_mass_g


def _parse_interval(lines):
    """Return ``(T_lo, T_hi, coeffs9)`` for one 3-line interval block."""
    T_lo = _f(lines[0][0:11])
    T_hi = _f(lines[0][11:22])
    n_coef = int(lines[0][22])
    if n_coef != 7:  # pragma: no cover - all standard records use 7 terms
        raise ValueError(f"thermo.inp: unsupported n_coef={n_coef} (expected 7)")
    vals = [
        _f(lines[1][0:16]),
        _f(lines[1][16:32]),
        _f(lines[1][32:48]),
        _f(lines[1][48:64]),
        _f(lines[1][64:80]),
        _f(lines[2][0:16]),
        _f(lines[2][16:32]),  # a6, a7
        _f(lines[2][48:64]),
        _f(lines[2][64:80]),  # b1, b2
    ]
    return T_lo, T_hi, np.array(vals, float)


def read_thermo_inp(path):
    """Parse ``thermo.inp`` into an ordered ``{name: Species}`` dict.

    Single-point records (interval count 0) and records outside the standard
    7-term layout are skipped; everything evaluable over a range is kept,
    gaseous and condensed alike (the phase flag is preserved in the note).
    """
    with open(path, "r") as fh:
        lines = fh.readlines()

    # Find the data start: the line that is exactly "thermo", then skip the
    # global temperature-range header line that follows it.
    start = 0
    for i, line in enumerate(lines):
        if line.strip().lower() == "thermo":
            start = i + 2
            break

    out = {}
    i = start
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()
        if not stripped or line[0] in "!#" or not line[0].isalpha():
            i += 1
            continue
        if stripped.upper().startswith("END"):
            i += 1
            continue

        name = line.split()[0]
        comment = line[len(name) :].strip()
        n_intervals, composition, molar_mass_g = _parse_record_2(lines[i + 1])

        if n_intervals < 1:
            # Reference-only / single-point record: not evaluable over a range.
            i += 3
            continue

        Tranges = []
        coeffs = []
        for k in range(n_intervals):
            blk = lines[i + 2 + 3 * k : i + 5 + 3 * k]
            T_lo, T_hi, c9 = _parse_interval(blk)
            if not Tranges:
                Tranges.append(T_lo)
            Tranges.append(T_hi)
            coeffs.append(c9)

        out[name] = Species(
            name=name,
            composition=composition,
            thermo=NASA9(Tranges, np.array(coeffs)),
            molar_mass=molar_mass_g * 1e-3,  # g/mol -> kg/mol
            note=comment,
        )
        i += 2 + 3 * n_intervals

    return out


class ThermoInp:
    """An easy, searchable handle on a parsed ``thermo.inp`` database."""

    def __init__(self, path="thermo.inp"):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"thermo.inp not found: {path!r}")
        self.path = path
        self.species = read_thermo_inp(path)

    def __contains__(self, name):
        return name in self.species

    def __getitem__(self, name):
        return self.species[name]

    def __len__(self):
        return len(self.species)

    @property
    def names(self):
        return list(self.species)

    def search(self, substring, case_sensitive=False):
        """Return species names containing ``substring``."""
        if case_sensitive:
            return [n for n in self.species if substring in n]
        s = substring.lower()
        return [n for n in self.species if s in n.lower()]

    def library(self, names=None, P_ref=None):
        """Build a :class:`SpeciesLibrary` from ``names`` (all if ``None``)."""
        if names is None:
            chosen = list(self.species.values())
        else:
            missing = [n for n in names if n not in self.species]
            if missing:
                raise KeyError(f"species not in {os.path.basename(self.path)}: {missing}")
            chosen = [self.species[n] for n in names]

        elements = []
        for sp in chosen:
            for el in sp.composition:
                if el not in elements:
                    elements.append(el)
        return SpeciesLibrary(
            elements=elements,
            species=chosen,
            P_ref=P_REF_BAR if P_ref is None else P_ref,
        )
