"""Reaction sets and the :class:`Mechanism` (species library + reactions).

The thermochemical *material database* is a :class:`~thermolib.species.SpeciesLibrary`, a
set of species with their NASA polynomials, and all that chemical equilibrium needs. A
**mechanism** is the *combination* of such a library with a set of **reactions** whose
participants refer to species in that library. Reactions are needed only for the
finite-rate path and for the shared-Gibbs ``K_c`` route; equilibrium and mixture
properties never need them.

Reaction stoichiometry and modified-Arrhenius data are carried; the native format is a
Cantera-YAML subset that round-trips, and an offline Cantera importer is provided.
``Mechanism`` proxies the library's thermo interface so it can be passed anywhere a
:class:`SpeciesLibrary` is expected.

Public: :class:`Reaction`, :class:`Mechanism`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import yaml

from .species import (
    SpeciesLibrary,
    _cantera_solution,
    _parse_native_doc,
)

__all__ = ["Reaction", "Mechanism"]


@dataclass
class Reaction:
    """Elementary reaction with modified-Arrhenius parameters.

    Stored for the finite-rate path. The rate machinery itself is not implemented; only the
    data and the shared ``K_c`` route (via species Gibbs energies) are wired.
    """

    reactants: dict  # species name -> stoichiometric coefficient
    products: dict
    A: float  # pre-exponential [units per mechanism convention]
    b: float  # temperature exponent
    Ea: float  # activation energy [J/mol]
    reversible: bool = True
    equation: str = ""


@dataclass
class Mechanism:
    """A species library together with a set of reactions over its species.

    ``library`` carries the species/thermo data (and the element matrix used by
    the equilibrium kernel); ``reactions`` is the kinetic data.  Thermo and
    sizing attributes are proxied to the library, so a ``Mechanism`` is accepted
    anywhere a :class:`SpeciesLibrary` is.
    """

    library: SpeciesLibrary
    reactions: list = field(default_factory=list)

    # -- proxy the library's thermo / sizing surface ---------------------
    @property
    def elements(self):
        return self.library.elements

    @property
    def species(self):
        return self.library.species

    @property
    def species_index(self):
        return self.library.species_index

    @property
    def element_index(self):
        return self.library.element_index

    @property
    def molar_masses(self):
        return self.library.molar_masses

    @property
    def element_matrix(self):
        return self.library.element_matrix

    @property
    def element_weights(self):
        return self.library.element_weights

    @property
    def P_ref(self):
        return self.library.P_ref

    @property
    def n_species(self):
        return self.library.n_species

    @property
    def n_elements(self):
        return self.library.n_elements

    def cp_R(self, T):
        return self.library.cp_R(T)

    def h_RT(self, T):
        return self.library.h_RT(T)

    def s_R(self, T):
        return self.library.s_R(T)

    def g_RT(self, T):
        return self.library.g_RT(T)

    # -- loaders ---------------------------------------------------------
    @classmethod
    def from_native(cls, path):
        """Load a native mechanism (species library + reactions)."""
        with open(path, "r") as fh:
            doc = yaml.safe_load(fh)
        return cls.from_dict(doc)

    @classmethod
    def from_dict(cls, doc):
        """Build a mechanism from an already-parsed native-YAML document."""
        elements, species, raw_reactions = _parse_native_doc(doc)
        library = SpeciesLibrary(elements=elements, species=species)
        reactions = [_reaction_from_dict(r) for r in raw_reactions]
        return cls(library=library, reactions=reactions)

    @classmethod
    def from_cantera(cls, source, phase_name=None):
        """Offline importer from a full Cantera mechanism."""
        gas = _cantera_solution(source)
        library = SpeciesLibrary.from_cantera(gas)
        reactions = [_reaction_from_cantera(rxn) for rxn in gas.reactions()]
        return cls(library=library, reactions=reactions)

    # -- writer ----------------------------------------------------------
    def to_native_dict(self):
        """Serialize to a native-YAML-compatible dict (round-trips with :meth:`from_dict`)."""
        doc = self.library.to_native_dict()
        if self.reactions:
            doc["reactions"] = [
                {
                    "equation": r.equation or _format_equation(r),
                    "rate-constant": {"A": r.A, "b": r.b, "Ea": r.Ea},
                }
                for r in self.reactions
            ]
        return doc

    def write_native(self, path):
        with open(path, "w") as fh:
            yaml.safe_dump(self.to_native_dict(), fh, sort_keys=False)


def _reaction_from_cantera(rxn):
    rate = getattr(rxn, "rate", None)
    A = getattr(rate, "pre_exponential_factor", float("nan")) if rate else float("nan")
    b = getattr(rate, "temperature_exponent", 0.0) if rate else 0.0
    # Cantera activation_energy is J/kmol -> J/mol.
    Ea = (getattr(rate, "activation_energy", 0.0) if rate else 0.0) * 1e-3
    return Reaction(
        reactants=dict(rxn.reactants),
        products=dict(rxn.products),
        A=A,
        b=b,
        Ea=Ea,
        reversible=rxn.reversible,
        equation=rxn.equation,
    )


def _reaction_from_dict(r):
    eq = r.get("equation", "")
    reactants, products = _parse_equation(eq) if eq else ({}, {})
    rc = r.get("rate-constant", {}) or {}
    return Reaction(
        reactants=reactants,
        products=products,
        A=rc.get("A", float("nan")),
        b=rc.get("b", 0.0),
        Ea=rc.get("Ea", 0.0),
        reversible="<=>" in eq or ("=" in eq and "=>" not in eq),
        equation=eq,
    )


def _parse_equation(eq):
    """Parse a reaction equation into reactant/product stoichiometry dicts."""
    for arrow in ("<=>", "=>", "="):
        if arrow in eq:
            lhs, rhs = eq.split(arrow)
            break
    else:
        return {}, {}

    def side(text):
        out = {}
        for term in text.split("+"):
            term = term.strip()
            if not term or term.upper() == "M":
                continue
            coeff = 1.0
            k = 0
            while k < len(term) and (term[k].isdigit() or term[k] == "."):
                k += 1
            if k > 0:
                coeff = float(term[:k])
                term = term[k:].strip()
            out[term] = out.get(term, 0.0) + coeff
        return out

    return side(lhs), side(rhs)


def _format_equation(r):
    """Render a reaction's stoichiometry as an equation string."""

    def side(d):
        return " + ".join(sp if nu == 1 else f"{nu:g} {sp}" for sp, nu in d.items())

    arrow = " <=> " if r.reversible else " => "
    return side(r.reactants) + arrow + side(r.products)
