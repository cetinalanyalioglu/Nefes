"""The public ``Thermo`` facade -- the uniform, backend-selectable API (A.9).

Network-agnostic: inputs and outputs are purely thermodynamic
(composition, ``T``/``h``, ``p``, derived properties) per R-A1.3.

A ``Thermo`` is built from a :class:`~thermolib.species.SpeciesLibrary` -- that
is all equilibrium and mixture properties need.  Passing a
:class:`~thermolib.mechanism.Mechanism` (a library *plus* reactions) is also
accepted; the reactions then enable the shared-Gibbs ``K_c`` route and the
finite-rate design hook.
"""

from __future__ import annotations

import numpy as np

from .backends import make_backend
from .constants import R_UNIVERSAL

__all__ = ["Thermo"]


class Thermo:
    """Uniform thermochemistry interface over a selectable backend.

    Example::

        lib  = SpeciesLibrary.from_native("h2o2.yaml")   # or ThermoInp(...).library(...)
        gas  = Thermo(lib, backend="kernel")             # Backend D (native kernel)
        props = gas.properties(Y, T, p)
        eq    = gas.equilibrate_HP(Z_elem, h, p)
    """

    def __init__(self, source, backend="kernel"):
        # Accept a SpeciesLibrary or a Mechanism (library + reactions).
        self.library = getattr(source, "library", source)
        self.reactions = getattr(source, "reactions", None)
        self.backend_name = backend
        self.backend = make_backend(backend, self.library)

    @property
    def mech(self):  # backward-compatible alias
        return self.library

    # -- properties ------------------------------------------------------
    def properties(self, Y, T, p):
        """Mixture properties at ``(Y, T, p)``: cp, cv, gamma, h, s, rho,
        a_frozen, ... (R-A3.2, R-A3.4 frozen part).
        """
        return self.backend.properties(Y, T, p)

    # -- equilibrium -----------------------------------------------------
    def equilibrate_HP(self, Z_elem, h, p, **kw):
        """HP equilibrium -> T, rho, Y, a_equilibrium, ... (R-A4.1)."""
        return self.backend.equilibrate_HP(Z_elem, h, p, **kw)

    def equilibrate_TP(self, Z_elem, T, p, **kw):
        """TP equilibrium (validation/reuse, R-A4.3)."""
        return self.backend.equilibrate_TP(Z_elem, T, p, **kw)

    # -- composition helpers (thermodynamic, network-agnostic) -----------
    def elemental_mass_fractions(self, Y):
        """Elemental mass fractions ``Z`` from species mass fractions ``Y``.

        ``Z_i = W_i * sum_j (a_ij Y_j / W_j)``.  Lets a consumer obtain the
        first-class elemental descriptor (D-2) from a species state.
        """
        Y = np.asarray(Y)
        Yn = Y / np.sum(Y)
        gram_atoms = self.library.element_matrix @ (Yn / self.library.molar_masses)
        Z = self.library.element_weights * gram_atoms
        return Z / np.sum(Z)

    def enthalpy_mass(self, Y, T):
        """Mixture specific enthalpy [J/kg] at ``(Y, T)`` (datum: D-1 absolute,
        formation-inclusive, as carried by the NASA polynomials)."""
        Y = np.asarray(Y)
        Yn = Y / np.sum(Y)
        return R_UNIVERSAL * T * np.sum(Yn * self.library.h_RT(T) / self.library.molar_masses)

    # -- finite-rate design hook (secondary target, A.5) -----------------
    def _require_reactions(self):
        if not self.reactions:
            raise ValueError(
                "this Thermo was built from a SpeciesLibrary with no reactions; "
                "reaction data (a Mechanism) is required for kinetic quantities. "
                "Build with Thermo(Mechanism.from_native(...)) or .from_cantera(...)."
            )
        return self.reactions

    def equilibrium_constants_Kc(self, T):
        """Concentration equilibrium constants ``K_c(T)`` per reaction.

        Derived from the *same* species Gibbs energies used by the equilibrium
        solver (detailed balance, R-A5.2 / R-A3.3), guaranteeing that a future
        finite-rate model relaxes exactly to this equilibrium model as t->inf.
        Provided now so the reverse-rate route is wired even though
        :meth:`net_rates` itself is a design hook in the MVP.
        """
        reactions = self._require_reactions()
        gRT = self.library.g_RT(T)  # dimensionless standard Gibbs
        idx = self.library.species_index
        # K_p = exp(-sum nu_j g_RT_j) (in p/P_ref); concentration form
        # K_c = K_p * (P_ref/(R T))^(dnu), in SI concentration units [mol/m^3].
        c0 = self.library.P_ref / (R_UNIVERSAL * T)
        Kc = []
        for rxn in reactions:
            dnu_g = 0.0
            dnu = 0.0
            for sp, nu in rxn.products.items():
                if sp in idx:
                    dnu_g += nu * gRT[idx[sp]]
                    dnu += nu
            for sp, nu in rxn.reactants.items():
                if sp in idx:
                    dnu_g -= nu * gRT[idx[sp]]
                    dnu -= nu
            Kp = np.exp(-dnu_g)
            Kc.append(Kp * c0**dnu)
        return np.array(Kc)

    def net_rates(self, Y, T, p):
        """Net molar production rates ``wdot(T, p, Y)`` -- *design hook only*.

        REQUIREMENTS A.5 (secondary target): the architecture is structured so
        that complex-analytic ``net_rates`` and their derivatives can be added
        later, with reverse rates from :meth:`equilibrium_constants_Kc` (R-A5.2).
        The forward/reverse rate assembly is intentionally not implemented in
        this equilibrium-focused MVP.
        """
        raise NotImplementedError(
            "net_rates is a forward-compatibility design hook (REQUIREMENTS A.5, "
            "secondary target). The MVP implements chemical equilibrium only. "
            "Reaction data and the shared K_c route (equilibrium_constants_Kc) "
            "are wired so this can be completed without an architectural change."
        )
