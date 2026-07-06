"""Public chemical-equilibrium entry points over the compiled kernel.

Element-potential (Lagrange-multiplier / CEA-style) equilibrium: the unknowns are element
potentials plus species moles, with element conservation as the constraint. HP (enthalpy,
pressure) and TP (temperature, pressure) solves are provided, each returning an
:class:`EquilibriumResult` with the derived mixture properties and both frozen and
equilibrium sound speeds.

These are thin wrappers over the single compiled engine in :mod:`nefes.thermo.kernel`
(``equilibrate_hp_cs`` / ``equilibrate_tp`` / ``equilibrium_sound_speed``); the network
solver calls the same kernel through :mod:`nefes.thermo.edge_state`.

Differentiation contract: the HP solve converges on the real parts, then splices the
imaginary part through the converged reduced matrix by the implicit-function theorem, so a
complex-step perturbation on ``h`` or ``p`` propagates exact derivatives. The TP path is
real-valued.

Algorithm reference: Gordon & McBride, "Computer Program for Calculation of Complex
Chemical Equilibrium Compositions and Applications", NASA RP-1311 (1994), Sections 2-3.

Public: :class:`EquilibriumResult`, :func:`equilibrate_HP`, :func:`equilibrate_TP`,
:func:`elemental_abundance`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .properties import mixture_properties
from .kernel import equilibrate_hp_cs, equilibrate_tp, equilibrium_sound_speed

__all__ = ["EquilibriumResult", "equilibrate_TP", "equilibrate_HP", "elemental_abundance"]


@dataclass
class EquilibriumResult:
    """Outcome of an equilibrium solve."""

    T: float
    p: float
    Y: np.ndarray  # mass fractions (full mechanism ordering)
    X: np.ndarray  # mole fractions
    n: np.ndarray  # moles per kg of mixture [mol/kg]
    rho: float
    properties: object  # MixtureState, including a_equilibrium
    iterations: int
    converged: bool

    @property
    def a_equilibrium(self):
        return self.properties.a_equilibrium

    @property
    def a_frozen(self):
        return self.properties.a_frozen


def elemental_abundance(lib, Z_elem):
    """Convert elemental *mass fractions* to gram-atoms per kg, ``b``.

    ``Z_elem`` may be a dict ``{element: mass_fraction}`` or an array aligned to
    ``lib.elements``.  Returns ``b`` with ``b[i] = Z_i / W_element_i``.
    """
    if isinstance(Z_elem, dict):
        Z = np.array([Z_elem.get(e, 0.0) for e in lib.elements])
    else:
        Z = np.asarray(Z_elem)
    return Z / lib.element_weights


def _product_subset(lib):
    """Gas-phase product arrays ``(prod_idx, coeffs, Tint, Af)`` the kernel solves over.

    Condensed feed species (e.g. a liquid fuel) carry their atoms into ``b`` but never
    appear as equilibrium products; an all-gas library leaves this the full set.  Mirrors
    the product slicing in :func:`nefes.thermo.edge_state.pack_equilibrium`.
    """
    coeffs, Tint = lib.nasa9_arrays()
    A = np.ascontiguousarray(lib.element_matrix, dtype=np.float64)
    prod_mask = np.asarray(getattr(lib, "product_mask", np.ones(lib.n_species, bool)), dtype=bool)
    prod_idx = np.nonzero(prod_mask)[0]
    prod_coeffs = np.ascontiguousarray(coeffs[prod_idx], dtype=np.float64)
    prod_Tint = np.ascontiguousarray(Tint[prod_idx], dtype=np.float64)
    prod_A = np.ascontiguousarray(A[:, prod_idx], dtype=np.float64)
    return prod_idx, prod_coeffs, prod_Tint, prod_A


def _finalize(lib, prod_idx, coeffs, Tint, Af, nj_prod, ntot, T, p, iterations, converged):
    """Assemble an :class:`EquilibriumResult` from the kernel's product-space solution."""
    n_full = np.zeros(lib.n_species, dtype=nj_prod.dtype)
    n_full[prod_idx] = nj_prod
    mass = n_full * lib.molar_masses
    Y = mass / np.sum(mass)
    props = mixture_properties(lib, Y, T, p)
    props.a_equilibrium = equilibrium_sound_speed(coeffs, Tint, Af, nj_prod, ntot, T, p)
    return EquilibriumResult(
        T=T,
        p=p,
        Y=Y,
        X=props.X,
        n=n_full,
        rho=props.rho,
        properties=props,
        iterations=iterations,
        converged=converged,
    )


def _cold_start(b0, Np):
    """Uniform gram-atom guess spread over the ``Np`` product species (real float64)."""
    return np.full(Np, float(np.sum(np.real(b0))) / (2.0 * Np))


def equilibrate_HP(lib, Z_elem, h, p, T_guess=2000.0):
    """Constant-enthalpy, constant-pressure (HP) chemical equilibrium.

    Given the elemental composition, the mixture specific enthalpy and the pressure, solve
    for the equilibrium temperature, density, and species composition. Complex-step in ``h``
    or ``p`` propagates exact derivatives (the imaginary part is spliced through the
    converged reduced system by the implicit-function theorem).

    Parameters
    ----------
    lib : SpeciesLibrary or Mechanism
        The species data (NASA polynomials and the element matrix).
    Z_elem : dict or numpy.ndarray
        Elemental *mass* fractions, either ``{element: fraction}`` or an array aligned to
        ``lib.elements``. May be complex for a complex-step through the composition.
    h : float or complex
        Mixture specific enthalpy [J/kg] (absolute, formation-inclusive datum).
    p : float or complex
        Pressure [Pa].
    T_guess : float, optional
        Initial temperature [K] for the Newton iteration (default 2000 K).

    Returns
    -------
    EquilibriumResult
        The converged temperature, density, mass/mole fractions, moles, and the derived
        :class:`MixtureState` (including the frozen and equilibrium sound speeds).
    """
    prod_idx, coeffs, Tint, Af = _product_subset(lib)
    b0 = elemental_abundance(lib, Z_elem)
    nj_init = _cold_start(b0, prod_idx.size)

    if np.iscomplexobj(b0) or isinstance(h, complex) or isinstance(p, complex):
        b0 = np.asarray(b0, dtype=np.complex128)
        nj_init = nj_init.astype(np.complex128)
        h = complex(h)
        p = complex(p)
    else:
        b0 = np.ascontiguousarray(b0, dtype=np.float64)

    T, nj_prod, ntot, flag, nit = equilibrate_hp_cs(coeffs, Tint, Af, b0, h, p, lib.P_ref, float(T_guess), nj_init)
    return _finalize(lib, prod_idx, coeffs, Tint, Af, nj_prod, ntot, T, p, nit, flag == 1)


def equilibrate_TP(lib, Z_elem, T, p):
    """Constant-temperature, constant-pressure (TP) chemical equilibrium.

    The fixed-temperature companion of :func:`equilibrate_HP`: the element potentials and
    species moles are solved at the prescribed ``T``. Real-valued (the TP path is not
    complex-stepped).

    Parameters
    ----------
    lib : SpeciesLibrary or Mechanism
        The species data (NASA polynomials and the element matrix).
    Z_elem : dict or numpy.ndarray
        Elemental *mass* fractions, either ``{element: fraction}`` or an array aligned to
        ``lib.elements``.
    T : float
        Temperature [K].
    p : float
        Pressure [Pa].

    Returns
    -------
    EquilibriumResult
        The equilibrium density, mass/mole fractions, moles, and the derived
        :class:`MixtureState` (including the frozen and equilibrium sound speeds) at ``T``.
    """
    prod_idx, coeffs, Tint, Af = _product_subset(lib)
    b0 = np.ascontiguousarray(np.real(elemental_abundance(lib, Z_elem)), dtype=np.float64)
    nj = _cold_start(b0, prod_idx.size)
    Tr, pr = float(np.real(T)), float(np.real(p))
    ntot, flag, nit = equilibrate_tp(coeffs, Tint, Af, b0, Tr, pr, lib.P_ref, nj)
    return _finalize(lib, prod_idx, coeffs, Tint, Af, nj, ntot, Tr, pr, nit, flag == 1)
