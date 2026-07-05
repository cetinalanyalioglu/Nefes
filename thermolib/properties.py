"""Mixture thermodynamic properties from species polynomials.

Mixture properties are composed from the per-species values. The frozen speed of sound
lives here; the equilibrium speed of sound lives in :mod:`thermolib.equilibrium`.

All functions are complex-analytic in ``T``, ``p`` and the composition vector ``Y`` (mass
fractions): they use only ``+ - * /``, integer powers, ``log`` and ``sqrt`` with a
positive real part, so complex-step differentiation propagates through them. Mole
fractions are floored inside ``log`` by a tiny constant to keep ``0 * log 0`` well defined
without branching.

Public: :class:`MixtureState`, :func:`mixture_properties`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import R_UNIVERSAL

__all__ = ["MixtureState", "mixture_properties"]

_X_FLOOR = 1e-300  # keeps log finite for absent species; contribution ~ Y*log ~ 0


@dataclass
class MixtureState:
    """Bundle of mixture properties at a thermodynamic point.

    Attributes are SI, mass-specific where applicable:
    ``T`` [K], ``p`` [Pa], ``rho`` [kg/m^3], ``W`` mean molar mass [kg/mol],
    ``cp``/``cv`` [J/(kg K)], ``gamma`` [-], ``h`` [J/kg], ``s`` [J/(kg K)],
    ``a_frozen`` frozen sound speed [m/s].  ``a_equilibrium`` is attached by the
    equilibrium solver when available.
    """

    T: float
    p: float
    Y: np.ndarray
    X: np.ndarray
    W: float
    rho: float
    cp: float
    cv: float
    gamma: float
    h: float
    s: float
    a_frozen: float
    a_equilibrium: float = None


def mixture_properties(lib, Y, T, p):
    """Compute mixture properties for mass fractions ``Y`` at ``(T, p)``.

    ``lib`` is a :class:`~thermolib.species.SpeciesLibrary` (or a ``Mechanism``,
    which proxies the same thermo interface).  ``Y`` need not be normalized; it is
    renormalized internally so the result is well defined for slightly off-sum
    inputs (e.g. mid-solve compositions).
    """
    Y = np.asarray(Y)
    Wk = lib.molar_masses
    Ysum = np.sum(Y)
    Yn = Y / Ysum

    inv_W = np.sum(Yn / Wk)
    W = 1.0 / inv_W  # mean molar mass [kg/mol]
    X = (Yn / Wk) * W  # mole fractions
    R_spec = R_UNIVERSAL / W  # specific gas constant [J/(kg K)]
    rho = p * W / (R_UNIVERSAL * T)  # ideal gas

    cpR = lib.cp_R(T)  # dimensionless, per species
    hRT = lib.h_RT(T)
    sR = lib.s_R(T)

    # Mass-specific mixture properties (species mass-specific value = dimensionless
    # * R / W_k, then Y-weighted).
    cp = R_UNIVERSAL * np.sum(Yn * cpR / Wk)
    h = R_UNIVERSAL * T * np.sum(Yn * hRT / Wk)

    # Entropy includes the ideal mixing term and pressure correction:
    #   s_k = (R/W_k) * (s_R_k - ln(X_k) - ln(p/P_ref))
    s = R_UNIVERSAL * np.sum(Yn / Wk * (sR - np.log(X + _X_FLOOR) - np.log(p / lib.P_ref)))

    cv = cp - R_spec
    gamma = cp / cv
    a_frozen = np.sqrt(gamma * R_spec * T)  # = sqrt(gamma * p / rho)

    return MixtureState(
        T=T,
        p=p,
        Y=Yn,
        X=X,
        W=W,
        rho=rho,
        cp=cp,
        cv=cv,
        gamma=gamma,
        h=h,
        s=s,
        a_frozen=a_frozen,
    )
