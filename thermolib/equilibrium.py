"""Native CEA-style chemical-equilibrium kernel.

Implements the **element-potential (Lagrange-multiplier / CEA-style)** formulation:
unknowns are element potentials plus species moles, with element conservation as the
constraint. Provides HP equilibrium and TP equilibrium, plus the **equilibrium speed of
sound** from the converged sensitivity block.

Differentiation contract: the solve converges on the *real* parts of the inputs, then, if
any input is complex, takes one undamped Newton step in log-variables from the converged
real state with the full complex inputs. Because the real residual is zero there, that
step is exactly the implicit-function-theorem sensitivity, so a complex-step perturbation
on ``h``, ``p`` or the composition propagates exact derivatives through the solve.

Algorithm reference: Gordon & McBride, "Computer Program for Calculation of Complex
Chemical Equilibrium Compositions and Applications", NASA RP-1311 (1994), Sections 2-3
(all-gas reduced equations and damping).

Public: :class:`EquilibriumResult`, :func:`equilibrate_HP`, :func:`equilibrate_TP`,
:func:`elemental_abundance`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import R_UNIVERSAL
from .properties import mixture_properties

__all__ = ["EquilibriumResult", "equilibrate_TP", "equilibrate_HP", "elemental_abundance"]

_TRACE_LN = -18.420681  # ln(1e-8): trace-species threshold (RP-1311 eq 3.2)
_TRACE_CTRL = 9.2103404  # = -ln(1e-4)


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


def _reduced_step(a, n, nt, T, p, gRT, hRT, cpR, b, h_target, P_ref):
    """Build and solve the reduced CEA system; return the log-corrections.

    ``a`` is the (E, S) element matrix of the *active* subset.  Returns
    ``(dln_nj, dln_n, dln_T)`` where ``dln_T`` is ``None`` for a TP solve
    (``h_target is None``).
    """
    E, S = a.shape
    hp = h_target is not None
    muRT = gRT + np.log(n / nt) + np.log(p / P_ref)

    an = a @ n  # (E,)  sum_j a_ij n_j
    n_mu = n * muRT
    A = np.zeros(
        (E + 1 + (1 if hp else 0),) * 2, dtype=np.result_type(n, nt, T, complex if np.iscomplexobj(n) else float)
    )
    rhs = np.zeros(A.shape[0], dtype=A.dtype)

    # Element block.
    A[:E, :E] = a @ (n[:, None] * a.T)  # sum_j a_ij a_kj n_j
    A[:E, E] = an
    A[E, :E] = an
    A[E, E] = np.sum(n) - nt
    rhs[:E] = b - an + a @ n_mu
    rhs[E] = nt - np.sum(n) + np.sum(n_mu)

    if hp:
        n_h = n * hRT
        ah = a @ n_h
        A[:E, E + 1] = ah
        A[E, E + 1] = np.sum(n_h)
        A[E + 1, :E] = ah
        A[E + 1, E] = np.sum(n_h)
        A[E + 1, E + 1] = np.sum(n * cpR) + np.sum(n_h * hRT)
        rhs[E + 1] = h_target / (R_UNIVERSAL * T) - np.sum(n_h) + np.sum(n_h * muRT)

    sol = np.linalg.solve(A, rhs)
    pi = sol[:E]
    dln_n = sol[E]
    dln_T = sol[E + 1] if hp else None

    dln_nj = a.T @ pi + dln_n - muRT
    if hp:
        dln_nj = dln_nj + hRT * dln_T
    return dln_nj, dln_n, dln_T


def _damping(n, nt, dln_nj, dln_n, dln_T):
    """CEA control factor limiting the size of a log-step (RP-1311 eq 3.1-3.2)."""
    a_nj = np.abs(np.real(dln_nj))
    ctrl = max(
        5.0 * abs(np.real(dln_n)),
        float(np.max(a_nj)) if a_nj.size else 0.0,
        abs(np.real(dln_T)) if dln_T is not None else 0.0,
    )
    lam = 1.0 if ctrl <= 2.0 else 2.0 / ctrl

    ln_ratio = np.real(np.log(n / nt))
    trace = (ln_ratio <= _TRACE_LN) & (np.real(dln_nj) > 0.0)
    if np.any(trace):
        num = -ln_ratio[trace] - _TRACE_CTRL
        den = np.real(dln_nj)[trace] - np.real(dln_n)
        lam2 = np.min(np.abs(num / den))
        lam = min(lam, lam2)
    return lam


def _solve_subset(lib, b, p, h_target, T_init, max_iter=400, tol=1e-11):
    """Core solver on the element/species subset with nonzero abundance.

    Returns ``(n_full, T, iterations, converged, active_species, a_sub)``.
    """
    P_ref = lib.P_ref
    afull = lib.element_matrix
    E_full, S_full = afull.shape
    b_real = np.real(b)
    bscale = np.max(b_real) if b_real.size else 1.0

    keep_el = np.where(b_real > 1e-13 * bscale)[0]
    drop_el = np.setdiff1d(np.arange(E_full), keep_el)
    # Keep species that contain no atoms of any dropped element.
    if drop_el.size:
        has_dropped = np.any(np.real(afull[drop_el, :]) != 0.0, axis=0)
    else:
        has_dropped = np.zeros(S_full, bool)
    # Restrict the active set to product-eligible species (gas phase in v1).  Condensed
    # feed species (e.g. liquid fuel) carry their atoms into ``b`` and their enthalpy into
    # ``h_target`` but never appear as products; an all-gas library leaves this a no-op.
    product = np.asarray(getattr(lib, "product_mask", np.ones(S_full, bool)), bool)
    keep_sp = np.where(~has_dropped & product)[0]

    a = afull[np.ix_(keep_el, keep_sp)]
    bsub = b[keep_el]
    Ssub = keep_sp.size

    # Initial guess: spread the available gram-atoms over the active species.
    n0 = (np.sum(b_real[keep_el]) / (2.0 * Ssub)) * np.ones(Ssub)
    n = n0.astype(np.result_type(b, float))
    nt = np.sum(n)
    T = T_init

    hp = h_target is not None
    # Phase 1: converge on the real parts.
    pr = np.real(p)
    br = np.real(bsub)
    hr = np.real(h_target) if hp else None

    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        gRT = lib.g_RT(T)[keep_sp]
        hRT = lib.h_RT(T)[keep_sp]
        cpR = lib.cp_R(T)[keep_sp]
        dln_nj, dln_n, dln_T = _reduced_step(a, n, nt, T, pr, gRT, hRT, cpR, br, hr, P_ref)
        lam = _damping(n, nt, dln_nj, dln_n, dln_T)
        n = n * np.exp(lam * dln_nj)
        nt = nt * np.exp(lam * dln_n)
        if hp:
            T = T * np.exp(lam * dln_T)

        resid = (
            float(np.max(np.abs(np.real(n) * np.real(dln_nj))) / np.real(nt))
            + abs(np.real(dln_n))
            + (abs(np.real(dln_T)) if hp else 0.0)
        )
        if resid < tol:
            converged = True
            break

    # Phase 2: propagate complex perturbations.
    if np.iscomplexobj(b) or np.iscomplexobj(p) or (hp and np.iscomplexobj(h_target)):
        n = n.astype(complex)
        nt = complex(nt)
        T = complex(T)
        for _ in range(3):
            gRT = lib.g_RT(T)[keep_sp]
            hRT = lib.h_RT(T)[keep_sp]
            cpR = lib.cp_R(T)[keep_sp]
            dln_nj, dln_n, dln_T = _reduced_step(a, n, nt, T, p, gRT, hRT, cpR, bsub, h_target, P_ref)
            n = n * np.exp(dln_nj)  # undamped (lam = 1)
            nt = nt * np.exp(dln_n)
            if hp:
                T = T * np.exp(dln_T)

    n_full = np.zeros(S_full, dtype=n.dtype)
    n_full[keep_sp] = n
    return n_full, T, it, converged


def _finalize(lib, n_full, T, p, iterations, converged):
    Wk = lib.molar_masses
    mass = n_full * Wk
    Y = mass / np.sum(mass)
    props = mixture_properties(lib, Y, T, p)
    props.a_equilibrium = _equilibrium_sound_speed(lib, n_full, T, p)
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


def equilibrate_TP(lib, Z_elem, T, p, max_iter=400, tol=1e-11):
    """Equilibrium at fixed temperature and pressure.

    ``lib``: a :class:`~thermolib.species.SpeciesLibrary` (or ``Mechanism``).
    ``Z_elem``: elemental mass fractions (dict or array aligned to ``lib.elements``).
    """
    b = elemental_abundance(lib, Z_elem)
    n_full, Tout, it, conv = _solve_subset(lib, b, p, None, T, max_iter=max_iter, tol=tol)
    return _finalize(lib, n_full, Tout, p, it, conv)


def equilibrate_HP(lib, Z_elem, h, p, T_guess=2000.0, max_iter=400, tol=1e-11):
    """HP equilibrium: given elemental composition, enthalpy and pressure, return
    ``T, rho, composition`` and derived properties.

    ``lib``: a :class:`~thermolib.species.SpeciesLibrary` (or ``Mechanism``).
    ``Z_elem``: elemental mass fractions (dict or array aligned to ``lib.elements``).
    """
    b = elemental_abundance(lib, Z_elem)
    n_full, Tout, it, conv = _solve_subset(lib, b, p, h, T_guess, max_iter=max_iter, tol=tol)
    return _finalize(lib, n_full, Tout, p, it, conv)


def _equilibrium_sound_speed(lib, n_full, T, p):
    """Equilibrium speed of sound from the converged sensitivity block.

    Uses the CEA reduced matrix evaluated at the converged composition and the analytic
    ``(d ln V/d ln T)_P`` / ``(d ln V/d ln P)_T`` / ``cp_eq`` to form the equilibrium
    ``gamma_s`` and ``a_eq = sqrt(gamma_s * p/rho)``. Real-arithmetic linear solves;
    reuse-friendly and complex-step compatible.
    """
    afull = lib.element_matrix
    n_real = np.real(n_full)
    active = n_real > 1e-30 * np.max(n_real)
    keep_sp = np.where(active)[0]
    # Keep elements that appear in the active species.
    keep_el = np.where(np.any(afull[:, keep_sp] != 0.0, axis=1))[0]
    a = afull[np.ix_(keep_el, keep_sp)]
    n = n_real[keep_sp]
    nt = np.sum(n)
    Tr = float(np.real(T))

    hRT = np.real(lib.h_RT(Tr))[keep_sp]
    cpR = np.real(lib.cp_R(Tr))[keep_sp]

    E = a.shape[0]
    M = np.zeros((E + 1, E + 1))
    an = a @ n
    M[:E, :E] = a @ (n[:, None] * a.T)
    M[:E, E] = an
    M[E, :E] = an
    M[E, E] = np.sum(n) - nt  # ~ 0 at convergence

    # Temperature sensitivities at constant p.
    bT = np.concatenate([-(a @ (n * hRT)), [-np.sum(n * hRT)]])
    yT = np.linalg.solve(M, bT)
    dlnnt_T = yT[E]
    dln_nj_T = a.T @ yT[:E] + dlnnt_T + hRT

    # Pressure sensitivities at constant T.
    bP = np.concatenate([an, [nt]])
    yP = np.linalg.solve(M, bP)
    dlnnt_P = yP[E]

    dlnV_dlnT = dlnnt_T + 1.0
    dlnV_dlnP = dlnnt_P - 1.0

    cp_eq = R_UNIVERSAL * (np.sum(n * cpR) + np.sum(n * hRT * dln_nj_T))
    Pv = nt * R_UNIVERSAL * Tr  # = p / rho [J/kg]
    cp_minus_cv = -(nt * R_UNIVERSAL) * dlnV_dlnT**2 / dlnV_dlnP
    cv_eq = cp_eq - cp_minus_cv
    gamma_s = -(cp_eq / cv_eq) / dlnV_dlnP
    return np.sqrt(gamma_s * Pv)
