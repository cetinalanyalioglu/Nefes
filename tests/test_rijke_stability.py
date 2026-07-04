"""Self-excited thermoacoustic instability of a Rijke tube (theory.md s12.4, s12.7).

A compact flame with an ``n-tau`` heat-release response drives a duct mode unstable.
The perturbation operator's source face ``S(omega)`` makes it non-self-adjoint, so the
eigenmodes acquire growth rates; for a band of time lags ``tau`` a mode crosses into
instability.

The reference is the **analytical compact-flame dispersion relation**: a two-region
(cold/hot) acoustic tube with a zero-mean-Mach jump

    p' continuous,    u_2' - u_1' = (gamma-1)/(gamma p A) * Q',
    Q' = Q_bar * n e^{-i omega tau} * u_1'(x_f)/u_bar_1,

whose ``det = 0`` gives the complex modal frequencies.  At low mean Mach (where the
analytic model is exact) Nefes reproduces the analytic frequencies and growth rates,
including the sign of the heat-release coupling -- the instability itself.

Run in the ``nefes`` env (numba); the reacting case also needs the thermolib data.
"""

import os
import warnings

import numpy as np
import pytest

from nefes.elements import catalog as cat
from nefes.elements.dynamic_source import n_tau_flame
from nefes.perturbation.operator.boundary_bc import PerturbationBC
from nefes.perturbation import eigenmodes
from nefes.solver import solve
from nefes.solver.report import states_table
from nefes.assembly.recover import ES_RHO, ES_C, ES_U, ES_P, ES_T
from nefes.thermo.api import thermo_state
from nefes.thermo.configure import perfect_gas, equilibrium

R_AIR, GAMMA = 287.0, 1.4
CP = GAMMA * R_AIR / (GAMMA - 1.0)
AREA = 0.01
L1, L2 = 0.6, 0.4  # cold (upstream) and hot (downstream) duct lengths [m]
MECH_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "thermolib", "data", "h2o2.yaml")


# --------------------------------------------------------------------------
# Analytical compact-flame dispersion relation (zero mean Mach)
# --------------------------------------------------------------------------


def _analytic_det(omega, means, Qdot, n, tau, gamma):
    """``det`` of the 4x4 [P1+, P1-, P2+, P2-] system; zero at a mode (e^{+i w t})."""
    rho1, c1, u1, p1, rho2, c2 = means
    k1, k2 = omega / c1, omega / c2
    Z1, Z2 = rho1 * c1, rho2 * c2
    # region 1 spans [-L1, 0], region 2 [0, L2], flame at x = 0
    eP1, eM1 = np.exp(1j * k1 * L1), np.exp(-1j * k1 * L1)
    eP2, eM2 = np.exp(-1j * k2 * L2), np.exp(1j * k2 * L2)
    Theta = (gamma - 1.0) * Qdot / (gamma * p1 * AREA) * (n * np.exp(-1j * omega * tau)) / (Z1 * u1)
    M = np.array(
        [
            [eP1, -eM1, 0.0, 0.0],  # inlet hard wall:  u1'(-L1) = 0
            [1.0, 1.0, -1.0, -1.0],  # flame: pressure continuous
            [-(1.0 / Z1 + Theta), (1.0 / Z1 + Theta), 1.0 / Z2, -1.0 / Z2],  # flame: velocity jump
            [0.0, 0.0, eP2, eM2],  # outlet open end:  p2'(L2) = 0
        ],
        dtype=complex,
    )
    return np.linalg.det(M)


def _analytic_mode(means, Qdot, n, tau, seed, gamma=GAMMA):
    """Newton-polish a complex mode of ``det A(omega) = 0`` from ``seed`` (rad/s)."""
    w = complex(seed)
    for _ in range(100):
        h = 1e-3 * (abs(w) + 1.0)
        d = _analytic_det(w, means, Qdot, n, tau, gamma)
        dp = (_analytic_det(w + h, means, Qdot, n, tau, gamma) - _analytic_det(w - h, means, Qdot, n, tau, gamma)) / (
            2.0 * h
        )
        if dp == 0:
            break
        step = d / dp
        w -= step
        if abs(step) < 1e-10:
            break
    return w.real / (2.0 * np.pi), -w.imag  # (freq Hz, growth 1/s)


# --------------------------------------------------------------------------
# Perfect-gas Rijke tube
# --------------------------------------------------------------------------


def _pg_rijke(n, tau, mdot=0.005, dT=400.0):
    """Cold air -> duct -> n-tau heat-release flame -> duct -> open end."""
    Qdot = mdot * CP * dT
    els = [
        cat.mass_flow_inlet(mdot, 300.0, perturbation_bc=PerturbationBC.hard_wall()),
        cat.duct(L1),
        cat.heat_release_flame(Qdot, dynamic_source=n_tau_flame(n, tau, ref_edge=1)),
        cat.duct(L2),
        cat.pressure_outlet(1.0e5, perturbation_bc=PerturbationBC.open_end()),
    ]
    edges = [(0, 1, AREA), (1, 2, AREA), (2, 3, AREA), (3, 4, AREA)]
    prob = cat.build_problem(perfect_gas(R_AIR, GAMMA), els, edges, mdot_ref=mdot, p_ref=1e5, h_ref=CP * 300.0)
    res = solve(prob)
    assert res.converged
    return prob, res.x, Qdot


def _means(prob, x):
    est = states_table(prob, x)
    return (est[ES_RHO, 1], est[ES_C, 1], est[ES_U, 1], est[ES_P, 1], est[ES_RHO, 2], est[ES_C, 2])


def _modes(prob, x, band=(60.0, 160.0)):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = eigenmodes(prob, x, freq_band=band, growth_band=(-200.0, 200.0), isentropic=True)
    return list(zip(r.freqs, r.growth_rates))


def test_passive_flame_matches_analytic_frequencies():
    """With no unsteady heat release the tube is a passive two-region resonator."""
    prob, x, Qdot = _pg_rijke(0.0, 0.0)
    means = _means(prob, x)
    fns = _modes(prob, x)
    assert fns, "no eigenmodes found"
    f_fns, g_fns = fns[0]
    f_an, g_an = _analytic_mode(means, Qdot, 0.0, 0.0, 2 * np.pi * f_fns)
    assert f_fns == pytest.approx(f_an, rel=5e-3)
    # passive growth is ~0 (only the O(Mach) mean-flow damping remains, < 2/s here)
    assert abs(g_fns) < 2.0
    assert abs(g_an) < 1e-3


def test_n_tau_flame_drives_self_excited_instability():
    """A destabilizing time lag makes the fundamental grow; Nefes matches the analytic root."""
    prob, x, Qdot = _pg_rijke(0.8, 4.0e-3)
    means = _means(prob, x)
    modes = _modes(prob, x)
    assert modes
    # the most unstable mode is the self-excited one
    f_fns, g_fns = max(modes, key=lambda m: m[1])
    assert g_fns > 0.0, f"expected a growing (unstable) mode, got growth {g_fns}"

    f_an, g_an = _analytic_mode(means, Qdot, 0.8, 4.0e-3, 2 * np.pi * f_fns + 1j * (-g_fns))
    assert g_an > 0.0  # the analytic root is unstable too (sign of the coupling)
    assert f_fns == pytest.approx(f_an, rel=5e-3)
    assert g_fns == pytest.approx(g_an, abs=max(2.0, 0.03 * abs(g_an)))


def test_n_tau_lag_sets_stability_band():
    """The same flame is stabilizing at one lag and destabilizing at another (n-tau band)."""
    prob_s, xs, Q = _pg_rijke(0.8, 1.5e-3)  # stabilizing
    prob_u, xu, _ = _pg_rijke(0.8, 4.0e-3)  # destabilizing
    g_stable = max(_modes(prob_s, xs), key=lambda m: m[1])[1]
    g_unstable = max(_modes(prob_u, xu), key=lambda m: m[1])[1]
    assert g_stable < 0.0 < g_unstable

    # the stabilizing lag matches the analytic damped root
    means = _means(prob_s, xs)
    fns = [m for m in _modes(prob_s, xs) if 80.0 < m[0] < 110.0][0]
    f_an, g_an = _analytic_mode(means, Q, 0.8, 1.5e-3, 2 * np.pi * fns[0] + 1j * (-fns[1]))
    assert fns[0] == pytest.approx(f_an, rel=5e-3)
    assert fns[1] == pytest.approx(g_an, abs=max(2.0, 0.03 * abs(g_an)))


# --------------------------------------------------------------------------
# Reacting (equilibrium) Rijke tube -- the n-tau source is identical for both flames
# --------------------------------------------------------------------------


def _h2_air():
    from thermolib import SpeciesLibrary, Thermo

    lib = SpeciesLibrary.from_native(MECH_PATH)
    gas = Thermo(lib)
    idx = lib.species_index
    moles = np.zeros(lib.n_species)
    moles[idx["H2"]], moles[idx["O2"]], moles[idx["N2"]] = 1.0, 0.5, 0.5 * 3.76
    Y = moles * lib.molar_masses
    Y /= Y.sum()
    return gas, Y, gas.elemental_mass_fractions(Y)


def _reacting_rijke(n, tau, mdot=0.02, Tin=300.0, p=1.0e5):
    from nefes.thermo.api import EQ_FROZEN, EQ_KERNEL

    gas, Y, Z = _h2_air()
    h_react = gas.enthalpy_mass(Y, Tin)
    fuel_air = {"H2": 1.0, "O2": 0.5, "N2": 0.5 * 3.76}
    els = [
        cat.mass_flow_inlet(mdot, Tin, composition=fuel_air, basis="mole", perturbation_bc=PerturbationBC.hard_wall()),
        cat.duct(L1),
        cat.equilibrium_flame(dynamic_source=n_tau_flame(n, tau, ref_edge=1)),
        cat.duct(L2),
        cat.pressure_outlet(
            p, Tt_backflow=Tin, composition=fuel_air, basis="mole", perturbation_bc=PerturbationBC.open_end()
        ),
    ]
    edges = [(0, 1, AREA), (1, 2, AREA), (2, 3, AREA), (3, 4, AREA)]
    # unburnt approach (edges 0,1) -> burnt products (edges 2,3): the flame ignites edge 2
    edge_models = [EQ_FROZEN, EQ_FROZEN, EQ_KERNEL, EQ_KERNEL]
    prob = cat.build_problem(
        equilibrium(gas.mech), els, edges, mdot_ref=mdot, p_ref=p, h_ref=h_react, edge_models=edge_models
    )
    res = solve(prob)
    assert res.converged
    return prob, res.x


# --------------------------------------------------------------------------
# First-principles analytic for the *equilibrium* flame (independent of the Nefes operator)
#
# The equilibrium flame conserves absolute total enthalpy, so its compact jump is NOT
# the perfect-gas contact discontinuity: it is mass-flux continuous (p' continuous,
# rho u' continuous), and the density responds to the (source-driven) total-enthalpy
# fluctuation through the *actual* equilibrium EOS derivatives -- not the perfect-gas
# caloric.  The dispersion relation below carries five waves (f1, g1 cold/isentropic;
# f2, g2, h2 hot with an entropy spot at the flame) and the same source coupling Nefes
# stamps (h_t2' = h_t1' + delta * n e^{-i w tau} * u1'/u1).  The caloric derivatives
# come from a complex step of the closure (thermo_state) and delta from the mean
# sensible heat release -- both physical inputs, so the match below validates the
# reacting perturbation operator, not a tautology.
# --------------------------------------------------------------------------


def _cp_eff(est_col):
    """Effective cp = gamma R/(gamma-1) from a mean edge state (sound-speed consistent)."""
    rho, c, p, T = est_col[ES_RHO], est_col[ES_C], est_col[ES_P], est_col[ES_T]
    gamma = rho * c * c / p
    return gamma * (p / (rho * T)) / (gamma - 1.0)


def _reacting_caloric(prob, x, est, e):
    """(a, b) = ((dh/drho)_p, (dh/dp)_rho) at edge ``e`` via a complex step of the closure."""
    mid = int(prob.edge_model[e])
    xi = np.ascontiguousarray(x[3 : 3 + prob.n_elem, e]).astype(np.complex128)
    ht = complex(x[2, e])
    p = float(est[ES_P, e])
    d = 1e-30
    drho_dh = thermo_state(mid, prob.tf, prob.ti, xi, ht + 1j * d, complex(p))[1].imag / d
    drho_dp = thermo_state(mid, prob.tf, prob.ti, xi, ht + 0j, p + 1j * d)[1].imag / d
    return 1.0 / drho_dh, -drho_dp / drho_dh


def _reacting_ref(prob, x):
    """Means, per-edge caloric (cold, hot) and source de-normalization ``delta`` for the analytic."""
    est = states_table(prob, x)
    m = (
        est[ES_RHO, 1],
        est[ES_C, 1],
        est[ES_U, 1],
        est[ES_P, 1],
        est[ES_RHO, 2],
        est[ES_C, 2],
        est[ES_U, 2],
        est[ES_P, 2],
    )
    cal1 = _reacting_caloric(prob, x, est, 1)
    cal2 = _reacting_caloric(prob, x, est, 2)
    delta = 0.5 * (_cp_eff(est[:, 1]) + _cp_eff(est[:, 2])) * (est[ES_T, 2] - est[ES_T, 1])
    return m, cal1, cal2, delta


def _reacting_det(omega, m, cal1, cal2, delta, n, tau):
    """``det`` of the 5x5 [f1, g1, f2, g2, h2] equilibrium-flame system (zero at a mode, e^{+i w t})."""
    rho1, c1, u1, p1, rho2, c2, u2, p2 = m
    a1, b1 = cal1
    a2, b2 = cal2
    Z1, Z2 = rho1 * c1, rho2 * c2
    F = n * np.exp(-1j * omega * tau)
    k1p, k1m = omega / (u1 + c1), omega / (c1 - u1)
    k2p, k2m = omega / (u2 + c2), omega / (c2 - u2)
    M = np.zeros((5, 5), dtype=complex)
    # inlet hard wall u1'(-L1) = 0
    M[0, 0], M[0, 1] = np.exp(1j * k1p * L1), -np.exp(-1j * k1m * L1)
    # outlet open p2'(L2) = 0
    M[1, 2], M[1, 3] = np.exp(-1j * k2p * L2), np.exp(1j * k2m * L2)
    # mass flux continuous: mdot' = A(u rho' + rho u');  rho1'=p1'/c1^2, rho2'=h2+p2'/c2^2
    M[2] = [
        AREA * (u1 * Z1 / c1**2 + rho1),
        AREA * (u1 * Z1 / c1**2 - rho1),
        -AREA * (u2 * Z2 / c2**2 + rho2),
        -AREA * (u2 * Z2 / c2**2 - rho2),
        -AREA * u2,
    ]
    # momentum continuous: p' + u^2 rho' + 2 rho u u'
    M[3] = [
        Z1 + u1**2 * Z1 / c1**2 + 2 * rho1 * u1,
        Z1 + u1**2 * Z1 / c1**2 - 2 * rho1 * u1,
        -(Z2 + u2**2 * Z2 / c2**2 + 2 * rho2 * u2),
        -(Z2 + u2**2 * Z2 / c2**2 - 2 * rho2 * u2),
        -(u2**2),
    ]
    # energy: h_t2' - h_t1' - delta F (u1'/u1) = 0,  h_t' = a rho' + b p' (KE dropped)
    ht1 = a1 * Z1 / c1**2 + b1 * Z1
    ht2 = a2 * Z2 / c2**2 + b2 * Z2
    src = delta * F / u1
    M[4] = [-ht1 - src, -ht1 + src, ht2, ht2, a2]
    return np.linalg.det(M)


def _reacting_mode(m, cal1, cal2, delta, n, tau, seed):
    """Newton-polish a complex mode of the equilibrium-flame ``det = 0`` from ``seed`` (rad/s)."""
    w = complex(seed)
    for _ in range(200):
        h = 1e-3 * (abs(w) + 1.0)
        d0 = _reacting_det(w, m, cal1, cal2, delta, n, tau)
        dp = (
            _reacting_det(w + h, m, cal1, cal2, delta, n, tau) - _reacting_det(w - h, m, cal1, cal2, delta, n, tau)
        ) / (2.0 * h)
        if dp == 0:
            break
        step = d0 / dp
        w -= step
        if abs(step) < 1e-9:
            break
    return w.real / (2.0 * np.pi), -w.imag


def test_reacting_flame_ignites_and_matches_analytic():
    """The reacting mean solve ignites and the passive fundamental matches the analytic (freq + growth)."""
    prob, x = _reacting_rijke(0.0, 0.0)
    est = states_table(prob, x)
    assert est[ES_T, 1] == pytest.approx(300.0, abs=5.0)  # unburnt approach
    assert est[ES_T, 2] > 2000.0  # burnt products
    modes = _modes(prob, x, band=(50.0, 200.0))  # fundamental band
    assert modes, "no eigenmodes found for the reacting tube"
    assert all(g < 0.0 for _f, g in modes)  # passive: damped (the intrinsic quasi-steady response)

    m, cal1, cal2, delta = _reacting_ref(prob, x)
    f_fns, g_fns = min(modes, key=lambda mm: abs(mm[0] - 106.0))
    f_an, g_an = _reacting_mode(m, cal1, cal2, delta, 0.0, 0.0, 2 * np.pi * f_fns)
    assert f_fns == pytest.approx(f_an, rel=5e-3)
    assert g_fns == pytest.approx(g_an, abs=max(2.0, 0.03 * abs(g_an)))


def test_reacting_n_tau_flame_matches_analytic_instability():
    """A destabilizing lag drives the reacting flame unstable; Nefes matches the analytic root."""
    # passive reference: damped
    prob0, x0 = _reacting_rijke(0.0, 0.0)
    assert max(g for _f, g in _modes(prob0, x0, band=(50.0, 200.0))) < 0.0

    prob, x = _reacting_rijke(0.8, 6.0e-3)
    modes = _modes(prob, x, band=(50.0, 200.0))
    assert modes
    f_fns, g_fns = max(modes, key=lambda mm: mm[1])
    assert g_fns > 0.0, f"expected an unstable reacting mode, got growth {g_fns}"

    m, cal1, cal2, delta = _reacting_ref(prob, x)
    f_an, g_an = _reacting_mode(m, cal1, cal2, delta, 0.8, 6.0e-3, 2 * np.pi * f_fns)
    assert g_an > 0.0  # the analytic root is unstable too (sign of the coupling)
    assert f_fns == pytest.approx(f_an, rel=5e-3)
    assert g_fns == pytest.approx(g_an, abs=max(2.0, 0.03 * abs(g_an)))


def test_composition_wave_convects_with_duct_phase():
    """A transported composition scalar gets the convective phase e^{-i w L/u} across a duct.

    The reacting network carries one mixture-fraction scalar.  In the full (non-isentropic)
    perturbation operator its duct transport row must read ``xi(head) = e^{-i w L/u} xi(tail)``
    -- the convected-wave phase, not the steady ``omega = 0`` convection it had before.  (It
    is decoupled under the isentropic stability mode, like the entropy wave, to keep the long
    transit time out of the acoustic spectrum.)
    """
    from nefes.perturbation.operator.operator import build_acoustic_blocks, assemble_acoustic

    prob, x = _reacting_rijke(0.0, 0.0)
    est = states_table(prob, x)
    blocks = build_acoustic_blocks(prob, x, isentropic=False)  # full operator: composition convects
    ns, E, tr0 = int(prob.n_solve), int(prob.n_edges), int(prob.transport_row0)
    # cold L1 duct (node 1): tail edge 0 -> head edge 1; composition scalar s = 1 (solve var 3)
    u = float(est[ES_U, 0])
    tau0 = L1 / u
    row = tr0 + 1 * E + 1  # composition (s=1) transport row on the head edge
    head_col, tail_col = ns * 1 + 3, ns * 0 + 3
    for f in (80.0, 137.0):
        w = 2.0 * np.pi * f
        A = assemble_acoustic(w, blocks, with_boundaries=True).tocsc()
        assert A[row, head_col] == pytest.approx(1.0, abs=1e-9)
        assert A[row, tail_col] == pytest.approx(-np.exp(-1j * w * tau0), abs=1e-9)
