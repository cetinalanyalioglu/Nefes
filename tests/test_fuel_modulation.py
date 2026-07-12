"""Fuel-flow modulation: the dynamic mass source and the equivalence-ratio instability.

A fuel injector (:func:`~nefes.elements.catalog.mass_source`) carrying a dynamic ``S(omega)``
on its injected mass flow models a fluctuating fuel feed.  The perturbation stamp must
modulate every source term the element carries -- mass, momentum (via the injection
velocity), the injected enthalpy, **and** the injected composition -- so the fuel pulse
sheds a convected mixture-fraction wave.  That wave convects to a downstream (static)
reacting flame, which burns it into an unsteady heat release: the classic *equivalence-ratio*
combustion-instability mechanism.

The tests anchor it:

1. The stamp factors equal ``mdot_src * dR/d(mdot_src)`` on every modulated row (mass,
   momentum, energy, each composition scalar) -- a finite-difference identity.
2. A forced fuel fluctuation generates a mixture-fraction wave at the injector (none
   upstream) that convects losslessly to the flame.
3. With a total-pressure (air) inlet and a mass-flow outlet (both inherited), the fuel-flow
   coupling destabilizes a mode -- and only because the convected composition wave is
   retained (``isentropic`` freezes it -> stable).

Run in the ``nefes`` env (numba); needs the bundled H2/air data.
"""

import os
import warnings

import numpy as np
import pytest

from nefes.assembly.assemble import residual
from nefes.assembly.recover import ES_AREA, ES_C, ES_M, ES_P, ES_RHO, ES_T, ES_U
from nefes.elements import catalog as cat
from nefes.elements.dynamic_source import mass_flow_response, n_tau, n_tau_lowpass
from nefes.perturbation import (
    CompositionalNoiseWarning,
    excite_perturbation,
    forced_response,
    open_loop_response,
    perturbation_response,
)
from nefes.perturbation.operator.boundary_bc import PerturbationBC
from nefes.perturbation.operator.characteristics import char_to_dx, edge_caloric
from nefes.perturbation.operator.operator import assemble_acoustic, build_acoustic_blocks
from nefes.perturbation.operator.stamps import build_source_stamps
from nefes.shell.build import build_problem
from nefes.solver import solve
from nefes.solver.report import states_table
from nefes.thermo.api import EQ_FROZEN, EQ_KERNEL
from nefes.thermo.configure import equilibrium

AREA = 0.02
MECH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nefes", "thermo", "data", "h2o2.yaml")
AIR = {"O2": 0.21, "N2": 0.79}
# node 0 inlet(air) | 1 duct | 2 fuel injector | 3 duct (convective lag) | 4 flame | 5 duct | 6 outlet
# edge 0 air | 1 | 2 injector-out | 3 flame-approach | 4 flame-out | 5
E_AIR, E_INJ_OUT, E_APPROACH, E_FLAME_OUT = 0, 2, 3, 4
N_INJECTOR = 2


def _air_enthalpy_datum():
    from nefes.thermo import SpeciesLibrary, Thermo

    lib = SpeciesLibrary.from_cantera(MECH)
    gas = Thermo(lib)
    idx = lib.species_index
    Y = np.zeros(lib.n_species)
    Y[idx["O2"]], Y[idx["N2"]] = 0.21, 0.79
    Y /= Y.sum()
    return gas, gas.enthalpy_mass(Y, 300.0)


def _rig(La=0.4, Lb=0.4, Lc=0.5, mdot_air=0.4, mdot_fuel=0.006, pt=1.2e5, ds=None, inlet_excite=False, inlet_bc=None):
    """Air -> duct -> H2 injector (optional dynamic fuel feed) -> duct -> reacting flame -> duct -> outlet."""
    gas, h_air = _air_enthalpy_datum()
    if inlet_bc is None:
        inlet_bc = PerturbationBC.anechoic(driven=("acoustic",)) if inlet_excite else PerturbationBC.inherit()
    els = [
        cat.total_pressure_inlet(pt, 300.0, composition=AIR, basis="mole", perturbation_bc=inlet_bc),
        cat.duct(La),
        cat.mass_source(mdot_fuel, 300.0, composition={"H2": 1.0}, basis="mole", dynamic_source=ds),
        cat.duct(Lb),
        cat.equilibrium_flame(),
        cat.duct(Lc),
        cat.mass_flow_outlet(mdot_air + mdot_fuel, perturbation_bc=PerturbationBC.inherit()),
    ]
    edges = [(0, 1, AREA), (1, 2, AREA), (2, 3, AREA), (3, 4, AREA), (4, 5, AREA), (5, 6, AREA)]
    edge_models = [EQ_FROZEN, EQ_FROZEN, EQ_FROZEN, EQ_FROZEN, EQ_KERNEL, EQ_KERNEL]
    prob = build_problem(
        equilibrium(gas.mech), els, edges, mdot_ref=mdot_air, p_ref=1e5, h_ref=h_air, edge_models=edge_models
    )
    return prob


def _converged(prob):
    res = solve(prob)
    assert res.converged, "mean solve failed"
    return res.x


# --------------------------------------------------------------------------
# 1. The stamp linearizes every injected source term (finite-difference identity)
# --------------------------------------------------------------------------


def test_fuel_stamp_matches_finite_difference():
    """Each modulated row's factor equals ``mdot_src * dR/d(mdot_src)`` (mass, momentum, energy, composition).

    The injected mass flow is bumped as a *static* parameter and the residual re-evaluated at
    the same frozen state, giving ``dR/d(mdot_src)`` directly; the dynamic-source stamp must
    reproduce it on every row it writes -- proof the fuel pulse perturbs all of the injector's
    source terms, not just mass + momentum.
    """
    mdot_fuel = 0.006
    ds = mass_flow_response(n_tau(0.7, 2e-3), ref_edge=E_AIR, quantity="u")
    prob = _rig(mdot_fuel=mdot_fuel, ds=ds)
    x = _converged(prob)

    stamps, _ = build_source_stamps(prob, x)
    inj = next(s for s in stamps if s.node == N_INJECTOR)
    assert len(inj.rows) >= 4  # mass, momentum, energy, and at least one composition scalar

    eps = 1e-4 * prob.var_scale[0]
    R0 = residual(prob, x, eps, 1e-6, 0.0)
    R1 = residual(_rig(mdot_fuel=mdot_fuel + 1e-6), x, eps, 1e-6, 0.0)
    dR = (R1 - R0) / 1e-6  # dR/d(mdot_src) at fixed state
    for r, fac in zip(inj.rows, inj.factors):
        assert fac / mdot_fuel == pytest.approx(dR[r], abs=1e-3, rel=1e-3)


# --------------------------------------------------------------------------
# 2. A fuel pulse sheds a convected mixture-fraction wave
# --------------------------------------------------------------------------


def test_fuel_pulse_generates_a_convected_mixture_wave():
    """A fuel-fraction wave appears at the injector (none upstream) and convects to the flame.

    Forcing the inlet drives the velocity-coupled fuel feed; the injected-stream fraction
    fluctuates only *downstream* of the injector and rides the convected wave to the flame --
    magnitude conserved, phase = the convective lag ``e^{-i omega L/u}`` (the equivalence-ratio
    transport that sets the instability's phase).
    """
    ds = mass_flow_response(n_tau(1.0, 2e-3), ref_edge=E_AIR, quantity="u")
    prob = _rig(ds=ds, inlet_excite=True)
    x = _converged(prob)
    est = states_table(prob, x)
    assert est[ES_T, E_FLAME_OUT] > 1200.0  # flame ignited

    ns = int(prob.n_solve)
    s_fuel = ns - 1  # last composition scalar = the injected ("source") stream fraction
    Lb, u = 0.4, float(est[ES_U, E_INJ_OUT])  # the injector -> flame duct and its mean speed
    freqs = np.array([150.0, 280.0, 400.0])
    omega = 2.0 * np.pi * freqs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fr = forced_response(prob, x, freqs, isentropic=False)
    z_air = fr.X[:, ns * E_AIR + s_fuel]
    z_inj = fr.X[:, ns * E_INJ_OUT + s_fuel]
    z_flame = fr.X[:, ns * E_APPROACH + s_fuel]
    assert np.all(np.abs(z_air) < 1e-12)  # no fuel-fraction fluctuation upstream of the injector
    assert np.all(np.abs(z_inj) > 1e-4)  # the injector sheds the wave
    assert np.allclose(np.abs(z_inj), np.abs(z_flame), rtol=1e-6)  # convection is lossless
    assert np.allclose(z_flame, z_inj * np.exp(-1j * omega * Lb / u), rtol=1e-6)  # convective phase lag


def test_forced_response_surfaces_scalar_waves():
    """``ForcedResponse.waves()`` returns the convected scalar wave alongside ``(f, g, h)``.

    Wave-family parity for reacting scalars: the per-edge wave vector carries one entry per
    transported scalar (named by ``prob.scalar_names``), surfaced directly from the network
    column -- the scalar is already its own convected wave -- so the same field accessor serves
    entropy and the scalars, no hand-indexing of ``X`` required.
    """
    ds = mass_flow_response(n_tau(1.0, 2e-3), ref_edge=E_AIR, quantity="u")
    prob = _rig(ds=ds, inlet_excite=True)
    x = _converged(prob)
    ns = int(prob.n_solve)
    assert ns > 3 and len(prob.scalar_names) == ns - 3  # genuinely reacting (scalars present)
    freqs = np.array([150.0, 280.0, 400.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fr = forced_response(prob, x, freqs, isentropic=False)
    # as many waves as unknowns, labelled (f, g, h) then the feed-stream names
    assert fr.n_char == ns
    assert fr.wave_labels[:3] == ("f", "g", "h")
    assert tuple(fr.wave_labels[3:]) == tuple(prob.scalar_names)
    # the surfaced scalar wave is exactly the raw network column (identity), for every scalar/edge
    for s in range(3, ns):
        for e in (E_AIR, E_INJ_OUT, E_APPROACH):
            assert np.allclose(fr.waves(e)[:, s], fr.X[:, ns * e + s])
    # genuinely nonzero (the fuel pulse sheds a mixture-fraction wave) and acoustics untouched
    assert np.max(np.abs(fr.waves(E_INJ_OUT)[:, ns - 1])) > 1e-4
    assert np.allclose(fr.reflection_at(E_AIR), fr.waves(E_AIR)[:, 1] / fr.waves(E_AIR)[:, 0])


def test_pure_composition_wave_radiates_sound_at_the_flame():
    """A composition wave alone -- no acoustic, no entropy forcing -- makes sound at the flame.

    The pinnacle of the complex-step approach: driving *only* a reacting-scalar (equivalence-ratio)
    wave at the inflow still generates acoustic waves, because the inherited linearization (the full
    algebraic Jacobian of the flame jump) carries the composition -> acoustic coupling -- direct
    combustion noise, with no hand-written scalar closure anywhere.  The response is linear in the
    drive amplitude (the scalar is the lone source), and -- the outlet being inherited, not a compact
    closure -- the narrowed compositional-noise warning must stay silent.
    """
    fuel = _rig().scalar_names[-1]  # the injected (H2 'source') stream's mixture fraction
    prob = _rig(inlet_bc=PerturbationBC.anechoic(driven=(fuel,)))  # drive ONLY the composition wave
    x = _converged(prob)
    assert states_table(prob, x)[ES_T, E_FLAME_OUT] > 1200.0  # flame ignited
    freqs = np.array([120.0, 350.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error", CompositionalNoiseWarning)  # inherited outlet -> no gap, no warning
        fr = forced_response(prob, x, freqs, isentropic=False)
    s = fr.wave_labels.index(fuel)
    assert np.all(np.abs(fr.waves(E_APPROACH)[:, s]) > 0.5)  # the composition wave convects to the flame
    # sound is generated at the flame with no acoustic forcing -- composition -> acoustic via J_alg
    f_out, g_out = fr.waves(E_FLAME_OUT)[:, 0], fr.waves(E_FLAME_OUT)[:, 1]
    assert np.all(np.abs(f_out) + np.abs(g_out) > 1.0)
    # genuinely linear in the drive amplitude: 2x composition wave -> 2x sound
    prob2 = _rig(inlet_bc=PerturbationBC.anechoic(driven=(fuel,), amplitudes={fuel: 2.0}))
    x2 = _converged(prob2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fr2 = forced_response(prob2, x2, freqs, isentropic=False)
    assert np.allclose(np.abs(fr2.waves(E_FLAME_OUT)[:, 0]), 2.0 * np.abs(f_out), rtol=1e-6)


def _nozzle_rig(nozzle_bc, *, mdot_h2=0.02, A_star=0.012):
    """Air -> duct -> H2 source -> duct -> compact choked nozzle (frozen: a pure composition spot)."""
    gas, h_air = _air_enthalpy_datum()
    els = [
        cat.mass_flow_inlet(0.4, 300.0, composition=AIR, basis="mole"),
        cat.duct(0.4),
        cat.mass_source(mdot_h2, 300.0, composition={"H2": 1.0}, basis="mole"),
        cat.duct(0.5),
        cat.choked_nozzle_outlet(A_star, perturbation_bc=nozzle_bc),
    ]
    edges = [(0, 1, AREA), (1, 2, AREA), (2, 3, AREA), (3, 4, AREA)]
    prob = build_problem(
        equilibrium(gas.mech), els, edges, mdot_ref=0.4, p_ref=1e5, h_ref=h_air, edge_models=[EQ_FROZEN] * 4
    )
    return prob


def _nozzle_reflection_row(prob, x, *, stamped):
    """``(M, R, R_s, R_xi)`` of the choked-nozzle outlet row in (f, g, h, scalars) coordinates.

    ``stamped`` reads the assembled operator row (an explicit closure overwrites it); otherwise the
    inherited ``J_alg`` row.  The 3 acoustic columns map through the reacting caloric; the scalar
    columns are the composition -> acoustic coefficients ``R_xi`` directly.
    """
    est = states_table(prob, x)
    cals = edge_caloric(prob, x)
    e, ns = 3, int(prob.n_solve)  # nozzle edge / solve width
    r0 = int(prob.node_row_ptr[4])  # nozzle node row (the acoustic to-specify wave)
    if stamped:
        row = np.asarray(
            assemble_acoustic(0.0, build_acoustic_blocks(prob, x), with_boundaries=True)
            .tocsr()[r0, ns * e : ns * e + ns]
            .todense()
        ).ravel()
    else:
        row = np.asarray(build_acoustic_blocks(prob, x).J_alg[r0, ns * e : ns * e + ns].todense()).ravel()
    rho, c, u, p, area = (float(est[k, e]) for k in (ES_RHO, ES_C, ES_U, ES_P, ES_AREA))
    cf, cg, ch = row[:3] @ char_to_dx(rho, c, u, area, cals[e])
    return float(est[ES_M, e]), -cf / cg, -ch / cg, -row[3:] / cg


def test_inherited_nozzle_carries_compositional_noise_analytic_closure_drops_it():
    """Compositional (indirect) noise at a compact nozzle: ``R_xi`` is captured by the inherited row.

    The inherited ``choked_nozzle_outlet`` linearizes the critical-mass-flux jump through its full
    composition dependence (complex-step), so its outlet row in characteristic coordinates is
    ``g = R f + R_s h + R_xi . xi`` with a *nonzero* composition column ``R_xi`` -- the compositional
    noise -- while still reproducing the Marble--Candel acoustic reflection ``R``.  The hand-written
    analytic :meth:`PerturbationBC.choked_nozzle` closure overwrites that row with a 3-wave
    ``(f, g, h)`` relation, so its ``R_xi`` is identically zero: the dropped coupling the narrowed
    :class:`CompositionalNoiseWarning` flags.  The analytic closure still matches the inherited ``R``,
    because it now takes the **backend-correct** effective gamma (``rho c^2 / p``) from the state.
    """
    prob_inh = _nozzle_rig(None)
    x_inh = solve(prob_inh).x
    assert prob_inh.scalar_names  # genuinely multi-stream (a composition that can fluctuate)
    M, R, R_s, R_xi = _nozzle_reflection_row(prob_inh, x_inh, stamped=False)

    # the inherited row reproduces the Marble--Candel reflection (subsonic approach) ...
    assert 0.0 < M < 1.0
    assert abs(R.imag) < 1e-9 and 0.6 < R.real < 1.0
    # ... AND carries a genuine composition -> acoustic coupling (compositional noise)
    assert np.max(np.abs(R_xi)) > 1.0

    prob_an = _nozzle_rig(PerturbationBC.choked_nozzle())
    x_an = solve(prob_an).x
    _, R_an, _, R_xi_an = _nozzle_reflection_row(prob_an, x_an, stamped=True)
    # the analytic closure drops compositional noise entirely: zero composition column ...
    assert np.max(np.abs(R_xi_an)) < 1e-9
    # ... yet its acoustic reflection now agrees with the inherited / Marble--Candel value, because it
    # reads the effective gamma off the state (the wrong perfect-gas K gamma gave ~15% error before).
    assert abs(R_an - R) < 0.01 * abs(R)


def test_scalar_scattering_matrix_port():
    """Scalar (composition) waves are measured at parity with entropy in the scattering matrices.

    A transported scalar is a passively-convected mode (it rides the mean speed ``u`` and is its
    own characteristic), so it now drives and reads back through ``perturbation_response`` exactly
    like the entropy wave: the response is the full ``n_solve x n_solve`` block, the scalar
    convects with phase ``exp(-i w L / u)`` on a pure duct and stays decoupled from the acoustics,
    and the multiport matrix carries a scalar port per inflow seat, labelled by the stream name.
    A genuine typo still raises ``ValueError``.
    """
    prob = _rig()
    x = _converged(prob)
    names = tuple(prob.scalar_names)
    assert names and prob.n_solve == 3 + len(names)  # reacting: scalar families exist
    freqs = np.array([200.0, 400.0])
    om = 2.0 * np.pi * freqs

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = perturbation_response(prob, x, freqs, excite=("acoustic", "entropy") + names)
        # full block: 2 acoustic + 1 entropy + one wave per transported scalar
        assert r.n_char == prob.n_solve and r.n == prob.n_solve
        assert tuple(r.cidx) == tuple(range(prob.n_solve))

        # the inlet duct (edges 0->1, length La=0.4) is a pure duct: each scalar convects at u
        # with phase exp(-i w L / u) and is decoupled from f / g / h.
        u = float(states_table(prob, x)[ES_U, 0])
        T = r.transfer_matrix(0, 1)
        for j in range(len(names)):
            c = 3 + j
            assert np.allclose(T[:, c, c], np.exp(-1j * om * 0.4 / u), atol=1e-7)
            for ac in (0, 1, 2):  # scalar <-> (f, g, h) decoupled on a pure duct
                assert np.allclose(T[:, ac, c], 0.0, atol=1e-7)
                assert np.allclose(T[:, c, ac], 0.0, atol=1e-7)

        # multiport: a scalar port per inflow seat, finite and stream-labelled
        S = r.multiport_scattering_matrix()
        inc, out = r.multiport_scattering_labels()
        assert S.shape == (freqs.size, len(out), len(inc))
        assert np.all(np.isfinite(S))
        assert any(name in lab for name in names for lab in inc)

        # excite_perturbation drives a single scalar family too
        f1 = excite_perturbation(prob, x, freqs, node=0, modes=("acoustic", names[-1]))
        assert np.all(np.isfinite(f1.waves(0)))

        # a genuine typo is still a plain ValueError
        with pytest.raises(ValueError, match="unknown wave family"):
            perturbation_response(prob, x, freqs, excite=("acoustic", "bogus"))

    # the BC driven= path continues to accept a scalar family (resolved at stamp time)
    PerturbationBC.anechoic(driven=(names[-1],))


# --------------------------------------------------------------------------
# 3. The equivalence-ratio instability (fuel flow -> heat release -> acoustics)
# --------------------------------------------------------------------------


def test_fuel_flow_coupling_destabilizes_a_mode():
    """The fuel-flow FTF drives a mode unstable -- and only with the convected composition wave.

    Nyquist on the fuel-injector return ratio: keeping the convected mixture wave
    (``isentropic=False``) the fuel-flow coupling encircles the critical point (an unstable
    mode); freezing it (``isentropic=True``) removes the convective lag and the encirclement,
    so the equivalence-ratio transport is the essential link.
    """
    ds = mass_flow_response(n_tau_lowpass(2.0, 1e-3, 400.0), ref_edge=E_AIR, quantity="u")
    prob = _rig(ds=ds)
    x = _converged(prob)
    freqs = np.linspace(0.0, 1400.0, 900)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = open_loop_response(prob, x, freqs, isentropic=False)
        iso = open_loop_response(prob, x, freqs, isentropic=True)
    assert full.closed and iso.closed
    assert full.n_unstable >= 1  # the fuel-flow coupling destabilizes a mode
    assert iso.n_unstable == 0  # frozen composition wave: no equivalence-ratio instability
    assert any(150.0 < c["freq_hz"] < 700.0 for c in full.crossings(tol=0.35))
