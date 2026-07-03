"""Entropy-wave generation at a flame and its conversion to sound at a nozzle.

The full *indirect (entropy) combustion noise* chain, exercised end to end on the
forced perturbation network (real frequencies, ``isentropic=False`` so the convected
entropy wave is retained):

1. **Generation.** An acoustic wave forced at the inlet drives the flame's unsteady
   heat release (an ``n-tau`` FTF). The fluctuating heat release creates an entropy
   (temperature) spot in the burnt gas -- there is *no* entropy upstream of the flame,
   so the entropy wave ``h`` appears across the flame.
2. **Transport.** The entropy spot convects down the hot duct at the mean speed ``u``:
   ``h(nozzle) = h(flame) * exp(-i omega L/u)``, magnitude conserved.
3. **Conversion.** At the compact choked nozzle the accelerating mean flow turns the
   entropy spot into a reflected acoustic wave (Marble--Candel): ``g = R f + R_s h``.
   The ``R_s h`` term is the indirect noise.

The default isentropic acoustic analysis pins ``h = 0`` away from the flame, so it
*misses* this entirely -- shown explicitly. Run in the ``nefes`` env (numba).
"""

import warnings

import numpy as np

from nefes.elements import catalog as cat
from nefes.elements.dynamic_source import n_tau_flame
from nefes.perturbation.operator.boundary_bc import PerturbationBC
from nefes.perturbation import forced_response
from nefes.solver import solve
from nefes.solver.report import states_table
from nefes.thermo.configure import perfect_gas
from nefes.assembly.derive import ES_U, ES_C, ES_M, ES_RHO

R_AIR, GAMMA = 287.0, 1.4
CP = GAMMA * R_AIR / (GAMMA - 1.0)
GM1 = GAMMA - 1.0
MDOT, AREA, A_STAR, L1, L2 = 0.4, 0.02, 0.012, 0.5, 0.5
# edge ids:  0,1 = cold (1 = flame approach, the FTF reference);  2 = flame outflow (hot),
# 3 = nozzle inlet (hot).  Entropy is generated on edge 2 and converts on edge 3.
E_COLD, E_FLAME_OUT, E_NOZZLE = 1, 2, 3
FREQS = np.linspace(50.0, 600.0, 12)


def _rig(n, tau, dT=600.0):
    """Acoustically-forced inlet -> cold duct -> n-tau flame -> hot duct -> choked nozzle.

    The flame + choked nozzle do not converge from the default cold guess, so the heat
    release is ramped (continuation) reusing the previous state -- same topology, same
    state layout.
    """
    ds = n_tau_flame(n, tau, ref_edge=E_COLD)

    def mk(Qdot):
        els = [
            cat.mass_flow_inlet(MDOT, 300.0, perturbation_bc=PerturbationBC.anechoic(driven=("acoustic",))),
            cat.duct(L1),
            cat.heat_release_flame(Qdot, dynamic_source=ds),
            cat.duct(L2),
            cat.choked_nozzle_outlet(A_STAR),
        ]
        edges = [(0, 1, AREA), (1, 2, AREA), (2, 3, AREA), (3, 4, AREA)]
        return cat.build_problem(perfect_gas(R_AIR, GAMMA), els, edges, mdot_ref=MDOT, p_ref=1e5, h_ref=CP * 300.0)

    x0 = None
    for ddT in (1.0, 150.0, 300.0, 450.0, dT):
        prob = mk(MDOT * CP * ddT)
        res = solve(prob, x0=x0)
        assert res.converged, f"mean solve failed at dT={ddT}"
        x0 = res.x
    return prob, res.x, states_table(prob, res.x)


def _response(prob, x, isentropic=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return forced_response(prob, x, FREQS, isentropic=isentropic)


def _marble_candel(est):
    """Marble--Candel ``(R, R_s)`` at the nozzle-inlet mean state (literature formulas)."""
    rho, c, M = (float(est[k, E_NOZZLE]) for k in (ES_RHO, ES_C, ES_M))
    R = (2.0 - GM1 * M) / (2.0 + GM1 * M)
    R_s = (c / rho) * M / (2.0 + GM1 * M)
    return R, R_s


# --------------------------------------------------------------------------
# 1. Generation: the flame turns upstream acoustic forcing into an entropy wave
# --------------------------------------------------------------------------


def test_flame_generates_entropy_from_upstream_forcing():
    """No entropy upstream of the flame; a finite entropy wave downstream of it."""
    prob, x, est = _rig(n=1.0, tau=2e-3)
    assert 0.0 < float(est[ES_M, E_FLAME_OUT]) < 1.0  # subsonic hot flow
    fr = _response(prob, x)
    h_cold = np.abs(fr.waves(E_COLD)[:, 2])  # entropy on the flame approach
    h_hot = np.abs(fr.waves(E_FLAME_OUT)[:, 2])  # entropy just downstream of the flame
    assert np.all(h_hot > 1e-5)  # genuinely generated at every frequency
    assert np.all(h_cold < 1e-10 * h_hot)  # nothing upstream -- the flame is the source


def test_unsteady_heat_release_modulates_generated_entropy():
    """The FTF is a genuine entropy-generation mechanism, not just the fixed-mean baseline.

    A fixed-power flame already makes some entropy (the mean heat release seen through the
    mass-flow fluctuation, ``n = 0``).  Turning on the unsteady heat release (``n > 0``)
    materially changes the entropy spot -- the feedback makes the closed-loop response a
    non-trivial function of the gain, so we assert a material, not monotone, change.
    """

    def h_hot(n):
        prob, x, _ = _rig(n=n, tau=2e-3)
        return _response(prob, x).waves(E_FLAME_OUT)[:, 2]

    h0, h2 = h_hot(0.0), h_hot(2.0)
    assert np.all(np.abs(h0) > 1e-5)  # the fixed-power flame alone already seeds entropy
    assert np.max(np.abs(h2 - h0)) > 0.1 * np.max(np.abs(h0))  # the FTF materially adds to it


# --------------------------------------------------------------------------
# 2. Transport: the entropy spot convects to the nozzle at the mean speed
# --------------------------------------------------------------------------


def test_entropy_convects_flame_to_nozzle():
    """``h(nozzle) = h(flame) * exp(-i omega L/u)`` -- magnitude conserved, pure convective lag."""
    prob, x, est = _rig(n=1.0, tau=2e-3)
    u_hot = float(est[ES_U, E_FLAME_OUT])
    fr = _response(prob, x)
    h_flame = fr.waves(E_FLAME_OUT)[:, 2]
    h_nozzle = fr.waves(E_NOZZLE)[:, 2]
    omega = 2.0 * np.pi * FREQS
    assert np.allclose(h_nozzle, h_flame * np.exp(-1j * omega * L2 / u_hot), rtol=1e-6, atol=1e-12)
    assert np.allclose(np.abs(h_nozzle), np.abs(h_flame), rtol=1e-9)  # convection is lossless


# --------------------------------------------------------------------------
# 3. Conversion: the nozzle turns the entropy spot into sound (indirect noise)
# --------------------------------------------------------------------------


def test_entropy_converts_to_indirect_noise_at_nozzle():
    """The reflected wave obeys ``g = R f + R_s h`` with the Marble--Candel coefficients.

    The ``R_s h`` (indirect, entropy) part is a substantial fraction of the reflected
    wave -- the entropy spot is radiating sound, not just convecting out.
    """
    prob, x, est = _rig(n=1.0, tau=2e-3)
    R, R_s = _marble_candel(est)
    fr = _response(prob, x)
    f, g, h = (fr.waves(E_NOZZLE)[:, k] for k in (0, 1, 2))
    assert np.allclose(g, R * f + R_s * h, rtol=1e-6, atol=1e-9)  # the closure converts entropy
    assert np.max(np.abs(R_s * h)) > 0.2 * np.max(np.abs(g))  # indirect noise genuinely matters


def test_isentropic_analysis_misses_the_indirect_noise():
    """The default isentropic mode pins ``h = 0`` away from the flame, so it drops the
    entropy-noise path: no entropy reaches the nozzle and the reflected wave differs."""
    prob, x, _ = _rig(n=1.0, tau=2e-3)
    fr_full = _response(prob, x, isentropic=False)
    fr_isen = _response(prob, x, isentropic=True)
    h_nozzle_isen = np.abs(fr_isen.waves(E_NOZZLE)[:, 2])
    assert np.all(h_nozzle_isen < 1e-12)  # entropy pinned out at the nozzle
    # the indirect noise is a real, missing contribution: the reflected wave changes
    g_full = fr_full.waves(E_NOZZLE)[:, 1]
    g_isen = fr_isen.waves(E_NOZZLE)[:, 1]
    assert np.max(np.abs(g_full - g_isen)) > 0.1 * np.max(np.abs(g_full))
