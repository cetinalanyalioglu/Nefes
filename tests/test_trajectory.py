"""Eigenvalue-trajectory continuation of the perturbation spectrum (nefes.perturbation.stability.trajectory).

The tool seeds the spectrum once with :func:`eigenmodes`, then continues each mode through a
parameter sweep by a predictor-corrector march.  These tests pin the two things continuation
must get right on a Rijke tube:

* a mode's path is *continuous and identity-preserving* -- the corrector tracks the same mode
  rather than re-matching independent spectra (verified against a reference eigenmode solve at
  an intermediate parameter value);
* the FTF-gain continuation ``n: 1 -> 0`` separates an **ITA** mode (growth dives as the flame
  is turned off) from a **cavity/acoustic** mode (parks on the passive duct resonance).

Run in the ``nefes`` env (numba).
"""

import warnings

import numpy as np
import pytest

from nefes.elements import catalog as cat
from nefes.elements.dynamic_source import n_tau_flame
from nefes.perturbation.operator.boundary_bc import PerturbationBC
from nefes.perturbation import eigenmodes, eigenvalue_trajectory, TrajectoryResult
from nefes.shell import Network
from nefes.thermo.configure import perfect_gas

R, GAMMA = 287.0, 1.4
CP = GAMMA * R / (GAMMA - 1.0)
AREA, L_COLD, L_HOT = 0.01, 0.25, 0.75
TT_IN, P_OUT, MDOT, DT = 300.0, 1.0e5, 0.001, 1000.0
QDOT = MDOT * CP * DT
TAU = 3.0e-3
BAND, GROWTH = (50.0, 700.0), (-500.0, 500.0)


def make_rijke(n, tau=TAU):
    """An *unsolved* near-stagnant Rijke network with an n-tau flame (gain ``n``)."""
    net = Network(perfect_gas(R, GAMMA), p_ref=P_OUT, T_ref=TT_IN)
    inlet = net.add(cat.mass_flow_inlet(MDOT, TT_IN, perturbation_bc=PerturbationBC.open_end()))
    cold = net.add(cat.duct(L_COLD))
    flame = net.add(cat.heat_release_flame(QDOT))
    hot = net.add(cat.duct(L_HOT))
    outlet = net.add(cat.pressure_outlet(P_OUT, perturbation_bc=PerturbationBC.open_end()))
    net.connect(inlet, cold, area=AREA)
    ref = net.connect(cold, flame, area=AREA)
    net.connect(flame, hot, area=AREA)
    net.connect(hot, outlet, area=AREA)
    net.set_dynamic_source(flame, n_tau_flame(n, tau, ref_edge=ref))
    return net


@pytest.fixture(scope="module")
def gain_sweep():
    """Continuation of the sub-700 Hz spectrum as the FTF gain is dialed 1 -> ~0."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return eigenvalue_trajectory(
            build=lambda n: make_rijke(n),
            params=np.linspace(1.0, 0.05, 49),
            freq_band=BAND,
            growth_band=GROWTH,
            isentropic=True,
            param_name="FTF gain n",
            max_step_halvings=6,
        )


def test_seed_modes_each_get_a_branch(gain_sweep):
    """Every seeded eigenmode is continued into exactly one branch, starting where it was seeded."""
    traj = gain_sweep
    assert isinstance(traj, TrajectoryResult)
    assert traj.n_branches == traj.seed.n_modes == 4
    seeded = np.sort(traj.seed.freqs)
    started = np.sort([b.freqs[0] for b in traj.branches])
    assert np.allclose(seeded, started, atol=1e-6)


def test_branches_are_continuous_and_identity_preserving(gain_sweep):
    """Each branch passes through the reference eigenvalue at an intermediate gain.

    Continuation must follow the *same* mode, not jump between modes: at a mid-sweep gain the
    tracked points reproduce an independent eigenmode solve (to a few Hz / a few 1/s).
    """
    traj = gain_sweep
    n_mid = 0.5
    # freq_band/growth_band are the semi-axes of an *ellipse*, and by mid-sweep one branch has
    # drifted onto its rim (growth ~ -487 against a -500 half-axis, at 0.28 of the frequency
    # half-axis).  The oracle gets a wider growth window so every branch has a mode to match.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = eigenmodes(
            make_rijke(n_mid).solve().problem,
            make_rijke(n_mid).solve().x,
            freq_band=BAND,
            growth_band=(2.0 * GROWTH[0], 2.0 * GROWTH[1]),
            isentropic=True,
        )
    ref_w = np.array([complex(2 * np.pi * f, -g) for f, g in zip(ref.freqs, ref.growth_rates)])
    for b in traj.branches:
        k = int(np.argmin(np.abs(b.params - n_mid)))
        w = complex(2 * np.pi * b.freqs[k], -b.growth[k])
        assert np.min(np.abs(ref_w - w)) / (2 * np.pi) < 5.0  # within 5 Hz-equivalent of a real mode


def test_gain_continuation_separates_ita_from_cavity(gain_sweep):
    """The ITA mode's growth dives as ``n -> 0``; the cavity mode parks on a passive resonance."""
    traj = gain_sweep
    # passive cavity spectrum (flame off) -- the acoustic resonances ITA modes do NOT belong to
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cav = eigenmodes(
            make_rijke(0.0).solve().problem,
            make_rijke(0.0).solve().x,
            freq_band=BAND,
            growth_band=GROWTH,
            isentropic=True,
        )
    cav_w = np.array([complex(2 * np.pi * f, -g) for f, g in zip(cav.freqs, cav.growth_rates)])

    def fate(seed_f):
        b = min(traj.branches, key=lambda br: abs(br.freqs[0] - seed_f))
        near = np.min(np.abs(cav_w - b.end)) / (2 * np.pi)
        drop = b.growth[0] - b.growth[-1]  # positive == growth fell over the sweep
        return near, drop

    # 162.6 Hz sits on the ITA ladder (2k+1)/(2 tau) ~ 167 Hz: growth must plunge, leaving any cavity mode
    near_ita, drop_ita = fate(162.6)
    assert drop_ita > 500.0
    assert near_ita > 50.0
    # 306 Hz is the duct mode: it lands on a passive cavity resonance with only a mild growth change
    near_cav, drop_cav = fate(306.1)
    assert near_cav < 25.0
    assert abs(drop_cav) < 200.0


def test_accepts_presolved_solution_and_branch_tangent(gain_sweep):
    """``build`` may return an already-solved solution; branches expose a finite sensitivity tangent."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        traj = eigenvalue_trajectory(
            build=lambda n: make_rijke(n).solve(),  # a Solution, not a Network
            params=np.linspace(1.0, 0.6, 9),
            freq_band=BAND,
            growth_band=GROWTH,
            isentropic=True,
        )
    assert traj.n_branches == 4
    b = traj.branches[0]
    assert b.tangent.shape == b.omega.shape
    assert np.all(np.isfinite(b.tangent))


def test_topology_change_is_rejected():
    """Continuation requires a fixed graph: a build that changes the edge count must error out."""

    def split_rijke(p):
        """Valid Rijke whose hot duct is split into two series ducts once ``p < 0.9`` (+1 edge)."""
        net = Network(perfect_gas(R, GAMMA), p_ref=P_OUT, T_ref=TT_IN)
        inlet = net.add(cat.mass_flow_inlet(MDOT, TT_IN, perturbation_bc=PerturbationBC.open_end()))
        cold = net.add(cat.duct(L_COLD))
        flame = net.add(cat.heat_release_flame(QDOT))
        hot = net.add(cat.duct(L_HOT if p >= 0.9 else 0.5 * L_HOT))
        outlet = net.add(cat.pressure_outlet(P_OUT, perturbation_bc=PerturbationBC.open_end()))
        net.connect(inlet, cold, area=AREA)
        ref = net.connect(cold, flame, area=AREA)
        net.connect(flame, hot, area=AREA)
        if p < 0.9:  # splice a second hot duct in series -> one extra edge / operator grows
            hot2 = net.add(cat.duct(0.5 * L_HOT))
            net.connect(hot, hot2, area=AREA)
            net.connect(hot2, outlet, area=AREA)
        else:
            net.connect(hot, outlet, area=AREA)
        net.set_dynamic_source(flame, n_tau_flame(1.0, TAU, ref_edge=ref))
        return net

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="topology"):
            eigenvalue_trajectory(
                build=split_rijke,
                params=np.linspace(1.0, 0.8, 5),
                freq_band=BAND,
                growth_band=GROWTH,
                isentropic=True,
            )


def test_repr_is_a_clean_summary(gain_sweep):
    """The repr is a compact human summary -- no raw array dumps."""
    text = repr(gain_sweep)
    assert text.startswith("TrajectoryResult")
    assert "branch" in text
    assert "array(" not in text and "\n  [" in text  # per-branch lines, not an ndarray repr
    assert isinstance(gain_sweep._repr_html_(), str)
