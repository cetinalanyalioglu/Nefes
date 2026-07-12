"""dP-scaled continuation start: size the artificial-friction coefficient to the domain dP.

The artificial-resistance continuation stamps an artificial pressure drop ``kappa * R_art * mdot``
into the interior pressure rows.  A constant ``R_art = 1`` (an absolute coefficient)
over-perturbs low-``dP`` / high-``mdot`` networks.  ``kappa_scale="dp"`` (the default) sets
``R_art = min(domain_max_dp / mdot_ref, 1)`` so the injected drop is capped at a fraction
``kappa`` of the real driving drop -- softening the friction where it would otherwise be too
strong while leaving healthy-``dP`` networks (``R_art = 1``) and the final exact stage
(``kappa = 0``) untouched.
"""

import numpy as np
import pytest

from nefes.elements import catalog as cat
from nefes.shell.build import build_problem
from nefes.solver import solve
from nefes.solver.control import domain_max_dp
from nefes.thermo.configure import perfect_gas

R_AIR, GAMMA = 287.0, 1.4
CP = GAMMA * R_AIR / (GAMMA - 1.0)
CFG = perfect_gas(R_AIR, GAMMA)
H_REF = CP * 300.0


def _duct(pt, p_out, mdot_ref, area=0.1, length=0.5):
    els = [cat.total_pressure_inlet(pt, 300.0), cat.duct(length), cat.pressure_outlet(p_out, 300.0)]
    return build_problem(CFG, els, [(0, 1, area), (1, 2, area)], mdot_ref, 101325.0, H_REF)


def _r_art(prob, mode="dp"):
    """Reproduce the friction resistance the solver would use, for assertions."""
    if mode == "absolute":
        return 1.0
    dp = domain_max_dp(prob)
    return min(dp / prob.var_scale[0], 1.0) if dp > 0.0 else 1.0


# --------------------------------------------------------------------------- #
# domain_max_dp helper
# --------------------------------------------------------------------------- #
def test_domain_max_dp_spans_pressure_boundaries():
    prob = _duct(1.20e5, 1.01325e5, mdot_ref=10.0)
    assert domain_max_dp(prob) == pytest.approx(1.20e5 - 1.01325e5)


def test_domain_max_dp_uses_widest_pair():
    # one inlet, two outlets at different pressures -> span is max minus min.
    els = [
        cat.total_pressure_inlet(1.10e5, 300.0),
        cat.splitter(),
        cat.pressure_outlet(1.05e5, 300.0),
        cat.pressure_outlet(0.98e5, 300.0),
    ]
    prob = build_problem(CFG, els, [(0, 1, 0.1), (1, 2, 0.1), (1, 3, 0.1)], 10.0, 101325.0, H_REF)
    assert domain_max_dp(prob) == pytest.approx(1.10e5 - 0.98e5)


def test_domain_max_dp_zero_when_mass_driven():
    # a single pressure reference (mass-flow inlet + one outlet) -> no a-priori drop.
    els = [cat.mass_flow_inlet(8.0, 300.0), cat.duct(0.5), cat.pressure_outlet(1.0e5, 300.0)]
    prob = build_problem(CFG, els, [(0, 1, 0.1), (1, 2, 0.1)], 8.0, 101325.0, H_REF)
    assert domain_max_dp(prob) == 0.0


# --------------------------------------------------------------------------- #
# Resistance sizing (the cap)
# --------------------------------------------------------------------------- #
def test_dp_scaling_softens_friction_for_low_dp_high_mdot():
    # dP = 75 Pa with a large reference flow (mdot_ref = 300): dp/mdot_ref = 0.25 < 1,
    # so the dP-scaled coefficient is a quarter of the absolute one.
    prob = _duct(1.0140e5, 1.01325e5, mdot_ref=300.0, area=0.3)
    assert domain_max_dp(prob) == pytest.approx(75.0)
    assert _r_art(prob, "dp") == pytest.approx(0.25)
    assert _r_art(prob, "absolute") == 1.0


def test_dp_scaling_is_noop_for_healthy_dp():
    # dP = 18.7 kPa with mdot_ref = 10: dp/mdot_ref >> 1, capped to 1 -> identical to absolute.
    prob = _duct(1.20e5, 1.01325e5, mdot_ref=10.0)
    assert _r_art(prob, "dp") == 1.0

    # and the solve path is byte-for-byte the absolute one (same iterations + state).
    r_dp = solve(prob, kappa_scale="dp")
    r_abs = solve(prob, kappa_scale="absolute")
    assert r_dp.converged and r_abs.converged
    assert r_dp.iterations == r_abs.iterations
    assert np.allclose(r_dp.x, r_abs.x, rtol=0, atol=0)


def test_mass_driven_falls_back_to_absolute():
    els = [cat.mass_flow_inlet(8.0, 300.0), cat.duct(0.5), cat.pressure_outlet(1.0e5, 300.0)]
    prob = build_problem(CFG, els, [(0, 1, 0.1), (1, 2, 0.1)], 8.0, 101325.0, H_REF)
    assert _r_art(prob, "dp") == 1.0  # no a-priori dP -> absolute coefficient


# --------------------------------------------------------------------------- #
# Equivalence and the final exact stage
# --------------------------------------------------------------------------- #
def test_modes_reach_the_same_solution():
    # The schedules differ only in the (vanishing) friction path; the kappa = 0 stage
    # is the exact equations, so both modes land on the same converged state.
    for prob in (_duct(1.0140e5, 1.01325e5, 300.0, area=0.3), _duct(1.20e5, 1.01325e5, 10.0)):
        r_dp = solve(prob, kappa_scale="dp")
        r_abs = solve(prob, kappa_scale="absolute")
        assert r_dp.converged and r_abs.converged
        assert np.allclose(r_dp.x, r_abs.x, rtol=1e-9, atol=1e-9)


def test_low_dp_network_converges_under_dp_scaling():
    prob = _duct(1.0140e5, 1.01325e5, mdot_ref=300.0, area=0.3)
    res = solve(prob)  # default kappa_scale="dp"
    assert res.converged


def test_invalid_kappa_scale_raises():
    prob = _duct(1.20e5, 1.01325e5, mdot_ref=10.0)
    with pytest.raises(ValueError, match="kappa_scale"):
        solve(prob, kappa_scale="relative")
