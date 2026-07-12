"""The subsonic-scope backstop and its global switch.

The steady residual admits a spurious supersonic isentropic root beside the physical subsonic
one at over-critical operating points; a cold seed can reach it.  ``nefes.config.enforce_subsonic``
(on by default) makes ``solve`` re-solve such a case onto the subsonic branch.  These tests pin
that a bare solve stays subsonic, that the switch restores the raw branch, and that genuine
choking (a real throat pinning at M = 1) is untouched.
"""

import warnings

import numpy as np
import pytest

import nefes
from nefes.elements import catalog as cat


# The converging nozzle whose cold solve lands on the supersonic branch at this back pressure:
# reservoir (2 bar) -> feed pipe -> isentropic contraction -> tailpipe -> 1.5 bar outlet.
PT_IN, T_IN, P_OUT = 2.0e5, 300.0, 1.5e5
A_FEED, A_THROAT = 0.020, 0.010


def _converging_nozzle(p_out=P_OUT):
    # The reference scales matter: mdot_ref = 5.0 (the shipped case's value) puts the cold seed
    # in the basin of the spurious supersonic root, which is exactly what the backstop must undo.
    return nefes.Network(
        nodes=[
            cat.total_pressure_inlet(PT_IN, T_IN, name="reservoir"),
            cat.duct(0.3, name="feed-pipe"),
            cat.isentropic_area_change(name="nozzle"),
            cat.duct(0.3, name="tailpipe"),
            cat.pressure_outlet(p_out, T_IN, name="back-pressure"),
        ],
        edges=[(0, 1, A_FEED), (1, 2, A_FEED), (2, 3, A_THROAT), (3, 4, A_THROAT)],
        p_ref=101325.0,
        T_ref=T_IN,
        mdot_ref=5.0,
    )


@pytest.fixture(autouse=True)
def _restore_flag():
    """Every test leaves the global switch back on its default."""
    yield
    nefes.config.enforce_subsonic = True


def test_default_solve_lands_subsonic():
    """A bare solve returns the physical subsonic branch (no manual seeding, no warning)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sol = _converging_nozzle().solve()
    assert sol.converged
    assert sol.field("M").max() < 1.0  # subsonic, not the M = 1.43 spurious branch
    assert not [w for w in caught if "supersonic" in str(w.message)]  # silent recovery


def test_switch_off_exposes_the_supersonic_branch():
    """With the guard off, the raw cold solve reaches the supersonic branch."""
    nefes.config.enforce_subsonic = False
    sol = _converging_nozzle().solve()
    assert sol.converged
    assert sol.field("M").max() > 1.2  # the spurious supersonic throat (~1.43)


def test_per_solve_override_beats_the_global():
    """``solve(enforce_subsonic=...)`` overrides the global flag either way."""
    net = _converging_nozzle()
    nefes.config.enforce_subsonic = False  # global off ...
    assert net.solve(enforce_subsonic=True).field("M").max() < 1.0  # ... local on wins
    nefes.config.enforce_subsonic = True  # global on ...
    assert net.solve(enforce_subsonic=False).field("M").max() > 1.2  # ... local off wins


def test_choking_is_not_disturbed():
    """A real throat still chokes: M pins at 1 and the mass flow saturates below critical."""
    R, gamma = 287.0, 1.4
    flux_star = np.sqrt(gamma / R) * (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
    mdot_max = PT_IN / np.sqrt(T_IN) * flux_star * A_THROAT

    net = _converging_nozzle()
    prev = None
    for ratio in (0.75, 0.53, 0.45, 0.30):
        sol = net.with_params({"back-pressure.p": ratio * PT_IN}).solve(x0=prev.x if prev else None)
        prev = sol
        assert sol.converged
        assert sol.field("M").max() <= 1.0 + 1e-2  # never supersonic
    # below critical the throat is sonic and the mass flow is capped at the analytic maximum
    assert sol.edge(2)["M"] == pytest.approx(1.0, abs=1e-2)
    assert sol.edge(2)["mdot"] == pytest.approx(mdot_max, rel=2e-3)


def test_subsonic_case_is_untouched_by_the_backstop():
    """A comfortably subsonic solve is identical with the guard on or off (no false recovery)."""
    net = _converging_nozzle(p_out=1.9e5)  # ratio 0.95, deeply subsonic
    on = net.solve(enforce_subsonic=True)
    off = net.solve(enforce_subsonic=False)
    assert on.field("M").max() < 0.5
    assert np.allclose(on.field("M"), off.field("M"), rtol=1e-10)
