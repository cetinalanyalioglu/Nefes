"""The subsonic-scope backstop and its global switch.

The steady residual admits a spurious supersonic isentropic root beside the physical subsonic
one at over-critical operating points; a cold seed can reach it.  ``nefes.config.enforce_subsonic``
(on by default) makes ``solve`` re-solve such a case onto the subsonic branch.  These tests pin
that a bare solve stays subsonic, that the switch restores the raw branch, and that genuine
choking (a real throat pinning at M = 1) is untouched.
"""

import time
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


def _resistance_free_ring():
    """Two bare (resistance-free) parallel paths between the same pair of junctions.

    ``in -> j0``; then ``j0 -> j1 -> j2`` and ``j0 -> j3 -> j2`` with no loss on the ring
    edges; then ``j2 -> out``.  The split between the two parallel paths is a circulation the
    mean-flow balances do not pin down, so a solve can grow it until the tiny ring edges run
    supersonic -- the failure the can-annular interconnector ring hit before its tubes were given
    real friction.  The backstop cannot recover it, which is exactly the case that must not be
    accepted.  The header nodes are junctions on the full-dump limit (``recovery = 0``), the
    conditioning-robust closure that carries the solve onto the spurious supersonic root.
    """
    return nefes.Network(
        nodes=[
            cat.mass_flow_inlet(10.0, 300.0, name="in"),
            cat.junction(recovery=0.0, name="j0"),
            cat.junction(recovery=0.0, name="j1"),
            cat.junction(recovery=0.0, name="j2"),
            cat.junction(recovery=0.0, name="j3"),
            cat.pressure_outlet(1.0e5, 300.0, name="out"),
        ],
        edges=[(0, 1, 0.05), (1, 2, 0.001), (2, 3, 0.001), (1, 4, 0.001), (4, 3, 0.001), (3, 5, 0.05)],
        p_ref=101325.0,
        T_ref=300.0,
        mdot_ref=10.0,
    )


def test_unremovable_supersonic_is_not_converged():
    """When the backstop cannot remove a supersonic edge, the result is reported NOT converged."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sol = _resistance_free_ring().solve()  # enforce on (default)
    assert not sol.converged  # a wildly supersonic branch is never accepted as converged
    assert [w for w in caught if "supersonic" in str(w.message)]  # and it says why


def test_switch_off_accepts_the_supersonic_branch_as_converged():
    """With the guard off the raw supersonic branch is returned as converged (the opt-out)."""
    sol = _resistance_free_ring().solve(enforce_subsonic=False)
    assert sol.converged
    assert sol.field("M").max() > 1.0  # the accepted branch really is supersonic


def test_rescued_solve_reports_its_total_time():
    """A rescued solve times the whole call, the nested re-solve included.

    This case reaches the spurious branch cold (``test_switch_off_exposes_the_supersonic_branch``
    pins the raw M = 1.43), so the default solve here pays for two solves.  ``elapsed`` must
    account for both rather than for the rescue leg alone.
    """
    _converging_nozzle().solve()  # warm the kernels: their one-off compilation is not solve time
    t0 = time.perf_counter()
    sol = _converging_nozzle().solve()
    span = time.perf_counter() - t0
    assert sol.converged
    assert sol.field("M").max() < 1.0  # the rescue ran
    assert 0.0 < sol.elapsed <= span
    assert sol.elapsed >= 0.5 * span  # the whole call, not one leg of it
