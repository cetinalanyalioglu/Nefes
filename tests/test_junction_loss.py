"""The junction's per-branch loss-coefficient closure (tabulated junction data).

When ``K`` is given, each branch is charged a total-pressure loss ``K_e * (p_t,e - p_e)`` on its
own dynamic head, sign-symmetric in the flow direction so both the combining (inflow) and dividing
(outflow) branches dissipate.  These tests pin the limits (``K = 0`` is exact lossless
total-pressure continuity; a larger coefficient dissipates more), the per-branch resolution (a
branch with a higher coefficient loses more total pressure), the broadcast/per-port equivalence,
the second-law admissibility (no branch gains total pressure, entropy production is non-negative),
and the input validation.

See ``test_junction.py`` for the geometry-free ``recovery`` closure and
``test_complex_step_safety.py`` for the complex-step sweep of this closure through flow reversal.
"""

import numpy as np
import pytest

import nefes
from nefes.elements import catalog as cat

R_AIR, GAMMA = 287.0, 1.4
CP = GAMMA * R_AIR / (GAMMA - 1.0)


def _gas():
    return nefes.perfect_gas(R_AIR, GAMMA)


def _entropy(T, p):
    return CP * np.log(T) - R_AIR * np.log(p)


def _distribution_network(manifold):
    """One inflow (edge 0) distributed to two pressure outlets (edges 1, 2); node 1 is manifold."""
    nodes = [
        cat.total_pressure_inlet(2.0e5, 300.0),
        manifold,
        cat.pressure_outlet(1.9e5),
        cat.pressure_outlet(1.92e5),
    ]
    edges = [(0, 1, 0.10), (1, 2, 0.03), (1, 3, 0.03)]
    return nefes.Network(_gas(), nodes, edges, p_ref=1.0e5, T_ref=300.0)


def test_zero_coefficient_is_exact_lossless_continuity():
    """``K = 0`` on every branch ties the total pressures equal exactly (the lossless manifold).

    With zero coefficients the sign-symmetric loss term vanishes identically, so the couplings
    reduce to exact total-pressure equality with no smoothing floor: every branch leaves at the
    inflow's own total pressure.
    """
    sol = _distribution_network(cat.junction(K=0.0)).solve()
    assert sol.converged
    pt = sol.field("p_t")
    # inflow (edge 0) and both outflow branches (edges 1, 2) share one total pressure, exactly
    assert pt[1] == pytest.approx(pt[0], rel=1e-9)
    assert pt[2] == pytest.approx(pt[0], rel=1e-9)


def test_larger_coefficient_dissipates_more():
    """Raising the (broadcast) loss coefficient lowers the distributed branches' total pressure."""
    out_pt = []
    for k in (0.0, 0.3, 0.8):
        sol = _distribution_network(cat.junction(K=k)).solve()
        assert sol.converged, k
        assert np.abs(sol.field("M")).max() < 1.0
        out_pt.append(0.5 * (sol.field("p_t")[1] + sol.field("p_t")[2]))
    assert out_pt[0] > out_pt[1] > out_pt[2]  # more loss -> less recovered total pressure


def test_per_branch_coefficients_charge_each_branch_distinctly():
    """A branch with a higher loss coefficient leaves at a lower total pressure than a lower one.

    Ports are in wired order: port 0 the inflow (edge 0), then the two outflow branches (edges 1,
    2).  With a heavier coefficient on branch 1 than branch 2, branch 1 loses more.
    """
    # equal back pressures so the only asymmetry between the branches is their loss coefficient
    nodes = [
        cat.total_pressure_inlet(2.0e5, 300.0),
        cat.junction(K=[0.1, 0.9, 0.2]),  # port 0 inflow, then branch 1 (heavy) and branch 2 (light)
        cat.pressure_outlet(1.9e5),
        cat.pressure_outlet(1.9e5),
    ]
    edges = [(0, 1, 0.10), (1, 2, 0.03), (1, 3, 0.03)]
    sol = nefes.Network(_gas(), nodes, edges, p_ref=1.0e5, T_ref=300.0).solve()
    assert sol.converged
    pt = sol.field("p_t")
    assert pt[1] < pt[2]  # the heavier-loss branch (edge 1) leaves with less total pressure


def test_broadcast_matches_a_uniform_per_branch_list():
    """A scalar ``K`` is the same element as a per-port list of that same coefficient."""
    scalar = _distribution_network(cat.junction(K=0.4)).solve()
    listed = _distribution_network(cat.junction(K=[0.4, 0.4, 0.4])).solve()
    assert scalar.converged and listed.converged
    assert np.allclose(scalar.field("mdot"), listed.field("mdot"), rtol=1e-9)
    assert np.allclose(scalar.field("p_t"), listed.field("p_t"), rtol=1e-9)


def test_merge_respects_the_second_law():
    """Merging two unequal streams through a loss-coefficient junction is admissible.

    The node total pressure stays at or below every inflow's and the merge generates entropy,
    whatever the branch coefficients.
    """
    nodes = [
        cat.total_pressure_inlet(2.2e5, 400.0),
        cat.total_pressure_inlet(2.0e5, 300.0),
        cat.junction(K=[0.3, 0.5, 0.2]),  # two inflows (edges 0, 1) + one outflow (edge 2)
        cat.pressure_outlet(1.8e5),
    ]
    edges = [(0, 2, 0.02), (1, 2, 0.02), (2, 3, 0.05)]
    sol = nefes.Network(_gas(), nodes, edges, p_ref=1.0e5, T_ref=350.0).solve()
    assert sol.converged
    assert sol.verify() == []
    assert np.abs(sol.field("M")).max() < 1.0

    pt = sol.field("p_t")
    assert pt[2] <= min(pt[0], pt[1]) * (1.0 + 1e-6)  # no manufactured total pressure

    mdot, T, p = sol.field("mdot"), sol.field("T"), sol.field("p")
    s = _entropy(T, p)
    sgen = mdot[2] * s[2] - (mdot[0] * s[0] + mdot[1] * s[1])
    assert sgen > 0.0  # an irreversible merge generates entropy


def test_validation_rejects_bad_coefficients():
    """Negative coefficients and a single-entry list are rejected at construction."""
    with pytest.raises(ValueError):
        cat.junction(K=-0.1)
    with pytest.raises(ValueError):
        cat.junction(K=[0.2, -0.3, 0.1])
    with pytest.raises(ValueError, match="broadcast"):
        cat.junction(K=[0.5])  # a length-1 list is ambiguous; a scalar broadcasts instead


def test_per_branch_list_length_must_match_the_port_count():
    """A per-branch coefficient list must carry exactly one entry per wired port."""
    # a 3-port junction given only two coefficients
    nodes = [
        cat.total_pressure_inlet(2.0e5, 300.0),
        cat.junction(K=[0.2, 0.3]),
        cat.pressure_outlet(1.9e5),
        cat.pressure_outlet(1.9e5),
    ]
    edges = [(0, 1, 0.10), (1, 2, 0.03), (1, 3, 0.03)]
    with pytest.raises(ValueError, match="one K per port"):
        nefes.Network(_gas(), nodes, edges, p_ref=1.0e5, T_ref=300.0).compile()
