"""Stress tests for the junction: the network's central manifold element.

The junction carries the most smoothing machinery of any element -- an inflow indicator, a smooth
minimum over the inflows, a dynamic-head flow envelope that keeps a stagnant branch clean, a
sign-symmetric per-branch loss, and a mode selector across three closures (recovery, tabulated
``K``, static pressure).  These tests hammer all of it: every closure across merge / distribute /
mixed / dead-leg / many-port topologies and randomized configurations, asserting the invariants
the closures promise -- mass conservation, no manufactured total pressure, non-negative entropy
production, a common static pressure (static-p closure), subsonic convergence, an eps-independent
perturbation response at a dead-leg branch, and a mean flow untouched by a stagnant branch.

They are deliberately more aggressive than ``test_junction`` / ``test_junction_loss`` (fuzzing,
many ports, extremes), so a regression in the element's delicate machinery is caught here.
"""

import numpy as np
import pytest

import nefes
from nefes.elements import catalog as cat
from nefes.perturbation import eigenmodes, modal_energy_balance
from nefes.perturbation.operator.boundary_bc import PerturbationBC
from nefes.shell.build import build_problem
from nefes.solver import solve
from nefes.thermo.configure import perfect_gas

R_AIR, GAMMA = 287.0, 1.4
CP = GAMMA * R_AIR / (GAMMA - 1.0)


def _gas():
    return nefes.perfect_gas(R_AIR, GAMMA)


def _entropy(T, p):
    return CP * np.log(T) - R_AIR * np.log(p)


# --------------------------------------------------------------------------- #
# The closures under test, as constructors of a manifold node.  Each takes the port count, which
# only the per-branch forms need (their parameter list is one entry per port).
# --------------------------------------------------------------------------- #
def _alternating_recovery(deg):
    """Recovery factors alternating the ideal and the full dump across the ports.

    The most mixed setting the closure admits: neighbouring branches sit at opposite ends of the
    dump-to-ideal range, so both loss terms act at once on different branches of the same node.
    """
    return [1.0 if i % 2 == 0 else 0.0 for i in range(deg)]


CLOSURES = {
    "recovery=1": lambda deg: cat.junction(recovery=1.0),
    "recovery=0.5": lambda deg: cat.junction(recovery=0.5),
    "recovery=0": lambda deg: cat.junction(recovery=0.0),
    "recovery alternating 1/0": lambda deg: cat.junction(recovery=_alternating_recovery(deg)),
    "K=0 (lossless)": lambda deg: cat.junction(K=0.0),
    "K=0.5 (broadcast)": lambda deg: cat.junction(K=0.5),
    "static_pressure": lambda deg: cat.junction(static_pressure=True),
}
# Closures that tie an effective total pressure (so "node p_t <= min inflow p_t" applies); the
# static-p header does not, and is checked separately.
TOTAL_P_CLOSURES = [k for k in CLOSURES if k != "static_pressure"]


def _pinned_merge(manifold, feeds, a_in=0.02, a_out=None, p_out=1.5e5, loss_K=8.0):
    """``N`` total-pressure feeds, each through a loss, merging through ``manifold`` to an outlet.

    The per-branch losses pin the split for every closure, so the merge is well posed even at the
    resistance-free ``recovery = 1`` / ``K = 0`` limits.  ``feeds`` is a list of ``(p_t, T_t)``.
    The outlet area defaults to the *total* inflow area (``N * a_in``) so the combined stream does
    not accelerate through an area contraction and stay subsonic across closures; the branch losses
    then carry the pressure drop.  Nodes: N feeds, N losses, manifold, outlet.
    """
    n = len(feeds)
    if a_out is None:
        a_out = n * a_in  # match total inflow area -> no contraction, stays subsonic
    nodes = [cat.total_pressure_inlet(pt, tt) for pt, tt in feeds]
    nodes += [cat.loss(loss_K) for _ in range(n)]
    m_idx = len(nodes)
    nodes.append(manifold)
    out_idx = len(nodes)
    nodes.append(cat.pressure_outlet(p_out))
    edges = []
    for i in range(n):
        edges.append((i, n + i, a_in))  # feed -> loss
    for i in range(n):
        edges.append((n + i, m_idx, a_in))  # loss -> manifold (inflow)
    edges.append((m_idx, out_idx, a_out))  # manifold -> outlet (outflow)
    net = nefes.Network(_gas(), nodes, edges, p_ref=1.0e5, T_ref=350.0)
    return net, m_idx, out_idx


def _node_edges(net, node):
    return [ei for ei, (t, h, _a) in enumerate(net._edges) if t == node or h == node]


def _assert_manifold_invariants(net, sol, m_idx, total_pressure=True):
    """Mass conservation, subsonic, and (for a total-pressure closure) no manufactured p_t."""
    assert sol.converged
    mdot, pt = sol.field("mdot"), sol.field("p_t")
    edges = _node_edges(net, m_idx)
    # signed mass balance at the node
    signed = 0.0
    for ei in edges:
        t, h, _a = net._edges[ei]
        signed += (1.0 if t == m_idx else -1.0) * mdot[ei]
    assert abs(signed) < 1e-6 * (abs(mdot).max() + 1.0)
    assert np.abs(sol.field("M")).max() < 1.0
    if total_pressure:
        # The node total pressure (what every OUTFLOW leaves at) stays at or below the weakest
        # inflow's; the non-weakest inflows legitimately arrive above it, so they are not bounded.
        inflow_pt, outflow_edges = [], []
        for ei in edges:
            t, h, _a = net._edges[ei]
            into_node = (mdot[ei] > 0) == (h == m_idx)  # mass actually entering the node
            if abs(mdot[ei]) < 1e-9:
                continue  # a dead branch is neither in nor out
            (inflow_pt.append(pt[ei]) if into_node else outflow_edges.append(ei))
        if inflow_pt:
            pt_min = min(inflow_pt)
            for ei in outflow_edges:
                assert pt[ei] <= pt_min * (1.0 + 1e-4), "manufactured total pressure on an outflow"


# --------------------------------------------------------------------------- #
# 1. Every closure conserves mass and respects its pressure invariant, on merges
#    of 2..5 feeds.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("closure", list(CLOSURES))
@pytest.mark.parametrize("n_feeds", [2, 3, 5])
def test_merge_conserves_mass_and_respects_invariants(closure, n_feeds):
    feeds = [(2.0e5 + 1.0e4 * i, 300.0 + 40.0 * i) for i in range(n_feeds)]
    net, m_idx, _ = _pinned_merge(CLOSURES[closure](n_feeds + 1), feeds)
    sol = net.solve()
    _assert_manifold_invariants(net, sol, m_idx, total_pressure=(closure in TOTAL_P_CLOSURES))


# --------------------------------------------------------------------------- #
# 2. Randomized fuzz: many merges, no closure ever manufactures total pressure
#    or generates negative entropy.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("closure", TOTAL_P_CLOSURES)
def test_fuzz_no_manufactured_total_pressure_or_negative_entropy(closure):
    rng = np.random.default_rng(20240607)
    checked = 0
    for _ in range(24):
        n = int(rng.integers(2, 5))
        feeds = [(rng.uniform(1.8e5, 2.6e5), rng.uniform(300.0, 500.0)) for _ in range(n)]
        p_out = float(rng.uniform(1.2e5, 1.6e5))
        loss_K = float(rng.uniform(4.0, 20.0))
        net, m_idx, out_idx = _pinned_merge(CLOSURES[closure](n + 1), feeds, p_out=p_out, loss_K=loss_K)
        sol = net.solve()
        if not sol.converged or np.abs(sol.field("M")).max() >= 1.0:
            continue  # skip an occasional ill-conditioned or transonic draw
        checked += 1
        _assert_manifold_invariants(net, sol, m_idx, total_pressure=True)
        # entropy production at the node >= 0
        mdot, T, p = sol.field("mdot"), sol.field("T"), sol.field("p")
        s = _entropy(T, p)
        edges = _node_edges(net, m_idx)
        sgen = 0.0
        for ei in edges:
            t, h, _a = net._edges[ei]
            out_of_node = (mdot[ei] > 0) == (t == m_idx)
            sgen += (1.0 if out_of_node else -1.0) * abs(mdot[ei]) * s[ei]
        assert sgen > -1e-6, f"{closure}: negative entropy production {sgen}"
    assert checked >= 12, f"{closure}: too few fuzz draws converged ({checked})"


# --------------------------------------------------------------------------- #
# 3. Many-port junctions (a header with 8 branches) converge merging and distributing.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("closure", list(CLOSURES))
def test_eight_port_merge_converges(closure):
    feeds = [(2.0e5 + 5.0e3 * i, 300.0 + 20.0 * i) for i in range(8)]
    net, m_idx, _ = _pinned_merge(CLOSURES[closure](9), feeds)
    sol = net.solve()
    _assert_manifold_invariants(net, sol, m_idx, total_pressure=(closure in TOTAL_P_CLOSURES))


@pytest.mark.parametrize("closure", list(CLOSURES))
def test_eight_way_distribution_converges(closure):
    # One feed distributed to 8 pressure outlets, each branch pinned by a loss.  The drive is
    # moderate and each branch carries a heavy loss so every closure -- including the low-loss
    # recovery = 1 / K = 0 limits, which flow fastest -- stays subsonic through the split.
    nodes = [cat.total_pressure_inlet(1.3e5, 300.0), CLOSURES[closure](9)]
    for _ in range(8):
        nodes += [cat.loss(12.0), cat.pressure_outlet(1.0e5)]
    edges = [(0, 1, 0.05)]
    for k in range(8):
        loss_node = 2 + 2 * k
        out_node = 3 + 2 * k
        edges.append((1, loss_node, 0.02))
        edges.append((loss_node, out_node, 0.02))
    net = nefes.Network(_gas(), nodes, edges, p_ref=1.0e5, T_ref=300.0)
    sol = net.solve()
    _assert_manifold_invariants(net, sol, 1, total_pressure=(closure in TOTAL_P_CLOSURES))


# --------------------------------------------------------------------------- #
# 4. The static-pressure header ties a common static pressure exactly.
# --------------------------------------------------------------------------- #
def test_static_pressure_ties_common_static_pressure():
    # a common-static-pressure header cannot itself set a pressure difference between branches, so
    # each branch carries a loss that drops the common junction pressure to its own outlet; the
    # test checks that the junction's incident edges all sit at one static pressure.
    nodes = [
        cat.total_pressure_inlet(1.2e5, 350.0),
        cat.junction(static_pressure=True),
        cat.loss(3.0),
        cat.pressure_outlet(1.0e5),
        cat.loss(3.0),
        cat.pressure_outlet(1.02e5),
        cat.loss(3.0),
        cat.pressure_outlet(1.01e5),
    ]
    edges = [(0, 1, 0.05), (1, 2, 0.02), (2, 3, 0.02), (1, 4, 0.02), (4, 5, 0.02), (1, 6, 0.02), (6, 7, 0.02)]
    net = nefes.Network(_gas(), nodes, edges, p_ref=1.0e5, T_ref=350.0)
    sol = net.solve()
    assert sol.converged
    p = sol.field("p")
    incident = _node_edges(net, 1)
    assert np.allclose([p[ei] for ei in incident], p[incident[0]], rtol=1e-6)  # one static pressure


def test_static_pressure_matches_incompressible_split():
    """Two symmetric branches off a common-static-pressure header carry equal flow."""
    nodes = [
        cat.total_pressure_inlet(1.15e5, 300.0),
        cat.junction(static_pressure=True),
        cat.pressure_outlet(1.0e5),
        cat.pressure_outlet(1.0e5),
    ]
    edges = [(0, 1, 0.05), (1, 2, 0.02), (1, 3, 0.02)]  # identical branches
    sol = nefes.Network(_gas(), nodes, edges, p_ref=1.0e5, T_ref=300.0).solve()
    assert sol.converged
    mdot = sol.field("mdot")
    assert mdot[1] == pytest.approx(mdot[2], rel=1e-6)  # symmetric split is 50/50


# --------------------------------------------------------------------------- #
# 5. The flow envelope: a dead-leg branch leaves the mean flow untouched and keeps the
#    perturbation response eps-independent (the fix for the dead-leg acoustic artifact).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("closure", ["recovery=1", "recovery=0", "K=0.5 (broadcast)"])
def test_dead_leg_wall_leaves_mean_flow_untouched(closure):
    def build(with_dead_leg):
        nodes = [
            cat.total_pressure_inlet(1.3e5, 300.0),
            CLOSURES[closure](3 if with_dead_leg else 2),
            cat.pressure_outlet(1.0e5),
        ]
        edges = [(0, 1, 0.05), (1, 2, 0.05)]
        if with_dead_leg:
            nodes.append(cat.wall())
            edges.append((1, 3, 0.05))
        return nefes.Network(_gas(), nodes, edges, p_ref=1.0e5, T_ref=300.0, mdot_ref=5.0).solve()

    plain = build(False)
    dead = build(True)
    assert plain.converged and dead.converged
    # the main branch mass flow is unchanged by adding a capped dead leg (to the smoothing floor)
    assert dead.field("mdot")[1] == pytest.approx(plain.field("mdot")[1], rel=1e-4)


def _side_branch_resonator(mdot_ref):
    """Quiescent Helmholtz resonator: a junction with a neck-duct -> cavity dead leg."""
    P0 = 1.0e5
    bc = PerturbationBC.mean_flow_open_end(driven=())
    els = [
        cat.total_pressure_inlet(P0, 300.0),
        cat.duct(0.05),
        cat.junction(),  # recovery = 1 default: the closure with the dead-leg artifact
        cat.duct(0.05),
        cat.pressure_outlet(P0, Tt_backflow=300.0, perturbation_bc=bc),
        cat.duct(0.02),
        cat.cavity(1.0e-3),
    ]
    edges = [(0, 1, 3e-3), (1, 2, 3e-3), (2, 3, 3e-3), (3, 4, 3e-3), (2, 5, 5e-4), (5, 6, 5e-4)]
    prob = build_problem(perfect_gas(R_AIR, GAMMA), els, edges, mdot_ref, P0, CP * 300.0)
    res = solve(prob)
    assert res.converged
    return prob, res.x


def test_dead_leg_resonator_growth_is_zero_and_eps_independent():
    """The lossless side-branch resonance has ~zero growth, independent of the smoothing scale.

    The dead-leg flow envelope removes the spurious O(1/eps) acoustic resistance the recovery
    closure would otherwise inject; because it is built from the branch dynamic head rather than
    the smoothing scale eps (set here by ``mdot_ref``), the growth rate does not drift with eps.
    """
    growths = []
    for mdot_ref in (0.25, 1.0, 4.0):
        res = eigenmodes(
            *_side_branch_resonator(mdot_ref),
            freq_band=(150.0, 400.0),
            growth_band=(-2000.0, 400.0),
            isentropic=True,
        )
        eb = modal_energy_balance(res, 0)
        assert eb.consistent  # the modal energy balance closes
        growths.append(eb.growth_rate)
    growths = np.array(growths)
    assert np.all(np.abs(growths) < 5.0), f"resonator growth not ~0: {growths}"
    assert np.ptp(growths) < 2.0, f"growth drifts with eps (not envelope-independent): {growths}"


# --------------------------------------------------------------------------- #
# 6. A branch that reverses direction keeps the invariants through the crossover.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("closure", ["recovery=0", "K=0.5 (broadcast)", "static_pressure"])
def test_branch_reversal_preserves_mass_and_subsonic(closure):
    # three-port junction fed from one inlet, with two pressure outlets at *different* pressures;
    # the lower-pressure side draws flow, and if the second outlet pressure exceeds the node it
    # reverses (backflow), which the manifold must handle.
    nodes = [
        cat.mass_flow_inlet(4.0, 300.0),
        CLOSURES[closure](3),
        cat.pressure_outlet(1.0e5, Tt_backflow=300.0),
        cat.pressure_outlet(1.25e5, Tt_backflow=300.0),  # high back pressure -> this branch reverses
    ]
    edges = [(0, 1, 0.05), (1, 2, 0.03), (1, 3, 0.03)]
    sol = nefes.Network(_gas(), nodes, edges, p_ref=1.0e5, T_ref=300.0, mdot_ref=4.0).solve()
    assert sol.converged
    mdot = sol.field("mdot")
    # net mass balance at the node holds whatever the branch directions
    assert abs(mdot[0] - mdot[1] - mdot[2]) < 1e-6 * (abs(mdot).max() + 1.0)
    assert np.abs(sol.field("M")).max() < 1.0


# --------------------------------------------------------------------------- #
# 7. A high-subsonic merge exercises the continuation across smoothing stages.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("closure", ["recovery=1", "recovery=0", "K=0.5 (broadcast)"])
def test_high_drive_merge_survives_continuation(closure):
    # unequal feeds run the merge to high subsonic, so the solve must ramp its smoothing (kappa,
    # eps) across continuation stages without the loss switching state between them; the outlet
    # area matches the total inflow so the merged stream stays subsonic across closures.
    feeds = [(1.8e5, 400.0), (1.6e5, 350.0), (1.5e5, 300.0)]
    net, m_idx, _ = _pinned_merge(CLOSURES[closure](4), feeds, p_out=1.0e5, loss_K=3.0)
    sol = net.solve()
    assert sol.converged
    assert np.abs(sol.field("M")).max() < 1.0
    _assert_manifold_invariants(net, sol, m_idx, total_pressure=(closure in TOTAL_P_CLOSURES))


# --------------------------------------------------------------------------- #
# 8. Broadcast K equals a uniform per-port list; recovery raises the merge entropy floor.
# --------------------------------------------------------------------------- #
def test_broadcast_K_equals_uniform_per_port_list():
    feeds = [(2.2e5, 400.0), (2.0e5, 300.0)]
    scal, _, _ = _pinned_merge(cat.junction(K=0.4), feeds)
    lst, _, _ = _pinned_merge(cat.junction(K=[0.4, 0.4, 0.4]), feeds)
    a, b = scal.solve(), lst.solve()
    assert a.converged and b.converged
    assert np.allclose(a.field("mdot"), b.field("mdot"), rtol=1e-9)
    assert np.allclose(a.field("p_t"), b.field("p_t"), rtol=1e-9)


def test_recovery_raises_merge_entropy_monotonically():
    feeds = [(2.2e5, 400.0), (2.0e5, 300.0)]

    def sgen(recovery):
        net, m_idx, out_idx = _pinned_merge(cat.junction(recovery=recovery), feeds)
        sol = net.solve()
        assert sol.converged
        mdot, T, p = sol.field("mdot"), sol.field("T"), sol.field("p")
        s = _entropy(T, p)
        edges = _node_edges(net, m_idx)
        g = 0.0
        for ei in edges:
            t, h, _a = net._edges[ei]
            out_of_node = (mdot[ei] > 0) == (t == m_idx)
            g += (1.0 if out_of_node else -1.0) * abs(mdot[ei]) * s[ei]
        return g

    lo, mid, hi = sgen(0.0), sgen(0.5), sgen(1.0)
    assert lo > mid > hi > -1e-6  # more recovery -> less dissipation, never negative
