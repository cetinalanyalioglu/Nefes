"""Verification: Greyvenstein & Laurie (1994) Example 3 -- the compressed-air network.

G. P. Greyvenstein and D. P. Laurie, "A segregated CFD approach to pipe network analysis,"
Int. J. Numer. Methods Eng. 37, 3685-3705 (1994).  Example 3 (Fig. 9, Tables III-IV) is the
paper's only compressible case: a 29-pipe compressed-air distribution network, constant Darcy
friction factor ``f = 0.03``, isothermal ``T = 288.15 K`` (15 C), two 6-bar supplies and
thirteen 3-bar demands (a pure Dirichlet pressure problem; the mass flows are the answer).

It is the natural Nefes match because the ``pipe`` element *is* the Darcy-Weisbach ``DUCT (+)
LOSS`` unification the paper motivates: one length-bearing element dropping total pressure by
``K = f L / D`` on the mean flow.  We build the network from Table III, solve it, and check the
published Table IV mass flows and node pressures.

The agreement is ~0.1 %; the small residual is physical and in Nefes's favour -- Nefes resolves the
dynamic pressure and the full compressible closure the paper's low-Mach (static ~ total,
frozen-temperature) model drops.  At the operating point the flow is firmly low-Mach
(M <~ 0.04), so those differences are O(M^2) and land in the 3rd-4th significant figure.

A NOTE ON TABLE IV: the column headed "Density (kg/m^3)" is actually the element **velocity**
(m/s) -- every entry equals ``V = mdot / (rho A)`` to the published digits, while the true
density ``rho = p / R T`` differs (e.g. element 1: table 8.56 = V, whereas rho = 6.78).  The
``test_reported_density_column_is_velocity`` test pins this down.
"""

import numpy as np
import pytest

from nefes.elements import catalog as cat
from nefes.shell.network import Network
from nefes.thermo.configure import perfect_gas

R_AIR, GAMMA, T_ISO = 287.0, 1.4, 288.15
FRICTION = 0.03
P_SUPPLY, P_DEMAND = 6.0e5, 3.0e5

# Table III -- element number -> (upstream node, downstream node, diameter [m], length [m]).
PIPES = {
    1: (1, 2, 0.019, 200),
    2: (2, 3, 0.01588, 400),
    3: (3, 5, 0.01588, 400),
    4: (5, 6, 0.010, 100),
    5: (6, 7, 0.010, 100),
    6: (6, 8, 0.010, 100),
    7: (8, 9, 0.010, 100),
    8: (8, 10, 0.010, 100),
    9: (5, 11, 0.01588, 400),
    10: (11, 12, 0.010, 100),
    11: (11, 13, 0.01588, 400),
    12: (14, 13, 0.019, 200),
    13: (13, 15, 0.01588, 400),
    14: (15, 16, 0.010, 100),
    15: (17, 15, 0.01588, 400),
    16: (17, 18, 0.010, 100),
    17: (18, 19, 0.010, 100),
    18: (18, 20, 0.010, 100),
    19: (21, 17, 0.01588, 400),
    20: (21, 22, 0.010, 100),
    21: (23, 21, 0.01588, 400),
    22: (23, 24, 0.010, 100),
    23: (24, 25, 0.010, 100),
    24: (26, 23, 0.01588, 400),
    25: (26, 27, 0.010, 100),
    26: (2, 26, 0.01588, 400),
    27: (3, 4, 0.010, 100),
    28: (5, 28, 0.010, 100),
    29: (24, 29, 0.010, 100),
}
SUPPLY_NODES = (1, 14)  # 6.0 bar
DEMAND_NODES = (4, 7, 9, 10, 12, 16, 19, 20, 22, 25, 27, 28, 29)  # 3.0 bar (node 28 included)

# Table IV -- element mass flow [g/s] (+ in the upstream->downstream sense) and the reported
# "Density" column, which is actually the element velocity [m/s].
PUB_MDOT_GPS = {
    1: 16.461,
    2: 8.033,
    3: 3.596,
    4: 3.375,
    5: 1.782,
    6: 1.593,
    7: 0.797,
    8: 0.797,
    9: -3.596,
    10: 4.437,
    11: -8.033,
    12: 16.461,
    13: 8.429,
    14: 4.135,
    15: -4.293,
    16: 2.800,
    17: 1.400,
    18: 1.400,
    19: -1.493,
    20: 2.986,
    21: 1.493,
    22: 2.800,
    23: 1.400,
    24: 4.293,
    25: 4.135,
    26: 8.429,
    27: 4.437,
    28: 3.816,
    29: 1.400,
}
PUB_VELOCITY_MPS = {
    1: 8.56,
    2: 7.19,
    3: 3.77,
    4: 10.07,
    5: 6.05,
    6: 5.37,
    7: 2.78,
    8: 2.78,
    9: -3.77,
    10: 13.14,
    11: -7.19,
    12: 8.56,
    13: 7.65,
    14: 12.47,
    15: -4.73,
    16: 8.77,
    17: 4.81,
    18: 4.81,
    19: -1.75,
    20: 9.60,
    21: 1.75,
    22: 8.77,
    23: 4.81,
    24: 4.73,
    25: 12.47,
    26: 7.65,
    27: 13.14,
    28: 11.72,
    29: 4.81,
}
# Table IV -- node pressure [bar].
PUB_PRESSURE_BAR = {
    1: 6.0,
    2: 5.2151,
    3: 4.1131,
    4: 3.0,
    5: 3.8546,
    6: 3.2057,
    7: 3.0,
    8: 3.0423,
    9: 3.0,
    10: 3.0,
    11: 4.1131,
    12: 3.0,
    13: 5.2151,
    14: 6.0,
    15: 3.9848,
    16: 3.0,
    17: 3.5975,
    18: 3.1286,
    19: 3.0,
    20: 3.0,
    21: 3.5478,
    22: 3.0,
    23: 3.5975,
    24: 3.1286,
    25: 3.0,
    26: 3.9848,
    27: 3.0,
    28: 3.0,
    29: 3.0,
}

_LEAVES = set(SUPPLY_NODES) | set(DEMAND_NODES)
_PAPER_NODES = sorted({n for e in PIPES.values() for n in e[:2]})
_INTERIOR = [n for n in _PAPER_NODES if n not in _LEAVES]


def build_network(formulation="darcy-weisbach"):
    """Assemble Example 3 as an Nefes ``Network``.

    Each paper *pipe* is a two-port :func:`~nefes.elements.catalog.pipe` (Darcy-Weisbach,
    ``K = f L / D``, circular area ``pi D^2 / 4``); each interior paper *node* is a
    :func:`~nefes.elements.catalog.junction` on its ``recovery = 0`` limit, which at low Mach
    ties the incident static pressures equal -- the static-pressure header the paper assumes;
    each supply/demand leaf is a :func:`~nefes.elements.catalog.total_pressure_inlet` /
    :func:`~nefes.elements.catalog.pressure_outlet`.

    Returns
    -------
    net : Network
    pipe_edges : dict[int, tuple[int, int, float]]
        Element number -> ``(in_edge, out_edge, area)``.  ``out_edge`` leaves the pipe toward
        its downstream node: its ``mdot`` is positive in the paper's upstream->downstream sense
        and its ``p`` is the downstream junction's static pressure.  ``in_edge`` enters from the
        upstream node; the two edges carry the pipe's upstream/downstream states.
    """
    net = Network(gas=perfect_gas(R_AIR, GAMMA), p_ref=4.5e5, T_ref=T_ISO, mdot_ref=0.033)
    node_of = {}
    for n in _INTERIOR:
        node_of[n] = net.add(cat.junction(recovery=0.0, name=f"n{n}"))
    for n in SUPPLY_NODES:
        node_of[n] = net.add(cat.total_pressure_inlet(P_SUPPLY, T_ISO, name=f"supply{n}"))
    for n in DEMAND_NODES:
        node_of[n] = net.add(cat.pressure_outlet(P_DEMAND, T_ISO, name=f"demand{n}"))

    pipe_edges = {}
    for k, (u, d, D, L) in PIPES.items():
        area = np.pi * D * D / 4.0
        pk = net.add(cat.pipe(L, D, FRICTION, name=f"p{k}", formulation=formulation))
        e_in = net.connect(node_of[u], pk, area)  # port 0: inflow from the upstream node
        e_out = net.connect(pk, node_of[d], area)  # port 1: outflow to downstream
        pipe_edges[k] = (e_in, e_out, area)
    return net, pipe_edges


@pytest.fixture(scope="module")
def solved():
    net, pipe_edges = build_network()
    sol = net.solve()
    return sol, pipe_edges


@pytest.fixture(scope="module")
def solved_momentum():
    net, pipe_edges = build_network("momentum")
    sol = net.solve()
    return sol, pipe_edges


def test_network_shape():
    # 29 pipes, 14 interior junctions, 15 terminals (2 supplies + 13 demands, node 28 a demand).
    assert len(PIPES) == 29
    assert len(_INTERIOR) == 14
    assert len(_LEAVES) == 15
    assert 28 in DEMAND_NODES  # the demand leaf the first-pass assessment missed


def test_converges(solved):
    sol, _ = solved
    # a pure pressure-driven mesh (14 junctions, no fixed mass flow) -- the regime the
    # artificial-resistance continuation targets; it should converge tightly.
    assert sol.converged
    assert sol.residual_norm < 1e-8


def test_mass_flows_match_table_iv(solved):
    sol, pipe_edges = solved
    mdot = sol.field("mdot")
    worst = 0.0
    for k in PIPES:
        _e_in, e_out, _area = pipe_edges[k]
        m_fns_gps = mdot[e_out] * 1000.0  # +ve in the upstream->downstream sense
        rel = abs(m_fns_gps - PUB_MDOT_GPS[k]) / abs(PUB_MDOT_GPS[k])
        worst = max(worst, rel)
    # observed ~0.09 %; the residual is the compressible/dynamic-pressure physics the paper drops.
    assert worst < 3e-3, f"worst mass-flow error {worst:.3%} exceeds 0.3 %"


def test_node_pressures_match_table_iv(solved):
    sol, pipe_edges = solved
    p = sol.field("p")  # static pressure; the junction ties it equal across incident edges
    worst = 0.0
    for n in _INTERIOR:
        _e_in, e_out, _area = pipe_edges[next(k for k, (u, d, _D, _L) in PIPES.items() if d == n)]
        p_fns_bar = p[e_out] / 1e5
        worst = max(worst, abs(p_fns_bar - PUB_PRESSURE_BAR[n]))
    # observed ~0.0018 bar (largest at the fast supply junctions): the O(M^2) total-vs-static gap.
    assert worst < 4e-3, f"worst node-pressure error {worst:.4f} bar exceeds 0.004 bar"


def test_low_mach_regime(solved):
    # Example 3 sits at M <~ 0.05 (local per-edge peak ~0.046 at the low-density pipe exits), so
    # Nefes's compressible closure and the paper's low-Mach model agree to O(M^2); confirm the
    # operating point is where that argument holds.
    sol, _ = solved
    mach = sol.field("M")
    assert np.max(np.abs(mach)) < 0.06


def test_momentum_formulation_remains_close_in_the_low_mach_regime(solved, solved_momentum):
    darcy, darcy_edges = solved
    momentum, momentum_edges = solved_momentum
    assert momentum.converged
    assert np.max(np.abs(momentum.field("M"))) < 0.06
    flow_differences = []
    published_flow_errors = []
    for k in PIPES:
        darcy_out = darcy_edges[k][1]
        momentum_out = momentum_edges[k][1]
        mdot_darcy = darcy.edge(darcy_out)["mdot"]
        mdot_momentum = momentum.edge(momentum_out)["mdot"]
        flow_differences.append(abs(mdot_momentum - mdot_darcy) / abs(mdot_darcy))
        published_flow_errors.append(abs(mdot_momentum * 1000.0 - PUB_MDOT_GPS[k]) / abs(PUB_MDOT_GPS[k]))
    # The distributed momentum closure is not the paper's lumped isothermal model, and the
    # header nodes are junctions on the recovery = 0 limit (common static pressure only to
    # O(M^2)); at M < 0.06 the combined departure keeps branch flows within 1.2% of the
    # authoritative Darcy solve and 1.3% of the published values.
    assert max(flow_differences) < 1.2e-2
    assert max(published_flow_errors) < 1.3e-2

    published_pressure_errors = []
    for n in _INTERIOR:
        k = next(k for k, (_u, d, _D, _L) in PIPES.items() if d == n)
        momentum_out = momentum_edges[k][1]
        published_pressure_errors.append(abs(momentum.edge(momentum_out)["p"] / 1e5 - PUB_PRESSURE_BAR[n]))
    # The header nodes tie the incident static pressures only to O(M^2) (the recovery = 0 limit,
    # not an exact static-pressure junction), which combines with the momentum-closure departure.
    assert max(published_pressure_errors) < 8.5e-3


def test_reported_density_column_is_velocity(solved):
    # Table IV's "Density (kg/m^3)" column is really the element velocity (m/s): the paper's
    # V = mdot / (rho_bar A) at the mean density (the pipe's reference velocity), which Nefes
    # reproduces, while the true density rho = p/RT does not match the column.
    sol, pipe_edges = solved
    mdot, rho, area = sol.field("mdot"), sol.field("rho"), sol.field("area")
    col_minus_rho = []
    for k in PIPES:
        e_in, e_out, _a = pipe_edges[k]
        rho_bar = 0.5 * (rho[e_in] + rho[e_out])  # mean-density basis, as in the pipe kernel
        v_fns = mdot[e_out] / (rho_bar * area[e_out])  # signed, up->down positive
        assert v_fns == pytest.approx(PUB_VELOCITY_MPS[k], abs=0.06), f"pipe {k}: V mismatch"
        col_minus_rho.append(abs(abs(PUB_VELOCITY_MPS[k]) - rho_bar) / rho_bar)
    # ...and the column is NOT the true density: as a set the two differ substantially (a few
    # pipes coincide numerically, but the mean gap is large), so the "Density" label is a misprint.
    assert np.mean(col_minus_rho) > 0.2
