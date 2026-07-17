"""The Network/Solution shell and YAML connectivity loader."""

import os

import numpy as np
import pytest

from nefes.elements import catalog as cat
from nefes.io import load_case, load_connectivity
from nefes.shell import Network
from nefes.thermo.configure import perfect_gas

R_AIR, GAMMA = 287.0, 1.4
CP = GAMMA * R_AIR / (GAMMA - 1.0)


def test_network_api_solves_nozzle():
    net = Network(perfect_gas(R_AIR, GAMMA), p_ref=101325.0, T_ref=300.0)
    a = net.add(cat.total_pressure_inlet(120000.0, 300.0))
    b = net.add(cat.isentropic_area_change())
    c = net.add(cat.pressure_outlet(101325.0, Tt_backflow=300.0))
    net.connect(a, b, 0.10)
    net.connect(b, c, 0.05)

    sol = net.solve()
    assert sol.converged
    exit_state = sol.edge(1)
    # subsonic exit at the spec pressure, isentropic (uniform total pressure)
    assert exit_state["p"] == pytest.approx(101325.0, rel=1e-6)
    assert exit_state["p_t"] == pytest.approx(120000.0, rel=1e-5)
    assert 0.0 < exit_state["M"] < 1.0


def test_mdot_ref_default_from_inlet():
    net = Network()
    net.add(cat.mass_flow_inlet(7.5, 300.0))
    net.add(cat.pressure_outlet(101325.0))
    net.connect(0, 1, 0.1)
    assert net._seed_mdot() == pytest.approx(7.5)


def test_h_ref_defaults_to_cp_t_ref():
    # With no explicit datum the enthalpy reference is the perfect-gas cp * T_ref, and it sets
    # the total-enthalpy variable scale on the compiled problem.
    net = Network(perfect_gas(R_AIR, GAMMA), T_ref=300.0)
    net.add(cat.mass_flow_inlet(5.0, 300.0))
    net.add(cat.pressure_outlet(101325.0))
    net.connect(0, 1, 0.05)
    assert net._seed_h() == pytest.approx(CP * 300.0)
    assert net.compile().var_scale[2] == pytest.approx(CP * 300.0)


def test_h_ref_explicit_override():
    # An explicit absolute-enthalpy datum (as the reacting closures need) overrides the
    # cp * T_ref fallback and threads through to the h_t variable scale.  It is a pure
    # rescaling, so the converged perfect-gas mean flow is unchanged.
    h0 = 1.5e6

    def build(h_ref):
        net = Network(perfect_gas(R_AIR, GAMMA), T_ref=300.0, h_ref=h_ref)
        net.add(cat.total_pressure_inlet(120000.0, 300.0))
        net.add(cat.isentropic_area_change())
        net.add(cat.pressure_outlet(101325.0, Tt_backflow=300.0))
        net.connect(0, 1, 0.10)
        net.connect(1, 2, 0.05)
        return net

    net = build(h0)
    assert net._seed_h() == pytest.approx(h0)
    assert net.compile().var_scale[2] == pytest.approx(h0)

    explicit = net.solve()
    default = build(None).solve()
    assert explicit.converged and default.converged
    assert explicit.edge(1)["M"] == pytest.approx(default.edge(1)["M"], rel=1e-8)
    assert explicit.edge(1)["p_t"] == pytest.approx(default.edge(1)["p_t"], rel=1e-8)


def test_edge_model_defaults_to_gas_model_on_every_edge():
    gas = perfect_gas(R_AIR, GAMMA)
    net = Network(gas)
    net.add(cat.mass_flow_inlet(5.0, 300.0))
    net.add(cat.duct(0.5))
    net.add(cat.pressure_outlet(101325.0))
    net.connect(0, 1, 0.05)
    net.connect(1, 2, 0.05)
    # no per-edge override -> the gas config's model id on every edge
    assert list(net.compile().edge_model) == [int(gas.model_id)] * 2


def test_edge_model_per_edge_override_threads_and_fills_default():
    # The per-edge model id is threaded verbatim (the frozen-vs-equilibrium split a flame needs);
    # edges left unset fall back to the gas config's default model id.
    gas = perfect_gas(R_AIR, GAMMA)
    sentinel = int(gas.model_id) + 17  # a distinct id, to prove the override is carried through
    net = Network(gas)
    net.add(cat.mass_flow_inlet(5.0, 300.0))
    net.add(cat.duct(0.5))
    net.add(cat.pressure_outlet(101325.0))
    net.connect(0, 1, 0.05, edge_model=sentinel)
    net.connect(1, 2, 0.05)
    assert list(net.compile().edge_model) == [sentinel, int(gas.model_id)]


def test_format_states_tabulates_every_edge():
    from nefes.solver import format_states

    net = Network(perfect_gas(R_AIR, GAMMA), p_ref=101325.0, T_ref=300.0)
    net.add(cat.total_pressure_inlet(120000.0, 300.0))
    net.add(cat.isentropic_area_change())
    net.add(cat.pressure_outlet(101325.0, Tt_backflow=300.0))
    net.connect(0, 1, 0.10)
    net.connect(1, 2, 0.05)
    sol = net.solve()

    text = format_states(sol.problem, sol.x)
    lines = text.splitlines()
    # header + rule + one row per edge
    assert len(lines) == 2 + sol.problem.n_edges
    assert lines[0].split()[0] == "edge"
    for field in ("mdot", "p_t", "area"):
        assert field in lines[0]
    # rows are indexed 0 .. n_edges-1 in order
    assert [ln.split()[0] for ln in lines[2:]] == [str(e) for e in range(sol.problem.n_edges)]


def test_print_states_subset_and_precision():
    from nefes.solver import format_states

    net = Network(perfect_gas(R_AIR, GAMMA), p_ref=101325.0, T_ref=300.0)
    net.add(cat.total_pressure_inlet(120000.0, 300.0))
    net.add(cat.isentropic_area_change())
    net.add(cat.pressure_outlet(101325.0, Tt_backflow=300.0))
    net.connect(0, 1, 0.10)
    net.connect(1, 2, 0.05)
    sol = net.solve()

    # an explicit edge subset, in the requested order, tabulates just those rows
    text = format_states(sol.problem, sol.x, edges=[1])
    body = text.splitlines()[2:]
    assert len(body) == 1 and body[0].split()[0] == "1"

    # precision controls the significant digits of the printed Mach number
    coarse = format_states(sol.problem, sol.x, edges=[1], precision=2)
    fine = format_states(sol.problem, sol.x, edges=[1], precision=8)
    assert f"{sol.edge(1)['M']:.2g}" in coarse
    assert f"{sol.edge(1)['M']:.8g}" in fine


def _nozzle_solution():
    net = Network(perfect_gas(R_AIR, GAMMA), p_ref=101325.0, T_ref=300.0)
    net.add(cat.total_pressure_inlet(120000.0, 300.0))
    net.add(cat.isentropic_area_change())
    net.add(cat.pressure_outlet(101325.0, Tt_backflow=300.0))
    net.connect(0, 1, 0.10)
    net.connect(1, 2, 0.05)
    return net.solve()


def test_format_residuals_labels_every_equation():
    from nefes.solver import format_residuals, residual_labels

    sol = _nozzle_solution()
    prob = sol.problem

    # one label per residual row, in row order, and the table carries exactly that many rows
    labels = residual_labels(prob)
    assert len(labels) == prob.n_eq
    # element (node) rows come first, then the per-edge h_t transport rows
    assert labels[0].startswith("node 0")
    assert labels[-1] == "edge 1 transport: h_t"

    text = format_residuals(prob, sol.x, sort=False)
    lines = text.splitlines()
    # header + rule + one row per equation + the trailing ||R_hat|| summary
    assert len(lines) == 2 + prob.n_eq + 1
    assert lines[0].split() == ["row", "equation", "residual", "scaled"]
    assert "IsentropicAreaChange: pressure" in text
    # the summary line reports the same global norm the solver converged to
    assert lines[-1].startswith("||R_hat|| =")
    assert f"({prob.n_eq} equations)" in lines[-1]


def test_residuals_dict_matches_global_norm():
    sol = _nozzle_solution()
    res = sol.residuals()
    assert len(res) == sol.problem.n_eq
    # the per-equation scaled residuals recompose the global convergence norm
    norm = float(np.linalg.norm(list(res.values())))
    assert norm == pytest.approx(sol.residual_norm, rel=1e-6, abs=1e-14)


def test_format_residuals_sort_and_top():
    from nefes.solver import format_residuals, residual_breakdown

    sol = _nozzle_solution()
    _labels, _R, R_hat = residual_breakdown(sol.problem, sol.x)

    # sorted + top=2 keeps only the two largest |scaled residual| equations, worst first
    text = format_residuals(sol.problem, sol.x, sort=True, top=2)
    body = text.splitlines()[2:-1]  # drop header, rule, and summary
    assert len(body) == 2
    shown = [int(line.split()[0]) for line in body]
    expected = list(np.argsort(-np.abs(R_hat))[:2])
    assert shown == expected


def test_residuals_label_composition_scalars():
    from nefes.solver import residual_labels
    from nefes.thermo.configure import perfect_gas_passive_scalars

    gas = perfect_gas_passive_scalars(2, names=["soot", "co2"])
    net = Network(gas, p_ref=101325.0, T_ref=300.0)
    net.add(cat.mass_flow_inlet(5.0, 300.0, composition=[0.3, 0.7]))
    net.add(cat.duct(0.5))
    net.add(cat.pressure_outlet(101325.0))
    net.connect(0, 1, 0.05)
    net.connect(1, 2, 0.05)
    sol = net.solve()

    labels = residual_labels(sol.problem)
    # transport rows carry the named scalars after the h_t rows, one per edge
    assert "edge 0 transport: soot" in labels
    assert "edge 1 transport: co2" in labels


def test_residual_groups_partition_and_quadrature():
    from nefes.solver import residual_breakdown, residual_groups

    sol = _nozzle_solution()
    labels, ids = residual_groups(sol.problem)
    # mass / pressure / energy are always present; no composition scalars here
    assert labels == ["mass", "pressure", "energy"]
    # every residual row is assigned to exactly one in-range group
    assert ids.shape == (sol.problem.n_eq,)
    assert ids.min() >= 0 and ids.max() < len(labels)

    # the per-group 2-norms combine in quadrature to the global convergence norm
    _lab, _R, R_hat = residual_breakdown(sol.problem, sol.x)
    group_norms = [np.linalg.norm(R_hat[ids == g]) for g in range(len(labels))]
    assert np.linalg.norm(group_norms) == pytest.approx(sol.residual_norm, rel=1e-6, abs=1e-14)


def test_residual_groups_name_composition_columns():
    from nefes.solver import residual_groups
    from nefes.thermo.configure import perfect_gas_passive_scalars

    gas = perfect_gas_passive_scalars(2, names=["CH4", "O2"])
    net = Network(gas, p_ref=101325.0, T_ref=300.0)
    net.add(cat.mass_flow_inlet(5.0, 300.0, composition=[0.05, 0.95]))
    net.add(cat.duct(0.5))
    net.add(cat.pressure_outlet(101325.0))
    net.connect(0, 1, 0.05)
    net.connect(1, 2, 0.05)
    sol = net.solve()

    labels, _ids = residual_groups(sol.problem)
    # the composition scalars become their own named columns after mass/pressure/energy
    assert labels == ["mass", "pressure", "energy", "CH4", "O2"]


def test_solve_verbose2_prints_grouped_table(capsys):
    net = Network(perfect_gas(R_AIR, GAMMA), p_ref=101325.0, T_ref=300.0)
    net.add(cat.total_pressure_inlet(120000.0, 300.0))
    net.add(cat.isentropic_area_change())
    net.add(cat.pressure_outlet(101325.0, Tt_backflow=300.0))
    net.connect(0, 1, 0.10)
    net.connect(1, 2, 0.05)
    net.solve(verbose=2)
    out = capsys.readouterr().out
    # the equation-kind header is printed (once per stage), not a per-iteration global norm
    assert "mass" in out and "pressure" in out and "energy" in out
    assert "total" in out  # the trailing gross-norm column (groups in quadrature)
    assert "||R_hat||=" in out  # the gross-residual stage summary is still printed
    # the header repeats per continuation stage (3 stages); count its occurrences
    assert out.count("it       mass") == 3
    # the last per-iteration "total" of a stage equals that stage's gross ||R_hat||
    lines = out.splitlines()
    last_total = None
    for ln in lines:
        toks = ln.split()
        if toks and toks[0].isdigit():
            last_total = float(toks[-1])  # the trailing total column
        elif ln.startswith("kappa=") and last_total is not None:
            gross = float(ln.split("||R_hat||=")[1].split(",")[0])
            assert last_total == pytest.approx(gross, rel=1e-9, abs=1e-18)
            last_total = None


def test_solve_verbose1_is_gross_only(capsys):
    net = Network(perfect_gas(R_AIR, GAMMA), p_ref=101325.0, T_ref=300.0)
    net.add(cat.total_pressure_inlet(120000.0, 300.0))
    net.add(cat.isentropic_area_change())
    net.add(cat.pressure_outlet(101325.0, Tt_backflow=300.0))
    net.connect(0, 1, 0.10)
    net.connect(1, 2, 0.05)
    net.solve(verbose=1)
    out = capsys.readouterr().out
    # verbose=1 keeps the gross residual and emits no per-equation grouped table
    assert "||R_hat||=" in out
    assert "it       mass" not in out


def test_solve_verbose_failure_dumps_per_equation(capsys):
    net = Network(perfect_gas(R_AIR, GAMMA), p_ref=101325.0, T_ref=300.0)
    net.add(cat.total_pressure_inlet(120000.0, 300.0))
    net.add(cat.isentropic_area_change())
    net.add(cat.pressure_outlet(101325.0, Tt_backflow=300.0))
    net.connect(0, 1, 0.10)
    net.connect(1, 2, 0.05)
    # a 1-iteration cap forces non-convergence; verbose=1 dumps the worst equations
    sol = net.solve(max_iter=1, verbose=1)
    assert not sol.converged
    out = capsys.readouterr().out
    assert "did not converge" in out
    assert "equation" in out


def test_edge_between_lookup():
    net = Network(perfect_gas(R_AIR, GAMMA))
    a = net.add(cat.mass_flow_inlet(5.0, 300.0, name="inlet"))
    b = net.add(cat.duct(0.5, name="duct"))
    c = net.add(cat.pressure_outlet(101325.0))
    e0 = net.connect(a, b, 0.05)
    e1 = net.connect(b, c, 0.05)
    # the returned edge ids are recoverable from the element pair
    assert net.edge_between(a, b) == e0
    assert net.edge_between(b, c) == e1
    # display names resolve like node indices, and may be mixed with them
    assert net.edge_between("inlet", "duct") == e0
    assert net.edge_between("duct", c) == e1
    # orientation is ignored: naming the pair either way returns the same edge id
    assert net.edge_between("duct", "inlet") == e0
    assert net.edge_between(c, b) == e1
    # node ids from array-valued lookups (numpy integers) resolve like plain ints
    assert net.edge_between(np.int64(a), np.int64(b)) == e0
    with pytest.raises(ValueError, match="no edge"):
        net.edge_between(a, c)
    with pytest.raises(KeyError):
        net.edge_between("inlet", "not-a-node")


def test_edge_between_rejects_ambiguous_pair():
    net = Network(perfect_gas(R_AIR, GAMMA))
    a = net.add(cat.mass_flow_inlet(5.0, 300.0))
    b = net.add(cat.duct(0.5))
    net.connect(a, b, 0.05)
    net.connect(b, a, 0.05)  # a second edge joining the pair, wired the other way
    # with orientation ignored, the antiparallel pair is genuinely ambiguous either way round
    with pytest.raises(ValueError, match="multiple edges"):
        net.edge_between(a, b)
    with pytest.raises(ValueError, match="multiple edges"):
        net.edge_between(b, a)


def test_element_name_inverts_element_index():
    net = Network(perfect_gas(R_AIR, GAMMA))
    a = net.add(cat.mass_flow_inlet(5.0, 300.0, name="inlet"))
    net.add(cat.pressure_outlet(101325.0, name="outlet"))
    # index -> label and label -> index are inverses; a label passes through unchanged
    assert net.element_name(a) == "inlet"
    assert net.element_index(net.element_name(a)) == a
    assert net.element_name("outlet") == "outlet"
    assert net.element_name(np.int64(a)) == "inlet"
    with pytest.raises(KeyError):
        net.element_name(99)


def _splitting_network():
    """inlet -> splitter -> two branches -> two outlets, with named elements."""
    net = Network(perfect_gas(R_AIR, GAMMA))
    a = net.add(cat.mass_flow_inlet(5.0, 300.0, name="in"))
    s = net.add(cat.splitter(name="split"))
    b = net.add(cat.duct(0.5, name="d1"))
    c = net.add(cat.duct(0.5, name="d2"))
    o1 = net.add(cat.pressure_outlet(101325.0, name="out1"))
    o2 = net.add(cat.pressure_outlet(101325.0, name="out2"))
    e_in = net.connect(a, s, 0.05)
    e_b = net.connect(s, b, 0.03)
    e_c = net.connect(s, c, 0.03)
    net.connect(b, o1, 0.03)
    net.connect(c, o2, 0.03)
    return net, {"a": a, "s": s, "b": b, "c": c, "e_in": e_in, "e_b": e_b, "e_c": e_c}


def test_edges_of_incident_and_direction():
    net, r = _splitting_network()
    s, e_in, e_b, e_c = r["s"], r["e_in"], r["e_b"], r["e_c"]
    # a splitter with one inflow and two outflows: all three are incident
    assert net.edges_of(s) == sorted([e_in, e_b, e_c])
    assert net.edges_of("split", "in") == [e_in]  # the element is the head
    assert net.edges_of("split", "out") == sorted([e_b, e_c])  # the element is the tail
    # name and index resolve to the same answer, and a leaf element has one edge
    assert net.edges_of("in") == net.edges_of(r["a"]) == [e_in]


def test_edges_of_rejects_bad_direction_and_name():
    net, _ = _splitting_network()
    with pytest.raises(ValueError, match="direction must be"):
        net.edges_of("split", "sideways")
    with pytest.raises(KeyError):
        net.edges_of("nope")


def test_nodes_of_inverts_edges_of():
    net, r = _splitting_network()
    # nodes_of returns (tail, head) and round-trips against edge_between for every edge
    for e in net.edges_of("split"):
        t, h = net.nodes_of(e)
        assert net.edge_between(t, h) == e
    assert net.nodes_of(r["e_b"]) == (r["s"], r["b"])
    with pytest.raises(IndexError):
        net.nodes_of(999)


def test_set_dynamic_source_deferred_attach():
    # Wire the network first, take the flame's reference edge from connect()'s return, then attach
    # the dynamic source -- no edge index is guessed before the topology exists.
    from nefes.elements.dynamic_source import n_tau_flame

    net = Network(perfect_gas(R_AIR, GAMMA), p_ref=1e5, T_ref=300.0, mdot_ref=0.006)
    inlet = net.add(cat.mass_flow_inlet(0.006, 300.0))
    cold = net.add(cat.duct(0.6))
    flame = net.add(cat.heat_release_flame(0.006 * CP * 400.0))
    hot = net.add(cat.duct(0.4))
    outlet = net.add(cat.pressure_outlet(1e5))
    net.connect(inlet, cold, 0.01)
    ref = net.connect(cold, flame, 0.01)
    net.connect(flame, hot, 0.01)
    net.connect(hot, outlet, 0.01)

    # before attaching, the flame carries no source
    assert net.compile().node_dynamic_source[flame] is None

    src = n_tau_flame(0.8, 4.0e-3, ref_edge=ref)
    assert net.set_dynamic_source(flame, src) == flame  # returns the node, for chaining
    prob = net.compile()
    assert prob.node_dynamic_source[flame] is src
    assert prob.node_dynamic_source[flame].terms[0].ref_edge == ref

    # mean flow ignores the source, so it still converges
    assert net.solve().converged


def test_warm_restart_is_cheaper():
    net = Network(perfect_gas(R_AIR, GAMMA))
    net.add(cat.total_pressure_inlet(115000.0, 300.0))
    net.add(cat.isentropic_area_change())
    net.add(cat.pressure_outlet(101325.0, Tt_backflow=300.0))
    net.connect(0, 1, 0.10)
    net.connect(1, 2, 0.06)

    cold = net.solve()
    assert cold.converged
    warm = net.solve(x0=cold.x)
    assert warm.converged
    # warm start from the converged state needs far fewer iterations
    assert warm.iterations < cold.iterations


CASE_YAML = os.path.join(os.path.dirname(__file__), "..", "examples", "converging_nozzle.yaml")


@pytest.mark.skipif(not os.path.exists(CASE_YAML), reason="example case not present")
def test_load_case_solves():
    net = load_case(CASE_YAML)
    sol = net.solve()
    assert sol.converged
    throat = sol.edge(2)  # feed(0), pipe(1), throat(2), tailpipe(3)
    # isentropic nozzle + inert constant-area ducts: total pressure uniform
    assert throat["p_t"] == pytest.approx(200000.0, rel=1e-4)
    assert throat["p"] == pytest.approx(150000.0, rel=1e-6)  # subsonic: exit p = spec
    assert 0.0 < throat["M"] < 1.0


@pytest.mark.skipif(not os.path.exists(CASE_YAML), reason="example case not present")
def test_load_case_preserves_ports():
    # The UI export pins ports via the handles: the pipe enters the area change at
    # port 0, the throat leaves it at port 1.  The compiled connectivity must
    # reflect exactly that (port 0 = target/in side, port 1 = source/out side).
    net = load_case(CASE_YAML)
    prob = net.compile()
    # inlet(0) -feed(0)-> Duct(1) -pipe(1)-> nozzle(2) -throat(2)-> Duct(3) -tailpipe(3)-> outlet(4)
    assert list(prob.tail_node) == [0, 1, 2, 3]
    assert list(prob.head_node) == [1, 2, 3, 4]
    # node 2 (the area change) sees edge1 at its port 0 and edge2 at its port 1
    sl = slice(prob.row_ptr[2], prob.row_ptr[3])
    assert list(prob.col_edge[sl]) == [1, 2]
    assert list(prob.orient[sl]) == [-1, 1]  # edge1 incoming, edge2 outgoing


def _contraction_case(tmp_path, sac_attrs):
    """Write a minimal UI-export case: inlet -> sudden contraction -> outlet.

    Edge 0 (area 0.09) enters the element at port 0, edge 1 (area 0.05) leaves it
    at port 1, so the forward flow contracts large -> small.  ``sac_attrs`` are the
    extra attributes carried by the SuddenAreaChange node (e.g. a contraction
    coefficient), exactly as the UI would export them.
    """
    import yaml

    doc = {
        "version": "2.0.0",
        "model": {
            "id": "nefes",
            "globalAttributes": {
                "gasConstant": 287.0,
                "heatCapacityRatio": 1.4,
                "referencePressure": 101325.0,
                "referenceTemperature": 300.0,
                "referenceMassFlow": 10.0,
            },
            "nodes": [
                {
                    "id": "in",
                    "type": "TotalPressureInlet",
                    "attributes": {"label": "in", "index": 0, "totalPressure": 120000.0, "totalTemperature": 300.0},
                },
                {"id": "sac", "type": "SuddenAreaChange", "attributes": {"label": "sac", "index": 1, **sac_attrs}},
                {
                    "id": "out",
                    "type": "PressureOutlet",
                    "attributes": {"label": "out", "index": 2, "pressure": 101325.0, "backflowTotalTemperature": 300.0},
                },
            ],
            "edges": [
                {
                    "id": "e0",
                    "source": "in",
                    "target": "sac",
                    "sourceHandle": "in-port-0",
                    "targetHandle": "sac-port-0",
                    "type": "flow",
                    "attributes": {"label": "big", "index": 0, "area": 0.09},
                },
                {
                    "id": "e1",
                    "source": "sac",
                    "target": "out",
                    "sourceHandle": "sac-port-1",
                    "targetHandle": "out-port-0",
                    "type": "flow",
                    "attributes": {"label": "small", "index": 1, "area": 0.05},
                },
            ],
        },
    }
    path = tmp_path / f"contraction_{len(sac_attrs)}.yaml"
    path.write_text(yaml.safe_dump(doc))
    return str(path)


def test_load_case_sudden_contraction_coefficient(tmp_path):
    # The UI exports the contraction coefficient as a node attribute; load_case must
    # thread it to the kernel.  With cc = 0.62 the reverse (large -> small) flow loses
    # total pressure K_c * (1/2 rho u^2)_small; omitting the attribute is loss-free.
    lossy = load_case(_contraction_case(tmp_path, {"contractionCoefficient": 0.62})).solve()
    free = load_case(_contraction_case(tmp_path, {})).solve()  # default cc = 1
    assert lossy.converged and free.converged

    up, dn = lossy.edge(0), lossy.edge(1)  # large (upstream) -> small (downstream)
    assert dn["M"] > up["M"] and dn["M"] < 1.0  # genuinely contracting, subsonic
    K_c = (1.0 / 0.62 - 1.0) ** 2
    q_small = 0.5 * dn["rho"] * dn["u"] ** 2
    assert up["p_t"] - dn["p_t"] == pytest.approx(K_c * q_small, rel=1e-5)

    # the attribute-less case defaults to the loss-free contraction
    assert free.edge(0)["p_t"] - free.edge(1)["p_t"] == pytest.approx(0.0, abs=1.0)


def _ui_case(tmp_path, name, nodes, edges, glob=None):
    """Write a minimal ``nefes`` UI export to ``tmp_path`` and return its path.

    ``nodes`` are ``(id, type, index, attrs)`` tuples and ``edges`` are
    ``(id, src, src_port, tgt, tgt_port, index, area)`` tuples, so a test can spell a
    small topology inline without hand-writing the reactflow handle strings.
    """
    import yaml

    g = {
        "gasConstant": 287.0,
        "heatCapacityRatio": 1.4,
        "referencePressure": 101325.0,
        "referenceTemperature": 300.0,
        "referenceMassFlow": 10.0,
    }
    g.update(glob or {})
    doc = {
        "version": "2.0.0",
        "model": {
            "id": "nefes",
            "globalAttributes": g,
            "nodes": [
                {"id": nid, "type": ntype, "attributes": {"label": nid, "index": idx, **attrs}}
                for (nid, ntype, idx, attrs) in nodes
            ],
            "edges": [
                {
                    "id": eid,
                    "source": src,
                    "target": tgt,
                    "sourceHandle": f"{src}-port-{sp}",
                    "targetHandle": f"{tgt}-port-{tp}",
                    "type": "flow",
                    "attributes": {"index": idx, "area": area},
                }
                for (eid, src, sp, tgt, tp, idx, area) in edges
            ],
        },
    }
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(doc))
    return str(path)


def test_load_multiport_case_conserves_mass(tmp_path):
    # A UI export with a splitter and a junction (multi-port elements) must load with
    # the right ports and conserve mass: a fixed feed splits through two unequal area
    # changes and remerges, so the two branches sum back to the feed and the outlet.
    # Per-node ports are one 0..d-1 run (left ports then right ports).
    nodes = [
        ("feed", "MassFlowInlet", 0, {"massFlowRate": 10.0, "totalTemperature": 300.0}),
        ("split", "LosslessSplitter", 1, {}),
        ("oa", "IsentropicAreaChange", 2, {}),
        ("ob", "IsentropicAreaChange", 3, {}),
        ("merge", "JunctionStaticP", 4, {}),
        ("out", "PressureOutlet", 5, {"pressure": 101325.0, "backflowTotalTemperature": 300.0}),
    ]
    edges = [
        ("e0", "feed", 0, "split", 0, 0, 0.10),
        ("e1", "split", 1, "oa", 0, 1, 0.05),
        ("e2", "split", 2, "ob", 0, 2, 0.05),
        ("e3", "oa", 1, "merge", 0, 3, 0.04),
        ("e4", "ob", 1, "merge", 1, 4, 0.02),
        ("e5", "merge", 2, "out", 0, 5, 0.10),
    ]
    sol = load_case(_ui_case(tmp_path, "multiport_split", nodes, edges)).solve()
    assert sol.converged
    mdot = sol.field("mdot")
    assert np.isfinite(mdot).all()
    # feed(e0) -> [branch e3, branch e4] -> outlet(e5); the split remerges losslessly
    assert mdot[3] + mdot[4] == pytest.approx(mdot[0], rel=1e-9)
    assert mdot[5] == pytest.approx(mdot[0], rel=1e-9)


def test_deferred_supersonic_raises(tmp_path):
    # Supersonic boundaries are out of the subsonic scope: a case carrying one is rejected.
    nodes = [
        ("res", "TotalPressureInlet", 0, {"totalPressure": 200000.0, "totalTemperature": 400.0}),
        ("exit", "SupersonicOutlet", 1, {"pressure": 18786.5}),
    ]
    edges = [("e0", "res", 0, "exit", 0, 0, 0.10)]
    path = _ui_case(tmp_path, "supersonic", nodes, edges, glob={"referencePressure": 200000.0})
    with pytest.raises(ValueError, match="deferred"):
        load_case(path)


def test_network_repr_summarizes_topology_and_thermo():
    from nefes.elements.dynamic_source import n_tau_flame

    net = Network(perfect_gas(R_AIR, GAMMA), p_ref=101325.0, T_ref=300.0)
    inlet = net.add(cat.mass_flow_inlet(0.5, 300.0, name="air-in"))
    duct = net.add(cat.duct(0.3, name="duct"))
    flame = net.add(cat.heat_release_flame(2.0e4, name="flame"))
    outlet = net.add(cat.pressure_outlet(101325.0, name="out"))
    net.connect(inlet, duct, 0.01)
    ref = net.connect(duct, flame, 0.01, name="pre-flame")
    net.connect(flame, outlet, 0.01)
    net.set_dynamic_source(flame, n_tau_flame(1.0, 3.0e-3, ref_edge=ref))

    text = repr(net)
    # size, thermo model and reference conditions
    assert "4 elements, 3 edges" in text
    assert "perfect gas" in text
    assert "mdot=0.5 kg/s (auto)" in text
    # element + edge listings with their names, and the dynamic-source marker on the flame
    assert "air-in" in text and "HeatReleaseFlame *" in text
    # explicitly chosen names are kept verbatim ("duct", "flame"); "pre-flame" is an explicit edge name
    assert "duct -> flame" in text and "pre-flame" in text
    assert "carries a dynamic S(omega) source" in text

    html = net._repr_html_()
    assert html.startswith("<div") and "<table" in html
    assert "<b>Network</b>" in html and "air-in" in html
    assert "4 elements, 3 edges" in html and "perfect gas" in html


def test_network_repr_handles_empty_and_truncates(recwarn):
    # An edge-less network reports mdot as n/a and emits no numpy warning while probing it.
    empty = repr(Network())
    assert "0 elements, 0 edges" in empty and "mdot=n/a" in empty
    assert len(recwarn) == 0

    # Long listings are truncated past the per-table cap, with a "... (N more)" footer.
    net = Network()
    prev = net.add(cat.mass_flow_inlet(0.5, 300.0, name="in"))
    for k in range(25):
        nxt = net.add(cat.duct(0.1, name=f"duct{k}"))
        net.connect(prev, nxt, 0.01)
        prev = nxt
    out = net.add(cat.pressure_outlet(101325.0, name="out"))
    net.connect(prev, out, 0.01)
    text = repr(net)
    assert "27 elements, 26 edges" in text
    assert "... (7 more)" in text and "... (6 more)" in text


def _connectivity_doc(tmp_path):
    """Write a connectivity-only UI export (a diamond) exercising the topology parser.

    Nodes and edges are listed out of ``index`` order on purpose, so the parser must
    key on the ``index`` attribute, not document order.  Per-node ports run ``0..d-1``
    across all incident edges (left/in ports first, then right/out).  Topology::

        N0 -e0-> N1 -e1-> N2
                  \\-e2-> N3 <-e3- N2

    N1 fans out (ports 1, 2) and N3 merges (ports 0, 1): a non-tree diamond.
    """
    import yaml

    doc = {
        "model": {
            "id": "generic-network",
            "nodes": [
                {"id": "N2", "attributes": {"index": 2}},
                {"id": "N0", "attributes": {"index": 0}},
                {"id": "N3", "attributes": {"index": 3}},
                {"id": "N1", "attributes": {"index": 1}},
            ],
            "edges": [
                {
                    "source": "N2",
                    "target": "N3",
                    "sourceHandle": "N2-port-1",
                    "targetHandle": "N3-port-1",
                    "attributes": {"index": 3},
                },
                {
                    "source": "N0",
                    "target": "N1",
                    "sourceHandle": "N0-port-0",
                    "targetHandle": "N1-port-0",
                    "attributes": {"index": 0},
                },
                {
                    "source": "N1",
                    "target": "N3",
                    "sourceHandle": "N1-port-2",
                    "targetHandle": "N3-port-0",
                    "attributes": {"index": 2},
                },
                {
                    "source": "N1",
                    "target": "N2",
                    "sourceHandle": "N1-port-1",
                    "targetHandle": "N2-port-0",
                    "attributes": {"index": 1},
                },
            ],
        },
    }
    path = tmp_path / "connectivity.yaml"
    path.write_text(yaml.safe_dump(doc))
    return str(path)


def test_connectivity_yaml_parses_topology(tmp_path):
    conn = load_connectivity(_connectivity_doc(tmp_path))
    assert conn.n_nodes == 4
    assert conn.n_edges == 4
    # endpoints are emitted in edge-index order after the sort
    assert list(conn.tail_node) == [0, 1, 1, 2]
    assert list(conn.head_node) == [1, 2, 3, 3]
    assert list(conn.tail_port) == [0, 1, 2, 1]
    assert list(conn.head_port) == [0, 0, 0, 1]
    assert int(conn.row_ptr[-1]) == 2 * conn.n_edges
