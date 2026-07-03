"""Phase 5 validation: the Network/Solution shell and YAML connectivity loader."""

import os

import numpy as np
import pytest

from nefes.shell import Network
from nefes.elements import catalog as cat
from nefes.thermo.configure import perfect_gas
from nefes.io import load_connectivity, load_case

R_AIR, GAMMA = 287.0, 1.4
CP = GAMMA * R_AIR / (GAMMA - 1.0)

DEMO_YAML = os.path.join(
    os.path.dirname(__file__),
    "..",
    "preliminary-study",
    "docs",
    "examples",
    "ConnectivityDemonstrator.yaml",
)


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
    from nefes.solver import residual_groups, residual_breakdown

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
    a = net.add(cat.mass_flow_inlet(5.0, 300.0))
    b = net.add(cat.duct(0.5))
    c = net.add(cat.pressure_outlet(101325.0))
    e0 = net.connect(a, b, 0.05)
    e1 = net.connect(b, c, 0.05)
    # the returned edge ids are recoverable from the element pair
    assert net.edge_between(a, b) == e0
    assert net.edge_between(b, c) == e1
    with pytest.raises(ValueError, match="no edge"):
        net.edge_between(a, c)


def test_edge_between_rejects_ambiguous_pair():
    net = Network(perfect_gas(R_AIR, GAMMA))
    a = net.add(cat.mass_flow_inlet(5.0, 300.0))
    b = net.add(cat.duct(0.5))
    net.connect(a, b, 0.05)
    net.connect(a, b, 0.05)  # a second parallel edge between the same ordered pair
    with pytest.raises(ValueError, match="multiple edges"):
        net.edge_between(a, b)


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


SHOWCASE = os.path.join(os.path.dirname(__file__), "..", "preliminary-study", "examples", "ui_showcase")


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
            "id": "fns-flow-network",
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

    # the attribute-less case defaults to the historical loss-free contraction
    assert free.edge(0)["p_t"] - free.edge(1)["p_t"] == pytest.approx(0.0, abs=1.0)


@pytest.mark.skipif(not os.path.exists(SHOWCASE), reason="UI showcase cases not present")
def test_load_multiport_showcase_conserves_mass():
    # A real UI export with splitters/junctions (multi-port elements) must load
    # with correct ports and conserve mass at the merge.
    net = load_case(os.path.join(SHOWCASE, "gas_turbine_splits.yaml"))
    sol = net.solve()
    assert sol.converged
    mdot = sol.field("mdot")
    assert np.isfinite(mdot).all()


@pytest.mark.skipif(not os.path.exists(SHOWCASE), reason="UI showcase cases not present")
def test_deferred_supersonic_raises():
    with pytest.raises(ValueError, match="deferred"):
        load_case(os.path.join(SHOWCASE, "cd_nozzle_supersonic.yaml"))


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
    # factory-default names are numbered from the start ("duct" -> "duct-1"); "pre-flame" is an explicit edge name
    assert "duct-1 -> flame-1" in text and "pre-flame" in text
    assert "carries a dynamic S(omega) source" in text

    html = net._repr_html_()
    assert html.startswith("<div") and "<table" in html
    assert "<b>Network</b>" in html and "air-in" in html


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


@pytest.mark.skipif(not os.path.exists(DEMO_YAML), reason="demonstrator YAML not present")
def test_demonstrator_yaml_connectivity():
    conn = load_connectivity(DEMO_YAML)
    assert conn.n_nodes == 6
    assert conn.n_edges == 7
    assert list(conn.tail_node) == [0, 1, 1, 2, 2, 3, 4]
    assert list(conn.head_node) == [1, 2, 3, 3, 4, 4, 5]
    assert list(conn.tail_port) == [0, 1, 2, 2, 1, 2, 2]
    assert list(conn.head_port) == [0, 0, 1, 0, 0, 1, 0]
    assert int(conn.row_ptr[-1]) == 2 * conn.n_edges
