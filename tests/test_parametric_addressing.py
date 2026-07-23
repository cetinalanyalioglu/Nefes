"""Nested parameter addressing: the scalar-parameter protocol, recursive inventory, layer tags.

Objects attached to elements (dynamic sources, perturbation boundary conditions, transfer
functions) expose their scalar knobs through the protocol in :mod:`nefes.elements.parametric`;
the inventory recurses into them, the write paths route through their functional copies, and
every row carries a solution-layer tag.  The eigenvalue-sensitivity feature consumes all of it:
perturbation-layer rows skip the mean-flow chain term, and the flame gain / time lag become
differentiable like any duct length.
"""

import warnings

import numpy as np
import pytest

import nefes
from nefes.assembly.assemble import residual
from nefes.elements import catalog as cat
from nefes.elements.dynamic_source import DynamicResponseTerm, DynamicSource, FiniteImpulseResponse, NTau
from nefes.elements.parametric import is_parametric

AREA = 0.004


def _build():
    return nefes.Network(
        nodes=[
            cat.mass_flow_inlet(0.05, 300.0, perturbation_bc=nefes.PerturbationBC.reflection(0.9)),
            cat.duct(0.3),
            cat.heat_release_flame(2.0e4, name="flame"),
            cat.duct(0.5),
            cat.pressure_outlet(1.0e5, perturbation_bc=nefes.PerturbationBC.open_end()),
        ],
        edges=[(0, 1, AREA), (1, 2, AREA), (2, 3, AREA), (3, 4, AREA)],
    )


def _with_flame(n=0.8, tau=2.0e-3):
    net = _build()
    ref = net.edge_between("duct-1", "flame")
    net.set_dynamic_source("flame", DynamicSource([DynamicResponseTerm(NTau(n, tau), ref)]))
    return net


@pytest.fixture(scope="module")
def flame_case():
    net = _with_flame()
    sol = net.solve()
    eigs = sol.eigenmodes(freq_band=(80.0, 600.0), isentropic=True)
    assert eigs.n_modes >= 3 and eigs.certified
    return net, sol, eigs


def test_protocol_round_trips_on_the_objects():
    """Each protocol object reports its knobs, reads them back, and copies functionally."""
    F = NTau(1.5, 3.0e-3)
    assert is_parametric(F)
    assert [d.name for d in F.param_descriptors()] == ["n", "tau"]
    F2 = F.with_value("tau", 4.0e-3)
    assert F.tau == 3.0e-3 and F2.tau == 4.0e-3  # functional, never in place

    fir = FiniteImpulseResponse([0.0, 1.0, 0.5], 1.0e-3)
    g = fir.with_value("gain", 2.0)
    assert np.isclose(complex(g(120.0)), 2.0 * complex(fir(120.0)))
    d = fir.with_value("delay", 1.5e-3)
    assert np.isclose(complex(d(120.0)), complex(fir(120.0)) * np.exp(-2j * np.pi * 120.0 * 1.5e-3))
    assert d.max_delay == fir.max_delay + 1.5e-3  # the stability contour clamp sees the shift

    src = DynamicSource([DynamicResponseTerm(F, 3)])
    assert [x.name for x in src.param_descriptors()] == ["gain", "n", "tau"]  # single term: promoted
    assert src.with_value("n", 2.5).terms[0].transfer.n.real == 2.5

    two = DynamicSource([DynamicResponseTerm(F, 3), DynamicResponseTerm(fir, 4)])
    assert two.get("terms[1].gain") == 1.0
    assert two.with_value("terms[0].tau", 5.0e-3).terms[0].transfer.tau == 5.0e-3
    with pytest.raises(KeyError):
        two.get("gain")  # ambiguous on a multi-term source: must use terms[k].

    bc = nefes.PerturbationBC.reflection(0.9)
    assert bc.get("magnitude") == pytest.approx(0.9)
    bc2 = bc.with_value("phase", np.pi / 4)
    assert np.isclose(complex(bc2.R), 0.9 * np.exp(1j * np.pi / 4))
    assert nefes.PerturbationBC.anechoic().param_descriptors() == ()  # nothing scalar to expose


def test_recursive_inventory_and_write_paths(flame_case):
    """Nested rows appear as float rows with layer tags; get/with_params round-trip them."""
    net, _sol, _eigs = flame_case
    inv = net.parameters()
    row = inv["flame.dynamic_source.tau"]
    assert row.kind == "float" and row.layer == "perturbation" and row.value == pytest.approx(2.0e-3)
    assert "inlet-1.perturbation_bc.magnitude" in inv.addresses

    pert = net.parameters(layer="perturbation")
    assert all(r.layer == "perturbation" for r in pert)
    assert "flame.dynamic_source.gain" in pert.addresses
    assert "inlet-1.mdot" not in pert.addresses

    net2 = net.with_params({"flame.dynamic_source.tau": 3.0e-3, "inlet-1.perturbation_bc.magnitude": 0.7})
    assert net2.get("flame.dynamic_source.tau") == pytest.approx(3.0e-3)
    assert net2.get("inlet-1.perturbation_bc.magnitude") == pytest.approx(0.7)
    assert net.get("flame.dynamic_source.tau") == pytest.approx(2.0e-3)  # base pristine

    with pytest.raises(KeyError):
        net.get("flame.dynamic_source.no_such_knob")
    with pytest.raises(KeyError):
        net.with_params({"duct-1.dynamic_source.gain": 1.0})  # a duct carries no source


def test_perturbation_layer_rows_leave_the_mean_residual_invariant(flame_case):
    """A perturbation-tagged parameter never enters a mean residual: the tag is physical."""
    net, sol, _eigs = flame_case
    x = sol.x
    eps = 1e-4 * sol.problem.var_scale[0]
    R0 = residual(sol.problem, x, eps, 1e-6)
    for addr, value in [("flame.dynamic_source.tau", 4.0e-3), ("inlet-1.perturbation_bc.magnitude", 0.5)]:
        prob_h = net.with_params({addr: value}).compile()
        assert np.array_equal(residual(prob_h, x, eps, 1e-6), R0), addr
    # and a storage volume on a manifold element, the other perturbation-tagged family
    net_j = nefes.Network(
        nodes=[
            cat.mass_flow_inlet(0.05, 300.0, perturbation_bc=nefes.PerturbationBC.hard_wall()),
            cat.duct(0.3),
            cat.junction(name="plenum"),
            cat.duct(0.3),
            cat.pressure_outlet(1.0e5),
        ],
        edges=[(0, 1, AREA), (1, 2, AREA), (2, 3, AREA), (3, 4, AREA)],
    )
    assert net_j.parameters()["plenum.volume"].layer == "perturbation"
    sol_j = net_j.solve()
    eps_j = 1e-4 * sol_j.problem.var_scale[0]
    R_j = residual(sol_j.problem, sol_j.x, eps_j, 1e-6)
    prob_v = net_j.with_params({"plenum.volume": 1.0e-4}).compile()
    assert np.array_equal(residual(prob_v, sol_j.x, eps_j, 1e-6), R_j)


def test_sensitivities_over_source_knobs_match_brute_force(flame_case):
    """The one-shot dω/d(gain) and dω/d(tau) match a full re-solve/re-search difference."""
    net, _sol, eigs = flame_case
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sens = eigs.sensitivities(layer="perturbation", scheme="central")
    i = int(np.argmax(eigs.growth_rates))  # the flame-driven mode reacts most cleanly

    def brute(**kw):
        h_n, h_tau = kw.get("n", 0.0), kw.get("tau", 0.0)
        ws = []
        for s in (+1.0, -1.0):
            e = (
                _with_flame(n=0.8 + s * h_n, tau=2.0e-3 + s * h_tau)
                .solve()
                .eigenmodes(freq_band=(80.0, 600.0), isentropic=True)
            )
            ws.append(e.omega[int(np.argmin(np.abs(e.freqs - eigs.freqs[i])))])
        return (ws[0] - ws[1]) / (2.0 * (h_n + h_tau))

    bf_n = brute(n=1e-4)
    dw_n = sens["flame.dynamic_source.n"][i]
    assert abs(dw_n - bf_n) <= 1e-3 * abs(bf_n)

    bf_tau = brute(tau=1e-7)
    dw_tau = sens["flame.dynamic_source.tau"][i]
    assert abs(dw_tau - bf_tau) <= 1e-3 * abs(bf_tau)

    # S is linear in both n and the term gain, so the two scaled sensitivities must agree:
    # n * dω/dn == gain * dω/dgain
    assert np.allclose(0.8 * sens["flame.dynamic_source.n"], 1.0 * sens["flame.dynamic_source.gain"], rtol=1e-6)

    # the chain skip is exact for these rows: freezing the mean changes nothing
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frozen = eigs.sensitivities(layer="perturbation", scheme="central", chain=False)
    assert np.allclose(frozen.dw_dp, sens.dw_dp, rtol=1e-12, atol=1e-12)
