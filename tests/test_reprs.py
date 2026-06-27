"""Result objects print a user-friendly summary, never a raw array dump.

Every data-bearing result/diagnostic class returned by the public API defines a
concise ``__repr__`` (and, for the table-like single-state results, an HTML
``_repr_html_`` for notebooks).  These tests guard that contract: no ``repr`` may
leak a NumPy ``array(...)`` dump, and each must name its class and survive on a
real solved network.
"""

import warnings

import numpy as np
import pytest

from fns.elements import catalog as cat
from fns.elements.dynamic_source import n_tau_flame
from fns.perturbation import (
    PerturbationBC,
    boundary_power,
    build_blocks,
    eigenmodes,
    excite_perturbation,
    find_terminals,
    forced_power_balance,
    forced_response,
    modal_energy_balance,
    open_loop_response,
    perturbation_response,
)
from fns.shell import Network
from fns.thermo.configure import perfect_gas

R_AIR, GAMMA = 287.0, 1.4
CP = GAMMA * R_AIR / (GAMMA - 1.0)


def _rijke(n=1.0, tau=0.003):
    """Inlet -> cold duct -> n-tau flame -> hot duct -> driven open end."""
    net = Network(perfect_gas(R_AIR, GAMMA), p_ref=1.0e5, T_ref=300.0, mdot_ref=0.006)
    bc = PerturbationBC.mean_flow_open_end(driven=("acoustic",))
    i_in = net.add(cat.mass_flow_inlet(0.006, 300.0))
    i_cold = net.add(cat.duct(0.6))
    i_flame = net.add(cat.heat_release_flame(0.006 * CP * 400.0))
    i_hot = net.add(cat.duct(0.4))
    i_out = net.add(cat.pressure_outlet(1.0e5, perturbation_bc=bc))
    net.connect(i_in, i_cold, 0.01)
    ref = net.connect(i_cold, i_flame, 0.01)
    net.connect(i_flame, i_hot, 0.01)
    net.connect(i_hot, i_out, 0.01)
    net.set_dynamic_source(i_flame, n_tau_flame(n, tau, ref_edge=ref))
    sol = net.solve()
    assert sol.converged
    return sol


@pytest.fixture(scope="module")
def rig():
    """A solved Rijke tube plus the analysis products whose reprs we test."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = _rijke()
        freqs = np.linspace(20.0, 400.0, 50)
        fr = forced_response(sol.problem, sol.x, freqs, isentropic=True)
        res = eigenmodes(sol.problem, sol.x, freq_band=(40.0, 320.0), growth_band=(-200.0, 200.0), isentropic=True)
        terms = find_terminals(sol.problem, sol.x)
        objects = {
            "SolveResult": sol.result,
            "ForcedResponse": fr,
            "ForcedPowerBalance": forced_power_balance(fr, sol.problem),
            "EigenmodeResult": res,
            "Contour": res.contour,
            "NetworkGeometry": res.geometry,
            "BoundaryPower": boundary_power(res, 0),
            "ModalEnergyBalance": modal_energy_balance(res, 0),
            "Terminal": terms[0],
            "PerturbationField": excite_perturbation(sol.problem, sol.x, freqs, node=terms[0].node),
            "PerturbationResponse": perturbation_response(sol.problem, sol.x, freqs),
            "PathField": res.field_along_network(0, variable="p")[0],
            "NyquistResponse": open_loop_response(sol.problem, sol.x, freqs),
            "AcousticBlocks": build_blocks(sol.problem, sol.x),
        }
    return objects


def test_repr_is_a_clean_summary(rig):
    """No repr leaks a raw NumPy array dump, and each names its own class."""
    for name, obj in rig.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            text = repr(obj)
        assert "array(" not in text, f"{name} repr dumps a numpy array:\n{text}"
        assert name in text.splitlines()[0], f"{name} repr should lead with its class name; got:\n{text}"


def test_repr_html_is_valid_when_present(rig):
    """Classes exposing ``_repr_html_`` return a non-empty HTML fragment with no array dump."""
    has_html = {"BoundaryPower", "ModalEnergyBalance", "NyquistResponse", "EigenmodeResult"}
    for name in has_html:
        obj = rig[name]
        assert hasattr(obj, "_repr_html_"), f"{name} should define _repr_html_"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            html = obj._repr_html_()
        assert html.strip().startswith("<"), f"{name} html is not a tag:\n{html}"
        assert "array(" not in html, f"{name} html dumps a numpy array"
        assert "<table" in html or "<div" in html, f"{name} html has no structure"
