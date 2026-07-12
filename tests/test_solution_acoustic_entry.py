"""The perturbation entry points accept a solved ``Solution`` in place of ``(problem, x)``.

A user working through the :class:`nefes.Network` / :class:`nefes.Solution` front door holds a
``Solution``, not the low-level ``(CompiledProblem, mean state)`` pair the perturbation routines
were originally written against.  The :func:`~nefes.perturbation._meanstate.accepts_solution`
decorator lets every routine take the ``Solution`` directly.  These tests pin that the
``Solution``-first call is accepted and returns exactly what the explicit-pair call returns.
"""

import warnings

import numpy as np
import pytest

import nefes
from nefes.elements import catalog as cat
from nefes.elements.dynamic_source import n_tau_flame
from nefes.perturbation import (
    PerturbationBC,
    eigenmodes,
    eigenvalue_trajectory,
    find_terminals,
    forced_response,
    nyquist_stability,
    nyquist_stability_map,
    open_loop_response,
    perturbation_response,
    verify_perturbation,
)


def _rijke():
    """A Rijke tube with a named n-tau flame, for the parameter-swept stability methods."""
    R, gamma = 287.0, 1.4
    cp = gamma * R / (gamma - 1.0)
    mdot, area = 0.005, 0.01
    return nefes.Network(
        nefes.perfect_gas(R, gamma),
        nodes=[
            cat.mass_flow_inlet(mdot, 300.0, perturbation_bc=PerturbationBC.hard_wall()),
            cat.duct(0.6),
            cat.heat_release_flame(mdot * cp * 400.0, name="fl", dynamic_source=n_tau_flame(0.8, 4.0e-3, ref_edge=1)),
            cat.duct(0.4),
            cat.pressure_outlet(1.0e5, perturbation_bc=PerturbationBC.open_end()),
        ],
        edges=[(0, 1, area), (1, 2, area), (2, 3, area), (3, 4, area)],
        mdot_ref=mdot,
        h_ref=cp * 300.0,
    )


@pytest.fixture
def driven_duct():
    """A short driven duct: anechoic-driven inlet, rigid outlet, converged."""
    net = nefes.Network(
        nodes=[
            cat.total_pressure_inlet(1.02e5, 300.0, perturbation_bc=PerturbationBC.anechoic(driven=("acoustic",))),
            cat.duct(0.5),
            cat.pressure_outlet(1.0e5, 300.0, perturbation_bc=PerturbationBC.hard_wall()),
        ],
        edges=[(0, 1, 0.01), (1, 2, 0.01)],
    )
    sol = net.solve()
    assert sol.converged
    return sol


def test_forced_response_accepts_solution(driven_duct):
    sol = driven_duct
    freqs = np.linspace(50.0, 1500.0, 40)
    from_sol = forced_response(sol, freqs).reflection_at(1)
    from_pair = forced_response(sol.problem, sol.x, freqs).reflection_at(1)
    assert np.allclose(from_sol, from_pair)


def test_perturbation_response_accepts_solution(driven_duct):
    sol = driven_duct
    freqs = np.linspace(50.0, 1500.0, 40)
    from_sol = perturbation_response(sol, freqs).transfer_matrix(0, 1)
    from_pair = perturbation_response(sol.problem, sol.x, freqs).transfer_matrix(0, 1)
    assert np.allclose(from_sol, from_pair)


def test_eigenmodes_accepts_solution(driven_duct):
    sol = driven_duct
    from_sol = eigenmodes(sol, freq_band=(100.0, 1500.0))
    from_pair = eigenmodes(sol.problem, sol.x, freq_band=(100.0, 1500.0))
    assert np.allclose(np.sort(from_sol.freqs), np.sort(from_pair.freqs))


def test_find_terminals_accepts_solution(driven_duct):
    sol = driven_duct
    assert len(find_terminals(sol)) == len(find_terminals(sol.problem, sol.x))


def test_verify_perturbation_accepts_solution(driven_duct):
    sol = driven_duct
    #  Neither call should raise on a valid subsonic network.
    verify_perturbation(sol)
    verify_perturbation(sol.problem, sol.x)


def test_nyquist_accepts_solution():
    """Nyquist needs a dynamic source: a Rijke tube with an n-tau flame."""
    R, gamma = 287.0, 1.4
    cp = gamma * R / (gamma - 1.0)
    mdot, area = 0.005, 0.01
    net = nefes.Network(
        nodes=[
            cat.mass_flow_inlet(mdot, 300.0, perturbation_bc=PerturbationBC.hard_wall()),
            cat.duct(0.6),
            cat.heat_release_flame(mdot * cp * 400.0, dynamic_source=n_tau_flame(0.8, 4.0e-3, ref_edge=1)),
            cat.duct(0.4),
            cat.pressure_outlet(1.0e5, perturbation_bc=PerturbationBC.open_end()),
        ],
        edges=[(0, 1, area), (1, 2, area), (2, 3, area), (3, 4, area)],
        mdot_ref=mdot,
        h_ref=cp * 300.0,
    )
    sol = net.solve()
    freqs = np.linspace(1.0, 520.0, 200)
    #  The band-edge-quiet warning is about count convergence, not the equality under test.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from_sol = nyquist_stability(sol, freqs).n_unstable
        from_pair = nyquist_stability(sol.problem, sol.x, freqs).n_unstable
        assert from_sol == from_pair
        #  open_loop_response takes the same first argument.
        assert np.allclose(open_loop_response(sol, freqs).D, open_loop_response(sol.problem, sol.x, freqs).D)


def test_problem_pair_still_works(driven_duct):
    """Passing the explicit (problem, mean state) pair remains valid (backward compatible)."""
    sol = driven_duct
    freqs = np.linspace(50.0, 800.0, 10)
    res = perturbation_response(sol.problem, sol.x, freqs)
    assert res.freqs.shape == freqs.shape


# -- bound OO methods delegate to the free functions --------------------------------------------------------------


def test_solution_eigenmodes_method(driven_duct):
    sol = driven_duct
    a = sol.eigenmodes(freq_band=(100.0, 1500.0)).freqs
    b = eigenmodes(sol, freq_band=(100.0, 1500.0)).freqs
    assert np.allclose(np.sort(a), np.sort(b))


def test_solution_forced_response_method(driven_duct):
    sol = driven_duct
    freqs = np.linspace(50.0, 1500.0, 30)
    assert np.allclose(sol.forced_response(freqs).reflection_at(1), forced_response(sol, freqs).reflection_at(1))


def test_solution_perturbation_response_method(driven_duct):
    sol = driven_duct
    freqs = np.linspace(50.0, 1500.0, 30)
    a = sol.perturbation_response(freqs).transfer_matrix(0, 1)
    b = perturbation_response(sol, freqs).transfer_matrix(0, 1)
    assert np.allclose(a, b)


def test_solution_nyquist_stability_method():
    net = _rijke()
    sol = net.solve()
    freqs = np.linspace(1.0, 520.0, 200)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert sol.nyquist_stability(freqs).n_unstable == nyquist_stability(sol, freqs).n_unstable


def test_network_eigenvalue_trajectory_method():
    net = _rijke()
    vals = np.linspace(0.5, 1.5, 4) * net.get("fl.Qdot")
    kw = dict(freq_band=(20.0, 520.0), growth_band=(-400.0, 400.0), isentropic=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from_method = net.eigenvalue_trajectory("fl.Qdot", vals, **kw)
        from_free = eigenvalue_trajectory(net.builder("fl.Qdot"), vals, param_name="fl.Qdot", **kw)
    assert from_method.n_branches == from_free.n_branches
    assert from_method.param_name == "fl.Qdot"  # defaults to the address


def test_network_nyquist_stability_map_method():
    net = _rijke()
    vals = np.linspace(0.5, 1.5, 4) * net.get("fl.Qdot")
    freqs = np.linspace(1.0, 520.0, 150)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from_method = net.nyquist_stability_map("fl.Qdot", vals, freqs)
        from_free = nyquist_stability_map(net.builder("fl.Qdot"), vals, freqs, param_name="fl.Qdot")
    assert np.array_equal(from_method.n_unstable, from_free.n_unstable)
    assert from_method.param_name == "fl.Qdot"
