"""Equilibrium warm-start: identical results, fewer Newton steps.

The per-edge HP-equilibrium solve is the dominant reacting cost.  Two mechanisms keep it cheap:

* the recovery folds the closure's static density into a single equilibrium solve, so the
  reacting edge runs the equilibrium once (closure + thermo state together);
* the Jacobian seeds each complex-step column's equilibrium from the freshly recovered base
  composition *and* temperature, so it converges in a couple of Newton steps.

Both are pure speed: the HP equilibrium is unique, so the converged state -- and the assembled
residual and Jacobian -- are independent of the warm start.
"""

import numpy as np
import pytest

from nefes.shell import Network
from nefes.elements import catalog as cat
from nefes.thermo.configure import equilibrium
from nefes.thermo.api import EQ_FROZEN, EQ_KERNEL
from nefes.chem.composition import equivalence_ratio_mixture
from nefes.assembly import assemble

thermolib = pytest.importorskip("thermolib")


@pytest.fixture(scope="module")
def reacting_solution():
    lib = thermolib.SpeciesLibrary.from_cea(species=["H2", "O2", "N2", "H2O", "OH", "H", "O", "HO2", "NO"])
    mix = equivalence_ratio_mixture(lib, {"H2": 1.0}, {"O2": 0.21, "N2": 0.79}, 1.0)
    net = Network(equilibrium(lib), p_ref=101325.0, T_ref=300.0, mdot_ref=1.0, h_ref=1.0e6)
    net.add(cat.mass_flow_inlet(1.0, 300.0, composition=mix, basis="mole"))
    net.add(cat.equilibrium_flame())
    net.add(cat.duct(0.3))
    net.add(cat.pressure_outlet(101325.0, Tt_backflow=300.0, composition=mix, basis="mole"))
    net.connect(0, 1, 0.05, edge_model=EQ_FROZEN)
    net.connect(1, 2, 0.05, edge_model=EQ_KERNEL)
    net.connect(2, 3, 0.05, edge_model=EQ_KERNEL)
    sol = net.solve()
    assert sol.converged
    return sol


def test_reacting_solve_converges(reacting_solution):
    # The headline path still solves cleanly with the warm-started Jacobian.
    assert reacting_solution.converged
    T = reacting_solution.table()[5]  # ES_T
    assert T[1] > 2000.0  # the burnt edge is hot (H2/air flame)


def test_jacobian_is_invariant_to_the_warm_start(reacting_solution, monkeypatch):
    """The warm start changes only the iteration count -- the assembled Jacobian is identical."""
    prob, x = reacting_solution.problem, reacting_solution.x
    J_warm = assemble.jacobian(prob, x, 1e-4, 1e-6).toarray()

    # disable the warm start (an (E, 0) cache): the Jacobian must come out the same to round-off
    monkeypatch.setattr(assemble, "_nj_cache_jacobian", lambda p: np.zeros((p.n_edges, 0)))
    J_cold = assemble.jacobian(prob, x, 1e-4, 1e-6).toarray()

    assert np.allclose(J_warm, J_cold, rtol=1e-7, atol=1e-8)


def test_residual_is_invariant_to_the_warm_start(reacting_solution):
    """The residual (warm start off by design) matches a re-evaluation."""
    prob, x = reacting_solution.problem, reacting_solution.x
    R1 = assemble.residual(prob, x, 1e-4, 1e-6)
    R2 = assemble.residual(prob, x, 1e-4, 1e-6)
    assert np.allclose(R1, R2, rtol=0.0, atol=0.0)  # bit-for-bit deterministic
    # at the converged state the (raw, unscaled) residual is small
    assert np.max(np.abs(R1)) < 1e-3


def test_cache_carries_moles_and_temperature(reacting_solution):
    # the per-edge Jacobian cache is Ns + 1 wide (moles + temperature) for a reacting problem
    prob = reacting_solution.problem
    cache = assemble._nj_cache_jacobian(prob)
    assert cache.shape == (prob.n_edges, int(prob.ti[1]) + 1)
