"""Linear-stability eigenmodes of a duct: the nonlinear eigenproblem det A(omega) = 0.

Runnable directly.  Solves the mean flow, then -- mirroring the forced-response
workflow (take a Solution, hand its problem/state to a perturbation driver) -- calls
``eigenmodes`` to find the network's free oscillations in a frequency band by Beyn's
contour-integral method (theory.md s12.7 (ii)).

Two cases:

1. A rigid-rigid (closed-closed) duct: lossless, so the modes are marginally stable
   at ``f_n = n c / 2L``.
2. The same duct with a partially absorbing outlet (``|R| < 1``): every mode now
   decays (negative growth rate) -- the passive-stability check.
"""

import numpy as np

from fns.shell import Network
from fns.elements import catalog as cat
from fns.thermo.configure import perfect_gas
from fns.derive import ES_U, ES_C
from fns.perturbation import PerturbationBC, eigenmodes

CFG = perfect_gas(287.0, 1.4)
L = 0.5  # duct length [m]


def _build(outlet_bc_elem):
    """[total-pressure inlet, rigid] -- duct(L) -- [outlet element]."""
    net = Network(CFG, p_ref=101325.0, T_ref=300.0, mdot_ref=5.0)
    net.add(cat.total_pressure_inlet(101325.0, 300.0, perturbation_bc=PerturbationBC.hard_wall()))
    net.add(cat.duct(L))
    net.add(outlet_bc_elem)
    net.connect(0, 1, 0.05)
    net.connect(1, 2, 0.05)
    sol = net.solve()
    assert sol.converged
    return sol


def _report(title, sol):
    c = float(sol.table()[ES_C, 0])
    u = float(sol.table()[ES_U, 0])
    print(f"\n{title}  (mean u = {u:.3g} m/s, c = {c:.1f} m/s)")
    print(f"  expected lossless modes  f_n = n c / 2L = {', '.join(f'{n * c / (2 * L):.1f}' for n in (1, 2, 3))} Hz")
    res = eigenmodes(sol.problem, sol.x, (0.5 * c / (2 * L), 3.5 * c / (2 * L)))
    cert = "yes" if res.certified else "NO"
    print(f"  found {res.n_modes} mode(s); argument-principle count = {res.expected} (complete? {cert}):")
    print(f"    {'freq [Hz]':>12} {'growth [1/s]':>14} {'damping':>10} {'stable?':>9} {'residual':>10}")
    for m in res.summary():
        print(
            f"    {m['freq_hz']:>12.2f} {m['growth_rate']:>14.4g} {m['damping_ratio']:>10.3e} "
            f"{(not m['unstable']):>9} {m['residual']:>10.1e}"
        )
    return res


if __name__ == "__main__":
    # 1. closed-closed (rigid wall both ends): lossless, marginal modes at n c / 2L.
    res_closed = _report("Rigid-rigid duct (lossless)", _build(cat.wall()))
    assert np.all(np.abs(res_closed.growth_rates) < 1.0)  # essentially marginal

    # 2. partially absorbing outlet: every mode decays (passive -> stable).
    res_lossy = _report(
        "Rigid inlet, absorbing outlet R = 0.6",
        _build(cat.pressure_outlet(101325.0, 300.0, perturbation_bc=PerturbationBC.reflection(0.6))),
    )
    assert np.all(res_lossy.growth_rates < 0.0)  # all decaying

    print("\nThe spectrum and mode shapes are plottable in a notebook:")
    print("    res.plot_spectrum().show()")
    print("    res.plot_mode(0).show()")
