"""The linear flow resistance element: a quiescent-capable acoustic resistance.

Mean flow: a total-pressure drop linear in the through-flow, ``Pt_in - Pt_out = R * mdot``.
Acoustics: because the drop is *linear* in the flow (not the quadratic dynamic head a
``loss`` uses), its damping survives the ``M -> 0`` (quiescent) limit, where a velocity-squared
loss vanishes.
"""

import numpy as np
import pytest

from nefes.assembly.recover import ES_MDOT, ES_PT
from nefes.elements import catalog as cat
from nefes.perturbation import PerturbationBC, perturbation_response
from nefes.shell import Network
from nefes.thermo.configure import perfect_gas

CFG = perfect_gas(287.0, 1.4)


def _flowing(elem, pt_in=106000.0, mdot_ref=3.0):
    net = Network(CFG, p_ref=101325.0, T_ref=300.0, mdot_ref=mdot_ref)
    net.add(cat.total_pressure_inlet(pt_in, 300.0))
    net.add(cat.duct(0.4))
    net.add(elem)
    net.add(cat.duct(0.4))
    net.add(cat.pressure_outlet(101325.0, 300.0))
    for a in range(4):
        net.connect(a, a + 1, 0.05)
    sol = net.solve()
    assert sol.converged
    return sol


def test_mean_drop_is_linear_in_mass_flow():
    for R in (3.0, 30.0, 300.0):
        e = _flowing(cat.linear_resistance(R)).table()
        mdot = float(e[ES_MDOT, 1])  # through-flow on the element's tail edge
        dpt = float(e[ES_PT, 1] - e[ES_PT, 2])  # total-pressure drop across the element
        assert dpt == pytest.approx(R * abs(mdot), rel=1e-6)


def test_zero_resistance_is_transparent():
    e = _flowing(cat.linear_resistance(0.0)).table()
    assert float(e[ES_PT, 1] - e[ES_PT, 2]) == pytest.approx(0.0, abs=1e-6)


def test_rejects_negative_resistance():
    with pytest.raises(ValueError, match="non-negative"):
        cat.linear_resistance(-5.0)


# -- the headline: acoustic damping that does not vanish at low Mach -----------


def _transmission_attenuation(elem, mdot):
    """``1 - |T_ff|`` of the forward acoustic wave across the element at 200 Hz.

    A mass-flow inlet sets the Mach directly from ``mdot``, so the low-``mdot`` case is
    near-quiescent.
    """
    net = Network(CFG, p_ref=101325.0, T_ref=300.0, mdot_ref=max(mdot, 1e-3))
    net.add(cat.mass_flow_inlet(mdot, 300.0, perturbation_bc=PerturbationBC.anechoic(driven=("acoustic",))))
    net.add(cat.duct(0.4))
    net.add(elem)
    net.add(cat.duct(0.4))
    net.add(cat.pressure_outlet(101325.0, 300.0, perturbation_bc=PerturbationBC.anechoic()))
    for a in range(4):
        net.connect(a, a + 1, 0.05)
    sol = net.solve()
    assert sol.converged
    resp = perturbation_response(sol.problem, sol.x, np.array([200.0]))
    T = resp.transfer_matrix(0, 3)[0]  # 2-port transfer edge 0 -> edge 3, characteristic basis
    return 1.0 - abs(T[0, 0])


def test_acoustic_damping_persists_at_low_mach_unlike_a_quadratic_loss():
    mid, low = 0.30, 0.03  # the low case is ~M = 0.0015 (near-quiescent)

    res_mid = _transmission_attenuation(cat.linear_resistance(300.0), mid)
    res_low = _transmission_attenuation(cat.linear_resistance(300.0), low)
    loss_mid = _transmission_attenuation(cat.loss(3.0), mid)
    loss_low = _transmission_attenuation(cat.loss(3.0), low)

    # the linear resistance damps about the same amount regardless of Mach...
    assert res_low > 0.5 * res_mid
    # ...while the quadratic loss's damping collapses toward zero as the flow stills...
    assert loss_low < 0.3 * loss_mid
    # ...so at the near-quiescent point only the linear resistance still absorbs sound.
    assert res_low > 3.0 * loss_low
