"""Acoustic intensity-along-ducts field (`intensity_along_network`)."""

import numpy as np
import pytest

from nefes.assembly.recover import ES_C, ES_M, ES_RHO
from nefes.elements import catalog as cat
from nefes.perturbation import (
    PerturbationBC,
    acoustic_intensity,
    eigenmodes,
    perturbation_response,
)
from nefes.shell import Network
from nefes.thermo.configure import perfect_gas

CFG = perfect_gas(287.0, 1.4)
LDUCT = 0.5


def _driven_duct(area=0.05, mdot_ref=5.0, refl=0.6):
    net = Network(CFG, p_ref=101325.0, T_ref=300.0, mdot_ref=mdot_ref)
    net.add(cat.total_pressure_inlet(120000.0, 300.0, perturbation_bc=PerturbationBC.reflection(refl)))
    net.add(cat.duct(LDUCT))
    net.add(cat.pressure_outlet(101325.0, 300.0, perturbation_bc=PerturbationBC.reflection(refl)))
    net.connect(0, 1, area)
    net.connect(1, 2, area)
    sol = net.solve()
    assert sol.converged
    return sol


def test_intensity_field_is_real_and_spans_the_duct():
    sol = _driven_duct()
    c = float(sol.table()[ES_C, 0])
    f1 = c / (2.0 * LDUCT)
    resp = perturbation_response(sol.problem, sol.x, np.array([f1]))
    fields = resp.intensity_along_network(f1, n_x=60)
    assert fields
    pf = fields[0]
    assert pf.x[-1] == pytest.approx(LDUCT)
    # The intensity is a real diagnostic.
    assert np.isrealobj(pf.values) or np.allclose(pf.values.imag, 0.0)


def test_intensity_is_constant_within_a_lossless_duct():
    # At a real drive frequency each wave magnitude |f|, |g| is constant along a uniform
    # duct, so the Myers intensity is flat within the duct (only |f|^2, |g|^2 enter).
    sol = _driven_duct()
    c = float(sol.table()[ES_C, 0])
    f_drive = 0.37 * c / (2.0 * LDUCT)  # an off-resonance frequency
    resp = perturbation_response(sol.problem, sol.x, np.array([f_drive]))
    pf = resp.intensity_along_network(f_drive, n_x=80)[0]
    vals = np.real(pf.values)
    scale = np.max(np.abs(vals)) or 1.0
    assert np.ptp(vals) < 1e-6 * scale  # flat to numerical precision


def test_intensity_matches_face_value_from_myers_formula():
    sol = _driven_duct()
    est = sol.table()
    c = float(est[ES_C, 0])
    f_drive = 0.42 * c / (2.0 * LDUCT)
    resp = perturbation_response(sol.problem, sol.x, np.array([f_drive]))
    pf = resp.intensity_along_network(f_drive, n_x=40)[0]

    # Recompute the intensity at the inlet face (edge 0) straight from the wave amplitudes,
    # superposing the sources the same way the field does (default: unit on the first source).
    inc = np.zeros(resp.X.shape[1], dtype=np.complex128)
    inc[0] = 1.0
    chars0 = resp._waves(0)[0] @ inc  # (n_char,)
    rho, cc, mach = float(est[ES_RHO, 0]), float(est[ES_C, 0]), float(est[ES_M, 0])
    expected = acoustic_intensity(rho, cc, mach, complex(chars0[0]), complex(chars0[1]))
    assert np.real(pf.values[0]) == pytest.approx(expected, rel=1e-6, abs=1e-9)


def test_energy_density_is_nonnegative():
    sol = _driven_duct()
    c = float(sol.table()[ES_C, 0])
    f_drive = 0.6 * c / (2.0 * LDUCT)
    resp = perturbation_response(sol.problem, sol.x, np.array([f_drive]))
    pf = resp.intensity_along_network(f_drive, energy_density=True, n_x=50)[0]
    assert np.all(np.real(pf.values) >= -1e-12)  # subsonic energy density is non-negative


def test_eigenmode_intensity_field_runs():
    net = Network(CFG, p_ref=101325.0, T_ref=300.0, mdot_ref=5.0)
    net.add(cat.total_pressure_inlet(101325.0, 300.0, perturbation_bc=PerturbationBC.hard_wall()))
    net.add(cat.duct(LDUCT))
    net.add(cat.wall())
    net.connect(0, 1, 0.05)
    net.connect(1, 2, 0.05)
    sol = net.solve()
    c = float(sol.table()[ES_C, 0])
    res = eigenmodes(sol.problem, sol.x, (0.5 * c / (2 * LDUCT), 1.5 * c / (2 * LDUCT)))
    fields = res.intensity_along_network(0, n_x=40)
    assert fields and np.isrealobj(np.real(fields[0].values))
