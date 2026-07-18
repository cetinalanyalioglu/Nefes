"""Eigenvalue sensitivities: the one-shot derivative against brute-force and analytic references.

The derivative ``d omega / d p`` from :func:`nefes.perturbation.eigenvalue_sensitivities` must
match (i) a full re-solve/re-search finite difference and (ii) the closed-form frequency laws of
a uniform duct (``f = n c / 2L`` gives ``df/dL = -f/L`` and ``df/dT = f/(2T)``).  The mean-flow
route is pinned separately: an inlet-temperature sensitivity flows *only* through the mean state,
so it vanishes with ``chain=False`` and must be exact with ``chain=True``.
"""

import warnings

import numpy as np
import pytest

import nefes
from nefes.elements import catalog as cat
from nefes.perturbation import SensitivityWarning

AREA = 0.004  # small enough for a mean Mach ~0.15, so the mean flow couples visibly


def _build(L=0.5, mdot=2.0, Tt=300.0):
    return nefes.Network(
        nodes=[
            cat.mass_flow_inlet(mdot, Tt, perturbation_bc=nefes.PerturbationBC.hard_wall()),
            cat.duct(L),
            cat.choked_nozzle_outlet(1e-3, name="throat"),
        ],
        edges=[(0, 1, AREA), (1, 2, AREA)],
    )


@pytest.fixture(scope="module")
def base():
    """Solved base network, its spectrum, and the central-scheme sensitivities."""
    sol = _build().solve()
    eigs = sol.eigenmodes(freq_band=(200.0, 900.0), isentropic=True)
    assert eigs.n_modes >= 2 and eigs.certified
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sens = eigs.sensitivities(scheme="central")
    return sol, eigs, sens


def _brute(kw, base_value, eigs, rel=1e-3):
    """Full re-solve / re-search central difference of mode 0 in ``omega``."""
    h = rel * base_value
    ws = []
    for s in (+h, -h):
        e = _build(**{kw: base_value + s}).solve().eigenmodes(freq_band=(200.0, 900.0), isentropic=True)
        ws.append(e.omega[int(np.argmin(np.abs(e.freqs - eigs.freqs[0])))])
    return (ws[0] - ws[1]) / (2.0 * h)


def test_matches_brute_force_and_analytic_laws(base):
    """The one-shot derivative reproduces the re-solve finite difference and the duct laws."""
    _sol, eigs, sens = base
    f0, L, T = eigs.freqs[0], 0.5, 300.0

    dw_L = sens["duct-1.length"][0]
    bf_L = _brute("L", L, eigs)
    assert abs(dw_L - bf_L) <= 1e-4 * abs(bf_L)
    # closed-form: f = n c / 2L, so df/dL = -f/L
    assert np.isclose(dw_L.real / (2.0 * np.pi), -f0 / L, rtol=2e-2)

    dw_T = sens["inlet-1.Tt"][0]
    bf_T = _brute("Tt", T, eigs)
    assert abs(dw_T - bf_T) <= 1e-4 * abs(bf_T)
    # closed-form: c ~ sqrt(T), so df/dT = f / (2T)
    assert np.isclose(dw_T.real / (2.0 * np.pi), f0 / (2.0 * T), rtol=2e-2)


def test_mean_flow_route_is_the_chain_term(base):
    """An inlet-temperature sensitivity acts only through the mean state: zero without the chain."""
    _sol, eigs, sens = base
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frozen = eigs.sensitivities(scheme="central", chain=False)
    assert abs(frozen["inlet-1.Tt"][0]) <= 1e-10 * abs(sens["inlet-1.Tt"][0]) + 1e-12
    # a duct length never touches the mean flow: with and without the chain agree
    assert np.isclose(frozen["duct-1.length"][0], sens["duct-1.length"][0], rtol=1e-8)


def test_parameter_selection_knobs(base):
    """``include``/``exclude`` narrow by glob pattern; ``params`` is explicit and validated."""
    _sol, eigs, _sens = base
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        only_lengths = eigs.sensitivities(include="*.length")
        no_lengths = eigs.sensitivities(exclude=["*.length", "*.marker", "e*.area"])
        explicit = eigs.sensitivities(params=["inlet-1.Tt"])
    assert only_lengths.addresses == ["duct-1.length"]
    assert all(not a.endswith(".length") for a in no_lengths.addresses)
    assert explicit.addresses == ["inlet-1.Tt"] and explicit.dw_dp.shape == (eigs.n_modes, 1)
    with pytest.raises(KeyError):
        eigs.sensitivities(params=["no-such.parameter"])


def test_zero_valued_parameter_is_probed_one_sided():
    """A parameter sitting at zero (an unset end correction) is probed with the absolute step."""
    sol = nefes.Network(
        nodes=[
            cat.mass_flow_inlet(2.0, 300.0, perturbation_bc=nefes.PerturbationBC.hard_wall()),
            cat.duct(0.4),
            cat.isentropic_area_change(name="iac"),
            cat.duct(0.2),
            cat.choked_nozzle_outlet(1e-3, name="throat"),
        ],
        edges=[(0, 1, AREA), (1, 2, AREA), (2, 3, 2.0 * AREA), (3, 4, 2.0 * AREA)],
    ).solve()
    eigs = sol.eigenmodes(freq_band=(150.0, 700.0), isentropic=True)
    assert eigs.n_modes >= 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sens = eigs.sensitivities(params=["iac.end_correction"])
    assert sens.addresses == ["iac.end_correction"]
    assert "iac.end_correction" not in sens.failed
    assert np.all(np.isfinite(sens.dw_dp))
    # the end correction adds effective length: it must move the frequency, not vanish
    assert np.abs(sens.dfreq_dp).max() > 1.0
    # ranked influence uses the probe step (the +1% column would hide a zero-valued parameter)
    assert sens.influence()[0] > 0.0


def test_unprobeable_parameter_is_recorded_and_warned(base):
    """A single edge area of a constant-area duct cannot move alone: skipped, warned, recorded."""
    _sol, eigs, _sens = base
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sens = eigs.sensitivities(params=["e0.area"])
    assert "e0.area" in sens.failed
    assert sens.n_params == 0
    assert any(isinstance(w.message, SensitivityWarning) for w in caught)


def test_result_reprs_and_plot(base):
    """Text and HTML reprs rank by influence; the plot builds a figure without error."""
    _sol, eigs, sens = base
    text = repr(sens)
    assert "EigenmodeSensitivityResult" in text and "destabilizing" in text
    assert sens.top(1)[0] in text
    html = sens._repr_html_()
    assert "EigenmodeSensitivityResult" in html and sens.top(1)[0] in html
    fig = sens.plot(top=5)
    assert len(fig.data) >= 1
    fig1 = sens.plot(modes=0)
    assert len(fig1.data) == 1


def test_low_level_result_needs_an_explicit_solution():
    """A spectrum from the raw ``eigenmodes(problem, x_bar)`` call carries no network reference."""
    from nefes.perturbation import eigenmodes

    sol = _build().solve()
    eigs = eigenmodes(sol.problem, sol.x, freq_band=(200.0, 900.0), isentropic=True)
    with pytest.raises(ValueError, match="solution"):
        eigs.sensitivities()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sens = eigs.sensitivities(solution=sol, params=["duct-1.length"], isentropic=True)
    assert sens.n_params == 1 and np.all(np.isfinite(sens.dw_dp))
