"""Spatial mode-shape reconstruction: the analytic intra-duct field (theory.md s12.3).

A ``DUCT`` is uniform and lossless, so its interior perturbation field is the duct
stamp's phase relation evaluated at every station -- exact, not a discretization.
The checks are against closed form:

* endpoint consistency -- the reconstructed field at a duct's two ends equals the
  stored face wave-amplitudes there;
* closed-closed quiescent fundamental -- ``p'(x) ~ cos(pi x / L)`` (rigid ends);
* open-closed quarter-wave -- ``p'(x) ~ sin(pi x / 2L)`` (pressure node at the open end);
* a serial multi-duct network -- developed length adds up, an area change shows a
  jump (with a marker), a directly-shared edge stays continuous;
* the animated figure carries frames + a phase slider + a play control, for both an
  eigenmode and a forced response.
"""

import warnings

import numpy as np
import pytest

from fns.shell import Network
from fns.elements import catalog as cat
from fns.thermo.configure import perfect_gas
from fns.derive import ES_U, ES_C
from fns.perturbation import (
    PerturbationBC,
    eigenmodes,
    perturbation_response,
)
from fns.perturbation.modeshape import VARIABLE_SPEC, build_geometry, resolve_specs, PathField
from fns.plotting import animate_mode_shape, AnimSeries

CFG = perfect_gas(287.0, 1.4)
LDUCT = 0.5


def _duct_net(inlet_bc, outlet_elem, *, pt_in=101325.0, L=LDUCT, area=0.05, mdot_ref=5.0):
    net = Network(CFG, p_ref=101325.0, T_ref=300.0, mdot_ref=mdot_ref)
    net.add(cat.total_pressure_inlet(pt_in, 300.0, perturbation_bc=inlet_bc))
    net.add(cat.duct(L))
    net.add(outlet_elem)
    net.connect(0, 1, area)
    net.connect(1, 2, area)
    sol = net.solve()
    assert sol.converged
    return net, sol


def _uc(sol, e=0):
    est = sol.table()
    return float(est[ES_U, e]), float(est[ES_C, e])


def test_endpoint_consistency_matches_face_waves():
    # the reconstructed field at the duct ends must equal the mode's stored face values.
    _, sol = _duct_net(PerturbationBC.hard_wall(), cat.wall())
    u, c = _uc(sol)
    f1 = c / (2.0 * LDUCT)
    res = eigenmodes(sol.problem, sol.x, (0.5 * f1, 1.5 * f1))
    i = int(np.argmin(np.abs(res.freqs - f1)))
    seg = res.geometry.ducts[0]

    # project the stored face waves to p' the same way the reconstruction does
    from fns.perturbation.characteristics import basis_block_from_state

    basis, comp, _ = VARIABLE_SPEC["p"]
    p_tail = (basis_block_from_state(basis, res.est[:, seg.e_tail], res.K, None) @ res.mode_waves(i, seg.e_tail))[comp]
    p_head = (basis_block_from_state(basis, res.est[:, seg.e_head], res.K, None) @ res.mode_waves(i, seg.e_head))[comp]

    pf = res.field_along_network(i, variable="p", n_x=64)[0]
    assert pf.x[0] == pytest.approx(0.0)
    assert pf.x[-1] == pytest.approx(LDUCT)
    assert pf.values[0] == pytest.approx(p_tail, rel=1e-9, abs=1e-9)
    assert pf.values[-1] == pytest.approx(p_head, rel=1e-9, abs=1e-9)


def test_closed_closed_pressure_is_cosine():
    # rigid-rigid quiescent fundamental: p'(x) = cos(pi x / L), antinodes at both walls.
    _, sol = _duct_net(PerturbationBC.hard_wall(), cat.wall())
    u, c = _uc(sol)
    assert abs(u) < 1e-9
    f1 = c / (2.0 * LDUCT)
    res = eigenmodes(sol.problem, sol.x, (0.5 * f1, 1.5 * f1))
    i = int(np.argmin(np.abs(res.freqs - f1)))
    pf = res.field_along_network(i, variable="p", n_x=101)[0]

    p = pf.values / pf.values[0]  # fix the arbitrary global scale/phase to the tail antinode
    assert np.max(np.abs(p.imag)) < 1e-6  # a pure standing wave -> real shape
    assert np.allclose(p.real, np.cos(np.pi * pf.x / LDUCT), atol=1e-3)
    mid = int(np.argmin(np.abs(pf.x - 0.5 * LDUCT)))
    assert abs(p[mid]) < 1e-3  # pressure node at the centre


def test_open_closed_pressure_is_sine():
    # open inlet (R = -1) + wall: quarter-wave p'(x) = sin(pi x / 2L), node at the open end.
    _, sol = _duct_net(PerturbationBC.open_end(), cat.wall())
    u, c = _uc(sol)
    f_qw = c / (4.0 * LDUCT)
    res = eigenmodes(sol.problem, sol.x, (0.5 * f_qw, 1.5 * f_qw))
    i = int(np.argmin(np.abs(res.freqs - f_qw)))
    pf = res.field_along_network(i, variable="p", n_x=101)[0]

    p = pf.values / pf.values[-1]  # normalize to the wall antinode
    assert np.max(np.abs(p.imag)) < 1e-6
    assert abs(p[0]) < 1e-3  # pressure node at the open inlet
    assert np.allclose(p.real, np.sin(np.pi * pf.x / (2.0 * LDUCT)), atol=2e-3)


def test_serial_multiduct_jump_and_markers():
    # inlet - duct(L1) - sudden area change - duct(L2) - pressure outlet, flowing -> a jump at the SAC.
    L1, L2 = 0.3, 0.4
    A1, A2 = 0.01, 0.05
    net = Network(CFG, p_ref=101325.0, T_ref=300.0, mdot_ref=2.0)
    net.add(cat.total_pressure_inlet(112000.0, 300.0, perturbation_bc=PerturbationBC.hard_wall()))
    net.add(cat.duct(L1))
    net.add(cat.sudden_area_change(name="expansion"))
    net.add(cat.duct(L2))
    net.add(cat.pressure_outlet(101325.0, 300.0, perturbation_bc=PerturbationBC.reflection(0.5)))
    for a, b, ar in [(0, 1, A1), (1, 2, A1), (2, 3, A2), (3, 4, A2)]:
        net.connect(a, b, ar)
    sol = net.solve()
    assert sol.converged
    u, c = _uc(sol)
    f1 = c / (2.0 * (L1 + L2))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = eigenmodes(sol.problem, sol.x, (0.6 * f1, 1.4 * f1))
    assert res.n_modes >= 1
    pf = res.field_along_network(0, variable="p", n_x=80)[0]

    assert pf.x[0] == pytest.approx(0.0)
    assert pf.x[-1] == pytest.approx(L1 + L2)
    # the area change is marked at its developed position
    labels = [lab for (_x, lab) in pf.markers]
    assert "expansion" in labels
    xs_sac = [x for (x, lab) in pf.markers if lab == "expansion"]
    assert xs_sac and xs_sac[0] == pytest.approx(L1, abs=1e-9)
    # a finite jump straddles the area change (two stations share x = L1 with different p')
    at_L1 = np.where(np.isclose(pf.x, L1))[0]
    assert at_L1.size == 2
    assert abs(pf.values[at_L1[1]] - pf.values[at_L1[0]]) > 1e-3 * np.max(np.abs(pf.values))


def test_directly_joined_ducts_stay_continuous():
    # two ducts sharing one edge (no element between) -> the field is continuous across the join.
    L1, L2 = 0.3, 0.2
    net = Network(CFG, p_ref=101325.0, T_ref=300.0, mdot_ref=5.0)
    net.add(cat.total_pressure_inlet(101325.0, 300.0, perturbation_bc=PerturbationBC.hard_wall()))
    net.add(cat.duct(L1))
    net.add(cat.duct(L2))
    net.add(cat.wall())
    net.connect(0, 1, 0.05)
    net.connect(1, 2, 0.05)  # shared edge: duct1 head == duct2 tail
    net.connect(2, 3, 0.05)
    sol = net.solve()
    assert sol.converged
    u, c = _uc(sol)
    f1 = c / (2.0 * (L1 + L2))
    res = eigenmodes(sol.problem, sol.x, (0.6 * f1, 1.4 * f1))
    pf = res.field_along_network(0, variable="p", n_x=120)[0]
    assert pf.x[-1] == pytest.approx(L1 + L2)
    # no jump at the join: the value at x = L1 is single-valued (within sampling)
    at_join = np.where(np.isclose(pf.x, L1, atol=1e-6))[0]
    vals = pf.values[at_join]
    assert np.max(np.abs(vals - vals[0])) < 1e-6 * np.max(np.abs(pf.values))


def test_eigenmode_animation_figure():
    _, sol = _duct_net(PerturbationBC.hard_wall(), cat.wall())
    u, c = _uc(sol)
    f1 = c / (2.0 * LDUCT)
    res = eigenmodes(sol.problem, sol.x, (0.5 * f1, 1.5 * f1))
    fig = res.animate_mode(0, variable="p", n_x=40, n_frames=12)
    assert len(fig.frames) == 12
    assert fig.layout.updatemenus  # play / pause buttons
    assert fig.layout.sliders  # phase slider
    # the animated trace's frame data are real and bounded by the envelope
    assert all(np.all(np.isreal(fr.data[0].y)) for fr in fig.frames)


def test_forced_response_animation_and_field():
    # a driven two-terminal duct: animate the forced spatial field at a frequency.
    _, sol = _duct_net(
        PerturbationBC.reflection(0.6),
        cat.pressure_outlet(101325.0, 300.0, perturbation_bc=PerturbationBC.reflection(0.6)),
        pt_in=120000.0,
    )
    u, c = _uc(sol)
    f1 = c / (2.0 * LDUCT)
    resp = perturbation_response(sol.problem, sol.x, np.array([f1]))
    fields = resp.field_along_network(f1, variable="p", n_x=40)
    assert fields and fields[0].x[-1] == pytest.approx(LDUCT)
    fig = resp.animate_field(f1, variable="u", n_x=40, n_frames=10)
    assert len(fig.frames) == 10
    assert fig.layout.updatemenus and fig.layout.sliders


def test_unknown_variable_rejected():
    _, sol = _duct_net(PerturbationBC.hard_wall(), cat.wall())
    u, c = _uc(sol)
    res = eigenmodes(sol.problem, sol.x, (0.5 * c / (2 * LDUCT), 1.5 * c / (2 * LDUCT)))
    with pytest.raises(ValueError, match="unknown variable"):
        res.field_along_network(0, variable="pressure")


def _anim_lines(fig):
    # the animated mode-shape lines carry a visible width; the envelope fills are width 0.
    return [t for t in fig.data if t.type == "scatter" and (t.line.width or 0) > 0]


def _envelope_bands(fig):
    return [t for t in fig.data if t.type == "scatter" and (t.line.width or 0) == 0]


def _closed_duct_modes(fmax_factor=1.5):
    _, sol = _duct_net(PerturbationBC.hard_wall(), cat.wall())
    c = _uc(sol)[1]
    f1 = c / (2.0 * LDUCT)
    res = eigenmodes(sol.problem, sol.x, (0.5 * f1, fmax_factor * f1))
    return res, f1


def test_resolve_specs_variable_and_basis():
    # a friendly variable list resolves one-to-one to (label, flavor, component)
    specs = resolve_specs(["p", "u"])
    assert [s[0] for s in specs] == [VARIABLE_SPEC["p"][2], VARIABLE_SPEC["u"][2]]
    # a basis expands to its three components, in order
    specs_b = resolve_specs(basis="primitive")
    assert len(specs_b) == 3
    assert all(flavor == "primitive" for (_lab, flavor, _c) in specs_b)
    assert [c for (_l, _f, c) in specs_b] == [0, 1, 2]
    with pytest.raises(ValueError, match="unknown variable"):
        resolve_specs("pressure")
    with pytest.raises(ValueError, match="unknown basis"):
        resolve_specs(basis="bogus")


def test_animate_multiple_variables():
    # overlay p' and u' for one mode: one animated line each, both in the legend.
    res, _ = _closed_duct_modes()
    fig = res.animate_mode(0, variable=["p", "u"], n_x=40, n_frames=8)
    lines = _anim_lines(fig)
    assert len(lines) == 2
    assert all(ln.showlegend for ln in lines)
    assert any("p'" in ln.name for ln in lines) and any("u'" in ln.name for ln in lines)
    assert len(fig.frames) == 8
    assert all(len(fr.data) == 2 for fr in fig.frames)
    assert all(np.all(np.isreal(d.y)) for fr in fig.frames for d in fr.data)


def test_animate_basis_expands_to_three_components():
    res, _ = _closed_duct_modes()
    fig = res.animate_mode(0, basis="primitive", n_x=40, n_frames=6)
    assert len(_anim_lines(fig)) == 3  # one line per flavor component


def test_animate_envelope_toggle():
    # envelope=True frames each line with a +/-|psi| band (two zero-width fills); off drops them.
    res, _ = _closed_duct_modes()
    on = res.animate_mode(0, variable="p", n_x=40, n_frames=6, envelope=True)
    off = res.animate_mode(0, variable="p", n_x=40, n_frames=6, envelope=False)
    assert len(_envelope_bands(on)) == 2 and len(_anim_lines(on)) == 1
    assert len(_envelope_bands(off)) == 0 and len(_anim_lines(off)) == 1


def test_animate_multiple_modes():
    # overlay the first two duct modes: one line each, titled "Modes ...".
    res, _ = _closed_duct_modes(fmax_factor=2.5)
    assert res.n_modes >= 2
    order = np.argsort(res.freqs)
    i0, i1 = int(order[0]), int(order[1])
    fig = res.animate_mode([i0, i1], variable="p", n_x=40, n_frames=8)
    lines = _anim_lines(fig)
    names = [ln.name for ln in lines]
    assert len(lines) == 2
    assert fig.layout.title.text.startswith("Modes")
    assert all("mode" in nm for nm in names)
    assert any(str(i0) in nm for nm in names) and any(str(i1) in nm for nm in names)
    assert all(np.all(np.isreal(d.y)) for fr in fig.frames for d in fr.data)


def test_primitive_phase_ratio_drives_each_series():
    # the phase_ratio advances each overlaid quantity at its own rate (the multi-mode core):
    # series A (ratio 1) sweeps e^{i theta}, series B (ratio 2) sweeps e^{2 i theta}.
    x = np.linspace(0.0, 1.0, 5)
    v = np.array([1.0, 1j, -1.0, 0.5, 0.25 + 0.1j], dtype=np.complex128)
    sa = AnimSeries(path_fields=[PathField("a", x, v, [])], label="a", phase_ratio=1.0)
    sb = AnimSeries(path_fields=[PathField("b", x, v, [])], label="b", phase_ratio=2.0)
    fig = animate_mode_shape([sa, sb], n_frames=4, normalize=False, envelope=False)
    thetas = np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False)
    for k, th in enumerate(thetas):
        assert np.allclose(fig.frames[k].data[0].y, np.real(v * np.exp(1j * th)))
        assert np.allclose(fig.frames[k].data[1].y, np.real(v * np.exp(2j * th)))


def test_forced_field_multi_variable_and_envelope_off():
    _, sol = _duct_net(
        PerturbationBC.reflection(0.6),
        cat.pressure_outlet(101325.0, 300.0, perturbation_bc=PerturbationBC.reflection(0.6)),
        pt_in=120000.0,
    )
    c = _uc(sol)[1]
    f1 = c / (2.0 * LDUCT)
    resp = perturbation_response(sol.problem, sol.x, np.array([f1]))
    fig = resp.animate_field(f1, basis="primitive", n_x=40, n_frames=6, envelope=False)
    assert len(_anim_lines(fig)) == 3
    assert not _envelope_bands(fig)


def test_animation_controls_and_chrome():
    # icon-only play/pause beside the slider, no legend frame, modebar tools stripped.
    res, _ = _closed_duct_modes()
    fig = res.animate_mode(0, variable="p", n_x=40, n_frames=6)
    btns = fig.layout.updatemenus[0].buttons
    assert [b.label for b in btns] == ["▶", "❚❚"]
    assert fig.layout.updatemenus[0].direction == "left"
    assert fig.layout.legend.borderwidth == 0
    assert "toImage" in fig.layout.modebar.remove and "zoom2d" in fig.layout.modebar.remove


def test_geometry_builder_direct():
    # build_geometry exposes the duct segments and edge endpoints used for layout.
    _, sol = _duct_net(PerturbationBC.hard_wall(), cat.wall())
    geo = build_geometry(sol.problem)
    assert geo.n_nodes == 3 and geo.n_edges == 2
    assert len(geo.ducts) == 1
    d = geo.ducts[0]
    assert d.node == 1 and d.e_tail == 0 and d.e_head == 1
    assert d.length == pytest.approx(LDUCT)
