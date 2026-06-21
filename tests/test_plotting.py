"""The FNS Plotly theme + complex-matrix viewers: structure and integrity."""

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import pytest

from fns.plotting import (
    COLORWAY,
    FNS_TEMPLATE_NAME,
    fns_template,
    use_fns_theme,
    plot_complex_matrix,
    plot_transfer_matrix,
    plot_scattering_matrix,
    scattering_axis_labels,
)


def _rand_matrix(n_freq=16, n=3, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_freq, n, n)) + 1j * rng.standard_normal((n_freq, n, n))


def test_template_registered_on_import():
    # importing fns.plotting must register the template without touching default
    assert FNS_TEMPLATE_NAME in pio.templates


def test_fns_template_is_well_formed():
    tmpl = fns_template()
    assert isinstance(tmpl, go.layout.Template)
    assert tuple(tmpl.layout.colorway) == tuple(COLORWAY)
    assert tmpl.layout.paper_bgcolor == "#ffffff"
    # hairline grid, no zero line: the deliberate "modern" choices
    assert tmpl.layout.xaxis.zeroline is False
    assert tmpl.layout.yaxis.showgrid is True


def test_use_fns_theme_sets_default():
    prev = pio.templates.default
    try:
        name = use_fns_theme()
        assert name == FNS_TEMPLATE_NAME
        assert pio.templates.default == FNS_TEMPLATE_NAME
    finally:
        pio.templates.default = prev


def test_palette_entries_are_unique_hex():
    assert len(COLORWAY) == len(set(COLORWAY))
    assert all(c.startswith("#") and len(c) == 7 for c in COLORWAY)


# -- complex-matrix viewers -------------------------------------------------


def test_flat_layout_is_default_for_2x2():
    freqs = np.linspace(50, 1500, 16)
    fig = plot_complex_matrix(_rand_matrix(16, 2), freqs)
    assert isinstance(fig, go.Figure)
    # flat: 2 rows x 4 entries, two traces (mag, phase) per entry
    assert len(fig.data) == 2 * 4
    # magnitudes are non-negative; the top-row traces carry |.|
    mags = [tr for tr in fig.data if np.all(np.asarray(tr.y) >= 0)]
    assert len(mags) >= 4


def test_grid_layout_is_default_for_3x3():
    freqs = np.linspace(50, 1500, 16)
    fig = plot_complex_matrix(_rand_matrix(16, 3), freqs)
    # grid: an entry-shaped 3x3 of (mag-over-phase) cells -> 6 subplot rows x 3 cols
    assert len(fig.data) == 2 * 9  # mag + phase per entry
    assert len(list(fig.select_yaxes())) == 2 * 3 * 3  # 18 panels (6 rows x 3 cols)


def test_entries_subset_limits_traces():
    freqs = np.linspace(50, 1500, 16)
    fig = plot_complex_matrix(_rand_matrix(16, 3), freqs, entries=[(0, 0), (2, 1)], layout="grid")
    assert len(fig.data) == 2 * 2  # only the two requested entries


def test_overlay_two_datasets_adds_legend_groups():
    freqs = np.linspace(50, 1500, 16)
    A, B = _rand_matrix(16, 2, seed=1), _rand_matrix(16, 2, seed=2)
    fig = plot_complex_matrix([A, B], freqs, names=["model", "meas"])
    assert len(fig.data) == 2 * (2 * 4)  # two datasets
    legend_names = {tr.legendgroup for tr in fig.data}
    assert legend_names == {"model", "meas"}


def test_transfer_preset_accepts_explicit_labels():
    # the free plotter only names axes from explicit symbols (it cannot convert the
    # matrix); primitive symbols read column (input) -> row (output).
    freqs = np.linspace(50, 1500, 16)
    prim = ("p'/ρc", "u'", "ρ'c/ρ")
    fig = plot_transfer_matrix(_rand_matrix(16, 3), freqs, labels=prim)
    titles = {a.text for a in fig.layout.annotations}
    assert "u'→u'" in titles  # diagonal
    assert "p'/ρc→u'" in titles  # (1,0): input p'/ρc -> output u'
    assert "ρ'c/ρ→ρ'c/ρ" in titles


def test_phase_in_degrees_scales_axis():
    freqs = np.linspace(50, 1500, 16)
    M = _rand_matrix(16, 2)
    deg = plot_complex_matrix(M, freqs, phase="deg")
    # some phase trace should exceed pi in magnitude once scaled to degrees
    assert any(np.max(np.abs(tr.y)) > np.pi for tr in deg.data)


def test_scattering_preset_runs():
    freqs = np.linspace(50, 1500, 16)
    fig = plot_scattering_matrix(_rand_matrix(16, 3), freqs)
    assert isinstance(fig, go.Figure)


def test_shape_validation():
    freqs = np.linspace(50, 1500, 16)
    with pytest.raises(ValueError):
        plot_complex_matrix(np.zeros((16, 3, 2), dtype=complex), freqs)  # non-square
    with pytest.raises(ValueError):
        plot_complex_matrix(_rand_matrix(16, 3), np.linspace(0, 1, 8))  # freq mismatch


# -- edge-subscripted, direction-aware labels -------------------------------


def _sub(sym, idx):
    return f"{sym}<sub>{idx}</sub>"


def test_entry_titles_read_input_to_output():
    # No edges: titles read column (input) -> row (output), so the full 3x3 carries
    # both orderings of an off-diagonal pair.
    freqs = np.linspace(50, 1500, 16)
    fig = plot_transfer_matrix(_rand_matrix(16, 3), freqs)  # default labels (f, g, h)
    titles = {a.text for a in fig.layout.annotations}
    assert {"f→f", "g→f", "f→g", "h→h"} <= titles


def test_edges_subscript_entry_titles():
    # The TODO case: a transfer matrix between edges 1 and 2 must read f_1 -> f_2,
    # with the input edge on the left and the output edge on the right.
    freqs = np.linspace(50, 1500, 16)
    fig = plot_transfer_matrix(_rand_matrix(16, 3), freqs, edges=(1, 2))  # default labels (f, g, h)
    titles = {a.text for a in fig.layout.annotations}
    assert f"{_sub('f', 1)}→{_sub('f', 2)}" in titles  # diagonal (0,0)
    assert f"{_sub('g', 1)}→{_sub('f', 2)}" in titles  # (0,1): input g at 1 -> output f at 2
    assert f"{_sub('f', 1)}→{_sub('g', 2)}" in titles  # (1,0): input f at 1 -> output g at 2


def test_explicit_row_col_labels_override():
    freqs = np.linspace(50, 1500, 16)
    fig = plot_complex_matrix(_rand_matrix(16, 2), freqs, row_labels=["out0", "out1"], col_labels=["in0", "in1"])
    titles = {a.text for a in fig.layout.annotations}
    assert "in1→out0" in titles  # entry (0,1): col 1 -> row 0


def test_scattering_axis_labels_tag_their_own_station():
    incoming = [("a", 0), ("b", 1)]  # f at station a, g at station b
    outgoing = [("a", 1), ("b", 0)]  # g at station a, f at station b
    row, col = scattering_axis_labels(incoming, outgoing, edges=(0, 2))
    assert col == [_sub("f", 0), _sub("g", 2)]
    assert row == [_sub("g", 0), _sub("f", 2)]


def test_scattering_preset_partition_labels_entries():
    freqs = np.linspace(50, 1500, 16)
    inc, out = [("a", 0), ("b", 1)], [("a", 1), ("b", 0)]
    fig = plot_scattering_matrix(_rand_matrix(16, 2), freqs, edges=(0, 2), partition=(inc, out))
    titles = {a.text for a in fig.layout.annotations}
    assert f"{_sub('f', 0)}→{_sub('g', 0)}" in titles  # (0,0): incoming f_a -> outgoing g_a


# -- magnitude axis scaling -------------------------------------------------


def test_preset_mag_range_rule():
    from fns.plotting.complex_matrix import _preset_mag_range

    assert _preset_mag_range(0.4 * np.ones((4, 2, 2), dtype=complex)) == (0.0, 1.0)
    assert _preset_mag_range(3.0 * np.ones((4, 2, 2), dtype=complex)) == (0.0, 3.0)


def test_explicit_mag_range_sets_every_magnitude_axis():
    freqs = np.linspace(50, 1500, 16)
    fig = plot_complex_matrix(_rand_matrix(16, 2), freqs, mag_range=(0.0, 1.0))
    ranges = [tuple(ax.range) for ax in fig.select_yaxes() if ax.range is not None]
    assert ranges.count((0.0, 1.0)) == 4  # one magnitude panel per 2x2 entry


def test_scattering_preset_uses_unit_band_when_subunit():
    freqs = np.linspace(50, 1500, 16)
    M = 0.5 * np.ones((16, 2, 2), dtype=complex)  # nothing exceeds 1
    fig = plot_scattering_matrix(M, freqs)
    ranges = [tuple(ax.range) for ax in fig.select_yaxes() if ax.range is not None]
    assert (0.0, 1.0) in ranges
