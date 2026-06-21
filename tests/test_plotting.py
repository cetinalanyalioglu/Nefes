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


def test_transfer_preset_labels_entries_from_basis():
    freqs = np.linspace(50, 1500, 16)
    fig = plot_transfer_matrix(_rand_matrix(16, 3), freqs, basis="char")
    titles = {a.text for a in fig.layout.annotations}
    assert "f→f" in titles and "f→g" in titles and "h→h" in titles


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
