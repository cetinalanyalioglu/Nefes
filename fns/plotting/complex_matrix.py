"""Notebook plotting of complex transfer / scattering matrices vs frequency.

Each entry of an ``N x N`` perturbation matrix is complex, so the honest view is
**magnitude and phase** versus frequency.  Two layouts cover the size range:

* ``"flat"`` -- entries laid in a single strip, magnitude on the top row and
  phase on the bottom (the classic 2x2 acoustic view: 2 rows x 4 columns).
* ``"grid"`` -- the entries arranged *as the matrix*: an ``N x N`` grid of cells,
  each cell a magnitude panel stacked over a phase panel.  This stays readable as
  ``N`` grows to 3 (full perturbation network) and beyond (reacting scalars).

``plot_complex_matrix`` is the workhorse; ``plot_transfer_matrix`` and
``plot_scattering_matrix`` are thin presets that label the axes from a
perturbation-variable flavor.  Multiple datasets (e.g. model vs measurement) can
be overlaid by passing lists.
"""

from itertools import cycle

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .theme import FNS_TEMPLATE_NAME, COLORWAY

# default per-index symbols by matrix size (used when no explicit labels given)
_DEFAULT_LABELS = {2: ("f", "g"), 3: ("f", "g", "h")}


def _as_list(x):
    return list(x) if isinstance(x, (list, tuple)) else [x]


def _entry_title(labels, i, j):
    if labels is None:
        return f"{i + 1}{j + 1}"
    return f"{labels[i]}→{labels[j]}"


def _normalize(matrices, freqs, names):
    matrices = _as_list(matrices)
    matrices = [np.asarray(M) for M in matrices]
    for M in matrices:
        if M.ndim != 3 or M.shape[1] != M.shape[2]:
            raise ValueError(f"each matrix must be (n_freq, N, N); got shape {M.shape}")
    freqs = _as_list(freqs)
    if len(freqs) == 1 and len(matrices) > 1:
        freqs = freqs * len(matrices)
    if len(freqs) != len(matrices):
        raise ValueError(f"{len(matrices)} matrices but {len(freqs)} frequency arrays")
    for M, fr in zip(matrices, freqs):
        if M.shape[0] != np.asarray(fr).size:
            raise ValueError(f"matrix first axis {M.shape[0]} != frequency size {np.asarray(fr).size}")
    if names is None:
        names = [f"#{k + 1}" for k in range(len(matrices))]
    elif len(_as_list(names)) != len(matrices):
        raise ValueError("number of names must match number of matrices")
    return matrices, [np.asarray(fr) for fr in freqs], _as_list(names)


def plot_complex_matrix(
    matrices,
    freqs,
    *,
    names=None,
    labels=None,
    entries=None,
    layout="auto",
    x_title="frequency",
    phase="rad",
    unwrap=False,
    title=None,
    height=None,
    width=None,
    showlegend=None,
):
    """Plot the magnitude and phase of every (selected) entry of a complex matrix.

    Parameters
    ----------
    matrices, freqs : array or list of arrays
        One ``(n_freq, N, N)`` complex stack (or several, overlaid) and the
        matching frequency axis (or one shared axis).
    names : list of str, optional
        Legend names, one per overlaid matrix.
    labels : sequence of str, optional
        Per-index variable symbols (e.g. ``("f", "g", "h")``); entry titles read
        ``label_i -> label_j``.  Defaults to characteristic symbols by size, else
        1-based indices.
    entries : list of (i, j), optional
        Restrict to these entries (default: all ``N*N``).
    layout : {"auto", "flat", "grid"}
        ``"flat"`` = one strip (mag row over phase row); ``"grid"`` = matrix-shaped
        cells.  ``"auto"`` picks flat for ``N <= 2``, grid otherwise.
    phase : {"rad", "deg"}
        Phase unit; magnitude is always linear.
    unwrap : bool
        Unwrap the phase along frequency (off by default, so the wrapped phase
        sits in a fixed band).
    """
    matrices, freqs, names = _normalize(matrices, freqs, names)
    N = matrices[0].shape[1]
    if any(M.shape[1] != N for M in matrices):
        raise ValueError("all overlaid matrices must have the same size")
    if labels is None:
        labels = _DEFAULT_LABELS.get(N)
    if entries is None:
        entries = [(i, j) for i in range(N) for j in range(N)]
    if layout == "auto":
        layout = "flat" if N <= 2 else "grid"
    if showlegend is None:
        showlegend = len(matrices) > 1

    ph_scale = 180.0 / np.pi if phase == "deg" else 1.0
    ph_title = "∠ (deg)" if phase == "deg" else "∠ (rad)"

    if layout == "flat":
        fig = _flat_axes(entries, labels, x_title, ph_title, height, width)
        _draw_flat(fig, matrices, freqs, names, entries, ph_scale, unwrap, showlegend)
    elif layout == "grid":
        fig = _grid_axes(N, entries, labels, x_title, ph_title, height, width)
        _draw_grid(fig, matrices, freqs, names, N, entries, ph_scale, unwrap, showlegend)
    else:
        raise ValueError(f"unknown layout {layout!r}; choose 'auto', 'flat' or 'grid'")

    fig.update_layout(
        template=FNS_TEMPLATE_NAME,
        title=title,
        showlegend=showlegend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        hovermode="x",
    )
    return fig


# -- flat layout: entries across columns, magnitude row over phase row ------


def _flat_axes(entries, labels, x_title, ph_title, height, width):
    ncol = len(entries)
    titles = []
    for i, j in entries:
        titles.append(_entry_title(labels, i, j))
    titles += [""] * ncol  # phase row has no per-cell title
    fig = make_subplots(
        rows=2,
        cols=ncol,
        shared_xaxes=True,
        subplot_titles=titles,
        vertical_spacing=0.08,
        horizontal_spacing=0.04,
    )
    fig.update_yaxes(title_text="|·|", row=1, col=1)
    fig.update_yaxes(title_text=ph_title, row=2, col=1)
    for c in range(1, ncol + 1):
        fig.update_xaxes(title_text=x_title, row=2, col=c)
    fig.update_layout(height=height or 420, width=width)
    return fig


def _draw_flat(fig, matrices, freqs, names, entries, ph_scale, unwrap, showlegend):
    colors = cycle(COLORWAY)
    for M, fr, nm in zip(matrices, freqs, names):
        color = next(colors)
        for k, (i, j) in enumerate(entries):
            mag = np.abs(M[:, i, j])
            ang = np.angle(M[:, i, j])
            if unwrap:
                ang = np.unwrap(ang)
            fig.add_trace(
                go.Scatter(x=fr, y=mag, name=nm, legendgroup=nm, showlegend=showlegend and k == 0, line_color=color),
                row=1,
                col=k + 1,
            )
            fig.add_trace(
                go.Scatter(x=fr, y=ang * ph_scale, name=nm, legendgroup=nm, showlegend=False, line_color=color),
                row=2,
                col=k + 1,
            )
    for k, (i, j) in enumerate(entries):
        fig.update_yaxes(range=_mag_range(matrices, i, j), row=1, col=k + 1)
        if not unwrap:
            fig.update_yaxes(range=_phase_range(ph_scale), row=2, col=k + 1)


# -- grid layout: an N x N matrix of (magnitude-over-phase) cells -----------


def _grid_axes(N, entries, labels, x_title, ph_title, height, width):
    titles = []
    for i in range(N):
        for j in range(N):  # magnitude sub-row: titled by entry
            titles.append(_entry_title(labels, i, j) if (i, j) in entries else "")
        titles += [""] * N  # phase sub-row
    fig = make_subplots(
        rows=2 * N,
        cols=N,
        shared_xaxes=True,
        subplot_titles=titles,
        vertical_spacing=0.04,
        horizontal_spacing=0.05,
    )
    for i in range(N):
        fig.update_yaxes(title_text="|·|", row=2 * i + 1, col=1)
        fig.update_yaxes(title_text=ph_title, row=2 * i + 2, col=1)
    for c in range(1, N + 1):
        fig.update_xaxes(title_text=x_title, row=2 * N, col=c)
    fig.update_layout(height=height or 230 * N, width=width)
    return fig


def _draw_grid(fig, matrices, freqs, names, N, entries, ph_scale, unwrap, showlegend):
    colors = cycle(COLORWAY)
    first = entries[0]
    for M, fr, nm in zip(matrices, freqs, names):
        color = next(colors)
        for i, j in entries:
            mag = np.abs(M[:, i, j])
            ang = np.angle(M[:, i, j])
            if unwrap:
                ang = np.unwrap(ang)
            fig.add_trace(
                go.Scatter(
                    x=fr, y=mag, name=nm, legendgroup=nm, showlegend=showlegend and (i, j) == first, line_color=color
                ),
                row=2 * i + 1,
                col=j + 1,
            )
            fig.add_trace(
                go.Scatter(x=fr, y=ang * ph_scale, name=nm, legendgroup=nm, showlegend=False, line_color=color),
                row=2 * i + 2,
                col=j + 1,
            )
    for i, j in entries:
        fig.update_yaxes(range=_mag_range(matrices, i, j), row=2 * i + 1, col=j + 1)
        if not unwrap:
            fig.update_yaxes(range=_phase_range(ph_scale), row=2 * i + 2, col=j + 1)


def _mag_range(matrices, i, j):
    """Anchor the magnitude axis at 0 so a near-constant entry reads as a flat line."""
    m = max(float(np.max(np.abs(M[:, i, j]))) for M in matrices)
    return [0.0, 1.1 * m if m > 0.0 else 1.0]


def _phase_range(ph_scale):
    """Fixed wrapped-phase band, padded slightly so points at +-pi are not clipped."""
    lim = np.pi * ph_scale * 1.03
    return [-lim, lim]


# -- presets ----------------------------------------------------------------


def plot_transfer_matrix(matrices, freqs, *, basis="char", labels=None, **kwargs):
    """Preset for transfer matrices: labels the entries from a variable flavor."""
    if labels is None:
        labels = _basis_labels(matrices, basis)
    return plot_complex_matrix(matrices, freqs, labels=labels, **kwargs)


def plot_scattering_matrix(matrices, freqs, *, labels=None, **kwargs):
    """Preset for scattering matrices (incoming -> outgoing wave amplitudes)."""
    return plot_complex_matrix(matrices, freqs, labels=labels, **kwargs)


def _basis_labels(matrices, basis):
    from ..perturbation.characteristics import BASIS_LABELS

    N = np.asarray(_as_list(matrices)[0]).shape[1]
    syms = BASIS_LABELS.get(basis)
    if syms is not None and len(syms) >= N:
        return tuple(syms[:N])
    return None
