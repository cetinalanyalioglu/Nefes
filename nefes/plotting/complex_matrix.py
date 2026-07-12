"""Notebook plotting of complex transfer / scattering matrices vs frequency.

Each entry of an ``N x N`` perturbation matrix is complex, so the proper view is
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

from ._deps import go, make_subplots
from .labels import mathify, tex
from .theme import COLORWAY, NEFES_TEMPLATE_NAME

# default per-index symbols by matrix size (used when no explicit labels given)
_DEFAULT_LABELS = {2: ("f", "g"), 3: ("f", "g", "h")}


def _as_list(x):
    return list(x) if isinstance(x, (list, tuple)) else [x]


def _sub(sym, idx):
    """Attach an edge/station subscript as a LaTeX fragment (``f`` at edge 1 -> ``{f}_{1}``).

    The symbol is grouped with braces so a composite label (``p'/\\rho c``) is
    subscripted as a whole, not just its last token.
    """
    return f"{{{sym}}}_{{{idx}}}"


def _math(s):
    """Render a LaTeX fragment as a Plotly label, honoring the global LaTeX toggle."""
    return mathify(s)


def _entry_title(row_labels, col_labels, i, j):
    """Title of matrix entry ``(i, j)`` read causally: input (column ``j``) -> output (row ``i``).

    Returns a LaTeX (MathJax) string.  For a transfer matrix ``v_b = T v_a`` the
    column is the variable at the input station and the row is the variable at the
    output station, so entry ``(i, j)`` reads ``col_j -> row_i`` (e.g. ``f_1 \\to
    f_2`` between edges 1 and 2).  Labels are expected to be LaTeX fragments.
    """
    if row_labels is None or col_labels is None:
        return _math(f"{i + 1}{j + 1}")
    if not col_labels[j]:  # single-axis overlay (e.g. source attribution): title by row alone
        return _math(row_labels[i])
    if not row_labels[i]:
        return _math(col_labels[j])
    return _math(f"{col_labels[j]} \\to {row_labels[i]}")


def _axis_labels(nrow, ncol, labels, row_labels, col_labels, edges):
    """Resolve per-row and per-column entry labels.

    Explicit ``row_labels``/``col_labels`` win.  Otherwise, with ``edges=(a, b)``
    the column variables are subscripted by the input edge ``a`` and the row
    variables by the output edge ``b`` (``f`` -> ``f_a`` / ``f_b``).  With neither,
    a single ``labels`` set is shared by both axes.  A shared symbol set only makes
    sense for a square matrix; a rectangular one without explicit labels falls back
    to numeric indices.
    """
    if row_labels is not None or col_labels is not None:
        return row_labels, col_labels
    if nrow != ncol:
        return None, None
    base = labels if labels is not None else _DEFAULT_LABELS.get(nrow)
    if edges is not None:
        a, b = edges
        syms = base if base is not None else [str(k + 1) for k in range(nrow)]
        return [_sub(s, b) for s in syms], [_sub(s, a) for s in syms]
    return base, base


def _resolve_mag_range(matrices, i, j, mag_range):
    """Magnitude y-range for entry ``(i, j)``: a fixed override or per-entry auto."""
    if mag_range is not None:
        return [float(mag_range[0]), float(mag_range[1])]
    return _mag_range(matrices, i, j)


def _preset_mag_range(matrices):
    """Shared magnitude range used by the transfer/scattering presets.

    ``(0, 1.05)`` when no entry exceeds unity (the natural band for reflection /
    transmission coefficients, with a little headroom so a coefficient sitting at
    ``1`` is not flush against the frame), otherwise ``(0, 1.05 * max |entry|)``.
    """
    m = max(float(np.max(np.abs(np.asarray(M)))) for M in _as_list(matrices))
    return (0.0, 1.05) if m <= 1.0 else (0.0, 1.05 * m)


def _normalize(matrices, freqs, names):
    matrices = _as_list(matrices)
    matrices = [np.asarray(M) for M in matrices]
    for M in matrices:
        if M.ndim != 3:
            raise ValueError(f"each matrix must be (n_freq, n_row, n_col); got shape {M.shape}")
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
    row_labels=None,
    col_labels=None,
    edges=None,
    entries=None,
    layout="auto",
    x_title=r"$f\;(\mathrm{Hz})$",
    phase="rad",
    unwrap=False,
    mag_range=None,
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
        Per-index variable symbols (e.g. ``("f", "g", "h")``) shared by both axes.
        Defaults to characteristic symbols by size, else 1-based indices.
    row_labels, col_labels : sequence of str, optional
        Per-axis labels, overriding ``labels``/``edges``.  Entry ``(i, j)`` is
        titled ``col_labels[j] -> row_labels[i]`` (input column -> output row).
    edges : tuple of (a, b), optional
        Subscript the column variables by the input edge ``a`` and the row
        variables by the output edge ``b``, so a transfer-matrix entry reads
        ``f_a -> f_b`` instead of the ambiguous ``f -> f``.
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
    mag_range : tuple of (lo, hi), optional
        Fixed magnitude y-range applied to every entry.  Default ``None`` anchors
        each entry at 0 and scales to its own peak.

    Returns
    -------
    plotly.graph_objects.Figure
        The assembled magnitude-over-phase figure.
    """
    matrices, freqs, names = _normalize(matrices, freqs, names)
    nrow, ncol = matrices[0].shape[1], matrices[0].shape[2]
    if any(M.shape[1:] != (nrow, ncol) for M in matrices):
        raise ValueError("all overlaid matrices must have the same shape")
    row_labels, col_labels = _axis_labels(nrow, ncol, labels, row_labels, col_labels, edges)
    if entries is None:
        entries = [(i, j) for i in range(nrow) for j in range(ncol)]
    if layout == "auto":
        layout = "flat" if max(nrow, ncol) <= 2 else "grid"
    if showlegend is None:
        showlegend = len(matrices) > 1

    ph_scale = 180.0 / np.pi if phase == "deg" else 1.0
    ph_title = r"$\angle\;(\mathrm{deg})$" if phase == "deg" else r"$\angle\;(\mathrm{rad})$"

    if layout == "flat":
        fig = _flat_axes(entries, row_labels, col_labels, x_title, ph_title, height, width)
        _draw_flat(fig, matrices, freqs, names, entries, ph_scale, unwrap, showlegend, mag_range)
    elif layout == "grid":
        fig = _grid_axes(nrow, ncol, entries, row_labels, col_labels, x_title, ph_title, height, width)
        _draw_grid(fig, matrices, freqs, names, entries, ph_scale, unwrap, showlegend, mag_range)
    else:
        raise ValueError(f"unknown layout {layout!r}; choose 'auto', 'flat' or 'grid'")

    fig.update_layout(
        template=NEFES_TEMPLATE_NAME,
        title=title,
        showlegend=showlegend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        hovermode="x",
    )
    return fig


# -- flat layout: entries across columns, magnitude row over phase row ------


def _flat_axes(entries, row_labels, col_labels, x_title, ph_title, height, width):
    ncol = len(entries)
    titles = []
    for i, j in entries:
        titles.append(_entry_title(row_labels, col_labels, i, j))
    titles += [""] * ncol  # phase row has no per-cell title
    fig = make_subplots(
        rows=2,
        cols=ncol,
        shared_xaxes=True,
        subplot_titles=titles,
        vertical_spacing=0.10,
        horizontal_spacing=0.06,
    )
    fig.update_yaxes(title_text=tex(r"$|\cdot|$"), row=1, col=1)
    fig.update_yaxes(title_text=tex(ph_title), row=2, col=1)
    for c in range(1, ncol + 1):
        fig.update_xaxes(title_text=tex(x_title), row=2, col=c)
    fig.update_layout(height=height or 440, width=width)
    return fig


def _draw_flat(fig, matrices, freqs, names, entries, ph_scale, unwrap, showlegend, mag_range=None):
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
        fig.update_yaxes(range=_resolve_mag_range(matrices, i, j, mag_range), row=1, col=k + 1)
        if not unwrap:
            fig.update_yaxes(range=_phase_range(ph_scale), row=2, col=k + 1)


# -- grid layout: an N x N matrix of (magnitude-over-phase) cells -----------


def _grid_axes(nrow, ncol, entries, row_labels, col_labels, x_title, ph_title, height, width):
    titles = []
    for i in range(nrow):
        for j in range(ncol):  # magnitude sub-row: titled by entry
            titles.append(_entry_title(row_labels, col_labels, i, j) if (i, j) in entries else "")
        titles += [""] * ncol  # phase sub-row
    fig = make_subplots(
        rows=2 * nrow,
        cols=ncol,
        shared_xaxes=True,
        subplot_titles=titles,
        vertical_spacing=0.05,
        horizontal_spacing=0.07,
    )
    for i in range(nrow):
        fig.update_yaxes(title_text=tex(r"$|\cdot|$"), row=2 * i + 1, col=1)
        fig.update_yaxes(title_text=tex(ph_title), row=2 * i + 2, col=1)
    for c in range(1, ncol + 1):
        fig.update_xaxes(title_text=tex(x_title), row=2 * nrow, col=c)
    fig.update_layout(height=height or 230 * nrow, width=width)
    return fig


def _draw_grid(fig, matrices, freqs, names, entries, ph_scale, unwrap, showlegend, mag_range=None):
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
        fig.update_yaxes(range=_resolve_mag_range(matrices, i, j, mag_range), row=2 * i + 1, col=j + 1)
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


_PRESET = object()  # sentinel: "apply the preset magnitude rule" (vs an explicit range)


def plot_transfer_matrix(matrices, freqs, *, labels=None, edges=None, mag_range=_PRESET, **kwargs):
    """Plot a perturbation transfer matrix versus frequency (magnitude over phase).

    A thin preset over :func:`plot_complex_matrix`: it applies the preset magnitude
    rule and, given the two station edges, subscripts each variable by its edge so an
    entry reads ``f₁ → f₂`` (input edge ``a`` to output edge ``b``) instead of the
    ambiguous ``f → f``.

    This is a *display* helper only -- it does not change the variables the matrix is
    written in.  Build the matrix in the variables you want first
    (``PerturbationResponse.transfer_matrix(a, b, basis=...)``); ``labels`` here only
    names the axes (default: characteristic symbols ``(f, g, h)``).

    Parameters
    ----------
    matrices : array or list of arrays
        One ``(n_freq, N, N)`` complex transfer-matrix stack, or several to overlay.
    freqs : array or list of arrays
        Matching frequency axis (or one shared axis) for the x-axis.
    labels : sequence of str, optional
        Per-variable symbols for the axes (default: ``(f, g, h)`` by size).  Set these
        to match the variables the matrix was built in.
    edges : tuple of (a, b), optional
        The two station edge ids.  Subscripts the input (column) variables by ``a``
        and the output (row) variables by ``b``, giving the ``f_a → f_b`` reading;
        without it the entries fall back to the bare ``f → f`` form.
    mag_range : tuple of (lo, hi), optional
        Fixed magnitude y-range.  By default the preset rule applies: ``(0, 1)``
        when no entry exceeds unity, otherwise ``(0, max |entry|)``.
    **kwargs
        Forwarded to :func:`plot_complex_matrix` (``layout``, ``phase``, ``unwrap``,
        ``x_title``, ``names``, ``title``, ``height``, ``width`` …).

    Returns
    -------
    plotly.graph_objects.Figure

    See Also
    --------
    nefes.perturbation.PerturbationResponse.plot_transfer_matrix :
        Method that converts the matrix to ``basis`` and labels it to match, supplying
        ``edges`` automatically from the station ids.

    Examples
    --------
    >>> T = resp.transfer_matrix(1, 2, basis="primitive")   # convert here
    >>> plot_transfer_matrix(T, resp.freqs, edges=(1, 2),
    ...                      labels=("p'/ρc", "u'", "ρ'c/ρ")).show()
    """
    if mag_range is _PRESET:
        mag_range = _preset_mag_range(matrices)
    kwargs.setdefault("title", "Transfer matrix (magnitude over phase)")
    return plot_complex_matrix(matrices, freqs, labels=labels, edges=edges, mag_range=mag_range, **kwargs)


def plot_scattering_matrix(
    matrices,
    freqs,
    *,
    edges=None,
    partition=None,
    labels=None,
    row_labels=None,
    col_labels=None,
    mag_range=_PRESET,
    **kwargs,
):
    """Plot a perturbation scattering matrix versus frequency (magnitude over phase).

    Preset over :func:`plot_complex_matrix` for an *incoming → outgoing* wave matrix.
    Unlike a transfer matrix, the rows and columns of a scattering matrix belong to
    *different* stations (a reflection sits at one face, a transmission crosses to the
    other), so the labels need the wave partition, not just the station pair: pass
    ``edges=(a, b)`` together with ``partition=(incoming, outgoing)`` and each entry is
    titled by its own station-subscripted waves (e.g. ``f₁ → g₁`` for a reflection at
    edge 1).

    This is a *display* helper only -- it does not change the variables the matrix is
    written in.  ``labels`` names the wave symbols (default: ``(f, g, h)``); set it to
    match whatever variables the matrix was built in.

    Parameters
    ----------
    matrices : array or list of arrays
        One ``(n_freq, N, N)`` complex scattering-matrix stack, or several to overlay.
    freqs : array or list of arrays
        Matching frequency axis (or one shared axis) for the x-axis.
    edges : tuple of (a, b), optional
        The two station edge ids; station ``"a"`` → ``a``, ``"b"`` → ``b``.  Used with
        ``partition`` to build the station-subscripted labels.
    partition : tuple of (incoming, outgoing), optional
        The wave partition from
        :meth:`nefes.perturbation.PerturbationResponse.scattering_labels`: two lists of
        ``(station, char_index)`` tags ordering the matrix columns (incoming) and rows
        (outgoing).
    labels : sequence of str, optional
        Per-wave symbols indexed by characteristic (default: ``(f, g, h)``).  Used to
        build the partition labels, or as the shared axis symbols when no
        ``partition``/``edges`` are given.
    row_labels, col_labels : sequence of str, optional
        Explicit axis labels (outgoing rows / incoming columns), overriding everything
        above.
    mag_range : tuple of (lo, hi), optional
        Fixed magnitude y-range.  By default the preset rule applies: ``(0, 1)`` when
        no entry exceeds unity (the natural band for reflection / transmission
        coefficients), otherwise ``(0, max |entry|)``.
    **kwargs
        Forwarded to :func:`plot_complex_matrix`.

    Returns
    -------
    plotly.graph_objects.Figure

    See Also
    --------
    nefes.perturbation.PerturbationResponse.plot_scattering_matrix :
        Method that fills in ``edges`` and ``partition`` automatically.

    Examples
    --------
    >>> S = resp.scattering_matrix(1, 2)
    >>> plot_scattering_matrix(S, resp.freqs, edges=(1, 2),
    ...                        partition=resp.scattering_labels(1, 2)).show()
    """
    if partition is not None and edges is not None and row_labels is None and col_labels is None:
        row_labels, col_labels = scattering_axis_labels(partition[0], partition[1], edges, labels)
    if mag_range is _PRESET:
        mag_range = _preset_mag_range(matrices)
    kwargs.setdefault("title", "Scattering matrix (magnitude over phase)")
    return plot_complex_matrix(
        matrices,
        freqs,
        labels=labels,
        row_labels=row_labels,
        col_labels=col_labels,
        mag_range=mag_range,
        **kwargs,
    )


def scattering_axis_labels(incoming, outgoing, edges, labels=None):
    """Per-axis scattering labels from a wave partition.

    Parameters
    ----------
    incoming, outgoing : sequence of (station, char)
        The ``("a"/"b", char_index)`` tags from
        ``PerturbationResponse.scattering_labels(a, b)``.  Columns of the scattering
        matrix are the incoming waves, rows the outgoing ones.
    edges : tuple of (a, b)
        The two station edge ids; station ``"a"`` -> ``a``, ``"b"`` -> ``b``.
    labels : sequence of str, optional
        Per-wave symbols indexed by characteristic (default: ``("f", "g", "h")``).

    Returns
    -------
    (row_labels, col_labels) : tuple of list of str
        Station-subscripted labels, ready for ``plot_scattering_matrix`` /
        ``plot_complex_matrix``.
    """
    a, b = edges
    syms = tuple(labels) if labels is not None else ("f", "g", "h")

    def lab(tag):
        station, ci = tag
        sym = syms[ci] if ci < len(syms) else str(ci + 1)
        return _sub(sym, a if station == "a" else b)

    return [lab(t) for t in outgoing], [lab(t) for t in incoming]
