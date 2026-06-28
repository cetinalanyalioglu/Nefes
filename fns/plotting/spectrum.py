"""Notebook plotting for the stability spectrum and mode shapes (theory.md s12.7).

Two views over an :class:`fns.perturbation.EigenmodeResult`:

* :func:`plot_spectrum` -- the eigenvalues in the (frequency, growth-rate) plane,
  split about the ``growth = 0`` stability boundary so unstable modes stand out.
* :func:`plot_mode_shape` -- one mode's wave amplitudes (magnitude over phase)
  along the network's edges.

Frequencies are on the x-axis in Hz, per the project convention.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .theme import FNS_TEMPLATE_NAME, COLORWAY
from .labels import mathify

_STABLE_COLOR = COLORWAY[0]  # blue
_UNSTABLE_COLOR = COLORWAY[4]  # red


def plot_spectrum(freqs, growth_rates, *, residuals=None, freq_unit="Hz", title="Eigenmode spectrum", **layout):
    """Plot the stability spectrum: growth rate versus modal frequency.

    Each mode is a marker at ``(frequency, growth rate)``; the dashed line at
    ``growth = 0`` is the stability boundary, with growing (unstable) modes above it
    drawn in a contrasting colour.

    Parameters
    ----------
    freqs : array_like
        Modal frequencies (Hz), shape ``(n_modes,)``.
    growth_rates : array_like
        Growth rates ``-Im(omega)`` (1/s); positive is unstable.
    residuals : array_like, optional
        Per-mode validation residual, shown in the hover text.
    freq_unit : str, optional
        Frequency-axis unit label (default ``"Hz"``).
    title : str, optional
        Figure title.
    **layout
        Forwarded to ``Figure.update_layout``.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    freqs = np.asarray(freqs, dtype=float)
    growth = np.asarray(growth_rates, dtype=float)
    unstable = growth > 0.0

    def _hover(mask):
        if residuals is None:
            return None
        r = np.asarray(residuals, dtype=float)[mask]
        return [f"residual = {v:.1e}" for v in r]

    fig = go.Figure()
    for mask, name, color, symbol in (
        (~unstable, "stable / decaying", _STABLE_COLOR, "circle"),
        (unstable, "unstable (growing)", _UNSTABLE_COLOR, "diamond"),
    ):
        if not np.any(mask):
            continue
        fig.add_trace(
            go.Scatter(
                x=freqs[mask],
                y=growth[mask],
                mode="markers",
                name=name,
                marker=dict(size=11, color=color, symbol=symbol, line=dict(width=1, color="white")),
                text=_hover(mask),
                hovertemplate="f = %{x:.4g} "
                + freq_unit
                + "<br>growth = %{y:.4g} 1/s"
                + ("<br>%{text}" if residuals is not None else "")
                + "<extra></extra>",
            )
        )
    fig.add_hline(y=0.0, line_dash="dash", line_color="#9aa5b1", line_width=1.4)
    fig.update_layout(
        template=FNS_TEMPLATE_NAME,
        title=title,
        xaxis_title=f"frequency [{freq_unit}]",
        yaxis_title="growth rate −Im(ω) [1/s]",
        showlegend=True,
    )
    fig.update_layout(**layout)
    return fig


def plot_mode_shape(shape, *, labels=None, positions=None, title="Mode shape", **layout):
    """Plot a mode's wave amplitudes along the edges: magnitude (top) over phase (bottom).

    Parameters
    ----------
    shape : ndarray
        Complex array ``(n_edges, n_char)`` -- one mode projected onto every edge
        (e.g. from :meth:`EigenmodeResult.mode_shape`).
    labels : sequence of str, optional
        Per-characteristic symbols (LaTeX fragments); defaults to ``("f", "g", "h")``.
    positions : array_like, optional
        x-axis positions per edge (default: edge index).
    title : str, optional
        Figure title.
    **layout
        Forwarded to ``Figure.update_layout``.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    shape = np.asarray(shape, dtype=np.complex128)
    n_edges, n_char = shape.shape
    x = np.arange(n_edges) if positions is None else np.asarray(positions, dtype=float)
    syms = list(labels) if labels is not None else ["f", "g", "h"][:n_char]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=("magnitude", "phase [rad]")
    )
    for k in range(n_char):
        color = COLORWAY[k % len(COLORWAY)]
        legend = mathify(syms[k]) if k < len(syms) else f"w{k}"
        fig.add_trace(
            go.Scatter(
                x=x,
                y=np.abs(shape[:, k]),
                mode="lines+markers",
                name=legend,
                line=dict(color=color),
                legendgroup=legend,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=np.angle(shape[:, k]),
                mode="lines+markers",
                name=legend,
                line=dict(color=color),
                legendgroup=legend,
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    fig.update_xaxes(title_text="edge", row=2, col=1)
    fig.update_layout(template=FNS_TEMPLATE_NAME, title=title)
    fig.update_layout(**layout)
    return fig
