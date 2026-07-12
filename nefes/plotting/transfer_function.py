"""Notebook plotting of complex functions of frequency (transfer functions, FTFs).

The scalar companion of :mod:`nefes.plotting.complex_matrix`: a single complex curve
``F(f)`` is read as **magnitude and phase** versus frequency (the default), or as a
**Nyquist** diagram (imaginary vs real, frequency as the path parameter).  Several
curves overlay -- model vs measurement, a parameter sweep, the terms of a flame
response -- by passing a list.

``plot_transfer_function`` accepts the value array directly, or a callable / FTF
object (anything with ``F(freqs) -> complex``, e.g.
:class:`nefes.elements.dynamic_source.TransferFunction`), evaluated on the given grid.
"""

from itertools import cycle

import numpy as np

from ._deps import go, make_subplots
from .labels import tex
from .theme import COLORWAY, NEFES_TEMPLATE_NAME


def _as_list(x):
    # a bare complex ndarray is one curve; a list/tuple of them is several
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _eval(func, freqs):
    """Evaluate one curve on ``freqs``: a callable/FTF is called, an array is returned as-is."""
    if callable(func):
        out = np.asarray(func(freqs), dtype=np.complex128)
        if out.shape == ():  # a constant collapsed to a scalar
            out = out * np.ones_like(freqs, dtype=np.complex128)
        return out
    out = np.asarray(func, dtype=np.complex128)
    if out.shape != np.asarray(freqs).shape:
        raise ValueError(f"value array has shape {out.shape} but freqs has shape {np.asarray(freqs).shape}")
    return out


def plot_transfer_function(
    funcs,
    freqs,
    *,
    names=None,
    phase="rad",
    unwrap=False,
    mag_range=None,
    nyquist=False,
    title=None,
    x_title=r"$f\;(\mathrm{Hz})$",
    height=None,
    width=None,
    showlegend=None,
):
    """Plot one or more complex functions of frequency.

    Parameters
    ----------
    funcs : array, callable, or list thereof
        A complex value array ``F(freqs)``, or a callable / transfer-function object
        evaluated on ``freqs`` (e.g.
        :class:`nefes.elements.dynamic_source.TransferFunction`).  Pass a list to overlay
        several curves.
    freqs : array_like
        Frequencies [Hz] for the x-axis (and the evaluation grid for callables).
    names : list of str, optional
        Legend names, one per overlaid curve.
    phase : {"rad", "deg"}, optional
        Phase unit (default ``"rad"``); magnitude is always linear.
    unwrap : bool, optional
        Unwrap the phase along frequency (default False, wrapped band).
    mag_range : tuple of (lo, hi), optional
        Fixed magnitude y-range; default anchors at 0 and scales to the peak.
    nyquist : bool, optional
        If True, draw a Nyquist diagram (imaginary vs real, equal aspect) instead of
        the magnitude/phase stack (default False).
    title : str, optional
        Figure title.
    x_title : str, optional
        x-axis title for the magnitude/phase layout (default ``f (Hz)``).
    height, width : int, optional
        Figure size in pixels.
    showlegend : bool, optional
        Force the legend on/off (default: on when more than one curve).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    funcs = _as_list(funcs)
    freqs = np.asarray(freqs, dtype=float)
    curves = [_eval(f, freqs) for f in funcs]
    if names is None:
        names = [f"#{k + 1}" for k in range(len(curves))]
    elif len(_as_list(names)) != len(curves):
        raise ValueError("number of names must match number of curves")
    names = _as_list(names)
    if showlegend is None:
        showlegend = len(curves) > 1

    if nyquist:
        fig = _draw_nyquist(curves, freqs, names, showlegend, height, width)
    else:
        fig = _draw_magphase(curves, freqs, names, phase, unwrap, mag_range, x_title, showlegend, height, width)

    fig.update_layout(
        template=NEFES_TEMPLATE_NAME,
        title=title,
        showlegend=showlegend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        hovermode="x" if not nyquist else "closest",
    )
    return fig


def _draw_magphase(curves, freqs, names, phase, unwrap, mag_range, x_title, showlegend, height, width):
    ph_scale = 180.0 / np.pi if phase == "deg" else 1.0
    ph_title = r"$\angle\;(\mathrm{deg})$" if phase == "deg" else r"$\angle\;(\mathrm{rad})$"
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    colors = cycle(COLORWAY)
    peak = 0.0
    for F, nm in zip(curves, names):
        color = next(colors)
        mag = np.abs(F)
        peak = max(peak, float(mag.max()) if mag.size else 0.0)
        ang = np.angle(F)
        if unwrap:
            ang = np.unwrap(ang)
        fig.add_trace(
            go.Scatter(x=freqs, y=mag, name=nm, legendgroup=nm, showlegend=showlegend, line_color=color),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=freqs, y=ang * ph_scale, name=nm, legendgroup=nm, showlegend=False, line_color=color),
            row=2,
            col=1,
        )
    fig.update_yaxes(title_text=tex(r"$|F|$"), row=1, col=1)
    fig.update_yaxes(title_text=tex(ph_title), row=2, col=1)
    fig.update_xaxes(title_text=tex(x_title), row=2, col=1)
    if mag_range is not None:
        fig.update_yaxes(range=[float(mag_range[0]), float(mag_range[1])], row=1, col=1)
    else:
        fig.update_yaxes(range=[0.0, 1.1 * peak if peak > 0.0 else 1.0], row=1, col=1)
    if not unwrap:
        lim = np.pi * ph_scale * 1.03
        fig.update_yaxes(range=[-lim, lim], row=2, col=1)
    fig.update_layout(height=height or 460, width=width)
    return fig


def _draw_nyquist(curves, freqs, names, showlegend, height, width):
    fig = go.Figure()
    colors = cycle(COLORWAY)
    for F, nm in zip(curves, names):
        color = next(colors)
        fig.add_trace(
            go.Scatter(
                x=F.real,
                y=F.imag,
                name=nm,
                legendgroup=nm,
                showlegend=showlegend,
                mode="lines+markers",
                marker=dict(size=4),
                line_color=color,
                customdata=freqs,
                hovertemplate="f=%{customdata:.4g} Hz<br>Re=%{x:.4g}<br>Im=%{y:.4g}<extra></extra>",
            )
        )
    fig.update_xaxes(title_text=tex(r"$\mathrm{Re}\,F$"), zeroline=True)
    fig.update_yaxes(title_text=tex(r"$\mathrm{Im}\,F$"), zeroline=True, scaleanchor="x", scaleratio=1.0)
    fig.update_layout(height=height or 480, width=width or 520)
    return fig
