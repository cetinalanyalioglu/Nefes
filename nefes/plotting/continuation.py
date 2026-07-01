"""Notebook plotting for analytic continuations of tabulated frequency data.

Two views over a :class:`nefes.elements.continuation.RationalFit` (or any tabulated/fit
pair):

* :func:`plot_fit` -- the continued curve over the original samples, magnitude over
  phase, so the fit quality is read at a glance (markers = data, line = fit).
* :func:`plot_pole_map` -- the fit's poles and zeros in the (frequency, growth) plane,
  with the stability search window shaded, so one can check no pole intrudes on the
  region the eigensolver sweeps.

Frequencies are on the x-axis in Hz, per the project convention; the pole map uses the
same ``(frequency, growth = -Im(omega))`` axes as :func:`nefes.plotting.plot_spectrum`.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .theme import NEFES_TEMPLATE_NAME, COLORWAY
from .labels import tex

_DATA_COLOR = COLORWAY[1]  # orange markers for the tabulated samples
_FIT_COLOR = COLORWAY[0]  # blue line for the continued curve
_POLE_COLOR = COLORWAY[4]  # red x for poles
_ZERO_COLOR = COLORWAY[0]  # blue o for zeros


def plot_fit(
    fit,
    *,
    freqs=None,
    data=True,
    phase="rad",
    unwrap=False,
    extend=0.0,
    n=400,
    title="Analytic continuation of tabulated data",
    height=None,
    width=None,
):
    """Overlay a continued curve on the tabulated samples it was fitted to.

    Parameters
    ----------
    fit : RationalFit or callable
        The continuation to draw.  A :class:`~nefes.elements.continuation.RationalFit`
        carries its own ``freqs``/``values`` (drawn as markers); any other callable is
        drawn as a line only (pass ``freqs`` to set the grid).
    freqs : array_like, optional
        Evaluation grid [Hz] for the fit line; default a dense grid spanning the data
        band (optionally widened by ``extend``).
    data : bool, optional
        Overlay the original tabulated samples as markers (default True; ignored when the
        samples are unavailable).
    phase : {"rad", "deg"}, optional
        Phase unit (default ``"rad"``).
    unwrap : bool, optional
        Unwrap the phase along frequency (default False).
    extend : float, optional
        Fractional band widening of the default fit grid beyond the data (e.g. ``0.25``
        draws 25% past each end to show the continuation's behaviour outside the data).
    n : int, optional
        Number of points on the fit line (default 400).
    title : str, optional
        Figure title.
    height, width : int, optional
        Figure size in pixels.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fd = np.asarray(getattr(fit, "freqs", []), dtype=float)
    vd = np.asarray(getattr(fit, "values", []), dtype=np.complex128)
    have_data = data and fd.size > 0

    if freqs is None:
        if fd.size == 0:
            raise ValueError("pass freqs: this fit carries no tabulated grid to span")
        lo, hi = float(fd.min()), float(fd.max())
        pad = extend * (hi - lo)
        freqs = np.linspace(lo - pad, hi + pad, n)
    freqs = np.asarray(freqs, dtype=float)
    curve = np.asarray(fit(freqs), dtype=np.complex128)

    ph_scale = 180.0 / np.pi if phase == "deg" else 1.0
    ph_title = r"$\angle\;(\mathrm{deg})$" if phase == "deg" else r"$\angle\;(\mathrm{rad})$"

    def _ph(z):
        a = np.angle(z)
        return np.unwrap(a) if unwrap else a

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    fig.add_trace(
        go.Scatter(x=freqs, y=np.abs(curve), name="continuation", legendgroup="fit", line_color=_FIT_COLOR),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=freqs,
            y=_ph(curve) * ph_scale,
            name="continuation",
            legendgroup="fit",
            showlegend=False,
            line_color=_FIT_COLOR,
        ),
        row=2,
        col=1,
    )
    if have_data:
        marker = dict(size=6, color=_DATA_COLOR, symbol="circle-open", line=dict(width=1.4, color=_DATA_COLOR))
        fig.add_trace(
            go.Scatter(x=fd, y=np.abs(vd), name="tabulated data", legendgroup="data", mode="markers", marker=marker),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=fd,
                y=_ph(vd) * ph_scale,
                name="tabulated data",
                legendgroup="data",
                showlegend=False,
                mode="markers",
                marker=marker,
            ),
            row=2,
            col=1,
        )
    fig.update_yaxes(title_text=tex(r"$|F|$"), row=1, col=1)
    fig.update_yaxes(title_text=tex(ph_title), row=2, col=1)
    fig.update_xaxes(title_text=tex(r"$f\;(\mathrm{Hz})$"), row=2, col=1)
    fig.update_layout(
        template=NEFES_TEMPLATE_NAME,
        title=title,
        height=height or 480,
        width=width,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        hovermode="x",
    )
    return fig


def plot_pole_map(
    fit,
    *,
    freq_band=None,
    growth_band=None,
    show_zeros=True,
    freq_range=None,
    growth_range=None,
    freq_unit="Hz",
    title="Continuation pole / zero map",
    height=None,
    width=None,
):
    """Plot a continuation's poles and zeros in the (frequency, growth) plane.

    A pole at complex frequency ``f_p`` plots at ``(Re(f_p), -2*pi*Im(f_p))`` -- the same
    ``(frequency, growth)`` axes as the eigenvalue spectrum, so the search window can be
    overlaid.  Poles below the ``growth = 0`` line are in the stable (decaying) half-plane
    (the desirable place for a causal response); a pole inside the shaded search window
    means the continuation is not trustworthy there.

    By default the view focuses on the **data band** (the frequencies the fit was built
    on): a delay-dominated fit scatters far-field poles decades away in frequency/growth,
    and showing them would squash the region of interest to a sliver.  The off-view marker
    count is annotated; pass ``freq_range`` / ``growth_range`` to override.

    Parameters
    ----------
    fit : RationalFit
        The fit whose ``poles`` (and ``zeros``) are drawn.
    freq_band : tuple of float, optional
        ``(f_lo, f_hi)`` of the stability search window [Hz]; shaded if given.
    growth_band : tuple of float, optional
        ``(g_lo, g_hi)`` of the search window [1/s]; combined with ``freq_band`` to shade
        the region the eigensolver sweeps.
    show_zeros : bool, optional
        Also draw the zeros (default True).
    freq_range, growth_range : tuple of float, optional
        Explicit axis ranges [Hz] / [1/s]; default focuses on the data band and the poles
        that fall within it (plus the search window).
    freq_unit : str, optional
        Frequency-axis unit label (default ``"Hz"``).
    title : str, optional
        Figure title.
    height, width : int, optional
        Figure size in pixels.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    poles = np.asarray(fit.poles, dtype=np.complex128)
    pf, pg = poles.real, -2.0 * np.pi * poles.imag
    zeros = np.asarray(fit.zeros, dtype=np.complex128)
    zf, zg = zeros.real, -2.0 * np.pi * zeros.imag

    # default view: the data band (x), and the growth of in-band *poles* + the search window (y).
    # Zeros are drawn but kept out of the y-autoscale -- a delay-dominated fit can fling a zero
    # to extreme growth and squash the region of interest.
    fd = np.asarray(getattr(fit, "freqs", []), dtype=float)
    if freq_range is None and fd.size:
        pad = 0.05 * (float(fd.max()) - float(fd.min()) + 1.0)
        freq_range = (float(fd.min()) - pad, float(fd.max()) + pad)
    if growth_range is None and freq_range is not None:
        in_x = (pf >= freq_range[0]) & (pf <= freq_range[1])
        gs = pg[in_x]
        if growth_band is not None:
            gs = np.concatenate([gs, np.asarray(growth_band, dtype=float)])
        gs = np.concatenate([gs, [0.0]])
        if gs.size:
            lo, hi = float(np.min(gs)), float(np.max(gs))
            m = 0.12 * (hi - lo + 1.0)
            growth_range = (lo - m, hi + m)

    fig = go.Figure()

    # shaded search window (drawn first, behind the markers)
    if freq_band is not None:
        g_lo, g_hi = (
            growth_band if growth_band is not None else (pg.min() if pg.size else -1.0, pg.max() if pg.size else 1.0)
        )
        fig.add_shape(
            type="rect",
            x0=freq_band[0],
            x1=freq_band[1],
            y0=g_lo,
            y1=g_hi,
            fillcolor="rgba(37,99,235,0.08)",
            line=dict(color="#9aa5b1", width=1.2, dash="dot"),
            layer="below",
        )
        fig.add_annotation(
            x=freq_band[1],
            y=g_hi,
            text="search window",
            showarrow=False,
            font=dict(size=11, color="#52606d"),
            xanchor="right",
            yanchor="bottom",
        )

    if show_zeros:
        fig.add_trace(
            go.Scatter(
                x=zf,
                y=zg,
                mode="markers",
                name="zeros",
                marker=dict(size=10, color=_ZERO_COLOR, symbol="circle-open", line=dict(width=1.6, color=_ZERO_COLOR)),
                hovertemplate="zero<br>f = %{x:.4g} " + freq_unit + "<br>growth = %{y:.4g} 1/s<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=pf,
            y=pg,
            mode="markers",
            name="poles",
            marker=dict(size=11, color=_POLE_COLOR, symbol="x-thin", line=dict(width=2.4, color=_POLE_COLOR)),
            hovertemplate="pole<br>f = %{x:.4g} " + freq_unit + "<br>growth = %{y:.4g} 1/s<extra></extra>",
        )
    )
    fig.add_hline(y=0.0, line_dash="dash", line_color="#9aa5b1", line_width=1.4)

    # note any markers pushed off-view by the focused range (a delay-dominated fit has far ones)
    if freq_range is not None and growth_range is not None:

        def _outside(f, g):
            return int(
                np.sum(~((f >= freq_range[0]) & (f <= freq_range[1]) & (g >= growth_range[0]) & (g <= growth_range[1])))
            )

        n_off = _outside(pf, pg) + (_outside(zf, zg) if show_zeros else 0)
        if n_off:
            fig.add_annotation(
                x=1.0,
                y=0.0,
                xref="paper",
                yref="paper",
                text=f"{n_off} far-field pole/zero(s) off view",
                showarrow=False,
                font=dict(size=10, color="#9aa5b1"),
                xanchor="right",
                yanchor="bottom",
            )

    fig.update_layout(
        template=NEFES_TEMPLATE_NAME,
        title=title,
        xaxis_title=f"frequency [{freq_unit}]",
        yaxis_title="growth rate −Im(ω) [1/s]",
        height=height or 460,
        width=width,
        showlegend=True,
    )
    if freq_range is not None:
        fig.update_xaxes(range=list(freq_range))
    if growth_range is not None:
        fig.update_yaxes(range=list(growth_range))
    return fig
