"""Animated spatial mode shapes along the developed length of a duct network.

Given the reconstructed per-path fields (:func:`nefes.perturbation.fields.modeshape.reconstruct_field`)
this draws the instantaneous physical perturbation ``Re{psi(x) e^{i theta}}`` against
developed length and sweeps the phase ``theta`` with a slider + play button, so the
standing/travelling wave animates.  A static envelope ``+/- |psi(x)|`` frames the
oscillation, and compact elements (area changes, junctions, terminals) are marked
where the field jumps.

The same primitive serves an eigenmode (one shape, complex ``omega``) and a forced
field (one shape per excited frequency): both reduce to a complex amplitude per
station, animated over one phase cycle.  Several quantities -- different variables,
different bases, *and* different modes -- can share one figure: each is an
:class:`AnimSeries` carrying its own per-path fields and a ``phase_ratio`` so that
modes of unequal frequency advance at the correct relative rate on a common
real-time axis (the slowest mode completes one cycle; the rest beat against it).
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np
import plotly.graph_objects as go

from .theme import NEFES_TEMPLATE_NAME, COLORWAY
from .labels import mathify

# Light fill for the +/- |psi| envelope band, keyed off the path's line colour.
_ENVELOPE_ALPHA = 0.14

# Cartesian modebar tool buttons removed from the animation figure -- the whole
# zoom/pan/screenshot overlay, leaving only the (config-controlled) plotly logo.
_MODEBAR_TOOLS = [
    "zoom2d",
    "pan2d",
    "select2d",
    "lasso2d",
    "zoomIn2d",
    "zoomOut2d",
    "autoScale2d",
    "resetScale2d",
    "toImage",
    "toggleSpikelines",
    "hoverClosestCartesian",
    "hoverCompareCartesian",
]


@dataclass
class AnimSeries:
    """One animated quantity: its per-path fields, a legend label, and a phase rate.

    Attributes
    ----------
    path_fields : list of PathField
        Reconstructed root->leaf fields for this quantity (one mode, one variable);
        see :func:`nefes.perturbation.fields.modeshape.reconstruct_field`.
    label : str
        Legend/title fragment identifying the quantity (LaTeX, e.g. ``"p'"`` or
        ``"mode 2 \\cdot u'"``); empty for the lone-series case.
    phase_ratio : float
        Phase advance per animation step relative to the reference series.  ``1.0`` for
        a single mode; a mode at frequency ``f_k`` uses ``f_k / f_ref`` so that all modes
        share a real-time axis on which the reference mode completes exactly one cycle.
    """

    path_fields: List = field(default_factory=list)
    label: str = ""
    phase_ratio: float = 1.0


def _hex_to_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _normalize(path_fields):
    """Global scale + reference phase so the peak |psi| reads real ``1`` at ``theta = 0``."""
    peak, ref = 0.0, 1.0 + 0.0j
    for pf in path_fields:
        if pf.values.size == 0:
            continue
        k = int(np.argmax(np.abs(pf.values)))
        if abs(pf.values[k]) > peak:
            peak = abs(pf.values[k])
            ref = pf.values[k]
    if peak <= 0.0:
        return 1.0 + 0.0j
    return peak * (ref / abs(ref))


def _as_series(series):
    """Coerce the input into a list of :class:`AnimSeries`.

    Accepts an already-built list of :class:`AnimSeries`, or a bare list of ``PathField``
    (the single-quantity case) which is wrapped into one unlabelled series.
    """
    if series and isinstance(series[0], AnimSeries):
        return list(series)
    return [AnimSeries(path_fields=list(series), label="")]


def _flatten(series, normalize):
    """Flatten (series x path) into per-trace records, normalized per series.

    Returns ``(traces, ymax)`` where each trace is a dict with ``x`` (developed length),
    ``v`` (complex, normalized), ``name`` (legend label), ``ratio`` (phase rate), and
    ``markers``.  Normalizing per series (not globally) keeps every quantity O(1) on a
    shared axis -- different variables have different units, and different modes have
    arbitrary scale.
    """
    traces = []
    ymax = 1e-12
    for s in series:
        scale = _normalize(s.path_fields) if normalize else (1.0 + 0.0j)
        multi_path = len(s.path_fields) > 1
        for pf in s.path_fields:
            v = pf.values / scale
            if s.label and multi_path:
                name = f"{mathify(s.label)} · {pf.name}"
            elif s.label:
                name = mathify(s.label)
            else:
                name = pf.name
            traces.append(dict(x=pf.x, v=v, name=name, ratio=float(s.phase_ratio), markers=pf.markers))
            if v.size:
                ymax = max(ymax, float(np.max(np.abs(v))))
    return traces, ymax


def animate_mode_shape(
    series,
    *,
    y_title="amplitude",
    title="Mode shape",
    n_frames=60,
    normalize=True,
    envelope=True,
    **layout,
):
    """Animate one or more spatial perturbation fields over one phase cycle.

    Parameters
    ----------
    series : list of AnimSeries or list of PathField
        The quantities to animate.  A list of :class:`AnimSeries` overlays several
        variables/bases/modes (each with its own ``phase_ratio``); a bare list of
        ``PathField`` is the single-quantity case (one mode, one variable).
    y_title : str, optional
        Pre-formatted y-axis title (the caller knows whether it is one variable or a mix).
    title : str, optional
        Figure title.
    n_frames : int, optional
        Phase samples over ``[0, 2 pi)`` of the reference series (default 60).
    normalize : bool, optional
        Scale each series so its peak magnitude is ``1`` and real at ``theta = 0``
        (default True); eigenvectors have arbitrary scale and global phase.
    envelope : bool, optional
        Draw the static ``+/- |psi(x)|`` band behind each animated line (default True);
        set False to remove the background shading.
    **layout
        Forwarded to ``Figure.update_layout``.

    Notes
    -----
    Plotly has no native "loop" for animations -- the play control runs through the
    phase frames once and stops; press it again (or drag the slider) to replay.  The
    figure removes the cartesian modebar tools (zoom/pan/screenshot/...) by default;
    to also drop the plotly logo, render with ``fig.show(config={"displayModeBar": False})``.

    Returns
    -------
    plotly.graph_objects.Figure
        A figure carrying animation frames, a phase slider, and a play/pause control.
    """
    series = _as_series(series)
    traces, ymax = _flatten(series, normalize)

    thetas = np.linspace(0.0, 2.0 * np.pi, n_frames, endpoint=False)
    fig = go.Figure()
    dynamic_idx = []  # trace indices the frames animate
    show_legend = len(traces) > 1

    for j, tr in enumerate(traces):
        color = COLORWAY[j % len(COLORWAY)]
        x, v = tr["x"], tr["v"]
        if envelope and v.size:
            mag = np.abs(v)
            #  GL lines (below) draw over this SVG band; the band is static, so the
            #  per-frame redraw barely touches it.
            fig.add_trace(go.Scatter(x=x, y=mag, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=-mag,
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=_hex_to_rgba(color, _ENVELOPE_ALPHA),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        #  SVG Scatter, not Scattergl: a webgl trace renders blank the moment an
        #  animation frame is applied (regl does not rebuild its scene on animate, so
        #  the line vanishes on the first slider step and never returns).  A few hundred
        #  interior stations redraw smoothly enough as SVG.
        fig.add_trace(
            go.Scatter(
                x=x,
                y=np.real(v),
                mode="lines",
                name=tr["name"],
                showlegend=show_legend,
                line=dict(color=color, width=2.5),
                hovertemplate="x = %{x:.4g} m<br>%{y:.3g}<extra>" + tr["name"] + "</extra>",
            )
        )
        dynamic_idx.append(len(fig.data) - 1)

    frames = []
    for theta in thetas:
        data = [go.Scatter(y=np.real(tr["v"] * np.exp(1j * tr["ratio"] * theta))) for tr in traces]
        frames.append(go.Frame(data=data, traces=dynamic_idx, name=f"{np.degrees(theta):.0f}"))
    fig.frames = frames

    pad = 1.08 * ymax
    fig.update_yaxes(range=[-pad, pad])
    for x0, lab in _unique_markers(traces):
        fig.add_vline(
            x=x0,
            line_width=1,
            line_dash="dot",
            line_color="#9aa5b1",
            annotation_text=lab,
            annotation_font_size=10,
            annotation_position="top",
        )

    fig.update_layout(
        template=NEFES_TEMPLATE_NAME,
        title=title,
        xaxis_title="developed length [m]",
        yaxis_title=y_title,
        legend=dict(borderwidth=0),  # no frame around the legend
        #  Strip the cartesian modebar overlay (zoom / pan / screenshot / select / ...).
        #  Only the plotly logo survives this -- hiding that too needs the render-time
        #  config={"displayModeBar": False}, which a Figure cannot carry on its own.
        modebar=dict(remove=_MODEBAR_TOOLS),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",  # play + pause side by side, tucked at the slider's left end
                showactive=False,
                x=0.1,
                xanchor="right",
                y=0,
                yanchor="top",
                pad=dict(r=10, t=87),
                buttons=[
                    dict(
                        label="▶",  # icon only
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=40, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=0),
                                mode="immediate",
                            ),
                        ],
                    ),
                    dict(
                        label="❚❚",  # icon only
                        method="animate",
                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                x=0.1,
                len=0.9,
                xanchor="left",
                y=0,
                yanchor="top",
                pad=dict(b=10, t=50),
                currentvalue=dict(prefix="phase = ", suffix="°", font=dict(size=12)),
                steps=[
                    dict(
                        method="animate",
                        label=f.name,
                        args=[
                            [f.name],
                            dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0)),
                        ],
                    )
                    for f in frames
                ],
            )
        ],
    )
    fig.update_layout(**layout)
    return fig


def _unique_markers(traces):
    """Union of compact-element markers across traces, de-duplicated by (x, label)."""
    seen = set()
    out = []
    for tr in traces:
        for x0, lab in tr["markers"]:
            key = (round(float(x0), 9), lab)
            if key in seen:
                continue
            seen.add(key)
            out.append((float(x0), lab))
    return out
