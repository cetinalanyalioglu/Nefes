"""Animated spatial mode shapes along the developed length of a duct network.

Given the reconstructed per-path fields (:func:`fns.perturbation.modeshape.reconstruct_field`)
this draws the instantaneous physical perturbation ``Re{psi(x) e^{i theta}}`` against
developed length and sweeps the phase ``theta`` with a slider + play button, so the
standing/travelling wave animates.  A static envelope ``+/- |psi(x)|`` frames the
oscillation, and compact elements (area changes, junctions, terminals) are marked
where the field jumps.

The same primitive serves an eigenmode (one shape, complex ``omega``) and a forced
field (one shape per excited frequency): both reduce to a complex amplitude per
station, animated over one phase cycle.
"""

import numpy as np
import plotly.graph_objects as go

from .theme import FNS_TEMPLATE_NAME, COLORWAY

# Light fill for the +/- |psi| envelope band, keyed off the path's line colour.
_ENVELOPE_ALPHA = 0.14


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


def animate_mode_shape(
    path_fields,
    *,
    var_label="p'",
    title="Mode shape",
    n_frames=48,
    normalize=True,
    envelope=True,
    freq_hz=None,
    **layout,
):
    """Animate the spatial perturbation field over one phase cycle.

    Parameters
    ----------
    path_fields : list of PathField
        Reconstructed fields, one per root->leaf path
        (:func:`fns.perturbation.modeshape.reconstruct_field`).
    var_label : str, optional
        LaTeX fragment for the plotted variable (e.g. ``"p'"``), used on the y-axis.
    title : str, optional
        Figure title.
    n_frames : int, optional
        Phase samples over ``[0, 2 pi)`` (default 48).
    normalize : bool, optional
        Scale so the peak magnitude is ``1`` and real at ``theta = 0`` (default True);
        eigenvectors have arbitrary scale and global phase, so this is usually wanted.
    envelope : bool, optional
        Draw the static ``+/- |psi(x)|`` band behind the animated line (default True).
    freq_hz : float, optional
        Frequency annotated on the play control (for context only).
    **layout
        Forwarded to ``Figure.update_layout``.

    Returns
    -------
    plotly.graph_objects.Figure
        A figure carrying animation frames, a phase slider, and a play/pause control.
    """
    scale = _normalize(path_fields) if normalize else (1.0 + 0.0j)
    fields = []
    ymax = 1e-12
    for pf in path_fields:
        v = pf.values / scale
        fields.append((pf, v))
        if v.size:
            ymax = max(ymax, float(np.max(np.abs(v))))

    thetas = np.linspace(0.0, 2.0 * np.pi, n_frames, endpoint=False)
    fig = go.Figure()
    dynamic_idx = []  # trace indices the frames animate

    for j, (pf, v) in enumerate(fields):
        color = COLORWAY[j % len(COLORWAY)]
        show = len(fields) > 1  # legend only meaningful with multiple paths
        if envelope and v.size:
            mag = np.abs(v)
            fig.add_trace(
                go.Scatter(
                    x=pf.x,
                    y=mag,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                    legendgroup=pf.name,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=pf.x,
                    y=-mag,
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=_hex_to_rgba(color, _ENVELOPE_ALPHA),
                    showlegend=False,
                    hoverinfo="skip",
                    legendgroup=pf.name,
                )
            )
        fig.add_trace(
            go.Scatter(
                x=pf.x,
                y=np.real(v),
                mode="lines",
                name=pf.name,
                legendgroup=pf.name,
                showlegend=show,
                line=dict(color=color, width=2.5),
                hovertemplate="x = %{x:.4g} m<br>" + f"${var_label}$ = " + "%{y:.3g}<extra></extra>",
            )
        )
        dynamic_idx.append(len(fig.data) - 1)

    frames = []
    for theta in thetas:
        data = [go.Scatter(y=np.real(v * np.exp(1j * theta))) for (_pf, v) in fields]
        frames.append(go.Frame(data=data, traces=dynamic_idx, name=f"{np.degrees(theta):.0f}"))
    fig.frames = frames

    pad = 1.08 * ymax
    fig.update_yaxes(range=[-pad, pad])
    for x0, lab in _unique_markers(path_fields):
        fig.add_vline(
            x=x0,
            line_width=1,
            line_dash="dot",
            line_color="#9aa5b1",
            annotation_text=lab,
            annotation_font_size=10,
            annotation_position="top",
        )

    play_label = "▶ play" if freq_hz is None else f"▶ play ({freq_hz:.4g} Hz)"
    fig.update_layout(
        template=FNS_TEMPLATE_NAME,
        title=title,
        xaxis_title="developed length [m]",
        yaxis_title=f"${var_label}$  (Re)",
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.0,
                y=1.12,
                xanchor="left",
                buttons=[
                    dict(
                        label=play_label,
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=60, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=0),
                                mode="immediate",
                            ),
                        ],
                    ),
                    dict(
                        label="❚❚ pause",
                        method="animate",
                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                x=0.12,
                len=0.88,
                xanchor="left",
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


def _unique_markers(path_fields):
    """Union of compact-element markers across paths, de-duplicated by (x, label)."""
    seen = set()
    out = []
    for pf in path_fields:
        for x0, lab in pf.markers:
            key = (round(float(x0), 9), lab)
            if key in seen:
                continue
            seen.add(key)
            out.append((float(x0), lab))
    return out
