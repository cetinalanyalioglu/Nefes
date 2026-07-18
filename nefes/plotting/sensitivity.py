"""Notebook plotting for eigenvalue sensitivities.

One view over an :class:`nefes.perturbation.EigenmodeSensitivityResult`:

* :func:`plot_sensitivities` -- a ranked horizontal bar chart of how much each setup
  parameter moves each mode's growth rate, with the destabilizing direction highlighted.
"""

import numpy as np

from ._deps import go
from .theme import COLORWAY, NEFES_TEMPLATE_NAME

_STABILIZING_COLOR = COLORWAY[0]  # blue, matches the stable markers of plot_spectrum
_DESTABILIZING_COLOR = COLORWAY[4]  # red, matches the unstable markers of plot_spectrum

_DEFAULT_TOP = 15


def plot_sensitivities(sens, *, modes=None, top=_DEFAULT_TOP, title="Eigenvalue sensitivities", **layout):
    """Ranked bar chart of growth-rate sensitivities, one bar group per mode.

    Each of the most influential parameters gets a horizontal bar per selected mode, whose
    length is the growth-rate change for a +1% parameter change (for a zero-valued
    parameter: for one probe step, so an unset volume or end correction still shows its
    leverage).  Bars to the right (positive) push the mode toward instability and are
    drawn in the unstable colour of :func:`plot_spectrum`; bars to the left stabilize.
    The frequency shift rides along in the hover text.

    Parameters
    ----------
    sens : EigenmodeSensitivityResult
        The mode-by-parameter derivative table.
    modes : int or sequence of int, optional
        Mode indices to show (default: all, capped at 6 to keep the bars readable).
    top : int, optional
        Number of parameters shown, ranked by their influence over the selected modes
        (default 15).
    title : str, optional
        Figure title.
    **layout
        Forwarded to ``Figure.update_layout``.

    Returns
    -------
    plotly.graph_objects.Figure

    See also
    --------
    nefes.perturbation.EigenmodeSensitivityResult.plot : the bound form.
    plot_spectrum : the spectrum this table differentiates.
    """
    if modes is None:
        mode_ids = list(range(min(sens.n_modes, 6)))
    elif np.isscalar(modes):
        mode_ids = [int(modes)]
    else:
        mode_ids = [int(i) for i in modes]
    if not mode_ids or not sens.n_params:
        raise ValueError("nothing to plot: the result holds no modes or no parameters")

    scale = sens._influence_scale()
    g = sens.dgrowth_dp * scale[None, :]  # growth change per +1% (per probe step if zero-valued)
    f = sens.dfreq_dp * scale[None, :]
    influence = np.abs(g[mode_ids]).max(axis=0)
    order = np.argsort(-influence, kind="stable")[:top][::-1]  # most influential on top

    labels = [sens.addresses[k] for k in order]
    single = len(mode_ids) == 1
    fig = go.Figure()
    for j, i in enumerate(mode_ids):
        vals = g[i, order]
        colors = np.where(vals > 0.0, _DESTABILIZING_COLOR, _STABILIZING_COLOR)
        opacity = 1.0 if single else 0.45 + 0.55 * (j + 1) / len(mode_ids)
        fig.add_trace(
            go.Bar(
                x=vals,
                y=labels,
                orientation="h",
                name=f"mode {i}: {sens.freqs[i]:.4g} Hz",
                marker=dict(color=colors, opacity=opacity, line=dict(width=0.5, color="white")),
                customdata=np.stack([f[i, order], np.full(order.size, sens.freqs[i])], axis=-1),
                hovertemplate="%{y}<br>Δgrowth = %{x:+.4g} 1/s"
                "<br>Δf = %{customdata[0]:+.4g} Hz"
                "<br>mode at %{customdata[1]:.4g} Hz<extra></extra>",
            )
        )
    fig.add_vline(x=0.0, line_color="#9aa5b1", line_width=1.4)
    fig.update_layout(
        template=NEFES_TEMPLATE_NAME,
        title=title,
        barmode="group",
        xaxis_title="growth-rate change per +1% of parameter [1/s]  (per probe step if value = 0)",
        yaxis_title=None,
        showlegend=not single,
        height=max(360, 28 * len(labels) * max(1, len(mode_ids) // 2) + 120),
    )
    fig.update_layout(**layout)
    return fig
