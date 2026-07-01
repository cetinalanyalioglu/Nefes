"""A modern, restrained Plotly theme for Nefes figures.

Design goals: a clean light canvas, hairline grid, no chart-junk, and a coherent
qualitative palette tuned for engineering line/scatter plots.  Registering the
template is a pure side effect of importing this module; nothing here depends on
the rest of Nefes, so it is safe to import in any notebook or script.
"""

import plotly.graph_objects as go
import plotly.io as pio

NEFES_TEMPLATE_NAME = "nefes"

# Qualitative palette: a saturated-but-soft set that reads well on white and stays
# distinguishable in greyscale print order.
COLORWAY = [
    "#2563eb",  # blue
    "#f97316",  # orange
    "#10b981",  # emerald
    "#8b5cf6",  # violet
    "#ef4444",  # red
    "#0ea5e9",  # sky
    "#eab308",  # amber
    "#ec4899",  # pink
]

_INK = "#1f2933"  # primary text
_MUTED = "#52606d"  # secondary text / tick labels
_GRID = "#eceff3"  # hairline grid
_AXIS = "#cbd2d9"  # axis lines / ticks
_PAPER = "#ffffff"
_PLOT = "#ffffff"

# Default body font for all Plotly text in the Nefes template.
FONT_FAMILY = "Arial"

# Figure title: centered, bold, slightly below the former 18 pt default.
_TITLE_FONT_SIZE = 16
_TICK_FONT_SIZE = 13
# Default Plotly standoff is 15 px; pull axis titles closer to tick labels.
_AXIS_TITLE_STANDOFF = 6


# Uniform line weight shared by every axis line, mirrored border and tick, so all
# four sides of every subplot read at the same thickness.  A slightly heavier weight
# gives each panel a crisper frame than the hairline grid inside it.
_LINE_W = 1.6


def _axis():
    return dict(
        showgrid=True,
        gridcolor=_GRID,
        gridwidth=1,
        zeroline=False,
        showline=True,
        linecolor=_AXIS,
        linewidth=_LINE_W,
        mirror=True,  # mirror the axis line to the opposite side -> a full box around each subplot
        ticks="outside",
        ticklen=5,
        tickwidth=_LINE_W,
        tickcolor=_AXIS,
        tickfont=dict(color=_MUTED, size=_TICK_FONT_SIZE),
        title=dict(font=dict(color=_INK, size=14), standoff=_AXIS_TITLE_STANDOFF),
        automargin=True,
    )


def nefes_template() -> go.layout.Template:
    """Build (without registering) the Nefes Plotly template."""
    return go.layout.Template(
        layout=dict(
            colorway=COLORWAY,
            font=dict(family=FONT_FAMILY, color=_INK, size=13),
            title=dict(
                font=dict(family=FONT_FAMILY, color=_INK, size=_TITLE_FONT_SIZE, weight="bold"),
                x=0.5,
                xanchor="center",
            ),
            paper_bgcolor=_PAPER,
            plot_bgcolor=_PLOT,
            colorscale=dict(sequential=[[0, "#eef2ff"], [0.5, "#60a5fa"], [1, "#1e3a8a"]]),
            xaxis=_axis(),
            yaxis=_axis(),
            margin=dict(l=70, r=30, t=60, b=60),
            legend=dict(
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor=_GRID,
                borderwidth=1,
                font=dict(color=_MUTED, size=12),
            ),
            hoverlabel=dict(
                bgcolor="#ffffff",
                bordercolor=_GRID,
                font=dict(family=FONT_FAMILY, color=_INK, size=12),
            ),
            hovermode="x unified",
        ),
        data=dict(
            scatter=[go.Scatter(line=dict(width=2.5), marker=dict(size=7, line=dict(width=0)))],
        ),
    )


def use_nefes_theme() -> str:
    """Register the Nefes template and make it the process-wide default.

    Returns the template name so callers can pass it explicitly if they prefer
    not to mutate the global default.
    """
    pio.templates[NEFES_TEMPLATE_NAME] = nefes_template()
    pio.templates.default = NEFES_TEMPLATE_NAME
    return NEFES_TEMPLATE_NAME


# Register on import so `template="nefes"` works without an explicit call; only the
# default is left untouched until use_nefes_theme() is called.
pio.templates[NEFES_TEMPLATE_NAME] = nefes_template()
