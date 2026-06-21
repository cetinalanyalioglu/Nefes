"""A modern, restrained Plotly theme for FNS figures.

Design goals: a clean light canvas, hairline grid, no chart-junk, and a coherent
qualitative palette tuned for engineering line/scatter plots.  Registering the
template is a pure side effect of importing this module; nothing here depends on
the rest of FNS, so it is safe to import in any notebook or script.
"""

import plotly.graph_objects as go
import plotly.io as pio

FNS_TEMPLATE_NAME = "fns"

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

_FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"


def _axis():
    return dict(
        showgrid=True,
        gridcolor=_GRID,
        gridwidth=1,
        zeroline=False,
        showline=True,
        linecolor=_AXIS,
        linewidth=1,
        ticks="outside",
        ticklen=5,
        tickcolor=_AXIS,
        tickfont=dict(color=_MUTED, size=12),
        title=dict(font=dict(color=_INK, size=14)),
        automargin=True,
    )


def fns_template() -> go.layout.Template:
    """Build (without registering) the FNS Plotly template."""
    return go.layout.Template(
        layout=dict(
            colorway=COLORWAY,
            font=dict(family=_FONT_FAMILY, color=_INK, size=13),
            title=dict(font=dict(family=_FONT_FAMILY, color=_INK, size=18), x=0.01, xanchor="left"),
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
                bordercolor=_AXIS,
                font=dict(family=_FONT_FAMILY, color=_INK, size=12),
            ),
            hovermode="x unified",
        ),
        data=dict(
            scatter=[go.Scatter(line=dict(width=2.5), marker=dict(size=7, line=dict(width=0)))],
        ),
    )


def use_fns_theme() -> str:
    """Register the FNS template and make it the process-wide default.

    Returns the template name so callers can pass it explicitly if they prefer
    not to mutate the global default.
    """
    pio.templates[FNS_TEMPLATE_NAME] = fns_template()
    pio.templates.default = FNS_TEMPLATE_NAME
    return FNS_TEMPLATE_NAME


# Register on import so `template="fns"` works without an explicit call; only the
# default is left untouched until use_fns_theme() is called.
pio.templates[FNS_TEMPLATE_NAME] = fns_template()
