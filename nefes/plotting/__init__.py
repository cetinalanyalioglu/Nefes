"""Plotly presentation layer for Nefes.

A single home for everything Plotly-related: a custom, modern theme (registered
as the ``"nefes"`` template) plus a small colour palette so every figure across the
examples and notebooks shares one consistent look.  Import side effects register
the template; call :func:`use_nefes_theme` to make it the process-wide default.

    from nefes.plotting import use_nefes_theme
    use_nefes_theme()        # all subsequent figures adopt the Nefes look

It also hosts the complex-matrix viewers used to read transfer / scattering
matrices in a notebook (magnitude over phase, with presets for the 2x2 acoustic
and 3x3 full perturbation networks)::

    from nefes.plotting import plot_transfer_matrix
    plot_transfer_matrix(resp.transfer_matrix(0, 1), resp.freqs).show()

Labels are MathJax (``$...$``) by default.  Where MathJax does not render (a plain
kernel, a static export), call :func:`use_latex(False) <nefes.plotting.use_latex>` to
switch every Nefes figure to a Unicode plain-text fallback.
"""

from .theme import COLORWAY, FONT_FAMILY, NEFES_TEMPLATE_NAME, nefes_template, use_nefes_theme
from .labels import use_latex, latex_enabled, mathify, tex, detex, tex_text
from .complex_matrix import (
    plot_complex_matrix,
    plot_transfer_matrix,
    plot_scattering_matrix,
    scattering_axis_labels,
)
from .transfer_function import plot_transfer_function
from .continuation import plot_fit, plot_pole_map
from .spectrum import plot_spectrum, plot_mode_shape
from .modeshape import animate_mode_shape, AnimSeries
from .topology import plot_network_topology

__all__ = [
    "COLORWAY",
    "FONT_FAMILY",
    "NEFES_TEMPLATE_NAME",
    "nefes_template",
    "use_nefes_theme",
    "use_latex",
    "latex_enabled",
    "mathify",
    "tex",
    "detex",
    "tex_text",
    "plot_complex_matrix",
    "plot_transfer_matrix",
    "plot_scattering_matrix",
    "scattering_axis_labels",
    "plot_transfer_function",
    "plot_fit",
    "plot_pole_map",
    "plot_spectrum",
    "plot_mode_shape",
    "animate_mode_shape",
    "AnimSeries",
    "plot_network_topology",
]
