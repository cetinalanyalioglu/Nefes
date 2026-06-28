"""Plotly presentation layer for FNS.

A single home for everything Plotly-related: a custom, modern theme (registered
as the ``"fns"`` template) plus a small colour palette so every figure across the
examples and notebooks shares one consistent look.  Import side effects register
the template; call :func:`use_fns_theme` to make it the process-wide default.

    from fns.plotting import use_fns_theme
    use_fns_theme()        # all subsequent figures adopt the FNS look

It also hosts the complex-matrix viewers used to read transfer / scattering
matrices in a notebook (magnitude over phase, with presets for the 2x2 acoustic
and 3x3 full perturbation networks)::

    from fns.plotting import plot_transfer_matrix
    plot_transfer_matrix(resp.transfer_matrix(0, 1), resp.freqs).show()

Labels are MathJax (``$...$``) by default.  Where MathJax does not render (a plain
kernel, a static export), call :func:`use_latex(False) <fns.plotting.use_latex>` to
switch every FNS figure to a Unicode plain-text fallback.
"""

from .theme import COLORWAY, FONT_FAMILY, FNS_TEMPLATE_NAME, fns_template, use_fns_theme
from .labels import use_latex, latex_enabled, mathify, tex, detex
from .complex_matrix import (
    plot_complex_matrix,
    plot_transfer_matrix,
    plot_scattering_matrix,
    scattering_axis_labels,
)
from .transfer_function import plot_transfer_function
from .spectrum import plot_spectrum, plot_mode_shape
from .modeshape import animate_mode_shape, AnimSeries
from .topology import plot_network_topology

__all__ = [
    "COLORWAY",
    "FONT_FAMILY",
    "FNS_TEMPLATE_NAME",
    "fns_template",
    "use_fns_theme",
    "use_latex",
    "latex_enabled",
    "mathify",
    "tex",
    "detex",
    "plot_complex_matrix",
    "plot_transfer_matrix",
    "plot_scattering_matrix",
    "scattering_axis_labels",
    "plot_transfer_function",
    "plot_spectrum",
    "plot_mode_shape",
    "animate_mode_shape",
    "AnimSeries",
    "plot_network_topology",
]
