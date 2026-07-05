"""Higher-order-mode cut-on frequencies: the plane-wave validity ceiling of a duct.

The whole Nefes acoustic layer is a **plane-wave** (one-dimensional) network: it
carries only the axial ``f``/``g``/``h`` waves and assumes the field is uniform
across each duct cross-section.  That assumption holds only **below the first
cut-on frequency**, where every higher-order (transverse) duct mode is evanescent.
Above it a spinning/radial mode propagates and the 1-D model silently loses
fidelity.

For a hard-walled duct of cross-sectional dimension ``d`` carrying mean flow at
Mach ``M``, the first higher-order mode cuts on at

    f_cut = f_cut0 * sqrt(1 - M^2),

where the quiescent ceiling ``f_cut0`` is

    circular (diameter d):  f_cut0 = 1.8412 * c / (pi * d)   (first zero of J1'),
    square    (side d):     f_cut0 = c / (2 * d).

The mean flow always **lowers** the ceiling (``sqrt(1 - M^2) <= 1``), so the
with-flow value is the conservative limit to keep analyses below.  This module
reports the per-duct cut-on across a solved network and the network-wide ceiling
(the widest, hence lowest-cut-on, cross-section).

Public: :func:`cuton_frequency` (single-section cut-on) and
:func:`duct_cuton_frequencies` (per-duct sweep of a solved network, returning a
:class:`CutOnReport` of :class:`DuctCutOn` entries).
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ...assembly.recover import ES_C, ES_M, ES_AREA

# First non-trivial root of J1'(x): the (m=1, n=0) mode of a hard-walled circular
# duct -- the lowest higher-order mode, hence the one that cuts on first.
ALPHA_CIRCULAR = 1.8411837813406593

_SECTIONS = ("circular", "square", "rectangular")


def _check_aspect(aspect):
    """Validate a rectangular aspect ratio (larger-to-smaller side, ``>= 1``)."""
    aspect = float(aspect)
    if not aspect >= 1.0:
        raise ValueError(f"aspect (width-to-height ratio) must be >= 1; got {aspect}")
    return aspect


def _transverse_span(area, section, aspect=1.0):
    """Cross-sectional dimension feeding the cut-on: circle diameter, or the larger rectangle side.

    For a rectangle of area ``area`` and side ratio ``aspect = a/b >= 1``, the larger side is
    ``a = sqrt(area * aspect)`` -- the dimension that sets the first (lowest) transverse mode.  The
    square is the ``aspect = 1`` special case.
    """
    if section == "circular":
        return 2.0 * np.sqrt(area / np.pi)  # diameter
    if section == "square":
        return np.sqrt(area)  # side
    if section == "rectangular":
        return np.sqrt(area * _check_aspect(aspect))  # larger side
    raise ValueError(f"section must be one of {_SECTIONS}; got {section!r}")


def cuton_frequency(area, c, mach=0.0, section="circular", aspect=1.0):
    """First higher-order-mode cut-on frequency [Hz] of a uniform duct.

    Parameters
    ----------
    area : float
        Cross-sectional area [m^2] (``> 0``).
    c : float
        Mean sound speed [m/s] (``> 0``).
    mach : float, optional
        Mean-flow Mach number; its magnitude lowers the ceiling by
        ``sqrt(1 - M^2)`` (default 0, the quiescent ceiling).
    section : {"circular", "square", "rectangular"}, optional
        Assumed cross-section shape (Nefes ducts store only an area).
    aspect : float, optional
        Width-to-height ratio (``>= 1``) for ``section="rectangular"``; ignored otherwise
        (default ``1.0``, a square).

    Returns
    -------
    float
        The first cut-on frequency [Hz]; the plane-wave model is valid below it.
    """
    area = float(area)
    c = float(c)
    if area <= 0.0 or c <= 0.0:
        raise ValueError(f"area and c must be positive; got area={area}, c={c}")
    d = _transverse_span(area, section, aspect)
    if section == "circular":
        f0 = ALPHA_CIRCULAR * c / (np.pi * d)
    else:  # square / rectangular: half-wavelength across the larger side
        f0 = c / (2.0 * d)
    m = abs(float(mach))
    flow = np.sqrt(max(1.0 - m * m, 0.0))  # subsonic v1; M->1 drives the ceiling to 0
    return float(f0 * flow)


@dataclass
class DuctCutOn:
    """The cut-on of one duct (edge) of a solved network."""

    edge: int
    name: str
    area: float
    span: float  # transverse dimension used (diameter for circular, side for square)
    c: float
    mach: float
    f_cuton: float  # first cut-on with the mean flow [Hz] -- the operational ceiling
    f_cuton_quiescent: float  # the M = 0 value, for reference


@dataclass
class CutOnReport:
    """Per-duct cut-on frequencies and the network-wide plane-wave ceiling."""

    section: str
    ducts: List[DuctCutOn] = field(default_factory=list)
    aspect: float = 1.0

    def _section_label(self) -> str:
        """The section descriptor shown in the headers (with the aspect ratio when rectangular)."""
        if self.section == "rectangular":
            return f"{self.section} section, aspect {self.aspect:g}"
        return f"{self.section} section"

    @property
    def f_cuton(self) -> float:
        """Network plane-wave validity ceiling [Hz]: the lowest duct cut-on."""
        if not self.ducts:
            return float("inf")
        return min(d.f_cuton for d in self.ducts)

    @property
    def limiting(self) -> Optional[DuctCutOn]:
        """The duct that sets the ceiling (lowest cut-on), or ``None`` if empty."""
        if not self.ducts:
            return None
        return min(self.ducts, key=lambda d: d.f_cuton)

    def __repr__(self) -> str:
        if not self.ducts:
            return "CutOnReport(no ducts)"
        lim = self.limiting
        head = (
            f"Cut-on report ({self._section_label()})\n"
            f"  plane-wave validity ceiling: f < {self.f_cuton:.1f} Hz "
            f"(edge {lim.edge} {lim.name!r}, span {lim.span:.4g} m, M {lim.mach:.3f})\n"
        )
        cols = (
            f"  {'edge':>4}  {'name':<12} {'area[m^2]':>10} {'span[m]':>9} "
            f"{'c[m/s]':>8} {'M':>6} {'f_cut[Hz]':>11} {'(M=0)':>10}\n"
        )
        rows = "".join(
            f"  {d.edge:>4}  {d.name[:12]:<12} {d.area:>10.4g} {d.span:>9.4g} {d.c:>8.1f} "
            f"{d.mach:>6.3f} {d.f_cuton:>11.1f} {d.f_cuton_quiescent:>10.1f}\n"
            for d in self.ducts
        )
        return head + cols + rows

    def _repr_html_(self) -> str:
        if not self.ducts:
            return "<div><b>CutOnReport</b> &middot; no ducts</div>"
        lim = self.limiting
        header = (
            "<div style='font-family:sans-serif;margin-bottom:4px'>"
            f"<b>Cut-on report</b> &nbsp;&middot;&nbsp; {self._section_label()} &nbsp;|&nbsp; "
            f"plane-wave validity ceiling <b>f &lt; {self.f_cuton:.1f} Hz</b> "
            f"(edge {lim.edge} {lim.name!r}, span {lim.span:.4g} m, M {lim.mach:.3f})</div>"
        )
        th = "padding:2px 8px;border-bottom:1px solid #ccc"
        cols = ("edge", "name", "area [m&sup2;]", "span [m]", "c [m/s]", "M", "f_cut [Hz]", "(M=0) [Hz]")
        head = "<tr>" + "".join(f"<th style='text-align:right;{th}'>{c}</th>" for c in cols) + "</tr>"
        body = []
        for d in self.ducts:
            flag = " style='background:#fff3cd'" if d.edge == lim.edge else ""
            cells = (
                str(d.edge),
                d.name,
                f"{d.area:.4g}",
                f"{d.span:.4g}",
                f"{d.c:.1f}",
                f"{d.mach:.3f}",
                f"{d.f_cuton:.1f}",
                f"{d.f_cuton_quiescent:.1f}",
            )
            tds = "".join(f"<td style='text-align:right;padding:2px 8px'>{c}</td>" for c in cells)
            body.append(f"<tr{flag}>" + tds + "</tr>")
        table = (
            "<table style='border-collapse:collapse;font-family:monospace;font-size:0.9em'>"
            + head
            + "".join(body)
            + "</table>"
        )
        return header + table


def duct_cuton_frequencies(prob, x, *, section="circular", aspect=1.0, names=None):
    """Per-duct cut-on frequencies of a solved network.

    Reads each edge's area, sound speed and Mach number from the converged
    mean-flow state and reports the first higher-order-mode cut-on.  The
    :attr:`CutOnReport.f_cuton` of the returned report is the network-wide
    plane-wave validity ceiling.

    Parameters
    ----------
    prob : CompiledProblem
        The compiled network (provides ``n_edges``).
    x : ndarray
        Converged mean-flow state (as returned by the solver).
    section : {"circular", "square", "rectangular"}, optional
        Assumed duct cross-section shape (default ``"circular"``).
    aspect : float, optional
        Width-to-height ratio (``>= 1``) for ``section="rectangular"``; ignored otherwise
        (default ``1.0``, a square).
    names : sequence of str, optional
        Per-edge names for the report (default ``"e{edge}"``).

    Returns
    -------
    CutOnReport
    """
    if section not in _SECTIONS:
        raise ValueError(f"section must be one of {_SECTIONS}; got {section!r}")
    if section == "rectangular":
        _check_aspect(aspect)
    from ...solver.report import states_table

    est = states_table(prob, x)
    ducts = []
    for e in range(prob.n_edges):
        area = float(est[ES_AREA, e])
        c = float(est[ES_C, e])
        mach = abs(float(est[ES_M, e]))
        name = names[e] if names is not None and e < len(names) else f"e{e}"
        ducts.append(
            DuctCutOn(
                edge=e,
                name=name,
                area=area,
                span=float(_transverse_span(area, section, aspect)),
                c=c,
                mach=mach,
                f_cuton=cuton_frequency(area, c, mach, section, aspect),
                f_cuton_quiescent=cuton_frequency(area, c, 0.0, section, aspect),
            )
        )
    return CutOnReport(section=section, ducts=ducts, aspect=float(aspect))
