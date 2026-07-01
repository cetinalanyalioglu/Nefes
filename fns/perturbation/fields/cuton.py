"""Higher-order-mode cut-on frequencies: the plane-wave validity ceiling of a duct.

The whole FNS acoustic layer is a **plane-wave** (one-dimensional) network: it
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
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ...assembly.derive import ES_C, ES_M, ES_AREA

# First non-trivial root of J1'(x): the (m=1, n=0) mode of a hard-walled circular
# duct -- the lowest higher-order mode, hence the one that cuts on first.
ALPHA_CIRCULAR = 1.8411837813406593

_SECTIONS = ("circular", "square")


def _transverse_span(area, section):
    """Cross-sectional dimension feeding the cut-on: circle diameter or square side."""
    if section == "circular":
        return 2.0 * np.sqrt(area / np.pi)  # diameter
    if section == "square":
        return np.sqrt(area)  # side
    raise ValueError(f"section must be one of {_SECTIONS}; got {section!r}")


def cuton_frequency(area, c, mach=0.0, section="circular"):
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
    section : {"circular", "square"}, optional
        Assumed cross-section shape (FNS ducts store only an area).

    Returns
    -------
    float
        The first cut-on frequency [Hz]; the plane-wave model is valid below it.
    """
    area = float(area)
    c = float(c)
    if area <= 0.0 or c <= 0.0:
        raise ValueError(f"area and c must be positive; got area={area}, c={c}")
    d = _transverse_span(area, section)
    if section == "circular":
        f0 = ALPHA_CIRCULAR * c / (np.pi * d)
    else:  # square
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
            f"Cut-on report ({self.section} section)\n"
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


def duct_cuton_frequencies(prob, x, *, section="circular", names=None):
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
    section : {"circular", "square"}, optional
        Assumed duct cross-section shape (default ``"circular"``).
    names : sequence of str, optional
        Per-edge names for the report (default ``"e{edge}"``).

    Returns
    -------
    CutOnReport
    """
    if section not in _SECTIONS:
        raise ValueError(f"section must be one of {_SECTIONS}; got {section!r}")
    from ...solver.control import states_table

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
                span=float(_transverse_span(area, section)),
                c=c,
                mach=mach,
                f_cuton=cuton_frequency(area, c, mach, section),
                f_cuton_quiescent=cuton_frequency(area, c, 0.0, section),
            )
        )
    return CutOnReport(section=section, ducts=ducts)
