"""Diagnostics and reporting for the mean-flow solver.

This module holds the output layer that sits beside the Newton control loop in
``control.py`` but is not part of it: recovery of the full per-edge state table,
the per-equation residual breakdown, the fixed-width / HTML formatters, and the
``_Reporter`` progress printer the control loop drives.  Keeping them here leaves
``control.py`` focused on the solve itself.
"""

from dataclasses import dataclass

import numpy as np

from ..assembly.assemble import residual
from ..elements.ids import (
    ELEMENT_TYPE_NAMES,
    KIND_MASS,
    KIND_NAMES,
    KIND_PRESSURE,
    row_kind_tags,
)
from ..thermo.api import PERFECT_GAS


@dataclass
class _Reporter:
    """Newton-progress printer (see ``solve``'s ``verbose``/``progress_interval``).

    ``level`` 0 is silent; 1 prints a one-line gross-residual summary per continuation
    stage; 2 additionally prints the scaled residual broken down by equation kind
    (mass, pressure, energy, then each composition scalar) every ``interval``
    iterations within a stage -- a column header once per stage, then the per-group
    2-norms on each iteration line.
    """

    level: int = 0
    interval: int = 1
    prob: object = None
    _grp: tuple = None  # cached (labels, ids, header, widths) for the per-iteration group table
    _IT_W: int = 4  # width of the leading iteration-index column

    def _groups(self):
        if self._grp is None:
            labels, ids = residual_groups(self.prob)
            header = labels + ["total"]  # trailing column: the gross ||R_hat|| (groups in quadrature)
            widths = [max(len(lab), 9) for lab in header]  # 9 fits a "-1.234e-05" magnitude
            self._grp = (labels, ids, header, widths)
        return self._grp

    def _row(self, first, cells, widths):
        parts = [first.rjust(self._IT_W)] + [c.rjust(w) for c, w in zip(cells, widths)]
        return "  " + "  ".join(parts)

    def stage_start(self, kappa, eps):
        if self.level >= 2:
            print(f"[kappa={kappa:<5g} eps={eps:.2e}]")
            _labels, _ids, header, widths = self._groups()
            print(self._row("it", header, widths))

    def iteration(self, it, R):
        if self.level < 2 or (it % self.interval != 0):
            return
        labels, ids, _header, widths = self._groups()
        if R is None:
            print(self._row(str(it), ["(non-physical)"], [len("(non-physical)")]))
            return
        R_hat = R / self.prob.res_scale
        cells = [f"{float(np.linalg.norm(R_hat[ids == g])):.3e}" for g in range(len(labels))]
        cells.append(f"{float(np.linalg.norm(R_hat)):.3e}")  # the gross norm (matches stage_end)
        print(self._row(str(it), cells, widths))

    def stage_end(self, kappa, it, norm, converged):
        if self.level >= 1:
            print(f"kappa={kappa:<5g} -> {it:3d} iters, ||R_hat||={norm:.3e}, converged={converged}")

    def failure(self, prob, x2d, kappa, top=10):
        """Dump the worst-converged equations after a failed solve (verbose >= 1)."""
        if self.level >= 1:
            shown = min(top, prob.n_eq)
            print(f"did not converge; {shown} largest residual(s) (equation-by-equation):")
            print(format_residuals(prob, x2d, kappa=kappa, top=top))


def states_table(prob, x2d, caloric=True):
    """Recover the full edge-state table (NS_EST, E) for diagnostics/output.

    By default the caloric-derivative columns (``ES_DHDRHO``/``ES_DHDP``) are filled too, per
    edge, from that edge's own thermo model (:func:`~nefes.assembly.recover.enrich_caloric`) --
    the partials the perturbation network needs -- so the table is fully populated for every
    consumer.  Pass ``caloric=False`` to skip them (mean-flow reporting, which does not use
    them and need not pay the reacting complex step).
    """
    from ..assembly.recover import NS_EST, enrich_caloric, recover_all

    est = np.zeros((NS_EST, prob.n_edges))
    nj_cache = np.zeros((prob.n_edges, 0))  # diagnostics: no warm start (single pass, robust uniform)
    marker_row = int(getattr(prob, "marker_row", -1))
    xc = np.ascontiguousarray(x2d)
    recover_all(prob.edge_model, prob.tf, prob.ti, xc, prob.area, prob.n_elem, marker_row, est, nj_cache)
    if caloric:
        enrich_caloric(prob.edge_model, prob.tf, prob.ti, xc, est, prob.n_elem, marker_row)
    return est


def _states_columns(prob, x2d, edges=None, precision=5):
    """Shared column extraction for the state-table formatters.

    Returns ``(headers, rows)`` where ``headers`` is the list of column titles
    (``"edge"`` followed by ``"<label> [<unit>]"`` per quantity) and ``rows`` is a
    list of pre-formatted string cells, one list per edge.
    """
    from ..assembly.recover import ES_AREA, ES_C, ES_HT, ES_M, ES_MDOT, ES_P, ES_PT, ES_RHO, ES_T, ES_U

    # (label, est-row index, unit) in edge-state-table column order
    cols = (
        ("mdot", ES_MDOT, "kg/s"),
        ("p", ES_P, "Pa"),
        ("h_t", ES_HT, "J/kg"),
        ("rho", ES_RHO, "kg/m^3"),
        ("u", ES_U, "m/s"),
        ("T", ES_T, "K"),
        ("c", ES_C, "m/s"),
        ("M", ES_M, "-"),
        ("p_t", ES_PT, "Pa"),
        ("area", ES_AREA, "m^2"),
    )
    est = states_table(prob, x2d, caloric=False)  # the mean-flow table needs no caloric partials
    if edges is None:
        edges = range(prob.n_edges)
    edges = [int(e) for e in edges]

    headers = ["edge"] + [f"{label} [{unit}]" for label, _idx, unit in cols]
    rows = [[str(e)] + [f"{est[idx, e]:.{precision}g}" for _label, idx, _unit in cols] for e in edges]
    return headers, rows


def format_states(prob, x2d, edges=None, precision=5):
    """Return a fixed-width table of the recovered per-edge mean-flow states.

    One row per edge (indexed by edge number) with the recovered flow quantities as columns:
    ``mdot``, ``p``, ``h_t``, ``rho``, ``u``, ``T``, ``c``, ``M``, ``p_t``, ``area``.

    Parameters
    ----------
    prob : CompiledProblem
        The compiled problem whose edges are tabulated.
    x2d : ndarray
        A converged (or trial) mean-flow state, shape ``(3 + n_elem, n_edges)``.
    edges : sequence of int, optional
        Edge indices to include, in the given order (default: every edge, ``0 .. n_edges - 1``).
    precision : int, optional
        Number of significant digits printed per value (default 5).

    Returns
    -------
    str
        A newline-joined, column-aligned table ready to print.
    """
    headers, rows = _states_columns(prob, x2d, edges=edges, precision=precision)
    widths = [max([len(headers[c])] + [len(r[c]) for r in rows]) for c in range(len(headers))]

    def _row(cells):
        return "  ".join(s.rjust(widths[c]) for c, s in enumerate(cells))

    lines = [_row(headers), _row(["-" * w for w in widths])] + [_row(r) for r in rows]
    return "\n".join(lines)


def format_states_html(prob, x2d, edges=None, precision=5):
    """Return an HTML ``<table>`` of the recovered per-edge mean-flow states.

    Same columns as :func:`format_states`, rendered as an HTML table for rich
    display in notebook environments.  See :func:`format_states` for the parameters.

    Returns
    -------
    str
        An HTML ``<table>`` element ready to hand to :class:`IPython.display.HTML`.
    """
    from html import escape

    headers, rows = _states_columns(prob, x2d, edges=edges, precision=precision)
    th = "; ".join(["text-align:right", "padding:2px 10px", "border-bottom:1px solid currentColor"])
    td = "; ".join(["text-align:right", "padding:2px 10px", "font-family:monospace"])
    head = "".join(f"<th style='{th}'>{escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td style='{td}'>{escape(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table style='border-collapse:collapse'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _in_notebook():
    """Return ``True`` when running inside a Jupyter/IPython kernel that renders HTML.

    Detects the ZMQ-based interactive shell used by Jupyter notebooks, JupyterLab and
    qtconsole; a plain IPython terminal or a bare interpreter returns ``False``.
    """
    try:
        from IPython import get_ipython

        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def print_states(prob, x2d, edges=None, precision=5, file=None):
    """Print the per-edge mean-flow state table to the screen.

    Thin wrapper over :func:`format_states`; see it for the column layout and parameters.
    In a notebook (and when ``file`` is not given) the table is rendered as rich HTML via
    :func:`format_states_html`; otherwise the fixed-width text table is forwarded to
    :func:`print` (``file`` defaults to ``sys.stdout``).
    """
    if file is None and _in_notebook():
        from IPython.display import HTML, display

        display(HTML(format_states_html(prob, x2d, edges=edges, precision=precision)))
        return
    print(format_states(prob, x2d, edges=edges, precision=precision), file=file)


def scalar_field_labels(prob):
    """Display labels for the transported composition scalars (band-1 rows ``3 .. n_solve-1``).

    The transported scalars depend on the thermo model:

    * a **reacting** (equilibrium) network transports one mixture fraction per distinct feed
      stream, plus the burnt marker when the closure is marker-gated.  The mixture fractions are
      labeled generically ``z1, z2, ...`` (in band order) and the marker ``marker`` -- the
      feed-stream labels are element names and read poorly in a residual table;
    * a **perfect gas with passive scalars** transports user-named scalars, kept verbatim (a
      ``scalar#c`` fallback covers an unnamed one).

    The marker is located by :attr:`CompiledProblem.marker_row` (``< 0`` when the network carries
    none), so a marker anywhere in the band is labeled correctly.

    Parameters
    ----------
    prob : CompiledProblem
        The compiled problem whose transported scalars are labeled.

    Returns
    -------
    list of str
        One label per composition scalar (length ``n_solve - 3``), in band order.
    """
    n_scalars = prob.n_solve - 3
    marker_c = int(getattr(prob, "marker_row", -1)) - 3  # composition-scalar ordinal of the marker
    reacting = prob.model_id != PERFECT_GAS
    scalars = prob.scalar_names or ()
    labels = []
    z = 0
    for c in range(n_scalars):
        if c == marker_c:
            labels.append("marker")
        elif reacting:  # reacting mixture fractions: generic z1, z2, ... (feed labels read poorly)
            z += 1
            labels.append(f"z{z}")
        else:  # perfect-gas passive scalars keep their user-given names
            labels.append(scalars[c] if c < len(scalars) and scalars[c] else f"scalar#{c}")
    return labels


def residual_labels(prob):
    """Human-readable label for every residual equation, in row order.

    The residual vector is laid out as the element (node) equations first -- each
    element contributes its band-1 algebraic rows (a mass balance plus pressure
    couplings for an interior element, or a single boundary row) -- followed by the
    per-edge advected-scalar transport equations (total enthalpy ``h_t`` for every
    edge, then each composition scalar for every edge).  This returns one label per
    row so a residual vector can be read equation-by-equation.

    Parameters
    ----------
    prob : CompiledProblem
        The compiled problem whose equation layout is described.

    Returns
    -------
    list of str
        ``prob.n_eq`` labels, in residual-row order.
    """
    names = prob.node_names or ()
    scalar_labels = scalar_field_labels(prob)
    nrp = prob.node_row_ptr
    labels = []
    for n in range(prob.n_nodes):
        rid = int(prob.node_rid[n])
        deg = int(nrp[n + 1] - nrp[n])
        type_name = ELEMENT_TYPE_NAMES.get(rid, f"residual#{rid}")
        label = names[n] if n < len(names) and names[n] else f"#{n}"
        for tag in row_kind_tags(rid, deg):
            labels.append(f"node {n} [{label}] {type_name}: {KIND_NAMES[tag]}")
    E = prob.n_edges
    for s in range(prob.n_solve - 2):
        # s=0 is the total-enthalpy (energy) transport row; s>=1 are the composition scalars
        field = "h_t" if s == 0 else scalar_labels[s - 1]
        for e in range(E):
            labels.append(f"edge {e} transport: {field}")
    return labels


def residual_groups(prob):
    """Group the residual rows by equation kind, for compact reporting.

    The per-equation residual is coarsened into a handful of physically meaningful
    groups: ``mass`` (every mass-balance / mass-flux row), ``pressure`` (every
    pressure / absolute-pressure row), ``energy`` (the per-edge total-enthalpy
    ``h_t`` transport rows), then one group per composition scalar (named by the
    feed-stream / mixture-fraction labels).  Each group's scaled-residual 2-norm
    combines in quadrature to the global convergence norm.

    Parameters
    ----------
    prob : CompiledProblem
        The compiled problem whose equation layout is grouped.

    Returns
    -------
    labels : list of str
        One label per group, in group-index order.
    ids : ndarray of int
        Length ``n_eq``; the group index of each residual row.
    """
    group_of_kind = {KIND_MASS: 0, KIND_PRESSURE: 1}
    nrp = prob.node_row_ptr
    ids = np.empty(prob.n_eq, dtype=np.int64)
    for n in range(prob.n_nodes):
        rid = int(prob.node_rid[n])
        r0 = int(nrp[n])
        for j, tag in enumerate(row_kind_tags(rid, int(nrp[n + 1] - nrp[n]))):
            ids[r0 + j] = group_of_kind[tag]
    # advected-scalar transport rows: s=0 is the energy (h_t) group, s>=1 the scalars
    E = prob.n_edges
    base = prob.transport_row0
    for s in range(prob.n_solve - 2):
        ids[base + s * E : base + (s + 1) * E] = 2 + s
    # one column per composition scalar: mixture fractions z1, z2, ... and the burnt marker
    labels = ["mass", "pressure", "energy"] + scalar_field_labels(prob)
    return labels, ids


def residual_breakdown(prob, x2d, kappa=0.0, eps=None):
    """Per-equation residual: ``(labels, R, R_hat)``.

    ``R`` is the raw residual in physical units; ``R_hat = R / res_scale`` is the
    nondimensional residual whose 2-norm the solver tests for convergence.  Together
    with :func:`residual_labels` this resolves the single global residual norm into
    its contribution from every equation.

    Parameters
    ----------
    prob : CompiledProblem
        The compiled problem to evaluate.
    x2d : ndarray
        A converged (or trial) mean-flow state, shape ``(n_solve, n_edges)``.
    kappa : float, optional
        Artificial-resistance continuation parameter (default ``0.0``, the exact equations).
    eps : float, optional
        Complementarity smoothing width.  Defaults to the continuation-stage width for
        ``kappa`` (``max(0.3*kappa, 1e-4) * mdot_ref``), matching what the solver used.

    Returns
    -------
    labels : list of str
        Per-row equation labels (see :func:`residual_labels`).
    R : ndarray
        Raw residual, length ``n_eq``.
    R_hat : ndarray
        Scaled residual, length ``n_eq``.
    """
    from .control import EPS_FB, _stage_eps

    if eps is None:
        eps = _stage_eps(prob.var_scale[0], kappa)
    R = residual(prob, x2d, eps, EPS_FB, kappa)
    R_hat = R / prob.res_scale
    return residual_labels(prob), R, R_hat


def format_residuals(prob, x2d, kappa=0.0, eps=None, sort=True, top=None, precision=4):
    """Return a fixed-width table of the residual, equation-by-equation.

    One row per equation: its index, label, raw residual (physical units), and scaled
    residual (the nondimensional value the convergence test sums).  A trailing summary
    line reports the scaled residual 2-norm so the global figure is still available.

    Parameters
    ----------
    prob : CompiledProblem
        The compiled problem to evaluate.
    x2d : ndarray
        A converged (or trial) mean-flow state, shape ``(n_solve, n_edges)``.
    kappa, eps : float, optional
        Continuation parameter and smoothing width; see :func:`residual_breakdown`.
    sort : bool, optional
        If ``True`` (default), order rows by descending ``|scaled residual|`` so the
        worst-converged equations come first; otherwise keep natural row order.
    top : int, optional
        Show only the first ``top`` rows after ordering (default: all rows).
    precision : int, optional
        Significant digits printed per residual value (default 4).

    Returns
    -------
    str
        A newline-joined, column-aligned table ready to print.
    """
    try:
        labels, R, R_hat = residual_breakdown(prob, x2d, kappa=kappa, eps=eps)
    except Exception:
        # The residual could not even be assembled: an edge recovered a non-finite state (a
        # failed h->T inversion, a diverged equilibrium solve), and the inner linear algebra
        # raised.  Rather than surface that opaque error, locate the offending edges and report
        # them -- this is the failed-solve diagnostic, so it must not itself crash.
        return _nonphysical_report(prob, x2d)
    order = np.argsort(-np.abs(R_hat)) if sort else np.arange(len(R_hat))
    if top is not None:
        order = order[: int(top)]

    headers = ["row", "equation", "residual", "scaled"]
    rows = [[str(int(i)), labels[i], f"{R[i]:.{precision}e}", f"{R_hat[i]:.{precision}e}"] for i in order]
    widths = [max([len(headers[c])] + [len(r[c]) for r in rows]) for c in range(len(headers))]
    # left-justify the text columns (row index, equation label), right-justify the numbers
    just = (str.ljust, str.ljust, str.rjust, str.rjust)

    def _row(cells):
        return "  ".join(just[c](s, widths[c]) for c, s in enumerate(cells))

    lines = [_row(headers), _row(["-" * w for w in widths])] + [_row(r) for r in rows]
    lines.append(f"||R_hat|| = {float(np.linalg.norm(R_hat)):.{precision}e}  ({len(R_hat)} equations)")
    return "\n".join(lines)


def _nonphysical_report(prob, x2d, top=10):
    """A readable diagnostic when a state is non-physical and the residual cannot be assembled.

    Recovers each edge in isolation and lists those whose recovered state raises or comes back
    non-finite, with the band-1 unknowns that produced it.  Used by :func:`format_residuals` as
    a graceful fallback so a failed solve reports *which* edge broke and with what state, instead
    of propagating the inner ``LinAlgError`` from the recovery.

    Parameters
    ----------
    prob : CompiledProblem
        The compiled problem to probe.
    x2d : ndarray
        The offending state, shape ``(n_solve, n_edges)``.
    top : int, optional
        Maximum number of offending edges to list (default 10).

    Returns
    -------
    str
        A newline-joined report of the non-physical edges.
    """
    from ..assembly.recover import NS_EST, recover_edge

    x = np.ascontiguousarray(x2d, dtype=np.float64)
    n_elem = int(prob.n_elem)
    mrow = int(getattr(prob, "marker_row", -1))
    has_marker = mrow >= 0
    names = prob.edge_names if getattr(prob, "edge_names", None) else ()
    bad = []
    for e in range(prob.n_edges):
        out = np.zeros(NS_EST)
        marker = float(x[mrow, e]) if has_marker else 0.0
        try:
            recover_edge(
                int(prob.edge_model[e]),
                prob.tf,
                prob.ti,
                x[0, e],
                x[1, e],
                x[2, e],
                prob.area[e],
                x[3 : 3 + n_elem, e],
                marker,
                out,
                np.zeros(0),
            )
            reason = "" if np.all(np.isfinite(out)) else "non-finite recovered state"
        except Exception as ex:  # the inner equilibrium / h->T solve raised
            reason = f"{type(ex).__name__}: {ex}"
        if reason:
            label = f"{names[e]} (edge {e})" if e < len(names) and names[e] else f"edge {e}"
            z = np.array2string(x[3 : 3 + n_elem, e], precision=4, separator=", ")
            bad.append(
                f"  {label}: {reason}\n"
                f"      mdot={x[0, e]:.4g}, p={x[1, e]:.6g}, h_t={x[2, e]:.6g}, "
                f"area={prob.area[e]:.4g}, Z={z}" + (f", marker={marker:.4g}" if has_marker else "")
            )
    if not bad:
        return (
            "residual could not be assembled, but every edge recovered a finite state "
            "(the failure is elsewhere; re-run with a debugger)."
        )
    head = f"state is non-physical: {len(bad)} edge(s) failed recovery (showing up to {top}):"
    return "\n".join([head] + bad[:top])


def print_residuals(prob, x2d, kappa=0.0, eps=None, sort=True, top=None, precision=4, file=None):
    """Print the residual broken down equation-by-equation.

    Thin wrapper over :func:`format_residuals`; see it for the column layout and
    parameters.  ``file`` is forwarded to :func:`print` (default ``sys.stdout``).
    """
    print(format_residuals(prob, x2d, kappa=kappa, eps=eps, sort=sort, top=top, precision=precision), file=file)
