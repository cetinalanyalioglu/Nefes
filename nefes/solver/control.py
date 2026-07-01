"""Newton control loop with nondimensionalization, globalization, and homotopy.

The solve runs a short vanishing-friction homotopy in the coefficient ``kappa``
(kappa): stages ``kappa in (0.1, 0.01, 0.0)`` each warm-started from the previous,
with the smoothing width ``eps = max(0.3*kappa, 1e-4) * mdot_ref``.  ``kappa`` scales
an artificial friction (a pressure drop proportional to mass flow) stamped into the
interior pressure rows; driving it to zero recovers the exact equations.  Each stage
is a damped Newton: a
sparse-LU step with backtracking line search on the scaled residual norm, and a
Levenberg-Marquardt fallback when the LU step stalls or the Jacobian is singular.
The final stage solves the exact equations (kappa = 0).
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np

from ..assembly.assemble import residual, jacobian
from ..elements.ids import (
    MASS_FLOW_INLET,
    PT_INLET,
    P_OUTLET,
    MASS_SOURCE,
    KIND_MASS,
    KIND_PRESSURE,
    KIND_NAMES,
    RESIDUAL_NAMES,
    row_kind_tags,
)
from ..thermo.api import EQ_KERNEL, PERFECT_GAS
from ..assembly.scaling import compose_scales, measure_inflow_scales
from .linear import newton_step, lm_step, scaled_system, col_scale

EPS_FB = 1e-5


def _stage_eps(mdot_ref, kappa):
    """Complementarity smoothing width for a homotopy stage (vanishes with ``kappa``)."""
    return max(0.3 * kappa, 1e-4) * mdot_ref


def domain_max_dp(prob):
    """Largest a-priori pressure drop set by the boundary conditions.

    The span between the highest and lowest absolute-pressure boundary reference
    (``total_pressure_inlet`` / ``pressure_outlet``).  Returns ``0.0`` when fewer than
    two such references exist -- a mass-driven network whose real pressure drop is not
    known until the flow is solved.  Used to scale the homotopy friction (and, later,
    the adaptive residual scales) to the real driving pressure differential.

    Parameters
    ----------
    prob : CompiledProblem
        The compiled network.

    Returns
    -------
    float
        ``max(p_ref) - min(p_ref)`` over the absolute-pressure boundaries, or ``0.0``.
    """
    refs = []
    for n in range(prob.n_nodes):
        if int(prob.node_rid[n]) in (PT_INLET, P_OUTLET):
            refs.append(float(prob.npar_f[prob.npar_fptr[n]]))
    if len(refs) < 2:
        return 0.0
    return max(refs) - min(refs)


def flatten(x2d):
    return np.ascontiguousarray(x2d).T.ravel()


def unflatten(flat, n_edges, n_solve=3):
    return np.ascontiguousarray(flat.reshape(n_edges, n_solve).T)


def initial_guess(prob, mdot0=None, p0=None, h0=None, z0=None):
    """Initial state; a small co-directional ``mdot`` by default.

    Each of ``mdot0``, ``p0``, ``h0`` may be a scalar (all edges) or an array of
    length ``n_edges`` (per-edge).  ``z0`` (composition rows ``3..n_solve``) may be
    ``None`` (zeros), a scalar, a ``(n_elem,)`` vector (same composition on every
    edge), or a ``(n_elem, n_edges)`` array (per-edge composition).  A reacting
    network usually needs **per-edge** ``h0``/``z0``: an unburnt air edge and a
    burnt edge sit at very different ``(h_t, Z)``, so a single uniform guess can
    leave a frozen ``h -> T`` inversion or an equilibrium solve far from any root.
    """
    mdot_ref, p_ref, h_ref = prob.var_scale[0], prob.var_scale[1], prob.var_scale[2]
    x = np.zeros((prob.n_solve, prob.n_edges))
    x[0, :] = 0.05 * mdot_ref if mdot0 is None else mdot0
    x[1, :] = p_ref if p0 is None else p0
    x[2, :] = h_ref if h0 is None else h0
    if prob.n_solve > 3:
        if z0 is None:
            x[3:, :] = 0.0
        else:
            z = np.asarray(z0, dtype=np.float64)
            if z.ndim == 2:
                x[3:, :] = z  # (n_elem, n_edges) per-edge composition
            else:
                x[3:, :] = z.reshape(-1, 1)  # (n_elem,) same composition on every edge
    return x


def auto_initial_guess(prob, mdot0=None, p0=None):
    """Physically-seeded per-edge state by propagating the feeds through the graph.

    A reacting network spans a wide enthalpy range -- an unburnt air edge sits at
    ``h_t ~ +1.9e3`` J/kg while CH4-laden / burnt edges sit at ``~ -2.2e5`` J/kg
    (the formation-inclusive datum, D-1) -- so a single uniform guess strands the
    frozen ``h -> T`` inversion or the burnt equilibrium far from any root.  This
    builds each edge's seed by **mass-weighted mixing along the network**, with no
    case-specific tuning:

    1. propagate the edge mass flow (inlets/sources inject; junctions sum; splitters
       divide), giving each edge a conserved ``mdot`` estimate;
    2. blend the advected scalars -- total enthalpy ``h_t`` and the feed-stream
       mixture fractions ``xi`` -- mass-weighted by that ``mdot``.

    Because every feed enters with its own ``h_t`` (formation + sensible at its
    injection ``T``) and ``xi = e_stream``, the blend lands each edge at exactly the
    adiabatic-mixing ``(h_t, xi)``: the burnt edge inherits the fuel+air mixture
    enthalpy automatically, and the equilibrium solve turns it into the flame
    temperature.  Conserved scalars mix linearly, so this is the right basin
    regardless of how negative the formation enthalpies are.
    """
    N = prob.n_nodes
    E = prob.n_edges
    n_solve = prob.n_solve
    nscal = n_solve - 2  # advected scalars: h_t (0) + mixture fractions (1..K)
    mdot_ref, p_ref, h_ref = prob.var_scale[0], prob.var_scale[1], prob.var_scale[2]
    md = float(mdot_ref if mdot0 is None else mdot0)

    tail = np.asarray(prob.tail_node)
    head = np.asarray(prob.head_node)
    rid = np.asarray(prob.node_rid)
    npar_f = np.asarray(prob.npar_f)
    ptr = np.asarray(prob.npar_fptr)

    # per-node injected (mdot, [h_t, xi...]) for the stream-introducing elements
    inj_mdot = np.zeros(N)
    inj_scal = np.zeros((N, nscal))
    has_inj = np.zeros(N)
    for n in range(N):
        pb = ptr[n]
        r = rid[n]
        if r == MASS_FLOW_INLET:
            inj_mdot[n] = npar_f[pb + 0]
            inj_scal[n] = npar_f[pb + 1 : pb + 1 + nscal]
            has_inj[n] = 1.0
        elif r == PT_INLET:
            inj_mdot[n] = md  # mdot is unknown at a pt-inlet; weight by the reference
            inj_scal[n] = npar_f[pb + 1 : pb + 1 + nscal]
            has_inj[n] = 1.0
        elif r == MASS_SOURCE:
            inj_mdot[n] = npar_f[pb + 0]
            inj_scal[n] = npar_f[pb + 2 : pb + 2 + nscal]
            has_inj[n] = 1.0

    out_count = np.zeros(N)
    np.add.at(out_count, tail, 1.0)
    max_sweeps = min(N + 5, 1000)

    # (1) propagate the edge mass flow
    edge_mdot = np.full(E, md)
    has_out = out_count[tail] > 0
    for _ in range(max_sweeps):
        node_in = inj_mdot * has_inj
        np.add.at(node_in, head, edge_mdot)
        new = edge_mdot.copy()
        new[has_out] = (node_in[tail] / np.maximum(out_count[tail], 1.0))[has_out]
        if np.allclose(new, edge_mdot, rtol=1e-12, atol=1e-12 * md):
            edge_mdot = new
            break
        edge_mdot = new

    # (2) blend the advected scalars, mass-weighted by the propagated mdot
    default = np.zeros(nscal)
    default[0] = h_ref
    inj_nodes = np.nonzero(has_inj)[0]
    if inj_nodes.size:
        default = inj_scal[inj_nodes[0]].copy()
    edge_scal = np.tile(default, (E, 1))
    inj_w = inj_mdot * has_inj
    for _ in range(max_sweeps):
        node_num = (inj_w[:, None] * inj_scal).copy()
        node_den = inj_w.copy()
        np.add.at(node_den, head, edge_mdot)
        np.add.at(node_num, head, edge_mdot[:, None] * edge_scal)
        good = node_den > 0.0
        node_mix = np.zeros((N, nscal))
        node_mix[good] = node_num[good] / node_den[good][:, None]
        new = edge_scal.copy()
        valid = has_out & good[tail]
        new[valid] = node_mix[tail][valid]
        if np.allclose(new, edge_scal, rtol=1e-12, atol=1e-12):
            edge_scal = new
            break
        edge_scal = new

    x = np.zeros((n_solve, E))
    x[0, :] = edge_mdot
    x[1, :] = float(p_ref if p0 is None else p0)
    x[2, :] = edge_scal[:, 0]
    if nscal > 1:
        x[3:, :] = edge_scal[:, 1:].T
    return x


@dataclass
class SolveResult:
    x: np.ndarray  # converged state, shape (3, E)
    converged: bool
    iterations: int
    residual_norm: float
    history: List[float] = field(default_factory=list)

    def __repr__(self) -> str:
        """One-line solver outcome: convergence, iteration count, residual, and state shape."""
        status = "converged" if self.converged else "NOT converged"
        shape = "x".join(str(s) for s in np.shape(self.x))
        return (
            f"SolveResult: {status} in {self.iterations} iteration(s), "
            f"residual_norm = {self.residual_norm:.3e}, state ({shape})"
        )


@dataclass
class _Reporter:
    """Newton-progress printer (see ``solve``'s ``verbose``/``progress_interval``).

    ``level`` 0 is silent; 1 prints a one-line gross-residual summary per homotopy
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


def _merit(prob, x2d, eps, kappa, res_scale):
    """Scaled residual 2-norm; +inf if the state is non-physical.

    Pressure must stay positive for every model.  The transported scalar in row 2
    is the absolute total enthalpy ``h_t``: for a perfect gas ``h_t = cp*Tt > 0``,
    but with the reacting backend's formation-inclusive datum (D-1) ``h_t`` is
    legitimately **negative** for fuels with a negative formation enthalpy (e.g.
    CH4), so the positivity guard applies only to the perfect gas.  Physicality of
    a reacting state is then enforced by the closure (a failed ``h -> T`` inversion
    raises and is caught below).
    """
    if np.any(x2d[1, :] <= 0.0):
        return np.inf, None
    if prob.model_id == PERFECT_GAS and np.any(x2d[2, :] <= 0.0):
        return np.inf, None
    try:
        R = residual(prob, x2d, eps, EPS_FB, kappa)
    except Exception:
        return np.inf, None
    if not np.all(np.isfinite(R)):
        return np.inf, None
    R_hat = R / res_scale
    return float(np.linalg.norm(R_hat)), R


def _solve_stage(prob, x2d, eps, kappa, tol, max_iter, history, res_scale, var_scale, reporter=None):
    vcol = col_scale(var_scale, prob.n_edges)
    lam = 1e-3
    norm, R = _merit(prob, x2d, eps, kappa, res_scale)
    for it in range(max_iter):
        history.append(norm)
        if reporter is not None:
            reporter.iteration(it, R)
        if norm < tol:
            return x2d, True, it, norm
        try:
            J = jacobian(prob, x2d, eps, EPS_FB, kappa)
        except np.linalg.LinAlgError:
            # A reacting (EQ_MARKER / EQ_KERNEL) complex-step column can drive the inner
            # equilibrium Newton to a non-finite state on a wild iterate (e.g. a flame drawn
            # both-in, whose auto-start is poor); the inner linear solve then raises.  Degrade
            # gracefully -- end this stage non-converged -- instead of crashing the whole solve.
            return x2d, False, it, norm
        J_hat, R_hat = scaled_system(J, R, vcol, res_scale)

        dy = newton_step(J_hat, R_hat)
        accepted = False
        if dy is not None:
            dx = unflatten(vcol * dy, prob.n_edges, prob.n_solve)
            alpha = 1.0
            for _ in range(30):
                x_try = x2d + alpha * dx
                n_try, R_try = _merit(prob, x_try, eps, kappa, res_scale)
                if n_try < (1.0 - 1e-4 * alpha) * norm:
                    x2d, norm, R = x_try, n_try, R_try
                    accepted = True
                    lam = max(lam * 0.5, 1e-12)
                    break
                alpha *= 0.5

        if not accepted:
            # Levenberg-Marquardt fallback with adaptive damping.
            for _ in range(40):
                dy = lm_step(J_hat, R_hat, lam)
                x_try = x2d + unflatten(vcol * dy, prob.n_edges, prob.n_solve)
                n_try, R_try = _merit(prob, x_try, eps, kappa, res_scale)
                if n_try < norm:
                    x2d, norm, R = x_try, n_try, R_try
                    accepted = True
                    lam = max(lam * 0.5, 1e-12)
                    break
                lam = min(lam * 4.0, 1e8)
            if not accepted:
                return x2d, False, it, norm
    return x2d, norm < tol, max_iter, norm


def solve(
    prob,
    x0=None,
    tol=1e-10,
    max_iter=80,
    kappa_stages=(0.1, 0.01, 0.0),
    kappa_scale="dp",
    adaptive_scale=True,
    verbose=0,
    progress_interval=1,
):
    """Solve the steady mean flow.  Returns a SolveResult (state shape (3, E)).

    Parameters
    ----------
    prob : Problem
        Compiled flow network.
    x0 : ndarray, optional
        Initial state, shape ``(3, E)`` (default: a uniform co-directional guess).
    tol : float, optional
        Convergence tolerance on the scaled residual 2-norm.
    max_iter : int, optional
        Maximum Newton iterations per homotopy stage.
    kappa_stages : sequence of float, optional
        Vanishing-friction homotopy schedule (dimensionless), warm-started in order.
    kappa_scale : {"dp", "absolute"}, optional
        How the schedule's artificial-friction coefficient is sized.  ``"dp"`` (default)
        multiplies each ``kappa`` by the friction resistance ``min(domain_max_dp(prob) /
        mdot_ref, 1)``, so the injected artificial pressure drop at the reference flow is
        capped at a fraction (``kappa``) of the real driving drop.  The cap means the
        scaling only *softens* the friction for low-``dP`` / high-``mdot`` networks (where
        the historical absolute drop ``kappa * mdot`` would over-perturb), and is a no-op
        for healthy-``dP`` networks (``r_art = 1``).  When no a-priori pressure drop is
        available (a mass-driven network, ``domain_max_dp == 0``) it falls back to the
        absolute coefficient.  ``"absolute"`` always uses that historical unit coefficient
        (artificial drop ``kappa * mdot``).  Either way ``eps`` tracks the dimensionless
        ``kappa``.
    adaptive_scale : bool, optional
        When ``True`` (default), the residual / variable scales are re-measured from the
        realized inflow at each homotopy stage (total inlet ``mdot`` for the mass rows, the
        mass-weighted mean inlet ``|h_t|`` for the energy rows) instead of the fixed compiled
        references -- so the nondimensionalization tracks the actual flow and the user need not
        supply ``mdot_ref`` / ``h_ref``.  The quiescent ``mdot = 0`` case falls back to the seed
        scales.  ``False`` uses the compiled ``prob.res_scale`` / ``prob.var_scale``.
    verbose : int or bool, optional
        Progress verbosity.  ``0``/``False`` is silent; ``1``/``True`` prints a
        one-line gross-residual summary per homotopy stage; ``2`` additionally prints
        the scaled residual broken down by equation kind (mass, pressure, energy, then
        each composition scalar) every ``progress_interval`` iterations within a stage.
    progress_interval : int, optional
        Iteration stride for the per-iteration prints at ``verbose >= 2``.

    Returns
    -------
    SolveResult
        The converged state and solve diagnostics.
    """
    mdot_ref = prob.var_scale[0]
    # artificial-friction resistance: scale the dimensionless kappa schedule so the
    # injected artificial dP is a fixed fraction of the real driving dP (kappa_scale
    # == "dp"); fall back to the historical unit coefficient when there is no a-priori
    # pressure drop or the caller asks for it explicitly.
    if kappa_scale == "dp":
        dp = domain_max_dp(prob)
        # cap at the absolute coefficient: dP-scaling only *softens* the friction for
        # low-dP / high-mdot networks (where dp/mdot_ref < 1, the artificial drop would
        # otherwise over-perturb), and leaves healthy-dP networks unchanged (r_art = 1),
        # so it never strengthens the friction enough to stiffen a near-choke stage.
        r_art = min(dp / mdot_ref, 1.0) if dp > 0.0 else 1.0
    elif kappa_scale == "absolute":
        r_art = 1.0
    else:
        raise ValueError(f"kappa_scale must be 'dp' or 'absolute'; got {kappa_scale!r}")
    if x0 is not None:
        x2d = np.array(x0, dtype=np.float64)
    elif prob.model_id == EQ_KERNEL:
        # reacting networks need a physically-seeded per-edge guess (wide h_t range)
        x2d = auto_initial_guess(prob)
    else:
        x2d = initial_guess(prob)
    # seed the burnt marker from the topology flood-fill (demoted to an initial guess; the
    # signed-flow transport self-corrects it).  Skipped when the caller supplies x0.
    if x0 is None and getattr(prob, "marker_row", -1) >= 0 and prob.marker_seed is not None:
        x2d[prob.marker_row, :] = prob.marker_seed
    reporter = _Reporter(level=int(verbose), interval=max(1, int(progress_interval)), prob=prob)
    history: List[float] = []
    total_it = 0
    converged = False
    norm = np.inf
    # seed scales from the compiled references; the adaptive path re-measures them from the
    # realized inflow at each homotopy stage (kept constant within a stage).
    seed_mass, p_scale, seed_h = float(prob.var_scale[0]), float(prob.var_scale[1]), float(prob.var_scale[2])
    degrees = np.diff(prob.row_ptr)
    for kappa in kappa_stages:
        eps = _stage_eps(mdot_ref, kappa)
        if adaptive_scale:
            mass, h = measure_inflow_scales(prob, x2d, seed_mass, seed_h)
            n_scalars = prob.n_solve - 3  # composition mixture fractions + the optional burnt marker
            res_scale, var_scale = compose_scales(prob.node_rid, degrees, prob.n_edges, n_scalars, mass, p_scale, h)
        else:
            res_scale, var_scale = prob.res_scale, prob.var_scale
        reporter.stage_start(kappa, eps)
        # eps tracks the dimensionless kappa; the kernel friction uses the scaled coefficient.
        x2d, converged, it, norm = _solve_stage(
            prob, x2d, eps, kappa * r_art, tol, max_iter, history, res_scale, var_scale, reporter
        )
        total_it += it
        reporter.stage_end(kappa, it, norm, converged)
        if not converged and kappa == 0.0:
            reporter.failure(prob, x2d, kappa)
            break
    return SolveResult(x=x2d, converged=converged, iterations=total_it, residual_norm=norm, history=history)


def states_table(prob, x2d):
    """Recover the full edge-state table (NS_EST, E) for diagnostics/output."""
    from ..assembly.derive import recover_all, NS_EST

    est = np.zeros((NS_EST, prob.n_edges))
    nj_cache = np.zeros((prob.n_edges, 0))  # diagnostics: no warm start (single pass, robust uniform)
    marker_row = int(getattr(prob, "marker_row", -1))
    recover_all(
        prob.edge_model, prob.tf, prob.ti, np.ascontiguousarray(x2d), prob.area, prob.n_elem, marker_row, est, nj_cache
    )
    return est


def _states_columns(prob, x2d, edges=None, precision=5):
    """Shared column extraction for the state-table formatters.

    Returns ``(headers, rows)`` where ``headers`` is the list of column titles
    (``"edge"`` followed by ``"<label> [<unit>]"`` per quantity) and ``rows`` is a
    list of pre-formatted string cells, one list per edge.
    """
    from ..assembly.derive import ES_MDOT, ES_P, ES_HT, ES_RHO, ES_U, ES_T, ES_C, ES_M, ES_PT, ES_AREA

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
    est = states_table(prob, x2d)
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
        from IPython.display import display, HTML

        display(HTML(format_states_html(prob, x2d, edges=edges, precision=precision)))
        return
    print(format_states(prob, x2d, edges=edges, precision=precision), file=file)


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
    scalars = prob.scalar_names or ()
    nrp = prob.node_row_ptr
    labels = []
    for n in range(prob.n_nodes):
        rid = int(prob.node_rid[n])
        deg = int(nrp[n + 1] - nrp[n])
        type_name = RESIDUAL_NAMES.get(rid, f"residual#{rid}")
        label = names[n] if n < len(names) and names[n] else f"#{n}"
        for tag in row_kind_tags(rid, deg):
            labels.append(f"node {n} [{label}] {type_name}: {KIND_NAMES[tag]}")
    E = prob.n_edges
    for s in range(prob.n_solve - 2):
        if s == 0:
            field = "h_t"  # the s=0 transport row carries total enthalpy
        elif (s - 1) < len(scalars) and scalars[s - 1]:
            field = scalars[s - 1]
        else:
            field = f"scalar#{s - 1}"
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
    scalars = prob.scalar_names or ()
    labels = ["mass", "pressure", "energy"]
    for s in range(prob.n_solve - 3):  # one column per composition scalar
        labels.append(scalars[s] if s < len(scalars) and scalars[s] else f"scalar#{s}")
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
        Vanishing-friction homotopy parameter (default ``0.0``, the exact equations).
    eps : float, optional
        Complementarity smoothing width.  Defaults to the homotopy-stage width for
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
        Homotopy parameter and smoothing width; see :func:`residual_breakdown`.
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
    labels, R, R_hat = residual_breakdown(prob, x2d, kappa=kappa, eps=eps)
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


def print_residuals(prob, x2d, kappa=0.0, eps=None, sort=True, top=None, precision=4, file=None):
    """Print the residual broken down equation-by-equation.

    Thin wrapper over :func:`format_residuals`; see it for the column layout and
    parameters.  ``file`` is forwarded to :func:`print` (default ``sys.stdout``).
    """
    print(format_residuals(prob, x2d, kappa=kappa, eps=eps, sort=sort, top=top, precision=precision), file=file)
