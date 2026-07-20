"""Newton control loop with nondimensionalization, globalization, and continuation.

The solve runs a short artificial-resistance continuation in the coefficient ``kappa``:
stages ``kappa in (0.1, 0.01, 0.0)``, each warm-started from the previous, with the
smoothing width ``eps = max(0.3*kappa, 1e-4) * mdot_ref``.  ``kappa`` scales an artificial
resistance (a fictitious pressure drop proportional to mass flow) stamped into the interior
pressure rows; it injects first-order flow sensitivity that regularizes the otherwise
singular Jacobian of the ``|mdot|*mdot`` pressure-loss law at rest, and driving it to zero
recovers the exact equations.  Each stage is a damped Newton: a sparse-LU step with a
backtracking line search on the scaled residual norm, and a Levenberg-Marquardt fallback
when the LU step stalls or the Jacobian is singular.  The final stage solves the exact
equations (``kappa = 0``).

The continuation and globalization follow standard treatments (Allgower & Georg,
*Numerical Continuation Methods*).  The artificial-resistance regularization addresses
the same zero-flow singularity that hydraulic-network solvers face: the ``|Q|^(n-1) Q``
loss law has vanishing flow derivative at rest, so the Jacobian degenerates there and
the global gradient algorithm of Todini & Pilati fails by division by zero.  Elhay &
Simpson (*J. Hydraul. Eng.* 137(10):1216-1224, 2011) prove the singularity -- it bites
when the zero-flow passages form a loop or a path between fixed-head nodes -- and
introduce the linear-surrogate remedy.  The difference here is that ``kappa`` is a
homotopy parameter driven to zero, so the final stage solves the unmodified equations
and the regularization leaves no trace in the converged state; a permanent local
surrogate instead perturbs the solution it converges to (Gorev et al., *Water Resour.
Manage.* 36(5):1679-1691, 2022).
"""

import time
import warnings
from dataclasses import dataclass, field
from typing import List

import numpy as np

from ..assembly.assemble import jacobian, residual
from ..assembly.recover import ES_M
from ..assembly.scaling import compose_scales, measure_inflow_scales
from ..config import config
from ..elements.ids import (
    CHOKED_NOZZLE_OUTLET,
    FLAME_HEAT_RELEASE,
    MASS_FLOW_INLET,
    MASS_SOURCE,
    P_OUTLET,
    PT_INLET,
)
from ..thermo.api import EQ_KERNEL, PERFECT_GAS, thermo_state
from .linear import col_scale, lm_step, newton_step, scaled_system, unflatten
from .report import _Reporter, states_table

# Mach above which an edge is treated as genuinely supersonic (outside the subsonic scope)
# rather than a smoothed sonic throat: the choking complementarity's Fischer-Burmeister floor
# lets a choked edge sit a hair above M = 1, so the threshold clears that band while catching
# the spurious supersonic branch (which runs well above it).
SUPERSONIC_TOL = 1.01

# Mach above which an unremovable supersonic edge is a clear scope violation rather than a
# marginal near-choke overshoot, so the solve is not reported as converged.  Between SUPERSONIC_TOL
# and this bound the edge is a hair past a sonic throat (e.g. an over-driven orifice) and is kept
# with a warning; above it the state is on the spurious/ill-posed supersonic branch a resistance-
# free loop or over-critical demand can produce (running to many times sonic), which must never be
# handed back as an accepted solution.
SUPERSONIC_REJECT = 1.5

# Fischer-Burmeister smoothing width for the choking complementarity residual
# (``fischer_burmeister(a, b, EPS_FB)`` in the area-change / pressure-outlet kernels).
# Its arguments are dimensionless margins -- a relative Mach deficit ``1 - M`` and a
# relative pressure gap ``(p - p_spec) / p_spec`` -- so a small fixed floor rounds the
# regime-switch corner equally well at any flow scale.  Unlike the continuation ``eps`` it
# does not taper with the schedule; it only regularizes the branch, never the equations.
EPS_FB = 1e-5

# Floor on the propagated mass flow, as a fraction of the reference, when the seed divides a
# flame's heat release by it to get the enthalpy rise Q_dot / |mdot|.  It only guards the seed's
# arithmetic where a network propagates no flow onto a flame's edge; the residual's own floor is
# the continuation's ``eps``.
MDOT_SEED_FLOOR = 1e-3

# Factor by which a flame seed's assumed mass flow may exceed the realized one before the solve
# reports the seed as misleading.  The seeded rise varies as Q_dot / mdot, so an over-stated flow
# understates the rise and starts the iteration on the steep side of that curve: a factor of a few
# is absorbed (the solve converges, taking more steps), while an order of magnitude can stall it.
# The threshold sits between the two, above the error a flame-blind estimate of a throttled flow
# incurs on its own and below the error that has been seen to stall a solve.
FLAME_SEED_MDOT_TOL = 10.0

# Globalization tuning for the damped Newton in ``_solve_stage``.  These are standard,
# solver-independent line-search / Levenberg-Marquardt defaults, not physics: the exact values
# only trade robustness against a few extra residual evaluations on hard iterates.
_LS_MAX_BACKTRACK = 30  # cap on step halvings in the Armijo line search (alpha down to ~2^-30)
_LS_SHRINK = 0.5  # step-length reduction per backtrack (alpha *= this)
_LS_ARMIJO = 1e-4  # Armijo sufficient-decrease coefficient
_LM_MAX_TRIES = 40  # cap on damping trials in the Levenberg-Marquardt fallback
_LM_INIT = 1e-3  # initial LM damping
_LM_INCREASE = 4.0  # damping growth when an LM trial is rejected (shorter, safer step)
_LM_DECREASE = 0.5  # damping relaxation after any accepted step (toward Gauss-Newton)
_LM_MIN = 1e-12  # damping floor
_LM_MAX = 1e8  # damping ceiling


def _stage_eps(mdot_ref, kappa):
    """Complementarity smoothing width for a continuation stage (vanishes with ``kappa``)."""
    return max(0.3 * kappa, 1e-4) * mdot_ref


def domain_max_dp(prob):
    """Largest a-priori pressure drop set by the boundary conditions.

    The span between the highest and lowest absolute-pressure boundary reference
    (``total_pressure_inlet`` / ``pressure_outlet``).  Returns ``0.0`` when fewer than
    two such references exist -- a mass-driven network whose real pressure drop is not
    known until the flow is solved.  Used to scale the artificial resistance (and, later,
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


def _choked_chamber_pressure(prob, node, edge_mdot, edge_ht, edge_xi):
    """Chamber (upstream) pressure at a compact choked-nozzle node, from its critical mass flux.

    The nozzle passes ``mdot = rho c A* (2/(gamma+1))^((gamma+1)/(2(gamma-1)))`` for the
    (near-stagnation) chamber state; since ``rho`` scales roughly linearly with pressure at fixed
    enthalpy, a few fixed-point steps recover the chamber pressure that carries the edge's seeded
    mass flow.  Returns ``None`` when the estimate is not well defined (no throat, no flow, or a
    non-physical closure state), so the caller simply drops this anchor.
    """
    row_ptr = np.asarray(prob.row_ptr)
    col_edge = np.asarray(prob.col_edge)
    npar_f = np.asarray(prob.npar_f)
    ptr = np.asarray(prob.npar_fptr)

    e0 = int(col_edge[row_ptr[node]])
    a_star = float(npar_f[ptr[node] + 0])
    mdot = abs(float(edge_mdot[e0]))
    if not (a_star > 0.0 and mdot > 0.0):
        return None
    h_t = float(edge_ht[e0])
    z_el = np.ascontiguousarray(edge_xi[:, e0]) if edge_xi is not None and edge_xi.shape[0] > 0 else np.zeros(0)

    pc = float(prob.var_scale[1])  # the reference pressure as the starting scale
    for _ in range(12):
        _, rho, c, _ = thermo_state(prob.model_id, prob.tf, prob.ti, z_el, h_t, pc)
        if not (np.isfinite(rho) and np.isfinite(c) and rho > 0.0 and c > 0.0):
            return None
        gamma = rho * c * c / pc
        if not (gamma > 1.0):
            return None
        gexp = (gamma + 1.0) / (2.0 * (gamma - 1.0))
        mdot_crit = rho * c * a_star * (2.0 / (gamma + 1.0)) ** gexp
        if not (mdot_crit > 0.0):
            return None
        pc_new = pc * (mdot / mdot_crit)
        if abs(pc_new - pc) <= 1e-6 * pc:
            pc = pc_new
            break
        pc = pc_new
    return pc if (np.isfinite(pc) and pc > 0.0) else None


def _max_boundary_pressure(prob):
    """Highest boundary or reference pressure in the network.

    Seeding every edge at this near-stagnation pressure starts the whole flow at a low Mach
    number, biasing the solve toward the physical subsonic branch -- the seed the subsonic-scope
    re-solve uses to escape a spurious supersonic root.
    """
    ptr = np.asarray(prob.npar_fptr)
    npar_f = np.asarray(prob.npar_f)
    rid = np.asarray(prob.node_rid)
    p_hi = float(prob.var_scale[1])  # the reference pressure
    for n in range(prob.n_nodes):
        r = int(rid[n])
        if r == PT_INLET or r == P_OUTLET:  # both carry their spec pressure at offset 0
            p_hi = max(p_hi, float(npar_f[int(ptr[n]) + 0]))
    return p_hi


def _boundary_pressure_seed(prob, edge_mdot, edge_ht, edge_xi, interior_default=None):
    """Per-edge static-pressure seed derived from the network's own boundary pressures.

    Whenever pressure information is present -- a total-pressure inlet, a static-pressure outlet, or
    a choked nozzle (whose chamber pressure is estimated from the critical mass flux) -- the seed is
    scaled to it instead of the gauge reference, so a 200-bar network starts near 200 bar rather than
    at ``p_ref``.  Boundary-incident edges take their own boundary pressure.  Interior edges take
    ``interior_default`` if given (used by the perfect-gas seed, whose recovery is robust to a plain
    reference-pressure interior and would be *misled* by the mean at a low-static-pressure throat),
    otherwise the mean of the anchors (the reacting seed, where an interior chamber sits near the
    boundary pressure).  Returns ``None`` when no boundary pressure is available (e.g. a purely
    mass-flow-driven network), so the caller keeps its reference-pressure default.
    """
    N, E = prob.n_nodes, prob.n_edges
    rid = np.asarray(prob.node_rid)
    npar_f = np.asarray(prob.npar_f)
    ptr = np.asarray(prob.npar_fptr)
    row_ptr = np.asarray(prob.row_ptr)
    col_edge = np.asarray(prob.col_edge)

    anchors = []  # (edge id, pressure)
    for n in range(N):
        r = int(rid[n])
        if r == PT_INLET or r == P_OUTLET:
            p_anchor = float(npar_f[int(ptr[n]) + 0])
        elif r == CHOKED_NOZZLE_OUTLET:
            p_anchor = _choked_chamber_pressure(prob, n, edge_mdot, edge_ht, edge_xi)
        else:
            continue
        if p_anchor is not None and np.isfinite(p_anchor) and p_anchor > 0.0:
            anchors.append((int(col_edge[row_ptr[n]]), p_anchor))

    if not anchors:
        return None
    rep = float(interior_default) if interior_default is not None else float(np.mean([p for _, p in anchors]))
    p_seed = np.full(E, rep)
    for e, p in anchors:
        p_seed[e] = p
    return p_seed


def initial_guess(prob, mdot0=None, p0=None, h0=None, z0=None):
    """Initial state; a small co-directional ``mdot`` by default.

    Each of ``mdot0``, ``p0``, ``h0`` may be a scalar (all edges) or an array of
    length ``n_edges`` (per-edge).  ``z0`` (composition rows ``3..n_solve``) may be
    ``None`` (zeros), a scalar, a ``(n_elem,)`` vector (same composition on every
    edge), or a ``(n_elem, n_edges)`` array (per-edge composition).  A reacting
    network usually needs **per-edge** ``h0``/``z0``: an unburnt air edge and a
    burnt edge sit at very different ``(h_t, Z)``, so a single uniform guess can
    leave a frozen ``h -> T`` inversion or an equilibrium solve far from any root.
    When the caller supplies no composition for a reacting network, this returns the
    same feed-mixing seed :func:`solve` uses (via :func:`auto_initial_guess`) rather
    than a zero-composition guess, which a reacting closure cannot evaluate (a mixture
    of zero total mass has no temperature).
    """
    # A reacting network seeded with zero composition is not evaluable -- the closure
    # divides by the mixture's total moles -- so unless the caller pins the composition
    # (z0), fall back to the physically-seeded per-edge guess solve() itself uses.
    if prob.model_id == EQ_KERNEL and z0 is None:
        x = auto_initial_guess(prob, mdot0=mdot0, p0=p0)
        if h0 is not None:
            x[2, :] = h0
        if getattr(prob, "marker_row", -1) >= 0 and prob.marker_seed is not None:
            x[prob.marker_row, :] = prob.marker_seed  # match solve()'s flood-fill marker seed
        return x
    mdot_ref, p_ref, h_ref = prob.var_scale[0], prob.var_scale[1], prob.var_scale[2]
    x = np.zeros((prob.n_solve, prob.n_edges))
    x[0, :] = 0.05 * mdot_ref if mdot0 is None else mdot0
    if p0 is not None:
        x[1, :] = p0
    else:
        # Seed the pressure from the network's own boundary pressures when available, not p_ref.
        # Interior edges keep the reference (the closed-form perfect-gas recovery is robust to it and
        # a low-static-pressure throat would be mis-seeded by the anchor mean).
        p_seed = _boundary_pressure_seed(prob, x[0, :], np.full(prob.n_edges, h_ref), None, interior_default=p_ref)
        x[1, :] = p_ref if p_seed is None else p_seed
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


def auto_initial_guess(prob, mdot0=None, p0=None, max_sweeps=1000):
    """Physically-seeded per-edge initial guess by mixing the feeds through the graph.

    Nefes carries absolute (formation-inclusive) enthalpies rather than sensible ones, so
    total enthalpy can differ enormously between edges -- an unburnt air edge and a burnt
    edge sit at very different ``h_t`` -- and a single uniform enthalpy guess is often too
    far from any root for a robust solve.  This routine performs a mass-weighted mixing
    estimate across the network to seed each edge's mass flow, total enthalpy, and reacting
    scalars: it propagates the feed mass flows (inlets/sources inject, junctions sum,
    splitters divide) and blends the advected scalars (``h_t`` and the feed mixture
    fractions) mass-weighted by that flow.  Because conserved scalars mix linearly, each
    edge lands at its adiabatic-mixing ``(h_t, xi)``, from which the closure recovers the
    temperature.  A heat-release flame adds its ``Q_dot / |mdot|`` enthalpy rise on top of
    that mixing, so the edges downstream of it are seeded burnt rather than unburnt.

    Parameters
    ----------
    prob : CompiledProblem
        The compiled reacting network.
    mdot0 : float, optional
        Reference mass flow for the propagation (default: the compiled ``mdot`` scale); also
        the assumed inflow at total-pressure inlets, whose ``mdot`` is not yet known.
    p0 : float, optional
        Uniform pressure seed (default: the compiled pressure scale).
    max_sweeps : int, optional
        Maximum relaxation sweeps for each of the two graph propagations (default 1000).  A
        warning is emitted, and the last iterate returned, if either propagation has not
        converged within this many sweeps (e.g. a very large or strongly cyclic network).

    Returns
    -------
    ndarray
        Initial state, shape ``(n_solve, n_edges)``.
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

    # per-node injected (mdot, [h_t, xi...]) for the stream-introducing elements, and the
    # heat release of any power-source flame (added to h_t downstream, not a stream of its own)
    inj_mdot = np.zeros(N)
    inj_scal = np.zeros((N, nscal))
    has_inj = np.zeros(N)
    node_qdot = np.zeros(N)
    for n in range(N):
        pb = ptr[n]
        r = rid[n]
        if r == FLAME_HEAT_RELEASE:
            node_qdot[n] = npar_f[pb + 0]
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

    # (1) propagate the edge mass flow; a topological propagation converges in at most the
    # longest path length, so the allclose break normally exits well before max_sweeps.
    edge_mdot = np.full(E, md)
    has_out = out_count[tail] > 0
    mdot_converged = False
    for _ in range(max_sweeps):
        node_in = inj_mdot * has_inj
        np.add.at(node_in, head, edge_mdot)
        new = edge_mdot.copy()
        new[has_out] = (node_in[tail] / np.maximum(out_count[tail], 1.0))[has_out]
        if np.allclose(new, edge_mdot, rtol=1e-12, atol=1e-12 * md):
            edge_mdot = new
            mdot_converged = True
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

    # A heat-release flame raises its outflow total enthalpy by Q_dot / |mdot| rather than
    # mixing a stream in, so the mass-weighted blend alone would carry the cold feed enthalpy
    # straight through it and leave every downstream edge at the unburnt temperature.  Because
    # the mass flow is already propagated, the rise is known per edge and added to h_t (scalar 0)
    # on each edge leaving a flame.
    edge_dh = node_qdot[tail] / np.maximum(np.abs(edge_mdot), MDOT_SEED_FLOOR * md)

    scal_converged = False
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
        new[valid, 0] += edge_dh[valid]
        if np.allclose(new, edge_scal, rtol=1e-12, atol=1e-12):
            edge_scal = new
            scal_converged = True
            break
        edge_scal = new

    if not (mdot_converged and scal_converged):
        warnings.warn(
            f"auto_initial_guess: graph propagation did not converge within max_sweeps={max_sweeps} "
            f"(mdot converged={mdot_converged}, scalars converged={scal_converged}); "
            "returning the last iterate as the seed.",
            stacklevel=2,
        )

    x = np.zeros((n_solve, E))
    x[0, :] = edge_mdot
    if p0 is not None:
        x[1, :] = float(p0)
    else:
        # Seed the pressure from the network's own boundary pressures (choked chambers included),
        # scaled by the propagated mass flow and mixed enthalpy, not the gauge reference.
        edge_xi = edge_scal[:, 1:].T if nscal > 1 else None
        p_seed = _boundary_pressure_seed(prob, edge_mdot, edge_scal[:, 0], edge_xi)
        x[1, :] = p_ref if p_seed is None else p_seed
    x[2, :] = edge_scal[:, 0]
    if nscal > 1:
        x[3:, :] = edge_scal[:, 1:].T
    return x


def _flame_divisor_edges(prob):
    """Each heat-release flame paired with the edge whose mass flow scales its enthalpy rise.

    The flame kernel raises its outflow total enthalpy by ``Q_dot / |mdot|`` taken on its first
    port (the outflow edge), and the seed estimates the same rise on the same edge, so this is
    the one flow whose seed error carries straight into the seeded temperature.
    """
    rid = np.asarray(prob.node_rid)
    row_ptr = np.asarray(prob.row_ptr)
    col_edge = np.asarray(prob.col_edge)
    return [(int(n), int(col_edge[row_ptr[n]])) for n in np.nonzero(rid == FLAME_HEAT_RELEASE)[0]]


def _warn_on_cold_flame_seed(prob, flame_edges, seed_mdot, x2d, mdot_ref):
    """Report a flame whose seeded mass flow badly overstates the one the solve reached.

    The seeded rise varies as ``Q_dot / mdot``, so a seed flow above the realized one seeds the
    flame cold, which is the expensive direction: it starts the iteration on the steep side of
    that curve.  Silent when the flow is prescribed (the seed is then exact) or when the estimate
    is merely a few times off, which the solve absorbs.
    """
    names = prob.node_names or ()
    worst = None
    for n, e in flame_edges:
        seeded = abs(float(seed_mdot[e]))
        realized = abs(float(x2d[0, e].real))
        if realized <= 0.0 or seeded <= realized * FLAME_SEED_MDOT_TOL:
            continue
        ratio = seeded / realized
        if worst is None or ratio > worst[0]:
            label = names[n] if n < len(names) and names[n] else f"#{n}"
            worst = (ratio, label, seeded, realized)
    if worst is None:
        return
    ratio, label, seeded, realized = worst
    others = f" ({len(flame_edges) - 1} other flame(s) not reported)" if len(flame_edges) > 1 else ""
    warnings.warn(
        f"the heat-release flame {label!r} seeded its total-enthalpy rise from a mass flow of "
        f"{seeded:.4g} kg/s, but the solve settled at {realized:.4g} kg/s on that edge, so the "
        f"seeded rise was about {ratio:.0f}x too small{others}. The rise varies as Q_dot / mdot, "
        f"so seeding a flame cold starts the iteration on the steep side of that curve and costs "
        f"extra steps, or stalls it outright. The seed takes this flow from the boundaries and "
        f"falls back to mdot_ref ({mdot_ref:.4g} kg/s) where no inlet prescribes it, so an "
        f"mdot_ref well above the true flow produces exactly this; leaving mdot_ref unset lets it "
        f"be estimated from the boundary specification instead.",
        stacklevel=3,
    )


@dataclass
class SolveResult:
    x: np.ndarray  # converged state, shape (3, E)
    converged: bool
    iterations: int
    residual_norm: float
    history: List[float] = field(default_factory=list)
    # Seconds spent in the solve that produced this state, measured on a monotonic clock.  The
    # first solve of a session carries the kernels' one-off compilation (see the note on
    # ``elapsed`` in ``solve``); a restored state that was never re-solved reports 0.0.
    elapsed: float = 0.0

    def __repr__(self) -> str:
        """One-line solver outcome: convergence, iteration count, residual, and state shape."""
        status = "converged" if self.converged else "NOT converged"
        shape = "x".join(str(s) for s in np.shape(self.x))
        return (
            f"SolveResult: {status} in {self.iterations} iteration(s), "
            f"residual_norm = {self.residual_norm:.3e}, state ({shape})"
        )


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
    """Drive one continuation stage to convergence with a globalized damped Newton.

    Solves the nondimensional system at fixed ``(eps, kappa)`` down to ``tol`` on the scaled
    residual 2-norm (the merit ``||R_hat||``, computed by :func:`_merit`).  Each iteration
    assembles the complex-step Jacobian, scales the linear system, and takes a step by two
    tiers:

    1. the full sparse-LU Newton step, with an Armijo backtracking line search on its length
       ``alpha`` (``alpha = 1`` is the full step) until the merit decreases sufficiently;
    2. if no Newton step is accepted -- the LU factorization failed / was singular, or every
       backtrack still increased the merit -- a Levenberg-Marquardt fallback whose damping
       ``lam`` is grown until a trial reduces the merit, giving a short, safe step out of the
       stall.  ``lam`` interpolates the step between Gauss-Newton (small ``lam``) and
       steepest descent (large ``lam``); it is relaxed on success and grown on rejection.

    Parameters
    ----------
    prob : CompiledProblem
        The compiled network.
    x2d : ndarray
        Current state, shape ``(n_solve, n_edges)`` (warm-started from the previous stage).
    eps, kappa : float
        Smoothing width and artificial-friction coefficient, held fixed for this stage.
    tol : float
        Convergence tolerance on the scaled residual 2-norm.
    max_iter : int
        Maximum Newton iterations.
    history : list of float
        Appended in place with the merit at each iteration.
    res_scale, var_scale : ndarray
        Residual / variable nondimensionalization used to scale the linear system.
    reporter : _Reporter, optional
        Progress printer.

    Returns
    -------
    x2d : ndarray
        The stage's final state.
    converged : bool
        Whether the merit fell below ``tol``.
    it : int
        Iterations taken.
    norm : float
        Final scaled residual 2-norm.
    """
    vcol = col_scale(var_scale, prob.n_edges)  # per-column variable scale (edge-major)
    lam = _LM_INIT
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

        # (1) full Newton step, then an Armijo backtracking line search on its length alpha.
        dy = newton_step(J_hat, R_hat)
        accepted = False
        if dy is not None:
            dx = unflatten(vcol * dy, prob.n_edges, prob.n_solve)
            alpha = 1.0
            for _ in range(_LS_MAX_BACKTRACK):
                x_try = x2d + alpha * dx
                n_try, R_try = _merit(prob, x_try, eps, kappa, res_scale)
                if n_try < (1.0 - _LS_ARMIJO * alpha) * norm:  # sufficient-decrease test
                    x2d, norm, R = x_try, n_try, R_try
                    accepted = True
                    lam = max(lam * _LM_DECREASE, _LM_MIN)  # a good full step -> relax LM damping
                    break
                alpha *= _LS_SHRINK  # shorten the step and retry

        # (2) Levenberg-Marquardt fallback with adaptive damping, when no Newton step helped.
        if not accepted:
            for _ in range(_LM_MAX_TRIES):
                dy = lm_step(J_hat, R_hat, lam)
                x_try = x2d + unflatten(vcol * dy, prob.n_edges, prob.n_solve)
                n_try, R_try = _merit(prob, x_try, eps, kappa, res_scale)
                if n_try < norm:  # accept any decrease; the step is already damped
                    x2d, norm, R = x_try, n_try, R_try
                    accepted = True
                    lam = max(lam * _LM_DECREASE, _LM_MIN)
                    break
                lam = min(lam * _LM_INCREASE, _LM_MAX)  # stronger damping -> shorter, safer step
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
    enforce_subsonic=None,
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
        Maximum Newton iterations per continuation stage.
    kappa_stages : sequence of float, optional
        Artificial-resistance continuation schedule (dimensionless), warm-started in order.
    kappa_scale : {"dp", "absolute"}, optional
        How the schedule's artificial-friction coefficient is sized.  ``"dp"`` (default)
        multiplies each ``kappa`` by the friction resistance ``min(domain_max_dp(prob) /
        mdot_ref, 1)``, so the injected artificial pressure drop at the reference flow is
        capped at a fraction (``kappa``) of the real driving drop.  The cap means the
        scaling only *softens* the friction for low-``dP`` / high-``mdot`` networks (where
        the constant absolute drop ``kappa * mdot`` would over-perturb), and is a no-op
        for healthy-``dP`` networks (``r_art = 1``).  When no a-priori pressure drop is
        available (a mass-driven network, ``domain_max_dp == 0``) it falls back to the
        absolute coefficient.  ``"absolute"`` always uses the constant unit coefficient
        (artificial drop ``kappa * mdot``).  Either way ``eps`` tracks the dimensionless
        ``kappa``.
    adaptive_scale : bool, optional
        When ``True`` (default), the residual / variable scales are re-measured from the
        realized inflow at each continuation stage (total inlet ``mdot`` for the mass rows, the
        mass-weighted mean inlet ``|h_t|`` for the energy rows) instead of the fixed compiled
        references -- so the nondimensionalization tracks the actual flow and the user need not
        supply ``mdot_ref`` / ``h_ref``.  The quiescent ``mdot = 0`` case falls back to the seed
        scales.  ``False`` uses the compiled ``prob.res_scale`` / ``prob.var_scale``.
    verbose : int or bool, optional
        Progress verbosity.  ``0``/``False`` is silent; ``1``/``True`` prints a
        one-line gross-residual summary per continuation stage; ``2`` additionally prints
        the scaled residual broken down by equation kind (mass, pressure, energy, then
        each composition scalar) every ``progress_interval`` iterations within a stage.
    progress_interval : int, optional
        Iteration stride for the per-iteration prints at ``verbose >= 2``.
    enforce_subsonic : bool or None, optional
        Whether to keep the returned mean flow on the physical subsonic branch (the present
        modeling scope).  ``None`` (default) follows the global ``nefes.config.enforce_subsonic``.
        When active, a converged solution carrying a genuinely supersonic edge (a spurious branch
        a cold start can reach) is re-solved once from a near-stagnation seed that lands the
        subsonic branch.  If that re-solve still cannot remove the supersonic edge, a marginal
        overshoot just past a sonic throat is kept with a warning, but a state running far past
        sonic (above ``SUPERSONIC_REJECT``, the spurious / ill-posed branch a resistance-free loop
        or over-critical demand produces) is returned with ``converged = False`` and a warning, so
        a wildly supersonic result is never handed back as accepted.  With the guard off the raw
        branch is returned as converged regardless of Mach number.

    Returns
    -------
    SolveResult
        The converged state and solve diagnostics, including the ``elapsed`` seconds the solve
        took.  That reading covers everything this call does (the seed, every continuation stage,
        and any subsonic-scope re-solve), so on the first solve of a session it also carries the
        one-off compilation of the kernels, which typically dwarfs the solve itself.  Time a
        second solve to measure the solver rather than the compiler.

    Notes
    -----
    ``elapsed`` is a measurement of this machine and this run, not a property of the network:
    unlike the converged state it is not reproducible, and it is deliberately not written to a
    case file.
    """
    t_start = time.perf_counter()
    mdot_ref = prob.var_scale[0]
    # artificial-friction resistance: scale the dimensionless kappa schedule so the
    # injected artificial dP is a fixed fraction of the real driving dP (kappa_scale
    # == "dp"); fall back to the constant unit coefficient when there is no a-priori
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
    elif prob.model_id == EQ_KERNEL or np.any(np.asarray(prob.node_rid) == FLAME_HEAT_RELEASE):
        # A physically-seeded per-edge guess is needed wherever h_t spans a wide range between
        # edges, which a uniform guess cannot straddle: a reacting network (unburnt and burnt
        # streams sit at very different absolute enthalpies) or a heat-release flame (whose
        # Q_dot/|mdot| jump leaves the downstream edges hundreds of kelvin above the feed).
        x2d = auto_initial_guess(prob)
    else:
        x2d = initial_guess(prob)
    # A flame's seeded rise is only as good as the mass flow the seed divided into it, so keep
    # that flow to compare against the realized one once the solve has run.  Only for a seed of
    # our own making: a caller-supplied x0 owes nothing to mdot_ref.
    flame_edges = _flame_divisor_edges(prob) if x0 is None else []
    seed_mdot = x2d[0, :].copy() if flame_edges else None
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
    # realized inflow at each continuation stage (kept constant within a stage).
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
        reporter.stage_start(kappa, eps, res_scale)
        # eps tracks the dimensionless kappa; the kernel friction uses the scaled coefficient.
        x2d, converged, it, norm = _solve_stage(
            prob, x2d, eps, kappa * r_art, tol, max_iter, history, res_scale, var_scale, reporter
        )
        total_it += it
        reporter.stage_end(kappa, it, norm, converged)
        if not converged and kappa == 0.0:
            reporter.failure(prob, x2d, kappa)
            break

    # Subsonic-scope backstop.  The steady residual admits a spurious *supersonic* isentropic
    # root beside the physical subsonic one at over-critical operating points, and a cold seed
    # can land on it; the choking model is unaffected (a real throat still pins at M = 1).  When
    # subsonic enforcement is on, a converged solution carrying a genuinely supersonic edge is
    # re-solved once from a near-stagnation seed, which reliably reaches the subsonic branch.  If
    # even that re-solve cannot remove the supersonic edge, the edge is kept but a warning is
    # raised; when it runs *far* past sonic (above SUPERSONIC_REJECT) the state is on the spurious
    # / ill-posed branch and is not reported as converged, so a wildly supersonic result is never
    # handed back as accepted -- the caller opts out with enforce_subsonic=False.
    enforce = config.enforce_subsonic if enforce_subsonic is None else bool(enforce_subsonic)
    if enforce and converged:
        m_max = float(np.max(np.abs(states_table(prob, x2d, caloric=False)[ES_M, :].real)))
        if m_max > SUPERSONIC_TOL:
            seed = initial_guess(prob, p0=_max_boundary_pressure(prob))  # low-Mach start
            recov = solve(
                prob,
                x0=seed,
                tol=tol,
                max_iter=max_iter,
                kappa_stages=kappa_stages,
                kappa_scale=kappa_scale,
                adaptive_scale=adaptive_scale,
                enforce_subsonic=False,
            )
            rec_m = float(np.max(np.abs(states_table(prob, recov.x, caloric=False)[ES_M, :].real)))
            if recov.converged and rec_m < m_max:
                x2d, converged, norm = recov.x, recov.converged, recov.residual_norm
                total_it += recov.iterations
                m_max = rec_m
            if m_max > SUPERSONIC_REJECT:
                converged = False  # far past sonic: the spurious branch, never accept it
                warnings.warn(
                    f"the steady solve carries a supersonic edge (max M = {m_max:.2f}) far past "
                    "sonic; the subsonic-scope re-solve could not remove it, so the result is "
                    "reported as not converged. The case is likely ill-posed (for example a "
                    "resistance-free loop) or genuinely supersonic, which is outside the present "
                    "(subsonic) scope. Set nefes.config.enforce_subsonic = False to accept it.",
                    stacklevel=2,
                )
            elif m_max > SUPERSONIC_TOL:
                warnings.warn(
                    f"the steady solve carries a marginally supersonic edge (max M = {m_max:.2f}); "
                    "the subsonic-scope re-solve could not pull it below sonic. It is kept as a "
                    "near-choke state, but is at the edge of the present (subsonic) scope.",
                    stacklevel=2,
                )
    if flame_edges:
        _warn_on_cold_flame_seed(prob, flame_edges, seed_mdot, x2d, mdot_ref)
    return SolveResult(
        x=x2d,
        converged=converged,
        iterations=total_it,
        residual_norm=norm,
        history=history,
        elapsed=time.perf_counter() - t_start,
    )
