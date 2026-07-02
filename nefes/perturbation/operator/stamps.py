"""Analytic acoustic stamps written onto ``A(omega)`` after ``J_alg + i*omega*M``.

Three faces (implementation-plan.md s8.2-8.3):

* ``stamp_propagation`` -- the **duct** phase relations ``P(omega)`` (theory.md
  s12.3), the only omega-dependent block in v1.  For each duct it replaces three
  rows (its two node rows + the head edge's transport row) with the
  characteristic phase relations, built diagonally in the wave amplitudes
  ``w = (f, g, h)`` and mapped to solution-variable rows through ``L_e``.
* ``stamp_sources`` -- the dynamic-source ``S(omega)`` face (theory.md s12.4): a
  flame's unsteady heat release on the downstream energy row, or a mass source's
  fluctuating injection on its node rows, each driven by a frequency-domain transfer
  function of a reference-edge fluctuation (:func:`build_source_stamps`).
* ``stamp_boundaries`` -- terminal reflection coefficients (reserved; the v1
  scattering driver imposes incoming waves at terminals instead, so a no-op).

``build_storage`` assembles the storage ``M`` block -- the ``d/dt integral_V U``
term dropped at steady state.  It is populated per element through a small
registry (``_STORAGE_BUILDERS``, keyed by residual id): a finite-volume
:func:`~nefes.elements.catalog.cavity` contributes its mass-row compliance, the
inline pressure elements their per-port compliance + series inertance, and the
manifolds their chamber compliance + per-branch neck inertance -- all from
length/volume inputs read off the same per-port machinery.  ``M`` enters the
operator as ``i*omega*M``, and its stored energy enters the acoustic-power
ledger (:func:`nefes.perturbation.fields.power._lumped_storage_energy`).

These run **above the @njit line** -- plain Python / SciPy.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .characteristics import dx_to_char, dq_to_dx
from .matrices import partition
from .verify import duct_nodes, verify_acoustic
from .terminals import find_terminals
from ...solver.control import states_table
from ...assembly.derive import ES_RHO, ES_C, ES_U, ES_P, ES_AREA, ES_MDOT, ES_T
from ...elements.ids import (
    CAVITY,
    FLAME_HEAT_RELEASE,
    ISEN_AREA_CHANGE,
    SUDDEN_AREA_CHANGE,
    LOSS,
    LINEAR_RESISTANCE,
    JUNCTION,
    SPLITTER,
)


@dataclass
class DuctStamp:
    """Frozen per-duct data for the ``P(omega)`` stamp (built once per sweep)."""

    e0: int  # tail-station edge (port 0, points into the duct)
    e1: int  # head-station edge (port 1, points out of the duct)
    L0: np.ndarray  # 3x3 dx_to_char at e0's mean state
    L1: np.ndarray  # 3x3 dx_to_char at e1's mean state
    tau_p: float  # L / (u + c)
    tau_m: float  # L / (c - u)
    tau_0: float  # L / u   (inf when quiescent)
    u: float  # mean axial velocity (>= 0 along the duct axis)
    row_f: int  # duct node row holding the downstream (f) phase relation
    row_g: int  # duct node row holding the upstream (g) phase relation
    row_h: int  # head edge's transport row, repurposed for the entropy (h) phase
    cols0: tuple  # the 3 columns of e0
    cols1: tuple  # the 3 columns of e1
    # transported composition scalars (s >= 1) convect at u like the entropy wave; each gets
    # the same downstream-edge transport row carrying xi(head) = P0 xi(tail).  Empty for an
    # inert (single-scalar, h_t-only) gas.  comp_cols0/1 are the scalar's tail/head columns.
    comp_rows: tuple = ()
    comp_cols0: tuple = ()
    comp_cols1: tuple = ()


def build_duct_stamps(prob, x_bar, K, u_floor=1e-8, cals=None):
    """Build the per-duct ``P(omega)`` data at the frozen mean state ``x_bar``.

    Runs ``verify_acoustic`` first (pinned orientation, subsonic, length > 0).
    ``cals`` (optional): per-edge caloric rows (:func:`characteristics.edge_caloric`)
    used in place of the perfect-gas ``K`` for a reacting/variable-gamma edge.
    """
    verify_acoustic(prob, x_bar)
    est = states_table(prob, x_bar)
    ns = int(prob.n_solve)
    stamps = []
    for n in duct_nodes(prob):
        base = int(prob.row_ptr[n])
        e0 = int(prob.col_edge[base])
        e1 = int(prob.col_edge[base + 1])
        length = float(prob.npar_f[int(prob.npar_fptr[n])])

        # the duct is constant-area and lossless: e0 and e1 share the mean state.
        rho = float(est[ES_RHO, e0])
        c = float(est[ES_C, e0])
        u = float(est[ES_U, e0])
        p = float(est[ES_P, e0])
        cal0 = None if cals is None else cals[e0]
        cal1 = None if cals is None else cals[e1]
        L0 = dx_to_char(rho, c, u, p, float(est[ES_AREA, e0]), K, cal0)
        L1 = dx_to_char(
            float(est[ES_RHO, e1]),
            float(est[ES_C, e1]),
            float(est[ES_U, e1]),
            float(est[ES_P, e1]),
            float(est[ES_AREA, e1]),
            K,
            cal1,
        )

        tau_p = length / (u + c)
        tau_m = length / (c - u)
        tau_0 = length / u if abs(u) > u_floor else np.inf

        # composition scalars (transport index s = 1 .. n_scalars-1; solve var 2+s) ride the
        # convected wave at u, just like the entropy/h_t scalar (s=0); their phase row lives on
        # the same downstream edge.  n_scalars = n_solve - 2 (s=0 is h_t).
        tr0 = int(prob.transport_row0)
        E = int(prob.n_edges)
        e_down = e0 if u < -u_floor else e1
        n_scalars = ns - 2
        comp_rows = tuple(tr0 + s * E + e_down for s in range(1, n_scalars))
        comp_cols0 = tuple(ns * e0 + (2 + s) for s in range(1, n_scalars))
        comp_cols1 = tuple(ns * e1 + (2 + s) for s in range(1, n_scalars))

        r0 = int(prob.node_row_ptr[n])
        stamps.append(
            DuctStamp(
                e0=e0,
                e1=e1,
                L0=L0,
                L1=L1,
                tau_p=tau_p,
                tau_m=tau_m,
                tau_0=tau_0,
                u=u,
                row_f=r0,
                row_g=r0 + 1,
                # entropy phase relation lives on the *downstream* edge's transport row
                # (head for forward/quiescent flow, tail under backflow), leaving the
                # genuine-inflow edge's transport row free for the boundary entropy seat.
                row_h=tr0 + (e0 if u < -u_floor else e1),
                cols0=tuple(ns * e0 + v for v in range(3)),
                cols1=tuple(ns * e1 + v for v in range(3)),
                comp_rows=comp_rows,
                comp_cols0=comp_cols0,
                comp_cols1=comp_cols1,
            )
        )
    return stamps


def _set_row(A, row, cols0, coeff0, cols1, coeff1):
    """Overwrite a full LIL row with two length-3 coefficient blocks."""
    A.rows[row] = []
    A.data[row] = []
    for c, v in zip(cols0, coeff0):
        A[row, c] = v
    for c, v in zip(cols1, coeff1):
        A[row, c] = v


def stamp_propagation(A, omega, duct_stamps, u_floor=1e-8, skip_entropy=False):
    """Apply the duct phase relations ``P(omega)`` to LIL matrix ``A`` in place.

    For each duct (tail station ``0`` -> head station ``1``):
        f1 = Pp*f0,   g0 = Pm*g1,   h1 = P0*h0,
    with ``Pp = exp(-i w tau_+)``, ``Pm = exp(-i w tau_-)``, ``P0 = exp(-i w
    tau_0)``.  At a quiescent duct (u ~ 0) the entropy wave is stationary and
    decoupled, so ``P0 = 1``.

    ``skip_entropy`` (set under isentropic assembly) omits the entropy (h) phase row
    entirely: it would only be overwritten by :func:`stamp_isentropic` with ``h = 0``,
    and at a complex ``omega`` its ``exp(-i w tau_0)`` can overflow needlessly.
    """
    for st in duct_stamps:
        Pp = np.exp(-1j * omega * st.tau_p)
        Pm = np.exp(-1j * omega * st.tau_m)

        # Row f:  f1 - Pp*f0 = 0
        _set_row(A, st.row_f, st.cols0, -Pp * st.L0[0, :], st.cols1, st.L1[0, :])
        # Row g:  g0 - Pm*g1 = 0
        _set_row(A, st.row_g, st.cols0, st.L0[1, :], st.cols1, -Pm * st.L1[1, :])
        if skip_entropy:
            # isentropic/stability mode: stamp_isentropic pins the entropy row to h = 0, and the
            # convected scalars (entropy + composition) are decoupled -- their transit time
            # tau_0 = L/u would overflow / pollute the acoustic spectrum at complex omega (the
            # composition rows keep their steady J_alg convection).  Re-enabled in the full
            # operator below (the forced-response path), where the convective phase matters.
            continue
        # Row h:  h1 - P0*h0 = 0
        P0 = np.exp(-1j * omega * st.tau_0) if abs(st.u) > u_floor else 1.0 + 0.0j
        _set_row(A, st.row_h, st.cols0, -P0 * st.L0[2, :], st.cols1, st.L1[2, :])
        # Composition scalars convect at u too (xi(head) = P0 xi(tail)); same convected wave
        # as the entropy/h_t scalar, so it rides the same decouple under isentropic mode.
        for row, c0, c1 in zip(st.comp_rows, st.comp_cols0, st.comp_cols1):
            _set_row(A, row, (c0,), (-P0,), (c1,), (1.0 + 0.0j,))


@dataclass
class SourceTerm:
    """One precomputed transfer-function term of a dynamic source ``S(omega)``.

    The term contributes ``factor * F(omega) * (coeff . x_ref)`` to a residual row,
    where ``coeff`` already folds the per-term ``gain`` and the ``1/phi_bar``
    normalization, and ``cols`` are its columns in the operator (the reference edge's
    solve variables).
    """

    transfer: object  # nefes.elements.dynamic_source.TransferFunction
    cols: np.ndarray  # int[k]   columns ns*ref_edge + v of the nonzero coefficients
    coeff: np.ndarray  # complex[k]  gain * functional / phi_bar at the reference edge


@dataclass
class SourceStamp:
    """Precomputed ``S(omega)`` stamp for one dynamic-source element (theory.md s12.4).

    Each frequency adds ``rows[r] += factors[r] * sum_k term_k`` -- the source
    *adds* to the rows ``J_alg`` already populates (it never overwrites them), so the
    feedback rides on top of the converged jump conditions.
    """

    rows: tuple  # residual rows the response feeds
    factors: np.ndarray  # real factor per row (e.g. -Q_bar/mdot for heat release)
    terms: list  # list of SourceTerm (summed)
    edges: tuple  # downstream edges to keep physical under isentropic (heat release)
    node: int  # the source element (for diagnostics)
    analytic: bool  # whether every transfer function is complex-analytic (stability)
    max_delay: float  # longest pure delay [s] across the terms (contour overflow clamp)


def _ref_functional(quantity, est_col, x_bar_col, K, scalar_names, cal=None):
    """Linear functional + mean of a reference quantity at one edge.

    Returns ``(idx, vec, mean)``: the solve-variable indices ``idx`` (within the edge
    block) carrying the nonzero coefficients ``vec`` of ``phi' = vec . x_edge``, and the
    mean value ``phi_bar`` used to normalize the fractional response.  ``cal`` (see
    :func:`characteristics.dq_to_dx`) supplies the reacting caloric coupling so that a
    velocity/density reference is extracted from ``(mdot', p', h_t')`` with the gas's
    *actual* caloric -- the perfect-gas ``K`` mis-extracts it for the reacting backend.
    """
    rho = float(est_col[ES_RHO])
    u = float(est_col[ES_U])
    p = float(est_col[ES_P])
    area = float(est_col[ES_AREA])
    mdot = float(est_col[ES_MDOT])
    if quantity == "mdot":
        return np.array([0]), np.array([1.0]), mdot
    if quantity == "p":
        return np.array([1]), np.array([1.0]), p
    if quantity in ("u", "rho"):
        # primitives (drho, du, dp) = inv(dq_to_dx) . (dmdot, dp, dht)
        inv = np.linalg.inv(dq_to_dx(rho, u, p, area, K, cal))
        row = 1 if quantity == "u" else 0
        mean = u if quantity == "u" else rho
        return np.array([0, 1, 2]), inv[row, :].astype(float), mean
    if quantity.startswith("Z:"):
        name = quantity[2:]
        if name not in scalar_names:
            raise ValueError(
                f"reference quantity {quantity!r} names an unknown composition scalar; "
                f"available: {list(scalar_names)}"
            )
        s = list(scalar_names).index(name)
        return np.array([3 + s]), np.array([1.0]), float(x_bar_col[3 + s])
    raise ValueError(f"unsupported reference quantity {quantity!r}")


def build_source_stamps(prob, x_bar, K, u_floor=1e-8, cals=None):
    """Precompute the ``S(omega)`` stamps for every dynamic-source element.

    Reads ``prob.node_dynamic_source`` (the :class:`~nefes.elements.dynamic_source.DynamicSource`
    descriptors) and resolves, at the frozen mean state, each element's target rows,
    the constant de-normalization factor, and per-term reference functionals.  Returns
    ``(stamps, flame_edges)`` where ``flame_edges`` are the downstream edges whose
    energy row must stay physical under the isentropic assembly (theory.md s12.4 --
    the active flame still adds heat even when convected entropy is dropped elsewhere).
    """
    srcs = getattr(prob, "node_dynamic_source", ()) or ()
    if not any(s is not None for s in srcs):
        return [], frozenset()

    est = states_table(prob, x_bar)
    x_bar = np.ascontiguousarray(x_bar)
    ns = int(prob.n_solve)
    tr0 = int(prob.transport_row0)
    scalar_names = tuple(getattr(prob, "scalar_names", ()) or ())

    def make_terms(desc):
        terms = []
        for t in desc.terms:
            e = int(t.ref_edge)
            cal = None if cals is None else cals[e]
            idx, vec, mean = _ref_functional(t.quantity, est[:, e], x_bar[:, e], K, scalar_names, cal)
            # guard only against a literal divide-by-zero; a small-but-finite mean (low
            # Mach) gives a large -- and physically correct -- fractional response 1/phi_bar.
            if abs(mean) <= 1e-30:
                raise ValueError(
                    f"dynamic source at node references {t.quantity!r} on edge {e}, whose mean "
                    f"value is ~ 0; the fractional response phi'/phi_bar is undefined there "
                    "(reference a flowing edge or a different quantity)"
                )
            coeff = (float(t.gain) / mean) * vec.astype(np.complex128)
            cols = np.array([ns * e + int(v) for v in idx], dtype=np.intp)
            terms.append(SourceTerm(transfer=t.transfer, cols=cols, coeff=coeff))
        return terms

    stamps = []
    flame_edges = set()
    for n in range(int(prob.n_nodes)):
        desc = srcs[n]
        if desc is None:
            continue
        base = int(prob.row_ptr[n])
        deg = int(prob.row_ptr[n + 1]) - base
        r0 = int(prob.node_row_ptr[n])
        pb = int(prob.npar_fptr[n])
        terms = make_terms(desc)

        if desc.target == "Qdot":
            if deg != 2:
                raise ValueError(f"heat-release dynamic source at node {n} must be a 2-port flame")
            # downstream (outflow) edge: the port whose oriented mdot leaves the node
            ports = [(int(prob.col_edge[base + i]), int(prob.orient[base + i])) for i in range(2)]
            outs = [(e, s) for (e, s) in ports if s * float(est[ES_MDOT, e]) > 0.0]
            ins = [(e, s) for (e, s) in ports if s * float(est[ES_MDOT, e]) <= 0.0]
            if len(outs) != 1 or len(ins) != 1:
                raise ValueError(
                    f"heat-release dynamic source at node {n} needs a single through-flow "
                    "direction (one inflow, one outflow edge) at the mean state"
                )
            e_out, s_out = outs[0]
            e_in, _s_in = ins[0]
            mdot_mag = s_out * float(est[ES_MDOT, e_out])  # > 0
            q_mean = _mean_heat_release(prob, n, pb, mdot_mag, est, e_in, e_out, desc.q_mean)
            delta = q_mean / mdot_mag  # specific enthalpy rise [J/kg]
            rows = (tr0 + e_out,)  # downstream total-enthalpy (energy) transport row
            factors = np.array([-delta], dtype=float)  # residual: h_t - H_donor - q'/mdot
            edges = (e_out,)
            flame_edges.add(e_out)
        else:  # target == "mdot": injected mass-flow modulation (a fluctuating injector)
            if deg != 2:
                raise ValueError(f"mass-flow dynamic source at node {n} must be a 2-port inline injector")
            a0 = float(est[ES_AREA, int(prob.col_edge[base])])
            mdot_src = desc.q_mean if desc.q_mean is not None else float(prob.npar_f[pb + 0])
            u_inj = float(prob.npar_f[pb + 1]) if int(prob.npar_fptr[n + 1]) - pb > 1 else 0.0
            # outflow port (oriented mdot leaving the node): the injected stream mixes into it
            ports = [(int(prob.col_edge[base + i]), int(prob.orient[base + i])) for i in range(2)]
            outs = [(e, s) for (e, s) in ports if s * float(est[ES_MDOT, e]) > 0.0]
            if len(outs) != 1:
                raise ValueError(
                    f"mass-flow dynamic source at node {n} needs a single through-flow direction "
                    "(one inflow, one outflow edge) at the mean state"
                )
            e_out, s_out = outs[0]
            mdot_out = s_out * float(est[ES_MDOT, e_out])  # > 0 (= inflow + injected, the mix weight)
            # mass row r0: residual ... - mdot_src';  momentum row r0+1: ... - mdot_src' u_inj / a0.
            rows = [r0, r0 + 1]
            factors = [-mdot_src, -mdot_src * u_inj / a0]
            # conserved-scalar mixing: a fuel pulse mdot_src' drags every advected scalar at the
            # outflow toward the injected stream.  The outflow mix is phi_out = (sum w_i phi_i +
            # mdot_src phi_src) / mdot_out, so d(mix)/d(mdot_src) = (phi_src - phi_out)/mdot_out and
            # the transport residual (phi_out - mix) gains  -mdot_src (phi_src - phi_out)/mdot_out.
            # s = 0 is the injected total enthalpy h_t,src (an enthalpy/entropy spot); s >= 1 are the
            # injected composition scalars (the equivalence-ratio wave the downstream flame burns).
            n_scalars = ns - 2
            E = int(prob.n_edges)
            for s in range(n_scalars):
                phi_src = float(prob.npar_f[pb + 2 + s])
                phi_out = float(x_bar[2 + s, e_out])
                rows.append(tr0 + s * E + e_out)
                factors.append(-mdot_src * (phi_src - phi_out) / mdot_out)
            rows = tuple(rows)
            factors = np.array(factors, dtype=float)
            # the injected enthalpy modulates the outflow entropy row; keep it physical under the
            # isentropic assembly (an active source edge, like a flame's heat-release row).
            edges = (e_out,)
            flame_edges.add(e_out)

        stamps.append(
            SourceStamp(
                rows=rows,
                factors=factors,
                terms=terms,
                edges=edges,
                node=n,
                analytic=desc.analytic,
                max_delay=desc.max_delay,
            )
        )
    return stamps, frozenset(flame_edges)


def _cp_from_state(est_col):
    """Effective specific heat ``cp`` [J/(kg K)] from one mean edge state.

    From the sound speed: the effective isentropic exponent is ``gamma = rho c^2 / p``
    and ``R = p / (rho T)``, so ``cp = gamma R / (gamma - 1)``.  Exact for a perfect
    gas; for the reacting closure it is the cp consistent with the (equilibrium/frozen)
    sound speed the acoustics already use -- and does not depend on the thermo ``tf``
    layout, which is *not* ``[cp, R, ...]`` for the reacting backend.
    """
    rho = float(est_col[ES_RHO])
    c = float(est_col[ES_C])
    p = float(est_col[ES_P])
    T = float(est_col[ES_T])
    gamma = rho * c * c / p
    Rgas = p / (rho * T)
    return gamma * Rgas / (gamma - 1.0)


def _mean_heat_release(prob, n, pb, mdot_mag, est, e_in, e_out, q_mean):
    """Mean heat release ``Q_bar`` [W] for the FTF de-normalization.

    Explicit ``q_mean`` wins.  Otherwise auto-derive from the converged mean flame:
    the perfect-gas heat-release flame carries its power as a parameter; any other
    flame uses the sensible-enthalpy rise ``mdot * cp_bar * (T_out - T_in)`` with
    ``cp_bar`` the mean of the per-edge effective ``cp`` (:func:`_cp_from_state`) -- a
    low-Mach approximation; pass ``q_mean`` for an exact value.
    """
    if q_mean is not None:
        return float(q_mean)
    if int(prob.node_rid[n]) == FLAME_HEAT_RELEASE:
        return float(prob.npar_f[pb + 0])  # Qdot [W], the kernel's own source
    cp_bar = 0.5 * (_cp_from_state(est[:, e_in]) + _cp_from_state(est[:, e_out]))
    dT = float(est[ES_T, e_out]) - float(est[ES_T, e_in])
    return mdot_mag * cp_bar * dT


def stamp_sources(A, omega, source_stamps):
    """Apply the dynamic-source face ``S(omega)`` to LIL matrix ``A`` in place.

    For each precomputed :class:`SourceStamp` and each of its rows ``r`` (factor
    ``fr``), adds ``fr * sum_k F_k(omega) * coeff_k`` into the reference columns --
    *accumulating* onto the rows ``J_alg`` already wrote (the steady jump linearized),
    so the feedback rides on top of them.  ``omega`` is angular (rad/s); transfer
    functions are evaluated at ``f = omega / 2 pi`` (Hz, project convention).
    """
    if not source_stamps:
        return
    freq = omega / (2.0 * np.pi)
    for st in source_stamps:
        # per-term scalar response F_k(f); broadcast scalar input -> scalar output
        scalars = [complex(np.asarray(term.transfer(freq)).reshape(-1)[0]) for term in st.terms]
        for r, fr in zip(st.rows, st.factors):
            for term, Fk in zip(st.terms, scalars):
                contrib = (fr * Fk) * term.coeff
                for c, v in zip(term.cols, contrib):
                    A[r, int(c)] = A[r, int(c)] + v


@dataclass
class TMStamp:
    """Precomputed transfer-matrix stamp for one ``TRANSFER_MATRIX`` element (theory.md s12.7).

    The element's acoustic rows are **overwritten** (like a duct's phase relations) with
    ``w_down = TM(omega) . w_up`` -- the user 2-port response replacing the linearized
    area-change jump.  ``w = (f, g, h)`` are the characteristic amplitudes read along each
    edge's own arrow: port ``e_up`` (the first incident edge) is the input, ``e_down`` the
    output.  Row ``i`` of the relation lands on ``rows[i]``; the ``h`` (entropy) relation of
    a 3x3 matrix sits on the flow-downstream edge's transport row, exactly like the duct.
    """

    rows: tuple  # residual rows overwritten: (node row 0, node row 1[, downstream transport])
    e_up: int  # arrow port 0 -- the transfer-matrix input edge
    e_down: int  # arrow port 1 -- the transfer-matrix output edge
    up_cols: tuple  # the 3 acoustic columns of e_up
    down_cols: tuple  # the 3 acoustic columns of e_down
    L_up: np.ndarray  # 3x3 dx_to_char at e_up (w_up = L_up . dx_up)
    L_down: np.ndarray  # 3x3 dx_to_char at e_down
    transfer: object  # nefes.perturbation.matrix.TransferMatrix (evaluated at f = omega/2pi)
    N: int  # matrix dimension: 2 (acoustic) or 3 (with entropy)
    node: int  # the element (for diagnostics)
    analytic: bool  # whether TM(omega) continues to complex frequency (stability)
    max_delay: float  # longest pure delay [s] carried by the continuation (contour clamp)


def build_tm_stamps(prob, x_bar, K, u_floor=1e-8, cals=None):
    """Precompute the transfer-matrix stamps for every ``TRANSFER_MATRIX`` element.

    Reads ``prob.node_transfer_matrix`` (the
    :class:`~nefes.perturbation.matrix.TransferMatrix` descriptors) and resolves, at the
    frozen mean state, each element's overwritten rows, the two faces' characteristic maps
    ``L_e`` and the arrow port ordering.  Returns the list of :class:`TMStamp`.
    """
    tms = getattr(prob, "node_transfer_matrix", ()) or ()
    if not any(t is not None for t in tms):
        return []
    est = states_table(prob, x_bar)
    ns = int(prob.n_solve)
    tr0 = int(prob.transport_row0)
    stamps = []
    for n in range(int(prob.n_nodes)):
        tm = tms[n]
        if tm is None or getattr(tm, "is_unknown", False):
            # an unknown-marker element stays acoustically an isentropic area change (its J_alg
            # rows) until identification resolves the matrix; nothing to stamp.
            continue
        base = int(prob.row_ptr[n])
        deg = int(prob.row_ptr[n + 1]) - base
        if deg != 2:
            raise ValueError(f"a transfer-matrix element at node {n} must be a 2-port; got degree {deg}")
        N = int(getattr(tm, "n", 0))
        if N not in (2, 3):
            raise ValueError(
                f"element transfer matrix at node {n} must be 2x2 (acoustic) or 3x3 (with entropy); got N={N}"
            )
        e0 = int(prob.col_edge[base])
        s0 = int(prob.orient[base])
        e1 = int(prob.col_edge[base + 1])
        s1 = int(prob.orient[base + 1])
        r0 = int(prob.node_row_ptr[n])

        def _L(e):
            return dx_to_char(
                float(est[ES_RHO, e]),
                float(est[ES_C, e]),
                float(est[ES_U, e]),
                float(est[ES_P, e]),
                float(est[ES_AREA, e]),
                K,
                None if cals is None else cals[e],
            )

        # flow-downstream edge (oriented mdot leaving the node): where the h-relation sits,
        # matching the duct convention (leaves the inflow edge's transport row free).
        mdot1 = s1 * float(est[ES_MDOT, e1])
        mdot0 = s0 * float(est[ES_MDOT, e0])
        e_down_flow = e1 if mdot1 > 0.0 else (e0 if mdot0 > 0.0 else e1)

        rows = (r0, r0 + 1, tr0 + e_down_flow) if N == 3 else (r0, r0 + 1)
        stamps.append(
            TMStamp(
                rows=rows,
                e_up=e0,
                e_down=e1,
                up_cols=tuple(ns * e0 + v for v in range(3)),
                down_cols=tuple(ns * e1 + v for v in range(3)),
                L_up=_L(e0),
                L_down=_L(e1),
                transfer=tm,
                N=N,
                node=n,
                analytic=bool(getattr(tm, "analytic", False)),
                max_delay=float(getattr(tm, "max_delay", 0.0)),
            )
        )
    return stamps


def _tm_block(transfer, L_up, N, freq):
    """The upstream coefficient block ``-(TM(freq) . L_up[:N])`` (shape ``N x 3``)."""
    T = np.asarray(transfer(freq), dtype=np.complex128)
    T = T.reshape(N, N) if T.size == N * N else T[0]
    return -(T @ L_up[:N, :])


def stamp_transfer_matrix(A, omega, tm_stamps, u_floor=1e-8, skip_entropy=False):
    """Apply the transfer-matrix relations to LIL matrix ``A`` in place (overwrite).

    For each element, evaluates ``TM(omega/2pi)`` and overwrites its rows with
    ``L_down . dx_down - TM . L_up . dx_up = 0`` (one row per characteristic).  Under
    ``skip_entropy`` (isentropic assembly) the entropy (``h``, row 2) relation is omitted --
    :func:`stamp_isentropic` pins that row to ``h = 0`` instead, so the element is treated as
    a 2-wave acoustic 2-port regardless of the matrix dimension.
    """
    if not tm_stamps:
        return
    freq = omega / (2.0 * np.pi)  # transfer matrices are in Hz (project convention)
    for st in tm_stamps:
        n_written = 2 if skip_entropy else st.N
        block_up = _tm_block(st.transfer, st.L_up, st.N, freq)  # (N, 3)
        for i in range(n_written):
            _set_row(A, st.rows[i], st.up_cols, block_up[i, :], st.down_cols, st.L_down[i, :].astype(np.complex128))


def _terminal_scalar_seats(prob, t, bc, e, specify, freq):
    """Driven reacting-scalar waves seated at a genuine-inflow terminal: ``(row, cols, coeff, rhs)``.

    A transported composition scalar convects at the mean speed ``u`` like the entropy wave, so
    it is to-specify exactly when the entropy wave is -- at a genuine inflow.  Seating it is a
    diagonal identity on its own transport row (``xi'_edge = amplitude``); the seat itself is
    decoupled, but the seated wave *does* radiate sound downstream wherever the linearization is
    inherited (the full Jacobian carries composition -> acoustic -- a flame, an area change, an
    inherited compact nozzle).  Raises if a scalar drive is requested where the convected waves are
    outgoing, or names a scalar the network does not transport.
    """
    families = [f for f in bc.driven if f not in ("acoustic", "entropy")]
    if not families:
        return ()
    names = tuple(getattr(prob, "scalar_names", ()) or ())
    if 2 not in specify:  # h (entropy/convected) index: the convected waves are arriving here
        raise ValueError(
            f"cannot drive scalar wave(s) {families} at this terminal: the convected waves leave the "
            "domain here -- drive a scalar only at a genuine inflow."
        )
    ns, E, tr0 = int(prob.n_solve), int(prob.n_edges), int(prob.transport_row0)
    seats = []
    for fam in families:
        if fam not in names:
            raise ValueError(f"unknown scalar wave family {fam!r}; the network transports {list(names)}.")
        j = names.index(fam)  # 0-based over the transported scalars
        row = tr0 + (j + 1) * E + e  # this scalar's transport row on the terminal edge
        col = ns * e + 3 + j  # this scalar's column (band-1 var 3 + j)
        seats.append((row, (col,), np.array([1.0 + 0.0j]), complex(bc._drive_amplitude(fam, freq))))
    return seats


def _terminal_closure(prob, est, K, t, bc, omega, cal=None):
    """Per to-specify wave at terminal ``t``: ``(row, cols, coeff_block, rhs)``.

    Builds the matrix closure ``w[specify] = A(omega) @ w[arriving] + b`` via
    :meth:`PerturbationBC.closure` over the mean-state wave partition
    (:func:`matrices.partition`), and maps each to-specify wave to its matrix row -- the acoustic
    wave on the boundary node row, the (inflow) entropy wave on the edge's transport row.  The
    length-3 coefficient block over the edge's acoustic columns is ``L_e[specify] - sum_j A[.,j]
    L_e[arriving_j]`` and ``rhs`` its forcing.  Any **driven reacting-scalar** waves are appended
    as diagonal seats on their transport rows (see :func:`_terminal_scalar_seats`).  ``cal`` (see
    :func:`characteristics.dq_to_dx`) is the terminal edge's reacting caloric row.
    """
    e = t.edge
    rho, c, u = float(est[ES_RHO, e]), float(est[ES_C, e]), float(est[ES_U, e])
    p, area = float(est[ES_P, e]), float(est[ES_AREA, e])
    m_out = (u / c) if not t.at_tail else (-u / c)  # outward-normal mean Mach
    specify, arriving = partition(u, c, "a" if t.at_tail else "b")
    freq = omega / (2.0 * np.pi)  # BC carriers (tables/callables) are in Hz; operator stays in omega
    Amat, bvec = bc.closure(freq, rho, c, u, m_out, K, specify, arriving, p=p)
    L_e = dx_to_char(rho, c, u, p, area, K, cal)
    acou_cols = tuple(int(prob.n_solve) * e + v for v in range(3))
    out = []
    for i, ch in enumerate(specify):
        row = t.row if ch in (0, 1) else int(prob.transport_row0) + e  # acoustic -> node, entropy -> transport
        coeff = L_e[ch, :].astype(np.complex128)
        for j, cha in enumerate(arriving):
            coeff = coeff - Amat[i, j] * L_e[cha, :]
        out.append((row, acou_cols, coeff, complex(bvec[i])))
    out.extend(_terminal_scalar_seats(prob, t, bc, e, specify, freq))
    return out


def stamp_boundaries(A, omega, prob, x_bar, cals=None):
    """Terminal closure face ``A(omega)`` (theory.md s12.4) onto LIL ``A``.

    Each single-port terminal carrying an explicit ``PerturbationBC`` (anything but
    ``inherit``) has the rows of its to-specify waves overwritten with the matrix
    closure ``w[specify] = A(omega) @ w[arriving] + b`` (``b`` built by
    :func:`boundary_forcing`).  The acoustic to-specify wave lands on the boundary node
    row; at an inflow (tail) terminal the incoming entropy wave is also seated, on that
    edge's transport row -- always a duct *tail* edge, so it never collides with the
    duct stamp's head-edge entropy phase (theory.md s6.2).  Terminals left at
    ``inherit`` keep their linearized mean boundary row from ``J_alg``.  ``cals``
    (optional): per-edge reacting caloric rows (:func:`characteristics.edge_caloric`).
    """
    node_bc = prob.node_bc
    if not node_bc:
        return
    est = states_table(prob, x_bar)
    K = float(prob.tf[0]) / float(prob.tf[1])
    for t in find_terminals(prob):
        bc = node_bc[t.node] if t.node < len(node_bc) else None
        if bc is None or not getattr(bc, "stamps_terminal", False):
            continue
        cal = None if cals is None else cals[t.edge]
        for row, cols, coeff, _rhs in _terminal_closure(prob, est, K, t, bc, omega, cal):
            _set_row(A, row, cols, coeff, (), ())


def stamp_isentropic(A, prob, est, K, skip_edges=(), cals=None):
    """Pin the entropy characteristic to zero on every edge: ``rho' = p'/c^2`` (isentropic).

    The entropy characteristic is ``h = rho' - p'/c^2`` (theory.md s9.1), so enforcing
    ``h_e = L_e[2, :] @ dx_e = 0`` on each edge ``e`` removes the convected entropy wave
    from ``A(omega)`` entirely -- the standard isentropic acoustic assumption, where density
    perturbations follow pressure alone.  Each edge's transport (entropy) row is overwritten
    with this constraint, so the operator keeps its size and the *same* solver / contour
    machinery applies unchanged.

    ``skip_edges`` are left physical (not pinned): the energy/transport rows an active
    flame writes its heat-release source onto (theory.md s12.4).  Dropping convected
    entropy in the ducts while keeping the flame's energy jump is the standard
    "acoustic network with a compact flame" model -- the flame still adds heat, but the
    entropy spot it sheds is not convected.

    The constraint is ``omega``-independent (no phase), so applied after the duct and
    boundary stamps it cleanly overrides whatever they wrote on the entropy rows, and it is
    folded into the frozen base of the fast assembler (:class:`operator._AssemblyPlan`).
    """
    skip = set(int(e) for e in skip_edges)
    ns = int(prob.n_solve)
    tr0 = int(prob.transport_row0)
    for e in range(int(prob.n_edges)):
        if e in skip:
            continue
        L_e = dx_to_char(
            float(est[ES_RHO, e]),
            float(est[ES_C, e]),
            float(est[ES_U, e]),
            float(est[ES_P, e]),
            float(est[ES_AREA, e]),
            K,
            None if cals is None else cals[e],
        )
        cols = tuple(ns * e + v for v in range(3))
        # transport row of edge e becomes  h_e = L_e[2, :] . dx_e = 0
        _set_row(A, tr0 + e, cols, L_e[2, :].astype(np.complex128), (), ())


def boundary_forcing(prob, x_bar, omega, cals=None):
    """Right-hand side ``b(omega)`` for the explicitly-closed terminals.

    The forcing of each to-specify wave (a driven acoustic wave on the node row, incoming
    entropy on the inflow-side transport row); zero everywhere else.  Mirrors the rows
    :func:`stamp_boundaries` overwrites, via the same :func:`_terminal_closure`.  ``cals``
    (optional): per-edge reacting caloric rows (:func:`characteristics.edge_caloric`).
    """
    b = np.zeros(prob.n_col, dtype=np.complex128)
    node_bc = prob.node_bc
    if not node_bc:
        return b
    est = states_table(prob, x_bar)
    K = float(prob.tf[0]) / float(prob.tf[1])
    for t in find_terminals(prob):
        bc = node_bc[t.node] if t.node < len(node_bc) else None
        if bc is None or not getattr(bc, "stamps_terminal", False):
            continue
        cal = None if cals is None else cals[t.edge]
        for row, _cols, _coeff, rhs in _terminal_closure(prob, est, K, t, bc, omega, cal):
            b[row] = rhs
    return b


@dataclass
class StorageStamp:
    """One element's contribution to the storage block ``M`` (frozen at the mean state).

    Each ``(row, col, val)`` triplet adds ``i*omega*val`` onto ``A[row, col]`` -- the
    linearized transient accumulation ``d/dt integral_V U`` that vanishes at steady
    state.  ``vals`` are the frequency-independent storage coefficients (the
    ``i*omega`` is applied at assembly).  The contribution **adds** onto the rows
    ``J_alg`` already populates (it never overwrites them), exactly like the dynamic
    source ``S(omega)``; so a storage element's conservation row stays the converged
    jump plus its accumulation term.
    """

    node: int  # the storage element (for diagnostics)
    rows: np.ndarray  # int[k]  residual rows that accumulate storage
    cols: np.ndarray  # int[k]  columns ns*edge + v of the storage coefficients
    vals: np.ndarray  # complex[k]  the M-entries (i*omega applied at assembly)


def _cavity_storage(prob, est, n, K=None, cals=None):
    """Compliance storage of a finite-volume cavity (:func:`~nefes.elements.catalog.cavity`).

    The cavity conserves mass: ``d/dt(rho_c V) = mass inflow``.  With the oriented port
    mass flow defined outward (the wall residual ``s0*mdot = 0`` of ``J_alg``) and the
    cavity gas compressing isentropically about the stagnant mean, the linearization is

        ``s0*mdot' + i*omega*V*drho' = 0``,   ``drho' = p'/c^2``,

    so the storage adds ``i*omega*(V/c^2)*p'`` onto the cavity's single mass row -- the
    lumped compliance ``C = V/(rho c^2)`` (the mass row ties ``mdot'`` to ``p'`` as
    ``mdot' = -i*omega*C*rho*p'``... i.e. volume velocity ``= i*omega*C*p'``).

    The energy equation is *not* an independent store in the adiabatic compact limit
    (theory.md s12.5): the isentropic relation ``drho' = p'/c^2`` it provides is folded
    straight into the mass storage here, which is why a single pressure-column entry
    suffices and the result is the same in the full 3-wave and isentropic operators.
    ``c`` is the local sound speed (the equilibrium/frozen one for a reacting cavity),
    so the compliance carries the actual thermodynamics; a stagnant cavity exchanges no
    convected entropy/composition (``u ~ 0``), so those characteristics are not stored.
    """
    base = int(prob.row_ptr[n])
    e0 = int(prob.col_edge[base])
    pb = int(prob.npar_fptr[n])
    V = float(prob.npar_f[pb])  # cavity volume (catalog.cavity fparams[0])
    ns = int(prob.n_solve)
    c = float(est[ES_C, e0])
    r0 = int(prob.node_row_ptr[n])  # the cavity's single (mass) residual row
    cols = np.array([ns * e0 + 1], dtype=np.intp)  # the pressure column of the cavity edge
    vals = np.array([V / (c * c)], dtype=np.complex128)
    rows = np.array([r0], dtype=np.intp)
    return StorageStamp(node=n, rows=rows, cols=cols, vals=vals)


# fparams offset where each inline 2-port element's storage block
# ``[l_up, l_down, end_correction]`` begins (after the element's own physics params).
_INLINE_STORAGE_OFFSET = {
    ISEN_AREA_CHANGE: 0,  # fparams = [l_up, l_down, end]
    SUDDEN_AREA_CHANGE: 1,  # fparams = [cc, l_up, l_down, end]
    LOSS: 2,  # fparams = [K, ref_port, l_up, l_down, end]
    LINEAR_RESISTANCE: 1,  # fparams = [R, l_up, l_down, end]
}


def _inline_storage(prob, est, n, K=None, cals=None):
    """Storage of an inline 2-port pressure element (area change / loss / resistance).

    The element's optional half-lengths ``l_up``/``l_down`` (catalog ``_storage_block``)
    give it both a **compliance** and an **inertance** (theory in
    ``scratch/inertance-end-correction-theory.md`` and theory.md s12.5):

    * **compliance** -- per-port reduced length on the mass row: each side stores
      ``l_i * A_i`` of gas, contributing ``+ l_i * A_i / c_i^2`` onto the mass row at that
      port's pressure column (the cavity's ``V/c^2``, distributed over the two ports that
      sit at *different* perturbation pressures);
    * **inertance** -- the series effective length ``L_eff = l_up + l_down + end_correction``
      on the pressure-drop row (``r0 + 1``), referenced to the throat (smaller) area:
      ``+ s0 * L_eff / A_ref`` onto the through-flow ``mdot`` column of port 0.  The sign
      ``s0 = orient[base]`` carries the port-0 orientation, because the Nefes pressure row is
      written in the upstream-minus-downstream sense with the through-flow ``-s0*mdot_e0``;
      the result is the reactive dual of the linear-resistance term ``-R*mdot_through``.

    ``end_correction`` adds to the inertance only (the entrained near-field mass the
    geometric length omits).  All lengths default to zero -> an empty stamp (the element is
    the lengthless jump it was before).
    """
    rid = int(prob.node_rid[n])
    off = _INLINE_STORAGE_OFFSET[rid]
    base = int(prob.row_ptr[n])
    pb = int(prob.npar_fptr[n])
    ns = int(prob.n_solve)
    e0 = int(prob.col_edge[base])
    s0 = int(prob.orient[base])
    e1 = int(prob.col_edge[base + 1])
    r_mass = int(prob.node_row_ptr[n])  # row 0: mass balance
    r_press = r_mass + 1  # row 1: the pressure-drop / momentum row
    l_up = float(prob.npar_f[pb + off])
    l_down = float(prob.npar_f[pb + off + 1])
    end = float(prob.npar_f[pb + off + 2])

    rows, cols, vals = [], [], []
    # compliance: per-port reduced length on the mass row (each port's pressure column)
    for e_i, l_i in ((e0, l_up), (e1, l_down)):
        if l_i > 0.0:
            A_i = float(est[ES_AREA, e_i])
            c_i = float(est[ES_C, e_i])
            rows.append(r_mass)
            cols.append(ns * e_i + 1)  # pressure variable
            vals.append(l_i * A_i / (c_i * c_i))
    # inertance: series effective length on the pressure row, throat-referenced
    L_eff = l_up + l_down + end
    if L_eff > 0.0:
        a0 = float(est[ES_AREA, e0])
        a1 = float(est[ES_AREA, e1])
        A_ref = a0 if a0 <= a1 else a1  # throat (smaller) area
        rows.append(r_press)
        cols.append(ns * e0 + 0)  # through-flow mass-flow variable of port 0
        vals.append(s0 * L_eff / A_ref)

    if not rows:
        return None
    return StorageStamp(
        node=n,
        rows=np.array(rows, dtype=np.intp),
        cols=np.array(cols, dtype=np.intp),
        vals=np.array(vals, dtype=np.complex128),
    )


def _manifold_storage(prob, est, n, K=None, cals=None):
    """Chamber compliance of a finite-volume manifold (a
    :func:`~nefes.elements.catalog.junction` or :func:`~nefes.elements.catalog.splitter` plenum).

    The manifold's ``fparams = [volume]`` (catalog ``_manifold_block``) give it one store
    (theory ``scratch/inertance-end-correction-theory.md`` s5): a non-zero chamber ``volume``
    is the lumped ``C = V/(rho c^2)``.  All ports share one pressure (the ``p_0 = p_i``
    coupling rows tie them), so it is a single ``+ V/c^2`` on the mass row at the port-0
    pressure column -- the cavity rule with through-flow.

    A branch's neck inertance is not a manifold parameter: model it as an explicit neck duct
    on the branch (its inertance then rides ``P(omega)``).  ``volume = 0`` (the default) ->
    no storage.  Inert in the mean flow (storage is the ``i*omega*M`` face only).
    """
    pb = int(prob.npar_fptr[n])
    base = int(prob.row_ptr[n])
    ns = int(prob.n_solve)
    r0 = int(prob.node_row_ptr[n])  # the manifold's mass-balance row

    # compliance: chamber volume on the common (port-0) pressure column
    V = float(prob.npar_f[pb])  # junction/splitter chamber volume (catalog _manifold_block fparams[0])
    if V <= 0.0:
        return None
    e0 = int(prob.col_edge[base])
    c0 = float(est[ES_C, e0])
    return StorageStamp(
        node=n,
        rows=np.array([r0], dtype=np.intp),
        cols=np.array([ns * e0 + 1], dtype=np.intp),  # common pressure column
        vals=np.array([V / (c0 * c0)], dtype=np.complex128),
    )


# Per-element storage builders, keyed by residual id.  A builder takes
# ``(prob, est, n, K, cals)`` and returns a :class:`StorageStamp` (or ``None``).  This
# is the extension point for the ``M`` block: a storage-bearing element registers its
# ``d/dt integral_V U`` contribution here.  The cavity is the 1-port compliance; the
# inline pressure elements carry per-port compliance + series inertance; the manifolds
# carry a chamber compliance (theory: scratch/inertance-end-correction-theory.md).
_STORAGE_BUILDERS = {
    CAVITY: _cavity_storage,
    ISEN_AREA_CHANGE: _inline_storage,
    SUDDEN_AREA_CHANGE: _inline_storage,
    LOSS: _inline_storage,
    LINEAR_RESISTANCE: _inline_storage,
    JUNCTION: _manifold_storage,
    SPLITTER: _manifold_storage,
}


def build_storage_stamps(prob, x_bar, K, cals=None):
    """Per-element storage stamps -- the contributors to the operator's ``M`` block.

    Iterates the network and asks each storage-bearing element (one registered in
    :data:`_STORAGE_BUILDERS` by residual id) for its ``d/dt integral_V U`` triplets at
    the frozen mean state ``x_bar``.

    Parameters
    ----------
    prob : CompiledProblem
        The compiled network.
    x_bar : ndarray
        Converged mean-flow solve state, shape ``(n_solve, E)``.
    K : float
        The perfect-gas caloric constant ``cp/R`` (fallback when ``cals`` is ``None``).
    cals : list, optional
        Per-edge reacting caloric rows (:func:`characteristics.edge_caloric`).

    Returns
    -------
    list of StorageStamp
        One per storage element (empty when the network carries none).
    """
    return storage_stamps_from_est(prob, states_table(prob, x_bar), K, cals)


def storage_stamps_from_est(prob, est, K=None, cals=None):
    """Per-element storage stamps from a precomputed mean edge-state table ``est``.

    The frozen-mean half of :func:`build_storage_stamps` (which calls this after building
    ``est`` from ``x_bar``).  Exposed so the acoustic-power ledger can recover each
    element's ``M`` contribution from a solved field without re-deriving ``est``.
    """
    if not _STORAGE_BUILDERS:
        return []
    stamps = []
    for n in range(int(prob.n_nodes)):
        builder = _STORAGE_BUILDERS.get(int(prob.node_rid[n]))
        if builder is None:
            continue
        st = builder(prob, est, n, K, cals)
        if st is not None and st.rows.size:
            stamps.append(st)
    return stamps


def build_storage(prob, x_bar, K=None, cals=None):
    """Storage block ``M`` (the ``d/dt integral_V U`` term dropped at steady state).

    Assembles the per-element storage stamps (:func:`build_storage_stamps`) into the
    sparse operator block that enters ``A(omega)`` as ``i*omega*M``.  All-zero (an
    empty CSC) when the network carries no storage element, so the fast assembler can
    test ``M.nnz`` to skip the storage fill.

    Parameters
    ----------
    prob : CompiledProblem
        The compiled network.
    x_bar : ndarray
        Converged mean-flow solve state, shape ``(n_solve, E)``.
    K : float, optional
        The perfect-gas caloric constant ``cp/R``; derived from ``prob.tf`` when omitted.
    cals : list, optional
        Per-edge reacting caloric rows (:func:`characteristics.edge_caloric`).

    Returns
    -------
    scipy.sparse.csc_matrix
        The storage block, shape ``(n_eq, n_col)``.
    """
    if K is None:
        K = float(prob.tf[0]) / float(prob.tf[1])
    n_eq, n_col = int(prob.n_eq), int(prob.n_col)
    stamps = build_storage_stamps(prob, x_bar, K, cals)
    if not stamps:
        return sp.csc_matrix((n_eq, n_col), dtype=np.complex128)
    rows = np.concatenate([st.rows for st in stamps])
    cols = np.concatenate([st.cols for st in stamps])
    vals = np.concatenate([st.vals for st in stamps]).astype(np.complex128)
    return sp.csc_matrix((vals, (rows, cols)), shape=(n_eq, n_col), dtype=np.complex128)
