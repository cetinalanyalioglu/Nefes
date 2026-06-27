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

``build_storage`` is the storage ``M`` hook: zero in v1 (no finite-volume
element), but the home for the ``d/dt integral_V U`` block.

These run **above the @njit line** -- plain Python / SciPy.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .characteristics import dx_to_char, dq_to_dx
from .matrices import partition
from .verify import duct_nodes, verify_acoustic
from .terminals import find_terminals
from ..solver.control import states_table
from ..derive import ES_RHO, ES_C, ES_U, ES_P, ES_AREA, ES_MDOT, ES_T
from ..elements.ids import ACOUSTIC_VOLUME, FLAME_HEAT_RELEASE


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

    transfer: object  # fns.elements.dynamic_source.TransferFunction
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

    Reads ``prob.node_dynamic_source`` (the :class:`~fns.elements.dynamic_source.DynamicSource`
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


def build_storage(prob, x_bar):
    """Storage block ``M`` (the ``d/dt integral_V U`` term dropped at steady state).

    Zero in v1 (no finite-volume element); a volumetric element would populate
    its conservation rows here via a complex-step of a transient-flux operator.
    """
    vol = [n for n in range(prob.n_nodes) if int(prob.node_acoustic_id[n]) == ACOUSTIC_VOLUME]
    if vol:
        raise NotImplementedError("finite-volume storage M is a reserved v1 provision")
    return sp.csc_matrix((prob.n_eq, prob.n_col), dtype=np.complex128)
