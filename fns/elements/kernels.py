"""@njit element residual and donor-enthalpy kernels (dtype-generic).

Ported from the prototype ``elements.py`` into a single switch on
``residual_id``.  Every residual is smooth in the flow state (only
``fns.smooth`` primitives; the lone ``.real`` comparisons are on areas, which
are fixed real parameters, never flow state).  ``stab`` is the vanishing-friction
homotopy coefficient added to interior pressure rows.
"""

from numba import njit

from ..smooth import smooth_step, smooth_pos, fischer_burmeister
from ..derive import ES_MDOT, ES_P, ES_HT, ES_RHO, ES_U, ES_M, ES_PT, ES_AREA
from .ids import (
    MASS_FLOW_INLET,
    PT_INLET,
    P_OUTLET,
    ISEN_AREA_CHANGE,
    SUDDEN_AREA_CHANGE,
    LOSS,
    JUNCTION,
    SPLITTER,
    DUCT,
)


@njit(cache=True)
def node_donor(n, rid, row_ptr, col_edge, orient, npar_f, npar_fptr, tf, eps, est):
    """Total enthalpy this element offers to an edge drawing from it."""
    base = row_ptr[n]
    deg = row_ptr[n + 1] - base
    cp = tf[0]
    if rid == MASS_FLOW_INLET or rid == PT_INLET or rid == P_OUTLET:
        return cp * npar_f[npar_fptr[n] + 1]  # cp * Tt(_backflow)
    # interior: mass-weighted smooth-upwind mix of the incoming port enthalpies
    acc = est[ES_MDOT, col_edge[base]] * 0.0
    w_sum = acc
    wh_sum = acc
    for i in range(deg):
        ei = col_edge[base + i]
        si = orient[base + i]
        mdot_in = -si * est[ES_MDOT, ei]
        w = smooth_pos(mdot_in, eps)
        w_sum = w_sum + w
        wh_sum = wh_sum + w * est[ES_HT, ei]
    return wh_sum / w_sum


@njit(cache=True)
def node_residual(n, rid, row_ptr, col_edge, orient, npar_f, npar_fptr, tf, eps, eps_fb, stab, est, R, node_row_ptr):
    """Write element n's ``deg`` residual rows into R starting at its row block."""
    base = row_ptr[n]
    deg = row_ptr[n + 1] - base
    r0 = node_row_ptr[n]
    pb = npar_fptr[n]

    if rid == MASS_FLOW_INLET:
        e0 = col_edge[base]
        s0 = orient[base]
        R[r0] = s0 * est[ES_MDOT, e0] - npar_f[pb + 0]
        return

    if rid == PT_INLET:
        e0 = col_edge[base]
        s0 = orient[base]
        mdot_out = s0 * est[ES_MDOT, e0]
        xi = smooth_step(mdot_out, eps)
        pt_spec = npar_f[pb + 0]
        R[r0] = xi * (est[ES_PT, e0] - pt_spec) + (1.0 - xi) * (est[ES_P, e0] - pt_spec)
        return

    if rid == P_OUTLET:
        e0 = col_edge[base]
        s0 = orient[base]
        mdot_in = -s0 * est[ES_MDOT, e0]
        m_in = -s0 * est[ES_M, e0]
        xi = smooth_step(mdot_in, eps)
        p_spec = npar_f[pb + 0]
        choked = fischer_burmeister(1.0 - m_in, (est[ES_P, e0] - p_spec) / p_spec, eps_fb) * p_spec
        R[r0] = xi * choked + (1.0 - xi) * (est[ES_PT, e0] - p_spec)
        return

    if rid == JUNCTION or rid == SPLITTER:
        acc = est[ES_MDOT, col_edge[base]] * 0.0
        for i in range(deg):
            acc = acc + orient[base + i] * est[ES_MDOT, col_edge[base + i]]
        R[r0] = acc
        e0 = col_edge[base]
        for i in range(1, deg):
            ei = col_edge[base + i]
            si = orient[base + i]
            if rid == JUNCTION:
                R[r0 + i] = est[ES_P, e0] - est[ES_P, ei] - stab * (si * est[ES_MDOT, ei])
            else:
                R[r0 + i] = est[ES_PT, e0] - est[ES_PT, ei] - stab * (si * est[ES_MDOT, ei])
        return

    # ---- two-port interior elements ----
    e0 = col_edge[base]
    s0 = orient[base]
    e1 = col_edge[base + 1]
    s1 = orient[base + 1]
    R[r0] = s0 * est[ES_MDOT, e0] + s1 * est[ES_MDOT, e1]  # mass balance
    stab_term = stab * (s1 * est[ES_MDOT, e1])  # on ports[1].mdot_out

    a0 = est[ES_AREA, e0].real
    a1 = est[ES_AREA, e1].real
    if a0 <= a1:
        se, ss = e0, s0  # small port
        la = a1  # large area
    else:
        se, ss = e1, s1
        la = a0

    if rid == DUCT:
        R[r0 + 1] = est[ES_PT, e0] - est[ES_PT, e1] - stab_term
        return

    if rid == ISEN_AREA_CHANGE:
        m_in = -ss * est[ES_M, se]  # Mach at small port, oriented into element
        sub_margin = 1.0 - m_in
        pt_large = est[ES_PT, e1] if se == e0 else est[ES_PT, e0]
        loss = (est[ES_PT, se] - pt_large) / est[ES_PT, se]
        row = fischer_burmeister(sub_margin, loss, eps_fb) * est[ES_PT, se]
        R[r0 + 1] = row - stab_term
        return

    if rid == SUDDEN_AREA_CHANGE:
        r_mom = (
            (est[ES_MDOT, e1] * est[ES_U, e1] + est[ES_P, e1] * a1)
            - (est[ES_MDOT, e0] * est[ES_U, e0] + est[ES_P, e0] * a0)
            - est[ES_P, se] * (a1 - a0)
        )
        r_mom = -r_mom / la
        r_isen = est[ES_PT, e0] - est[ES_PT, e1]
        mdot_in_small = -ss * est[ES_MDOT, se]
        # xi switches momentum (forward expansion: Borda loss) <-> isentropic
        # (contraction).  NOTE: even when xi is saturated (|mdot| >> eps, so the
        # mean flow is pure Borda), the switch's *derivative* leaks the large loss
        # residual (r_mom - r_isen) into the frozen perturbation Jacobian, biasing
        # the jump off the exact Borda by O(eps).  When the flow is one-directional,
        # set this element's eps small (ElementSpec.eps) to recover the exact jump.
        xi = smooth_step(mdot_in_small, eps)
        R[r0 + 1] = xi * r_mom + (1.0 - xi) * r_isen - stab_term
        return

    if rid == LOSS:
        K = npar_f[pb + 0]
        rho_avg = 0.5 * (est[ES_RHO, e0] + est[ES_RHO, e1])
        denom = rho_avg * a0
        u_ref = est[ES_MDOT, e0] / denom
        u_abs = (u_ref * u_ref + (eps / denom) ** 2) ** 0.5
        q_signed = 0.5 * rho_avg * u_ref * u_abs
        R[r0 + 1] = est[ES_PT, e0] - est[ES_PT, e1] - K * q_signed - stab_term
        return
