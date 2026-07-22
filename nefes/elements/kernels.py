"""@njit element residual and donor kernels (dtype-generic).

A single switch on ``residual_id`` writes either an element's steady residual
rows (``node_residual``) or the value it donates to an adjacent edge for a given
advected scalar (``node_donor``, applied to every scalar, not only enthalpy).
Every residual is smooth in the flow state so the complex-step Jacobian stays
exact.  ``kappa`` is the artificial-resistance continuation coefficient driven
to zero by the solver.
"""

from numba import njit

from ..assembly.recover import ES_AREA, ES_C, ES_M, ES_MDOT, ES_P, ES_PT, ES_RHO, ES_U
from ..assembly.smooth import fischer_burmeister, smooth_abs, smooth_min, smooth_pos, smooth_step
from .ids import (
    CAVITY,
    CHOKED_NOZZLE_OUTLET,
    DUCT,
    FLAME_EQUILIBRIUM,
    FLAME_HEAT_RELEASE,
    FORCED_SPLITTER,
    ISEN_AREA_CHANGE,
    JUNCTION,
    LINEAR_RESISTANCE,
    LOSS,
    MASS_FLOW_INLET,
    MASS_FLOW_OUTLET,
    MASS_SOURCE,
    MIXER,
    P_OUTLET,
    PIPE,
    PT_INLET,
    SPLITTER,
    SUDDEN_AREA_CHANGE,
    TRANSFER_MATRIX,
    WALL,
)

# Mixer smooth-minimum tuning, as fractions of a port total pressure.  The ideal
# (recovery = 1) merge holds the node total pressure at the smallest inflow total pressure;
# MIX_MIN_SMOOTH is the smoothing width of that minimum (a small under-estimate that keeps the
# weakest feed feasible), and MIX_MIN_SEED seeds the running minimum above every port so the
# first inflow sets it.
MIX_MIN_SMOOTH = 0.005
MIX_MIN_SEED = 0.5


@njit(cache=True)
def node_donor(n, rid, s, marker_s, row_ptr, col_edge, orient, npar_f, npar_fptr, eps, mdot_e, phi_e):
    """Value of advected scalar ``s`` element ``n`` offers to an edge drawing from it.

    Advected scalars are band-1 rows: ``s = 0`` is total enthalpy, ``s >= 1`` the conserved
    composition scalars, and ``s == marker_s`` (when ``>= 0``) the transported burnt marker.
    Boundaries return their prescribed value, scalar-transparent elements pass the interior
    value, and interior elements return a mass-weighted smooth-upwind mix.

    Parameters
    ----------
    n : int
        Element (node) index.
    rid : int
        Residual id selecting the element type.
    s : int
        Advected-scalar index.
    marker_s : int
        Scalar index of the burnt marker, or negative if none.
    row_ptr : ndarray
        CSR-style offsets into ``col_edge``/``orient`` per node.
    col_edge : ndarray
        Incident edge index of each port.
    orient : ndarray
        Port orientation sign (+1/-1).
    npar_f : ndarray
        Flat float-parameter array.
    npar_fptr : ndarray
        Per-node offset into ``npar_f``.
    eps : float
        Smoothing scale of the upwind weighting.
    mdot_e : ndarray
        Per-edge mass flow.
    phi_e : ndarray
        Per-edge value of scalar ``s``.

    Returns
    -------
    float or complex
        The donated scalar value (dtype follows the flow state).
    """
    base = row_ptr[n]
    deg = row_ptr[n + 1] - base
    pb = npar_fptr[n]
    if rid == WALL or rid == CAVITY or rid == MASS_FLOW_OUTLET or rid == CHOKED_NOZZLE_OUTLET:
        # Scalar-transparent: the stagnant/wall leg inherits the interior scalar, so the
        # smooth-upwind transport row collapses to the interior donor.  The mass-flow and
        # choked outlets are outflow-only and prescribe no external scalar.
        return phi_e[col_edge[base]]
    if rid == MASS_FLOW_INLET or rid == PT_INLET or rid == P_OUTLET:
        # Boundary value: total enthalpy at pb+1 (from Tt at build time) then composition.
        return npar_f[pb + 1 + s]
    # Interior: mass-weighted smooth-upwind mix of the incoming port scalars.  The burnt
    # marker is the exception -- a reachability label, not a conserved quantity, so it rides
    # a sticky noisy-OR: a fresh stream must never dilute a burnt one.
    acc = phi_e[col_edge[base]] * 0.0
    w_sum = acc
    wphi_sum = acc
    unburnt = acc + 1.0  # marker noisy-OR: prod_i (1 - theta_i * b_i) over incoming ports
    for i in range(deg):
        ei = col_edge[base + i]
        si = orient[base + i]
        mdot_in = -si * mdot_e[ei]
        w = smooth_pos(mdot_in, eps)
        w_sum = w_sum + w
        wphi_sum = wphi_sum + w * phi_e[ei]
        if s == marker_s:
            # theta in [0, 1] is the smooth upwind indicator; a burnt incoming port
            # (theta * b -> 1) drives the product to 0 so the outgoing marker saturates to 1.
            theta = smooth_step(mdot_in, eps)
            unburnt = unburnt * (1.0 - theta * phi_e[ei])

    if s == marker_s:
        # Sticky burnt marker.  Endpoints are exact: all-fresh incoming gives b = 0, any
        # fully-burnt incoming gives b = 1.  The flat gate marker_gate keeps it out of the
        # acoustic operator.
        if rid == FLAME_EQUILIBRIUM:
            # Equilibrium flame outflow is fully burnt (b = 1); a constant donor, so
            # acoustically silent.
            return unburnt * 0.0 + 1.0
        if rid == MASS_SOURCE:
            # The injected stream is always incoming (theta = 1): fresh air leaves the OR
            # unchanged, injected burnt gas (e.g. EGR) sets it.
            return 1.0 - unburnt * (1.0 - npar_f[pb + 2 + s])
        return 1.0 - unburnt

    # Mass-averaged scalars: total enthalpy (s = 0) and mixture fractions (s >= 1).
    if rid == MASS_SOURCE:
        # Inline injection: mass-weighted mix of the interior inflow and the injected stream
        # (mass npar_f[pb+0], scalar npar_f[pb+2+s]).  The injected mass is always incoming,
        # so it conserves the advected scalar across the source.
        msrc = npar_f[pb + 0]
        w_sum = w_sum + msrc
        wphi_sum = wphi_sum + msrc * npar_f[pb + 2 + s]
        return wphi_sum / w_sum
    mix = wphi_sum / w_sum
    if rid == FLAME_HEAT_RELEASE and s == 0:
        # Heat-addition flame: raise the outflow total enthalpy by Q_dot / |mdot|.  smooth_abs
        # floors |mdot| so the jump stays bounded and smooth at zero flow; composition scalars
        # (s >= 1) pass through unchanged.
        Qdot = npar_f[pb + 0]
        mdot_mag = smooth_abs(mdot_e[col_edge[base]], eps)
        return mix + Qdot / mdot_mag
    return mix


@njit(cache=True)
def node_residual(n, rid, row_ptr, col_edge, orient, npar_f, npar_fptr, eps, eps_fb, kappa, est, R, node_row_ptr):
    """Write element ``n``'s ``deg`` residual rows into ``R`` at its row block.

    Parameters
    ----------
    n : int
        Element (node) index.
    rid : int
        Residual id selecting the element type.
    row_ptr : ndarray
        CSR-style offsets into ``col_edge``/``orient`` per node.
    col_edge : ndarray
        Incident edge index of each port.
    orient : ndarray
        Port orientation sign (+1/-1).
    npar_f : ndarray
        Flat float-parameter array.
    npar_fptr : ndarray
        Per-node offset into ``npar_f``.
    eps : float
        Smoothing scale of the upwind switches.
    eps_fb : float
        Smoothing scale of the Fischer-Burmeister choke complementarity.
    kappa : float
        Artificial-resistance continuation coefficient.
    est : ndarray
        Recovered edge-state table indexed by ``ES_*`` rows.
    R : ndarray
        Residual vector, written in place.
    node_row_ptr : ndarray
        Per-node offset of the element's row block in ``R``.

    Returns
    -------
    None
    """
    base = row_ptr[n]
    deg = row_ptr[n + 1] - base
    r0 = node_row_ptr[n]
    pb = npar_fptr[n]

    if rid == MASS_FLOW_INLET:
        e0 = col_edge[base]
        s0 = orient[base]
        R[r0] = s0 * est[ES_MDOT, e0] - npar_f[pb + 0]
        return

    if rid == WALL or rid == CAVITY:
        # WALL and CAVITY share the mean residual: impermeable, no mass crosses the face.
        # The cavity differs only acoustically (its finite volume populates the storage block M).
        e0 = col_edge[base]
        s0 = orient[base]
        R[r0] = s0 * est[ES_MDOT, e0]
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

    if rid == MASS_FLOW_OUTLET:
        # Prescribed outflow rate: -s0*mdot is the mass leaving the domain.  The acoustic
        # counterpart is mdot' = 0, a constant-mass-flow termination.
        e0 = col_edge[base]
        s0 = orient[base]
        R[r0] = -s0 * est[ES_MDOT, e0] - npar_f[pb + 0]
        return

    if rid == CHOKED_NOZZLE_OUTLET:
        # Compact choked nozzle of throat area A* (= npar_f[pb+0]): the throat is sonic, so the
        # outflow equals the critical mass flux for the interior isentropic total state.  The
        # application plane stays subsonic (M = 1 sits in the lumped throat), so the acoustic
        # operator is non-degenerate.
        #   mdot_out = rho_t c_t A* (2/(gamma+1))^((gamma+1)/(2(gamma-1)))
        # with rho_t, c_t the stagnation density/sound-speed and gamma = rho c^2 / p.
        e0 = col_edge[base]
        s0 = orient[base]
        rho = est[ES_RHO, e0]
        c = est[ES_C, e0]
        p = est[ES_P, e0]
        A_star = npar_f[pb + 0]
        mdot_out = -s0 * est[ES_MDOT, e0]
        M = -s0 * est[ES_M, e0]  # outflow-positive approach Mach
        gamma = rho * c * c / p
        stag = 1.0 + 0.5 * (gamma - 1.0) * M * M  # 1 + (g-1)/2 M^2
        rho_t = rho * stag ** (1.0 / (gamma - 1.0))
        c_t = c * stag**0.5
        expc = (gamma + 1.0) / (2.0 * (gamma - 1.0))
        mdot_crit = rho_t * c_t * A_star * (2.0 / (gamma + 1.0)) ** expc
        R[r0] = mdot_out - mdot_crit
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
                R[r0 + i] = est[ES_P, e0] - est[ES_P, ei] - kappa * (si * est[ES_MDOT, ei])
            else:
                R[r0 + i] = est[ES_PT, e0] - est[ES_PT, ei] - kappa * (si * est[ES_MDOT, ei])
        return

    if rid == MIXER:
        # Merge manifold that obeys the second law by never handing an outflow more total
        # pressure than the feeds possess.  Mass balance, then (deg - 1) rows tying each port's
        # *effective* total pressure to port 0's.  A port's effective total pressure removes its
        # inflow loss, interpolated by sigma = npar_f[pb+0] in [0, 1] between the full dump loss
        # (sigma = 0: the whole dynamic head p_t - p) and the ideal loss (sigma = 1: only the
        # excess over the weakest inflow, p_t - min_inflow p_t):
        #   loss_k = chi_k * [ (1 - sigma)*(p_t,k - p_k) + sigma*(p_t,k - pt_min) ]
        # with chi_k the smooth inflow indicator (1 in, 0 out) and pt_min the smooth minimum of
        # the inflow total pressures.  So sigma = 1 recovers the lossless splitter when
        # distributing (one inflow -> its own p_t, zero loss) and the minimum-entropy mixer when
        # merging (the outlet leaves at the weakest feed's total pressure), sigma = 0 fully
        # dissipates the dynamic head (a plenum), and at low Mach it collapses to the
        # common-pressure junction for any sigma.
        e0 = col_edge[base]
        sigma = npar_f[pb + 0]
        # A port total pressure sets the (relative) smoothing / seed scales; it is kept complex
        # (not .real) because it depends on the flow state, so the complex-step derivative must
        # carry it.
        pt_ref = est[ES_PT, e0]
        delta = MIX_MIN_SMOOTH * pt_ref
        excl = MIX_MIN_SEED * pt_ref
        # First pass: mass balance, and the smooth minimum of the inflow total pressures.  Each
        # outflow is lifted clear of the minimum by excl (its inflow weight chi ~ 0), so the
        # minimum is taken over the inflows only; smooth_min under-estimates, keeping the
        # feasibility ceiling on the safe side (pt_min <= every clear inflow's p_t).
        acc = est[ES_MDOT, e0] * 0.0
        pt_min = acc
        for i in range(deg):
            ei = col_edge[base + i]
            si = orient[base + i]
            acc = acc + si * est[ES_MDOT, ei]
            chi = smooth_step(-si * est[ES_MDOT, ei], eps)  # inflow indicator (1 in, 0 out)
            q = est[ES_PT, ei] + (1.0 - chi) * excl  # an outflow (chi -> 0) sits at p_t + excl
            pt_min = q if i == 0 else smooth_min(pt_min, q, delta)
        R[r0] = acc
        # Second pass: each port's effective total pressure, tied to port 0's.  smooth_pos floors
        # the ideal-loss term at zero, so a transiently over-estimated pt_min can never make the
        # inflow gain total pressure (which would pump the flow backward); at the converged state
        # pt_min <= p_t and the floor is inactive.
        s0 = orient[base]
        chi0 = smooth_step(-s0 * est[ES_MDOT, e0], eps)
        loss0 = chi0 * (
            (1.0 - sigma) * (est[ES_PT, e0] - est[ES_P, e0]) + sigma * smooth_pos(est[ES_PT, e0] - pt_min, delta)
        )
        pteff0 = est[ES_PT, e0] - loss0
        for i in range(1, deg):
            ei = col_edge[base + i]
            si = orient[base + i]
            chi = smooth_step(-si * est[ES_MDOT, ei], eps)
            loss = chi * (
                (1.0 - sigma) * (est[ES_PT, ei] - est[ES_P, ei]) + sigma * smooth_pos(est[ES_PT, ei] - pt_min, delta)
            )
            pteff = est[ES_PT, ei] - loss
            R[r0 + i] = pteff0 - pteff - kappa * (si * est[ES_MDOT, ei])
        return

    if rid == FORCED_SPLITTER:
        # Flow divider: one inflow at port 0 and (deg - 1) outflows.  The first (deg - 2)
        # outflows each carry a fixed fraction beta_i = npar_f[pb + i - 1] of the port-0
        # inflow; the last outflow carries the remainder at total-pressure continuity.  Every
        # row is linear in the flow state, so the complex-step Jacobian is exact and, with the
        # inflow direction fixed, no upwind switch is needed.
        e0 = col_edge[base]
        s0 = orient[base]
        acc = est[ES_MDOT, e0] * 0.0
        for i in range(deg):
            acc = acc + orient[base + i] * est[ES_MDOT, col_edge[base + i]]
        R[r0] = acc  # net mass balance
        mdot_in = -s0 * est[ES_MDOT, e0]  # mass entering the node at port 0
        for i in range(1, deg - 1):
            ei = col_edge[base + i]
            si = orient[base + i]
            beta = npar_f[pb + i - 1]
            # mass leaving the node at port i (= si*mdot_i) equals beta * inflow
            R[r0 + i] = si * est[ES_MDOT, ei] - beta * mdot_in
        ei = col_edge[base + deg - 1]
        si = orient[base + deg - 1]
        R[r0 + deg - 1] = est[ES_PT, e0] - est[ES_PT, ei] - kappa * (si * est[ES_MDOT, ei])
        return

    # ---- two-port interior elements ----
    e0 = col_edge[base]
    s0 = orient[base]
    e1 = col_edge[base + 1]
    s1 = orient[base + 1]
    R[r0] = s0 * est[ES_MDOT, e0] + s1 * est[ES_MDOT, e1]  # mass balance
    kappa_term = kappa * (s1 * est[ES_MDOT, e1])  # on ports[1].mdot_out

    a0 = est[ES_AREA, e0].real
    a1 = est[ES_AREA, e1].real
    if a0 <= a1:
        se, ss = e0, s0  # small port
        la = a1  # large area
    else:
        se, ss = e1, s1
        la = a0

    if rid == DUCT:
        R[r0 + 1] = est[ES_PT, e0] - est[ES_PT, e1] - kappa_term
        return

    if rid == FLAME_HEAT_RELEASE or rid == FLAME_EQUILIBRIUM:
        # Constant-area flame: mass balance + the momentum equation (rho u^2 + p) A = const,
        # the exact thin-flame jump.  Orientation-robust: s_i signs the net momentum+pressure
        # flux to zero, and mom_i = rho_i u_i^2 is analytic.  Equal areas, so a0 normalizes both.
        #
        # FLAME_HEAT_RELEASE: the perfect-gas closure carries kinetic energy, so mass, momentum
        #   and energy (with the Q_dot donor source) are all exact.
        # FLAME_EQUILIBRIUM: mass and energy (h_t) are exact; the reacting closure uses h ~ h_t
        #   (drops u^2/2), so the recovered rho/u carry an O(M^2) bias and the momentum jump is
        #   exact to O(M^4).
        # No heat-release source sits on this row -> acoustically passive in J_alg.
        mom0 = est[ES_MDOT, e0] * est[ES_U, e0] / a0
        mom1 = est[ES_MDOT, e1] * est[ES_U, e1] / a0
        R[r0 + 1] = s0 * (mom0 + est[ES_P, e0]) + s1 * (mom1 + est[ES_P, e1]) - kappa_term
        return

    if rid == MASS_SOURCE:
        # Inline mass injection (constant area).  Mass: the net outflow exceeds the inflow by
        # the injected mdot_src (npar_f[pb+0]).  Momentum: the constant-area balance carries the
        # injected axial momentum mdot_src * u_inj (npar_f[pb+1]); u_inj = 0 is transverse
        # injection.  Energy and composition enter through the donor mix above.  mom_i is analytic.
        mdot_src = npar_f[pb + 0]
        u_inj = npar_f[pb + 1]
        R[r0] = R[r0] - mdot_src  # mass balance: s0*mdot0 + s1*mdot1 = mdot_src
        mom0 = est[ES_MDOT, e0] * est[ES_U, e0] / a0
        mom1 = est[ES_MDOT, e1] * est[ES_U, e1] / a0
        R[r0 + 1] = s0 * (mom0 + est[ES_P, e0]) + s1 * (mom1 + est[ES_P, e1]) - mdot_src * u_inj / a0 - kappa_term
        return

    if rid == ISEN_AREA_CHANGE or rid == TRANSFER_MATRIX:
        # TRANSFER_MATRIX shares the isentropic area-change mean jump (mass + energy conserved,
        # isentropic, area change allowed); it differs only in the perturbation layer, where its
        # acoustic rows are replaced by a user transfer matrix.
        m_in = -ss * est[ES_M, se]  # Mach at small port, oriented into element
        sub_margin = 1.0 - m_in
        pt_large = est[ES_PT, e1] if se == e0 else est[ES_PT, e0]
        loss = (est[ES_PT, se] - pt_large) / est[ES_PT, se]
        row = fischer_burmeister(sub_margin, loss, eps_fb) * est[ES_PT, se]
        R[r0 + 1] = row - kappa_term
        return

    if rid == SUDDEN_AREA_CHANGE:
        r_mom = (
            (est[ES_MDOT, e1] * est[ES_U, e1] + est[ES_P, e1] * a1)
            - (est[ES_MDOT, e0] * est[ES_U, e0] + est[ES_P, e0] * a0)
            - est[ES_P, se] * (a1 - a0)
        )
        r_mom = -r_mom / la
        # Reverse (large -> small) contraction: the jet necks to a vena contracta of area
        # Cc*A_small then Borda-expands back to the small pipe, the only lossy step, giving a
        # downstream-referenced total-pressure loss K_c*(1/2 rho u^2)_small with K_c = (1/Cc - 1)^2.
        # Cc = 1 (default) is the loss-free contraction.  The (1/2 rho u^2) head is the
        # incompressible reduction of the Borda balance, so the jump is accurate to O(M^2).
        cc = npar_f[pb + 0]
        k_contr = (1.0 / cc - 1.0) ** 2
        q_small = 0.5 * est[ES_RHO, se] * est[ES_U, se] * est[ES_U, se]
        loss_dir = 1.0 if se == e1 else -1.0  # orient the PT drop onto the small port
        r_isen = (est[ES_PT, e0] - est[ES_PT, e1]) - loss_dir * k_contr * q_small
        mdot_in_small = -ss * est[ES_MDOT, se]
        # xi switches momentum (forward expansion: Borda loss) <-> the contraction branch.  Even
        # when xi is saturated, its derivative leaks the residual gap (r_mom - r_isen) into the
        # frozen perturbation Jacobian by O(eps); set this element's eps small (ElementSpec.eps)
        # for one-directional flow to recover the exact jump.
        xi = smooth_step(mdot_in_small, eps)
        R[r0 + 1] = xi * r_mom + (1.0 - xi) * r_isen - kappa_term
        return

    if rid == LOSS:
        K = npar_f[pb + 0]
        # ref_port (npar_f[pb+1]) selects which port's area the loss coefficient K is referenced
        # to; matters only when the ports differ in area.
        ar = a0 if npar_f[pb + 1] < 0.5 else a1
        rho_avg = 0.5 * (est[ES_RHO, e0] + est[ES_RHO, e1])
        denom = rho_avg * ar
        # Through-flow, positive in the e0 -> e1 sense; the port orientation keeps the loss sign
        # correct regardless of how the two edges were wired.
        mdot_through = -s0 * est[ES_MDOT, e0]
        u_ref = mdot_through / denom
        u_abs = (u_ref * u_ref + (eps / denom) ** 2) ** 0.5
        q_signed = 0.5 * rho_avg * u_ref * u_abs
        R[r0 + 1] = est[ES_PT, e0] - est[ES_PT, e1] - K * q_signed - kappa_term
        return

    if rid == PIPE:
        # Length-bearing constant-area friction pipe.  fparams =
        # [length, diameter, friction_factor, formulation], with the Darcy coefficient
        # K = f * L / D.  The formulation slot is a build-time constant (never a function of the
        # flow state), so branching on it does not break the complex-step Jacobian.
        K = npar_f[pb + 2] * npar_f[pb + 0] / npar_f[pb + 1]  # f * L / D
        mdot_through = -s0 * est[ES_MDOT, e0]  # +ve in the e0 -> e1 sense
        if npar_f[pb + 3] < 0.5:
            # Darcy-Weisbach (Greyvenstein-Laurie): the total-pressure head, the same smooth
            # signed quadratic as LOSS.  Constant area, so a0 normalizes the through-flow.
            rho_avg = 0.5 * (est[ES_RHO, e0] + est[ES_RHO, e1])
            denom = rho_avg * a0
            u_ref = mdot_through / denom
            u_abs = (u_ref * u_ref + (eps / denom) ** 2) ** 0.5
            q_signed = 0.5 * rho_avg * u_ref * u_abs
            R[r0 + 1] = est[ES_PT, e0] - est[ES_PT, e1] - K * q_signed - kappa_term
            return
        # Momentum: the segment balance of the static-pressure and axial-momentum flux
        # (p + rho u^2) against the distributed wall head, the latter approximated by the
        # endpoint average of the signed (1/2) rho u|u|.  Written entirely in port indices, so
        # it is invariant to how the two edges were wired.  Refining a chain of these segments
        # converges to compressible Fanno flow, which the lumped head cannot reach.
        flux0 = est[ES_P, e0] + est[ES_MDOT, e0] * est[ES_U, e0] / a0
        flux1 = est[ES_P, e1] + est[ES_MDOT, e1] * est[ES_U, e1] / a1
        mdot_abs = smooth_abs(mdot_through, eps)
        q0 = 0.5 * mdot_through * mdot_abs / (est[ES_RHO, e0] * a0 * a0)
        q1 = 0.5 * mdot_through * mdot_abs / (est[ES_RHO, e1] * a1 * a1)
        R[r0 + 1] = flux0 - flux1 - K * 0.5 * (q0 + q1) - kappa_term
        return

    if rid == LINEAR_RESISTANCE:
        # Linear flow resistance: Pt_in - Pt_out = R_lin * mdot_through (R_lin >= 0, npar_f[pb+0]).
        # Unlike the quadratic LOSS it stays active at zero mean flow (a screen/perforate/damper
        # in a quiescent network).  mdot_through is +ve in the e0 -> e1 sense, so the drop reverses
        # with the flow; linear in the flow state, so the complex-step Jacobian is exact.
        r_lin = npar_f[pb + 0]
        mdot_through = -s0 * est[ES_MDOT, e0]
        R[r0 + 1] = est[ES_PT, e0] - est[ES_PT, e1] - r_lin * mdot_through - kappa_term
        return
