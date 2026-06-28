"""@njit element residual and donor-enthalpy kernels (dtype-generic).

Ported from the prototype ``elements.py`` into a single switch on
``residual_id``.  Every residual is smooth in the flow state (only
``fns.smooth`` primitives; the lone ``.real`` comparisons are on areas, which
are fixed real parameters, never flow state).  ``kappa`` (kappa) is the
vanishing-friction homotopy coefficient: it stamps an artificial pressure drop
``kappa * mdot`` into interior pressure rows and is driven to zero by the solver.
"""

from numba import njit

from ..smooth import smooth_step, smooth_pos, smooth_abs, fischer_burmeister
from ..derive import ES_MDOT, ES_P, ES_RHO, ES_U, ES_M, ES_PT, ES_AREA, ES_C
from .ids import (
    MASS_FLOW_INLET,
    PT_INLET,
    P_OUTLET,
    MASS_FLOW_OUTLET,
    CHOKED_NOZZLE_OUTLET,
    WALL,
    CAVITY,
    ISEN_AREA_CHANGE,
    SUDDEN_AREA_CHANGE,
    LOSS,
    JUNCTION,
    SPLITTER,
    DUCT,
    FLAME_HEAT_RELEASE,
    FLAME_EQUILIBRIUM,
    MASS_SOURCE,
    LINEAR_RESISTANCE,
)


@njit(cache=True)
def node_donor(n, rid, s, row_ptr, col_edge, orient, npar_f, npar_fptr, tf, eps, mdot_e, phi_e):
    """Value of advected scalar ``s`` this element offers to an edge drawing from it.

    The advected scalars are band-1 rows ``2 .. 2 + n_scalars`` -- ``s = 0`` is
    total enthalpy ``h_t``, ``s >= 1`` are the conserved composition scalars
    ``Z_el[s-1]`` (reactive-flow D-2/D-4).  ``mdot_e`` and ``phi_e`` are the per-edge
    mass-flow and scalar-``s`` rows; element float params hold the boundary value
    (``Tt`` then the per-element feed/backflow composition, in order).
    """
    base = row_ptr[n]
    deg = row_ptr[n + 1] - base
    pb = npar_fptr[n]
    if rid == WALL or rid == CAVITY or rid == MASS_FLOW_OUTLET or rid == CHOKED_NOZZLE_OUTLET:
        # Scalar-transparent: the element offers the edge its own scalar value, so the
        # smooth-upwind transport row (theta = 1/2 at mdot = 0) collapses to the
        # interior donor -- the stagnant/wall leg simply inherits it (theory.md s12.6).
        # The mass-flow and choked-nozzle outlets are outflow-only (no backflow), so they
        # prescribe no external scalar -- the edge keeps the interior value either way.
        return phi_e[col_edge[base]]
    if rid == MASS_FLOW_INLET or rid == PT_INLET or rid == P_OUTLET:
        # boundary params carry the absolute total enthalpy h_t at pb+1 (converted
        # from Tt at build time, per backend) then the feed/backflow composition.
        return npar_f[pb + 1 + s]
    # interior: mass-weighted smooth-upwind mix of the incoming port scalar values
    acc = phi_e[col_edge[base]] * 0.0
    w_sum = acc
    wphi_sum = acc
    for i in range(deg):
        ei = col_edge[base + i]
        si = orient[base + i]
        mdot_in = -si * mdot_e[ei]
        w = smooth_pos(mdot_in, eps)
        w_sum = w_sum + w
        wphi_sum = wphi_sum + w * phi_e[ei]
    if rid == MASS_SOURCE:
        # Inline injection: the outflow scalar is the mass-weighted mix of the
        # interior incoming flow and the injected stream (mass-flow npar_f[pb+0],
        # scalar value npar_f[pb+2+s]: s=0 is the injected total enthalpy h_t,src,
        # s>=1 the injected elemental composition Z_src[s-1]).  The injected mass is
        # always incoming, so it adds a constant positive weight -- which conserves
        # the advected scalar across the source: mdot_out*phi_out = mdot_in*phi_in
        # + mdot_src*phi_src.  Complex-step-safe (npar_f are fixed real params).
        msrc = npar_f[pb + 0]
        w_sum = w_sum + msrc
        wphi_sum = wphi_sum + msrc * npar_f[pb + 2 + s]
        return wphi_sum / w_sum
    mix = wphi_sum / w_sum
    if rid == FLAME_HEAT_RELEASE and s == 0:
        # Heat-addition flame: raise the outflow's total enthalpy by Q_dot / |mdot|.
        # |mdot| is floored by smooth_abs so the jump stays bounded and smooth at
        # zero flow (complex-step-safe); at the operating point |mdot| >> eps the
        # floor is inert and the rise is exactly Q_dot / mdot.  Composition scalars
        # (s >= 1) pass through unchanged.
        Qdot = npar_f[pb + 0]
        mdot_mag = smooth_abs(mdot_e[col_edge[base]], eps)
        return mix + Qdot / mdot_mag
    return mix


@njit(cache=True)
def node_residual(n, rid, row_ptr, col_edge, orient, npar_f, npar_fptr, tf, eps, eps_fb, kappa, est, R, node_row_ptr):
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

    if rid == WALL or rid == CAVITY:
        # WALL and CAVITY share the mean residual: impermeable, no mass crosses the face.
        # The cavity differs only acoustically -- its finite volume populates the storage
        # block M (a compliance), stamped above the @njit line; the steady state is a wall.
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
        # Prescribed outflow mass rate: -s0*mdot is the mass leaving the domain.  The
        # acoustic counterpart (inherited J_alg row) is mdot' = 0 -- a constant-mass-flow
        # acoustic termination.
        e0 = col_edge[base]
        s0 = orient[base]
        R[r0] = -s0 * est[ES_MDOT, e0] - npar_f[pb + 0]
        return

    if rid == CHOKED_NOZZLE_OUTLET:
        # Compact choked nozzle of throat area A* (= npar_f[pb+0]) lumped just downstream:
        # the throat is sonic, so the outflow equals the critical mass flux for the
        # (interior, isentropic) total state.  The application plane stays subsonic -- the
        # M = 1 point is in the lumped throat, not the domain -- so the acoustic operator
        # is non-degenerate, and the inherited linearization is the compact choked-nozzle
        # (Marble--Candel) reflection, entropy coupling included.
        #   mdot_out = rho_t c_t A* (2/(gamma+1))^((gamma+1)/(2(gamma-1)))
        # with rho_t, c_t the stagnation density/sound-speed from the local (rho, c, M) and
        # gamma = rho c^2 / p (the local isentropic exponent; the equilibrium one when reacting).
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
        # Constant-area flame: mass balance + the **momentum** equation
        # (rho u^2 + p) A = const, i.e. the exact thin-flame jump (no low-Mach
        # static-/total-pressure shortcut).  Orientation-robust: s_i signs the net
        # (momentum + pressure) flux out of the element to zero;  mom_i = mdot_i
        # u_i / A = rho_i u_i^2 is analytic, so complex step stays clean.  The
        # areas are equal (constant-area flame), so a0 normalizes both ports.
        #
        # FLAME_HEAT_RELEASE: the perfect-gas closure carries kinetic energy, so
        #   mass + momentum + (energy with the Q_dot donor source) are all exact.
        # FLAME_EQUILIBRIUM: energy (h_t) and mass are exact; the reacting closure
        #   currently uses h ~ h_t (drops u^2/2), so the recovered rho/u carry an
        #   O(M^2) bias and the momentum jump is exact to O(M^4) -- the residual KE
        #   coupling is the documented next refinement.
        # No heat-release source sits on this row -> acoustically passive in J_alg.
        mom0 = est[ES_MDOT, e0] * est[ES_U, e0] / a0
        mom1 = est[ES_MDOT, e1] * est[ES_U, e1] / a0
        R[r0 + 1] = s0 * (mom0 + est[ES_P, e0]) + s1 * (mom1 + est[ES_P, e1]) - kappa_term
        return

    if rid == MASS_SOURCE:
        # Inline mass injection (constant area).  Mass: the net outflow exceeds the
        # inflow by the injected mass-flow mdot_src (npar_f[pb+0]).  Momentum: the
        # exact constant-area balance (rho u^2 + p) carries the injected axial
        # momentum mdot_src * u_inj (npar_f[pb+1]); u_inj = 0 is normal (transverse)
        # injection -- mass added with no axial momentum (momentum flux still
        # continuous).  Energy and composition enter through the donor mix above, so
        # the source is the conserved-scalar injection the dynamic S(omega) phase
        # will later modulate.  mom_i = mdot_i u_i / a0 = rho u^2 is analytic.
        mdot_src = npar_f[pb + 0]
        u_inj = npar_f[pb + 1]
        R[r0] = R[r0] - mdot_src  # mass balance: s0*mdot0 + s1*mdot1 = mdot_src
        mom0 = est[ES_MDOT, e0] * est[ES_U, e0] / a0
        mom1 = est[ES_MDOT, e1] * est[ES_U, e1] / a0
        R[r0 + 1] = s0 * (mom0 + est[ES_P, e0]) + s1 * (mom1 + est[ES_P, e1]) - mdot_src * u_inj / a0 - kappa_term
        return

    if rid == ISEN_AREA_CHANGE:
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
        # Reverse (large -> small) contraction branch.  The jet necks to a vena
        # contracta of area Cc*A_small, then Borda-expands back to the small pipe;
        # that re-expansion is the only lossy step, giving a downstream-referenced
        # total-pressure loss  K_c * (1/2 rho u^2)_small  with  K_c = (1/Cc - 1)^2.
        # Cc = 1 (default) is the loss-free contraction and recovers exact total-
        # pressure continuity (the historical behaviour).
        #
        # O(M^2) NOTE: the (1/2 rho u^2) head is the *incompressible* reduction of
        # the Borda momentum balance, so this jump is accurate only to O(M^2).  A
        # dedicated contraction element that resolves the vena-contracta state (and
        # so stays exact at higher Mach) is planned -- see catalog.sudden_area_change.
        cc = npar_f[pb + 0]
        k_contr = (1.0 / cc - 1.0) ** 2
        q_small = 0.5 * est[ES_RHO, se] * est[ES_U, se] * est[ES_U, se]
        loss_dir = 1.0 if se == e1 else -1.0  # orient the PT drop onto the small port
        r_isen = (est[ES_PT, e0] - est[ES_PT, e1]) - loss_dir * k_contr * q_small
        mdot_in_small = -ss * est[ES_MDOT, se]
        # xi switches momentum (forward expansion: Borda loss) <-> the contraction
        # branch.  NOTE: even when xi is saturated (|mdot| >> eps, so the mean flow
        # is one regime), the switch's *derivative* leaks the residual gap
        # (r_mom - r_isen) into the frozen perturbation Jacobian, biasing the jump
        # by O(eps).  When the flow is one-directional, set this element's eps small
        # (ElementSpec.eps) to recover the exact jump.
        xi = smooth_step(mdot_in_small, eps)
        R[r0 + 1] = xi * r_mom + (1.0 - xi) * r_isen - kappa_term
        return

    if rid == LOSS:
        K = npar_f[pb + 0]
        # ref_port selects which port's area/velocity the loss coefficient K is
        # referenced to (catalog.loss); only matters when the ports differ in area.
        ar = a0 if npar_f[pb + 1] < 0.5 else a1
        rho_avg = 0.5 * (est[ES_RHO, e0] + est[ES_RHO, e1])
        denom = rho_avg * ar
        # Through-flow, positive in the e0 -> e1 sense (mass balance: the inflow at
        # port 0 equals the outflow at port 1).  Using the port orientation keeps
        # the loss sign correct regardless of how the two edges were wired.
        mdot_through = -s0 * est[ES_MDOT, e0]
        u_ref = mdot_through / denom
        u_abs = (u_ref * u_ref + (eps / denom) ** 2) ** 0.5
        q_signed = 0.5 * rho_avg * u_ref * u_abs
        R[r0 + 1] = est[ES_PT, e0] - est[ES_PT, e1] - K * q_signed - kappa_term
        return

    if rid == LINEAR_RESISTANCE:
        # A linear flow resistance: total pressure drops in proportion to the through-flow,
        # Pt_in - Pt_out = R_lin * mdot_through (R_lin >= 0, npar_f[pb+0]).  Unlike the quadratic
        # LOSS (which vanishes with the mean dynamic head), this is linear in mdot, so it stays
        # active in the linearized/acoustic problem even at zero mean flow -- the resistance of a
        # screen / perforate / damper in a quiescent network.  mdot_through is signed +ve in the
        # e0 -> e1 sense, so the drop reverses with the flow; linear in the flow state -> the
        # complex-step Jacobian is exact (no smoothing needed).
        r_lin = npar_f[pb + 0]
        mdot_through = -s0 * est[ES_MDOT, e0]
        R[r0 + 1] = est[ES_PT, e0] - est[ES_PT, e1] - r_lin * mdot_through - kappa_term
        return
