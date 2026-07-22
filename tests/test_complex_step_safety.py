"""Complex-step safety: the analytic (complex-step) Jacobian must equal a real
finite-difference Jacobian across every element type.

The hard constraint is that all residual math is complex-step-safe:
smooth, complex-analytic, no ``abs``/``min``/``max``/branch on the flow state.
The Jacobian is built by complex-step differentiation (``x + 1j*h``), which
returns the *correct* derivative ONLY for an analytic residual.  A real central
finite difference, by contrast, sees the actual function value -- so the two
agree to FD truncation accuracy for an analytic residual and *diverge* the
moment a non-analytic operation (``abs``, ``min``, a branch on the flow state)
slips into the residual stack.  Comparing them is therefore a direct detector
for a complex-step-unsafe implementation.

``test_cs_detector_flags_nonanalytic`` proves the detector actually fires: for
``|x|`` the complex-step "derivative" is wrong while the finite difference is
right, so the two disagree.  The element tests then assert agreement on the real
residual, covering every element kernel.

Two layers of coverage:

* a combined network that runs every element together at a solved operating
  point (and a random off-root state), and
* a **per-kernel regime sweep** (``PROBES`` / ``test_kernel_complex_step_safe_
  across_regimes``): each element gets a minimal isolated network driven across
  forward, reverse, near-zero and near-choke flow, where a non-analytic branch
  is most likely to be exposed.  ``test_every_element_kernel_is_swept`` is a
  roll-call: it fails if a newly added element type has no sweep probe, so the
  safety net cannot silently miss a kernel.
"""

import numpy as np
import pytest

from nefes.assembly.assemble import jacobian, residual
from nefes.assembly.recover import ES_M
from nefes.elements import catalog as cat
from nefes.elements.ids import (
    CAVITY,
    CHOKED_NOZZLE_OUTLET,
    DUCT,
    ELEMENT_TYPE_NAMES,
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
    SUPERSONIC_INLET,
    SUPERSONIC_OUTLET,
    TRANSFER_MATRIX,
    WALL,
)
from nefes.elements.kernels import node_donor
from nefes.shell.build import build_problem
from nefes.solver import solve
from nefes.solver.linear import col_scale, scaled_system
from nefes.solver.report import states_table
from nefes.thermo.configure import perfect_gas

R_AIR, GAMMA = 287.0, 1.4
CP = GAMMA * R_AIR / (GAMMA - 1.0)
EPS = 1e-3 * 40.0
EPS_FB = 1e-5
CS_H = 1e-30


def _fd_jacobian(prob, x2d, eps, eps_fb, kappa=0.0, rel=1e-6):
    """Dense real central finite-difference Jacobian (no complex step).

    Column ``n_solve*e + v`` is ``dR/dx[v, e]`` from a central difference with a
    per-variable step ``rel * var_scale[v]``.
    """
    n, E = prob.n_solve, prob.n_edges
    J = np.zeros((prob.n_eq, n * E))
    steps = np.empty(n)
    steps[:3] = rel * prob.var_scale
    if n > 3:
        steps[3:] = rel  # element-storage unknowns (none for a perfect gas)
    for e in range(E):
        for v in range(n):
            h = steps[v]
            xp = x2d.copy()
            xm = x2d.copy()
            xp[v, e] += h
            xm[v, e] -= h
            Rp = residual(prob, xp, eps, eps_fb, kappa)
            Rm = residual(prob, xm, eps, eps_fb, kappa)
            J[:, n * e + v] = (Rp - Rm) / (2.0 * h)
    return J


def _scaled(prob, J):
    """Nondimensionalize a dense Jacobian the way the Newton loop does."""
    vcol = col_scale(prob.var_scale, prob.n_edges)
    J_hat, _ = scaled_system(J, np.zeros(prob.n_eq), vcol, prob.res_scale)
    return np.asarray(J_hat.todense())


def _all_elements_network(inlet):
    """A single network exercising every element kernel.

    Layout (areas in m^2)::

        inlet -e0-> iac -e1-> splitter ==> loss -> duct -> junction -> outlet
                                       ==> sac --------------> junction
                                       ==> wall (dead leg, mdot = 0)

    The splitter feeds two flowing branches plus an impermeable wall leg, so the
    wall edge sits at mdot = 0 -- the C1-through-zero corner -- inside the same
    Jacobian comparison.
    """
    cfg = perfect_gas(R_AIR, GAMMA)
    A0, A1, A2 = 0.20, 0.12, 0.16
    elements = [
        inlet,  # 0
        cat.isentropic_area_change(),  # 1
        cat.splitter(),  # 2
        cat.loss(2.5),  # 3
        cat.duct(0.4),  # 4
        cat.sudden_area_change(cc=0.7),  # 5
        cat.junction(),  # 6
        cat.pressure_outlet(1.0e5),  # 7
        cat.wall(),  # 8
    ]
    edges = [
        (0, 1, A0),  # e0  inlet -> iac
        (1, 2, A1),  # e1  iac -> splitter (area change A0 -> A1)
        (2, 3, A1),  # e2  splitter -> loss
        (3, 4, A1),  # e3  loss -> duct      (loss: constant area)
        (4, 6, A1),  # e4  duct -> junction  (duct: constant area)
        (2, 5, A1),  # e5  splitter -> sac
        (5, 6, A2),  # e6  sac -> junction   (area change A1 -> A2)
        (6, 7, A2),  # e7  junction -> outlet
        (2, 8, A1),  # e8  splitter -> wall  (dead leg, mdot = 0)
    ]
    return build_problem(cfg, elements, edges, mdot_ref=40.0, p_ref=101325.0, h_ref=CP * 300.0)


INLETS = [
    cat.mass_flow_inlet(20.0, 300.0),
    cat.total_pressure_inlet(1.06e5, 300.0),
]


@pytest.mark.parametrize("inlet", INLETS, ids=["mass_flow_inlet", "pt_inlet"])
def test_cs_jacobian_matches_finite_difference(inlet):
    # Compare at the genuine operating point so the comparison is physical and
    # the wall leg actually sits at mdot = 0.
    prob = _all_elements_network(inlet)
    res = solve(prob)
    assert res.converged

    Jcs = _scaled(prob, jacobian(prob, res.x, EPS, EPS_FB).toarray())
    Jfd = _scaled(prob, _fd_jacobian(prob, res.x, EPS, EPS_FB))

    # Central FD is good to ~1e-9 on the scaled (O(1)) entries; a non-analytic
    # residual would blow this past any reasonable tolerance.
    assert np.allclose(Jcs, Jfd, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("inlet", INLETS, ids=["mass_flow_inlet", "pt_inlet"])
def test_cs_jacobian_matches_fd_off_operating_point(inlet):
    # Same agreement away from the root, where residuals are large -- exercises
    # the kernels over a broader slice of state space, including a reversed edge.
    prob = _all_elements_network(inlet)
    rng = np.random.default_rng(7)
    x = np.zeros((3, prob.n_edges))
    x[0, :] = rng.uniform(-15.0, 22.0, size=prob.n_edges)
    x[0, 0] = 24.0  # keep the inlet edge firmly forward
    x[1, :] = rng.uniform(9.0e4, 1.2e5, size=prob.n_edges)
    x[2, :] = CP * rng.uniform(295.0, 320.0, size=prob.n_edges)

    Jcs = _scaled(prob, jacobian(prob, x, EPS, EPS_FB).toarray())
    Jfd = _scaled(prob, _fd_jacobian(prob, x, EPS, EPS_FB))
    assert np.allclose(Jcs, Jfd, rtol=1e-5, atol=1e-6)


def test_residual_c1_through_zero_flow_all_edges():
    # The smooth-upwind enthalpy transport is the one place mdot crosses zero in
    # the residual; its mdot-derivative must be continuous at mdot = 0 on EVERY
    # edge (a sign()/branch would make the one-sided derivatives disagree).
    prob = _all_elements_network(INLETS[0])
    x = np.zeros((3, prob.n_edges))
    x[0, :] = 6.0
    x[1, :] = 1.0e5
    x[2, :] = CP * 305.0

    def dR_dmdot(edge, m0):
        xc = x.astype(np.complex128)
        xc[0, edge] = complex(m0, CS_H)
        return residual(prob, xc, EPS, EPS_FB).imag / CS_H

    # Sample close to zero: the smooth-switch curvature near mdot = 0 is
    # O(1/eps^2), so the one-sided derivatives only coincide in the limit; a
    # genuine kink (sign/branch) would instead leave a finite, offset-independent
    # gap between them.
    for e in range(prob.n_edges):
        d_lo = dR_dmdot(e, -1e-8)
        d_hi = dR_dmdot(e, +1e-8)
        scale = np.abs(d_hi).max() + 1.0
        assert np.allclose(d_lo, d_hi, atol=1e-6 * scale), f"edge {e} not C1 at mdot=0"


def test_cs_detector_flags_nonanalytic():
    # Guard on the guard: complex-step disagrees with finite difference for a
    # non-analytic function, so the comparison above genuinely catches a
    # complex-step-unsafe residual rather than passing vacuously.
    f = np.abs
    x0 = 0.3
    cs = f(complex(x0, CS_H)).imag / CS_H  # == 0 for abs: the WRONG derivative
    fd = (f(x0 + 1e-6) - f(x0 - 1e-6)) / 2e-6  # == 1: the right one
    assert not np.isclose(cs, fd, atol=1e-3)

    # ... while for an analytic function the two agree, confirming the detector
    # does not simply reject everything.
    g = np.exp
    cs_g = g(complex(x0, CS_H)).imag / CS_H
    fd_g = (g(x0 + 1e-6) - g(x0 - 1e-6)) / 2e-6
    assert np.isclose(cs_g, fd_g, rtol=1e-6)


# --------------------------------------------------------------------------
# Per-kernel regime sweep: every element type, in isolation, across regimes.
# --------------------------------------------------------------------------
#
# Each probe puts one element type in a minimal network.  The state is imposed
# directly (not solved), so the sweep can drive the element through forward,
# reverse, near-zero and near-choke flow -- the regimes where a non-analytic
# branch (a supersonic ``if``, a directional ``abs``, ...) tends to hide.  The
# residual stack is analytic at every (positive-p, positive-h) state regardless
# of mass balance, so unbalanced probe states are fine and intended.

PA, TT, PT_BC, P_OUT = 0.10, 300.0, 1.2e5, 1.0e5  # probe area / BCs
H_REF = CP * TT


def _probe_mass_flow_inlet():
    # mass_flow_inlet -> duct -> outlet
    els = [cat.mass_flow_inlet(18.0, TT), cat.duct(), cat.pressure_outlet(P_OUT)]
    return build_problem(perfect_gas(R_AIR, GAMMA), els, [(0, 1, PA), (1, 2, PA)], 30.0, PT_BC, H_REF)


def _probe_pt_inlet():
    els = [cat.total_pressure_inlet(PT_BC, TT), cat.duct(), cat.pressure_outlet(P_OUT)]
    return build_problem(perfect_gas(R_AIR, GAMMA), els, [(0, 1, PA), (1, 2, PA)], 30.0, PT_BC, H_REF)


def _probe_p_outlet():
    els = [cat.total_pressure_inlet(PT_BC, TT), cat.duct(), cat.pressure_outlet(P_OUT)]
    return build_problem(perfect_gas(R_AIR, GAMMA), els, [(0, 1, PA), (1, 2, PA)], 30.0, PT_BC, H_REF)


def _probe_isen_area_change():
    # gentle contraction so the small (downstream) edge stays subsonic at near-choke
    els = [cat.total_pressure_inlet(PT_BC, TT), cat.isentropic_area_change(), cat.pressure_outlet(P_OUT)]
    return build_problem(perfect_gas(R_AIR, GAMMA), els, [(0, 1, PA), (1, 2, 0.85 * PA)], 30.0, PT_BC, H_REF)


def _probe_transfer_matrix():
    # the TRANSFER_MATRIX element shares the isentropic-area-change mean-flow residual
    # (its transfer matrix acts only in the perturbation layer, above the @njit line), so
    # the same gentle contraction sweep exercises its kernel through near-choke flow.
    els = [cat.total_pressure_inlet(PT_BC, TT), cat.transfer_matrix_element(), cat.pressure_outlet(P_OUT)]
    return build_problem(perfect_gas(R_AIR, GAMMA), els, [(0, 1, PA), (1, 2, 0.85 * PA)], 30.0, PT_BC, H_REF)


def _probe_sudden_area_change():
    # forward flow expands (Borda), reverse flow contracts (vena-contracta) --
    # the sweep's sign flip exercises BOTH branches of the momentum<->loss switch
    els = [cat.total_pressure_inlet(PT_BC, TT), cat.sudden_area_change(cc=0.7), cat.pressure_outlet(P_OUT)]
    return build_problem(perfect_gas(R_AIR, GAMMA), els, [(0, 1, 0.85 * PA), (1, 2, PA)], 30.0, PT_BC, H_REF)


def _probe_loss():
    # loss straddling an area change, referenced to the downstream (smaller) port:
    # exercises the ref_port branch and the orientation-signed through-flow.
    els = [cat.total_pressure_inlet(PT_BC, TT), cat.loss(2.5, ref_port=1), cat.pressure_outlet(P_OUT)]
    return build_problem(perfect_gas(R_AIR, GAMMA), els, [(0, 1, PA), (1, 2, 0.8 * PA)], 30.0, PT_BC, H_REF)


def _probe_duct():
    els = [cat.total_pressure_inlet(PT_BC, TT), cat.duct(0.5), cat.pressure_outlet(P_OUT)]
    return build_problem(perfect_gas(R_AIR, GAMMA), els, [(0, 1, PA), (1, 2, PA)], 30.0, PT_BC, H_REF)


def _probe_linear_resistance():
    # linear total-pressure drop Pt0 - Pt1 = R * mdot; linear in the flow state, so the
    # complex-step Jacobian is exact across forward / reverse / near-zero through-flow.
    els = [cat.total_pressure_inlet(PT_BC, TT), cat.linear_resistance(250.0), cat.pressure_outlet(P_OUT)]
    return build_problem(perfect_gas(R_AIR, GAMMA), els, [(0, 1, PA), (1, 2, PA)], 30.0, PT_BC, H_REF)


def _probe_pipe(formulation="darcy-weisbach"):
    # length-bearing pipe (DUCT + LOSS): the Darcy-Weisbach friction head q ~ u*|u| with
    # the smooth-abs floor must stay analytic through forward / reverse / near-zero /
    # near-choke flow (the same signed quadratic as LOSS, here with K = f*L/D).
    els = [
        cat.total_pressure_inlet(PT_BC, TT),
        cat.pipe(0.5, 0.3, 0.02, formulation=formulation),
        cat.pressure_outlet(P_OUT),
    ]
    return build_problem(perfect_gas(R_AIR, GAMMA), els, [(0, 1, PA), (1, 2, PA)], 30.0, PT_BC, H_REF)


def _probe_junction():
    # two inflow legs merging (junction: shared static pressure)
    els = [
        cat.total_pressure_inlet(PT_BC, TT),
        cat.total_pressure_inlet(PT_BC, TT),
        cat.junction(),
        cat.pressure_outlet(P_OUT),
    ]
    edges = [(0, 2, PA), (1, 2, PA), (2, 3, PA)]
    return build_problem(perfect_gas(R_AIR, GAMMA), els, edges, 30.0, PT_BC, H_REF)


def _probe_splitter():
    # one inflow splitting into two legs (splitter: shared total pressure)
    els = [cat.total_pressure_inlet(PT_BC, TT), cat.splitter(), cat.pressure_outlet(P_OUT), cat.pressure_outlet(P_OUT)]
    edges = [(0, 1, PA), (1, 2, PA), (1, 3, PA)]
    return build_problem(perfect_gas(R_AIR, GAMMA), els, edges, 30.0, PT_BC, H_REF)


def _probe_mixer():
    # two inflow legs merging (mixer: shared effective total pressure).  The
    # inflow indicator smooth_step and the dynamic-head term (p_t - p) must stay analytic
    # through the forward / reverse / near-zero / near-choke sweep on every port.
    els = [
        cat.total_pressure_inlet(PT_BC, TT),
        cat.total_pressure_inlet(PT_BC, TT),
        cat.mixer(0.5),  # exercise both the dump and the minimum-inflow loss terms
        cat.pressure_outlet(P_OUT),
    ]
    edges = [(0, 2, PA), (1, 2, PA), (2, 3, PA)]
    return build_problem(perfect_gas(R_AIR, GAMMA), els, edges, 30.0, PT_BC, H_REF)


def _probe_forced_splitter():
    # one inflow forced-split into two legs: the first outflow carries 40% of the
    # inflow rate, the second (remainder) keeps total-pressure continuity.  Every row
    # is linear in the flow state, so the complex-step Jacobian is exact across the
    # forward / reverse / near-zero / near-choke sweep (no upwind switch).
    els = [
        cat.total_pressure_inlet(PT_BC, TT),
        cat.forced_splitter([0.4]),
        cat.pressure_outlet(P_OUT),
        cat.pressure_outlet(P_OUT),
    ]
    edges = [(0, 1, PA), (1, 2, PA), (1, 3, PA)]
    return build_problem(perfect_gas(R_AIR, GAMMA), els, edges, 30.0, PT_BC, H_REF)


def _probe_wall():
    # wall on a dead leg off a splitter (mdot = 0 at the wall edge)
    els = [cat.total_pressure_inlet(PT_BC, TT), cat.splitter(), cat.pressure_outlet(P_OUT), cat.wall()]
    edges = [(0, 1, PA), (1, 2, PA), (1, 3, PA)]
    return build_problem(perfect_gas(R_AIR, GAMMA), els, edges, 30.0, PT_BC, H_REF)


def _probe_cavity():
    # finite-volume cavity on a dead leg off a splitter (mdot = 0 at the cavity edge):
    # its mean residual is the wall's, so the complex-step sweep covers the same kernel.
    els = [cat.total_pressure_inlet(PT_BC, TT), cat.splitter(), cat.pressure_outlet(P_OUT), cat.cavity(2.0e-3)]
    edges = [(0, 1, PA), (1, 2, PA), (1, 3, PA)]
    return build_problem(perfect_gas(R_AIR, GAMMA), els, edges, 30.0, PT_BC, H_REF)


def _probe_heat_release_flame():
    # heat-addition flame (constant area): the smooth-abs |mdot| floor in the
    # h_t donor must stay analytic through the reverse / near-zero flow sweep.
    els = [cat.total_pressure_inlet(PT_BC, TT), cat.heat_release_flame(5.0e5), cat.pressure_outlet(P_OUT)]
    return build_problem(perfect_gas(R_AIR, GAMMA), els, [(0, 1, PA), (1, 2, PA)], 30.0, PT_BC, H_REF)


def _probe_mass_flow_outlet():
    # prescribed-mass-flow outlet: R = mdot_out - mdot_spec (linear, but the sweep still
    # exercises the recovery on its edge through reverse / near-zero / near-choke flow).
    els = [cat.total_pressure_inlet(PT_BC, TT), cat.duct(), cat.mass_flow_outlet(18.0)]
    return build_problem(perfect_gas(R_AIR, GAMMA), els, [(0, 1, PA), (1, 2, PA)], 30.0, PT_BC, H_REF)


def _probe_choked_nozzle_outlet():
    # compact choked-nozzle outlet: the critical-mass-flux residual carries stagnation
    # ratios stag^p and (2/(g+1))^p that must stay analytic through reverse / near-zero /
    # near-choke flow (stag = 1 + (g-1)/2 M^2 > 0 for any sign of M).  Throat < outlet area.
    els = [cat.total_pressure_inlet(PT_BC, TT), cat.duct(), cat.choked_nozzle_outlet(0.7 * PA)]
    return build_problem(perfect_gas(R_AIR, GAMMA), els, [(0, 1, PA), (1, 2, PA)], 30.0, PT_BC, H_REF)


def _probe_mass_source():
    # inline mass injection (constant area) with a nonzero injection velocity, so
    # both the mass source on the balance row and the momentum source term are
    # exercised; the donor mixes the injected h_t with the smooth-upwind interior
    # flow, which must stay analytic through reverse / near-zero flow.
    els = [
        cat.total_pressure_inlet(PT_BC, TT),
        cat.mass_source(6.0, 320.0, None, u_inj=40.0),
        cat.pressure_outlet(P_OUT),
    ]
    return build_problem(perfect_gas(R_AIR, GAMMA), els, [(0, 1, PA), (1, 2, PA)], 30.0, PT_BC, H_REF)


# focus element type -> minimal probe network
PROBES = {
    MASS_FLOW_INLET: _probe_mass_flow_inlet,
    PT_INLET: _probe_pt_inlet,
    P_OUTLET: _probe_p_outlet,
    ISEN_AREA_CHANGE: _probe_isen_area_change,
    TRANSFER_MATRIX: _probe_transfer_matrix,
    SUDDEN_AREA_CHANGE: _probe_sudden_area_change,
    LOSS: _probe_loss,
    LINEAR_RESISTANCE: _probe_linear_resistance,
    PIPE: _probe_pipe,
    DUCT: _probe_duct,
    JUNCTION: _probe_junction,
    SPLITTER: _probe_splitter,
    MIXER: _probe_mixer,
    FORCED_SPLITTER: _probe_forced_splitter,
    WALL: _probe_wall,
    CAVITY: _probe_cavity,
    FLAME_HEAT_RELEASE: _probe_heat_release_flame,
    MASS_SOURCE: _probe_mass_source,
    MASS_FLOW_OUTLET: _probe_mass_flow_outlet,
    CHOKED_NOZZLE_OUTLET: _probe_choked_nozzle_outlet,
}

# Supersonic boundaries are out of the subsonic scope and have no residual kernel
# yet, so they are exempt from the sweep.
DEFERRED_RIDS = {SUPERSONIC_INLET, SUPERSONIC_OUTLET}
# The reacting equilibrium flame needs the absolute-enthalpy datum and a physical
# reacting state, so the perfect-gas regime sweep here (which imposes h_t = cp*T
# states) cannot drive it.  It is complex-step-validated against a real
# finite-difference Jacobian at the converged operating point in tests/test_flame.py.
REACTIVE_RIDS = {FLAME_EQUILIBRIUM}
IMPLEMENTED_RIDS = set(ELEMENT_TYPE_NAMES) - DEFERRED_RIDS - REACTIVE_RIDS


def _sweep_states(prob):
    """Imposed states spanning forward / reverse / near-zero / near-choke flow.

    Also returns, per state, the peak |M| any edge reaches, so the test can
    assert the sweep genuinely visited a high-subsonic (choke-sensitive) regime
    without straying supersonic (where central FD would lose accuracy).
    """
    E = prob.n_edges
    levels = [38.0, 30.0, 12.0, 1e-3, -1e-3, -12.0, -30.0, -38.0]  # mdot, near-choke -> reverse
    states = []
    for m in levels:
        x = np.zeros((3, E))
        x[0, :] = m
        x[1, :] = 1.05e5
        x[2, :] = H_REF
        states.append(x)
    # a few mixed states (per-edge variation, modest range to stay subsonic)
    rng = np.random.default_rng(2024)
    for _ in range(4):
        x = np.zeros((3, E))
        x[0, :] = rng.uniform(-22.0, 22.0, size=E)
        x[1, :] = rng.uniform(9.5e4, 1.25e5, size=E)
        x[2, :] = CP * rng.uniform(295.0, 320.0, size=E)
        states.append(x)
    return states


def test_every_element_kernel_is_swept():
    # Roll-call: every implemented element type must appear in at least one
    # probe network, so adding a kernel without a complex-step sweep fails here.
    covered = set()
    for build in PROBES.values():
        covered |= set(int(r) for r in build().node_rid.tolist())
    missing = IMPLEMENTED_RIDS - covered
    assert not missing, "no complex-step sweep covers: " + ", ".join(ELEMENT_TYPE_NAMES[r] for r in sorted(missing))


def _assert_complex_step_matches_fd(prob, label):
    """Sweep a probe network and require the complex-step Jacobian to equal central FD."""
    peak_subsonic_M = 0.0
    for x in _sweep_states(prob):
        M = np.abs(states_table(prob, x)[ES_M])
        # keep the comparison in the regime where central FD is trustworthy;
        # the residual is analytic beyond it, but FD truncation is not.
        if np.any(M >= 0.97):
            continue
        peak_subsonic_M = max(peak_subsonic_M, float(M.max()))

        Jcs = _scaled(prob, jacobian(prob, x, EPS, EPS_FB).toarray())
        Jfd = _scaled(prob, _fd_jacobian(prob, x, EPS, EPS_FB))
        assert np.allclose(Jcs, Jfd, rtol=1e-5, atol=1e-6), f"{label} CS!=FD at mdot={x[0]}"

    # the sweep must actually have reached a high-subsonic (choke-sensitive)
    # state, else "passing" would just mean we never stressed the kernel
    assert peak_subsonic_M > 0.8, f"{label} sweep never near-choke (maxM={peak_subsonic_M:.2f})"


@pytest.mark.parametrize("rid", sorted(PROBES), ids=lambda r: ELEMENT_TYPE_NAMES[r])
def test_kernel_complex_step_safe_across_regimes(rid):
    _assert_complex_step_matches_fd(PROBES[rid](), ELEMENT_TYPE_NAMES[rid])


def test_momentum_pipe_kernel_complex_step_safe_across_regimes():
    # The pipe carries two closures behind one residual id, so the roll-call sweep above
    # (keyed on the id) reaches only the default one; the momentum branch is swept here.
    _assert_complex_step_matches_fd(_probe_pipe("momentum"), "Pipe (momentum)")


# --------------------------------------------------------------------------
# Burnt-marker transport: the sticky noisy-OR donor.  This is a scalar-branch of
# node_donor (keyed on ``s == marker_s``), not a distinct element rid, so it cannot
# ride the per-rid PROBES sweep above -- it gets its own complex-step check here.
# --------------------------------------------------------------------------


def _marker_donor(mdot, phi, rid=JUNCTION, npar_f=None):
    """``node_donor`` for the burnt marker at one node whose ``deg`` ports all feed it.

    ``orient = -1`` makes the node the head of every port, so a port with ``mdot > 0`` is
    an inflow (``mdot_in = +mdot``).  ``s = marker_s = 0`` forces the noisy-OR marker
    branch (the ``0.5``-centered gate lives elsewhere; this is the transport law).
    """
    deg = mdot.shape[0]
    row_ptr = np.array([0, deg], dtype=np.int64)
    col_edge = np.arange(deg, dtype=np.int64)
    orient = -np.ones(deg, dtype=np.int64)
    if npar_f is None:
        npar_f = np.zeros(0, dtype=mdot.dtype)
    npar_fptr = np.array([0, npar_f.shape[0]], dtype=np.int64)
    return node_donor(0, rid, 0, 0, row_ptr, col_edge, orient, npar_f, npar_fptr, EPS, mdot, phi)


def test_marker_noisy_or_donor_complex_step_matches_fd():
    # The sticky burnt-marker donor b_out = 1 - prod_i (1 - theta_i * b_i) must be
    # complex-analytic in every marker b_i AND every mass flow m_i (theta rides mdot),
    # through forward / reverse / near-zero flow and at intermediate marker values.
    H, d = 1e-30, 1e-6
    flow_sets = [
        np.array([30.0, 18.0, -25.0]),  # two in, one out
        np.array([1e-3, -1e-3, 12.0]),  # near-zero, both signs
        np.array([-30.0, -18.0, -9.0]),  # all reverse (all out)
        np.array([22.0, 0.0, 7.0]),  # a dead (mdot = 0) port
    ]
    marker_sets = [
        np.array([1.0, 0.0, 0.0]),  # bimodal: a burnt port meets fresh ones
        np.array([0.3, 0.7, 0.5]),  # intermediate (the transient regime)
        np.array([0.0, 0.0, 0.0]),  # all fresh (endpoint: must give exactly 0)
    ]
    for m in flow_sets:
        for b in marker_sets:
            if np.all(b == 0.0):  # endpoint exactness: no numerical creep off zero
                assert abs(_marker_donor(m.copy(), b.copy())) < 1e-14
            for j in range(3):
                unit = np.eye(3)[j]
                phic = b.astype(np.complex128)
                phic[j] += 1j * H
                cs_b = _marker_donor(m.astype(np.complex128), phic).imag / H
                fd_b = (_marker_donor(m, b + d * unit) - _marker_donor(m, b - d * unit)) / (2 * d)
                assert cs_b == pytest.approx(fd_b, rel=1e-6, abs=1e-9), f"d/db_{j} at m={m}, b={b}"
                mc = m.astype(np.complex128)
                mc[j] += 1j * H
                cs_m = _marker_donor(mc, b.astype(np.complex128)).imag / H
                fd_m = (_marker_donor(m + d * unit, b) - _marker_donor(m - d * unit, b)) / (2 * d)
                assert cs_m == pytest.approx(fd_m, rel=1e-6, abs=1e-9), f"d/dm_{j} at m={m}, b={b}"


def test_marker_mass_source_donor_is_sticky_and_analytic():
    # A mass source injecting FRESH gas (b_src = 0) into a burnt interior stream must not
    # dilute the marker (b_out -> 1); injecting burnt gas (b_src = 1, e.g. EGR) sets it.
    # npar_f = [msrc, u_inj, b_src] and s = 0 -> the branch reads b_src at pb + 2.
    H, d = 1e-30, 1e-6
    m = np.array([20.0])  # one interior incoming port
    fresh = np.array([0.0, 0.0, 0.0])  # injected b_src = 0
    burnt = np.array([0.0, 0.0, 1.0])  # injected b_src = 1
    # A burnt port lands b = 1 only to within the smooth-upwind floor O(eps^2/mdot^2)
    # (here ~1e-6); the marker is bimodal to eps, like a converged solve.  The all-fresh
    # cases below are exact (no floor), which is the no-creep guarantee that matters.
    burnt_floor = 2e-6
    # burnt interior + fresh injection stays burnt (the RQL quench case)
    assert _marker_donor(m, np.array([1.0]), MASS_SOURCE, fresh) == pytest.approx(1.0, abs=burnt_floor)
    # fresh interior + fresh injection stays fresh (exact)
    assert abs(_marker_donor(m, np.array([0.0]), MASS_SOURCE, fresh)) < 1e-14
    # fresh interior + injected burnt gas turns it burnt
    assert _marker_donor(m, np.array([0.0]), MASS_SOURCE, burnt) == pytest.approx(1.0, abs=1e-12)
    # analytic in the interior marker
    bi = 0.4
    phic = np.array([bi], dtype=np.complex128)
    phic[0] += 1j * H
    cs = _marker_donor(m.astype(np.complex128), phic, MASS_SOURCE, fresh.astype(np.complex128)).imag / H
    fd = (
        _marker_donor(m, np.array([bi + d]), MASS_SOURCE, fresh)
        - _marker_donor(m, np.array([bi - d]), MASS_SOURCE, fresh)
    ) / (2 * d)
    assert cs == pytest.approx(fd, rel=1e-6, abs=1e-9)
