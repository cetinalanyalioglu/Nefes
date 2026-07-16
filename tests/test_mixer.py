"""The mixer: a variable-port manifold that obeys the second law.

The static-pressure ``junction`` ties every port to a common static pressure, which at a fast
port hands the branch its velocity head as extra total pressure (more than the feed carries) --
free energy the second law forbids.  The ``mixer`` ties every port to a common
*effective* total pressure instead: each inflow gives up a loss on entering, so the node total
pressure never rises above the feeds and the mass-averaged outflow entropy never falls below the
feed mean.  ``recovery`` sets that loss between the least-dissipative ideal (``1``, the default:
the outlet at the minimum inflow total pressure, the lossless splitter when distributing and the
minimum-entropy limit when merging) and the full dump (``0``, the robust plenum).

These tests pin the guarantees (non-negative entropy production, no manufactured total
pressure), the limits (low-Mach merge -> junction, high-recovery distribution -> splitter,
recovery lowers the merge entropy toward the minimum), the well-posedness diagnostic for an
under-pinned high-recovery merge, the parameter addressing and YAML round-trip, and that the
acoustic operator accepts the new element.
"""

import warnings

import numpy as np
import pytest

import nefes
from nefes.assembly.assemble import residual
from nefes.assembly.recover import ES_M, ES_P, ES_PT
from nefes.elements import catalog as cat
from nefes.perturbation import PerturbationBC, perturbation_response
from nefes.shell.build import build_problem
from nefes.shell.diagnostics import diagnose_mixers
from nefes.solver import solve
from nefes.solver.report import states_table

R_AIR, GAMMA = 287.0, 1.4
CP = GAMMA * R_AIR / (GAMMA - 1.0)


def _gas():
    return nefes.perfect_gas(R_AIR, GAMMA)


def _entropy(T, p):
    """Specific entropy of the perfect gas (the additive constant cancels in a difference)."""
    return CP * np.log(T) - R_AIR * np.log(p)


def _merge_network(manifold, pt_hi=2.2e5, pt_lo=2.0e5, tt_hi=400.0, tt_lo=300.0, a_in=0.02, a_out=0.05, p_out=1.8e5):
    """Two total-pressure feeds merging through ``manifold`` into one pressure outlet.

    Node order: 0, 1 feeds; 2 manifold; 3 outlet.  Edges: e0 (0->2), e1 (1->2), e2 (2->3), so
    the two feeds flow into the manifold and edge 2 carries the merged stream out.
    """
    nodes = [
        cat.total_pressure_inlet(pt_hi, tt_hi),
        cat.total_pressure_inlet(pt_lo, tt_lo),
        manifold,
        cat.pressure_outlet(p_out),
    ]
    edges = [(0, 2, a_in), (1, 2, a_in), (2, 3, a_out)]
    return nefes.Network(_gas(), nodes, edges, p_ref=1.0e5, T_ref=350.0)


def _node_entropy_production(sol, in_edges, out_edges):
    """Entropy generated at a node: sum of (mass flux * s) leaving minus entering."""
    mdot, T, p = sol.field("mdot"), sol.field("T"), sol.field("p")
    s = _entropy(T, p)
    out = sum(mdot[e] * s[e] for e in out_edges)
    inn = sum(mdot[e] * s[e] for e in in_edges)
    return out - inn


def test_merge_converges_and_respects_second_law():
    """A merge of two unequal streams converges, stays subsonic, and generates entropy."""
    sol = _merge_network(cat.mixer(0.0)).solve()
    assert sol.converged, (sol.residual_norm, sol.print_residuals())
    assert sol.verify() == []
    assert np.abs(sol.field("M")).max() < 1.0

    sgen = _node_entropy_production(sol, in_edges=(0, 1), out_edges=(2,))
    assert sgen > 0.0  # adiabatic mixing generates entropy

    # No manufactured total pressure: the merged stream leaves at or below every feed.
    pt = sol.field("p_t")
    assert pt[2] <= min(pt[0], pt[1]) * (1.0 + 1e-6)


def test_junction_manufactures_total_pressure_where_mixer_does_not():
    """The documented failure: a slow plenum feeding a fast branch.

    The static-pressure junction hands the fast branch more total pressure than the feed
    carries (entropy production goes negative); the mixer keeps the branch at or
    below the feed and generates entropy.
    """

    def build(manifold):
        # slow feed -> manifold -> [slow branch (large area), fast branch (small area, low back p)]
        nodes = [
            cat.total_pressure_inlet(2.0e5, 300.0),
            manifold,
            cat.pressure_outlet(1.95e5),
            cat.pressure_outlet(1.1e5),
        ]
        edges = [(0, 1, 0.10), (1, 2, 0.10), (1, 3, 0.010)]
        return nefes.Network(_gas(), nodes, edges, p_ref=1.0e5, T_ref=300.0).solve()

    jun = build(cat.junction())
    mix = build(cat.mixer(0.0))
    assert jun.converged and mix.converged

    feed_pt = jun.field("p_t")[0]
    # The junction manufactures total pressure on the fast branch (edge 2 = node1->node3).
    assert jun.field("p_t")[2] > feed_pt * 1.05
    assert _node_entropy_production(jun, in_edges=(0,), out_edges=(1, 2)) < 0.0  # second-law violation

    # The mixer does not: the fast branch stays at or below the feed, entropy grows.
    assert mix.field("p_t")[2] <= feed_pt * (1.0 + 1e-6)
    assert _node_entropy_production(mix, in_edges=(0,), out_edges=(1, 2)) > 0.0


def _restricted_merge_problem(manifold, a=0.05, K=50.0):
    """Two feeds merging through ``manifold``, then a downstream loss to a pressure outlet.

    The loss carries the pressure drop, so the manifold ports run at low (but not vanishing)
    Mach.  Built through the low-level ``build_problem`` so its :class:`CompiledProblem` (and
    hence its residual) is directly evaluable.  Node order: 0, 1 feeds; 2 manifold; 3 loss; 4
    outlet.  Edge 2 (2->3) is the manifold's single outflow.
    """
    els = [
        cat.total_pressure_inlet(2.02e5, 320.0),
        cat.total_pressure_inlet(2.00e5, 300.0),
        manifold,
        cat.loss(K),
        cat.pressure_outlet(1.0e5),
    ]
    edges = [(0, 2, a), (1, 2, a), (2, 3, a), (3, 4, a)]
    return build_problem(_gas(), els, edges, mdot_ref=5.0, p_ref=1.0e5, h_ref=CP * 310.0)


def test_reduces_to_junction_residual_at_low_mach():
    """As the port Mach falls the mixer and junction kernels coincide.

    The two residuals differ only in the manifold rows, and there by exactly the outflow
    dynamic head ``p_t - p`` (the term the mixer dumps and the junction keeps).  So
    the mixer solution nearly solves the *junction* problem, with a residual equal to
    that dynamic head, which is ``O(M^2)`` and vanishes as ``M -> 0``.
    """
    prob_mix = _restricted_merge_problem(cat.mixer(0.0))
    prob_jun = _restricted_merge_problem(cat.junction())  # identical topology and state layout
    res = solve(prob_mix)
    assert res.converged
    x = res.x

    eps = 1.0e-4 * 5.0  # the converged smoothing scale (max(0.3*kappa, 1e-4) * mdot_ref, kappa -> 0)
    r_mix = residual(prob_mix, x, eps, 1.0e-5, 0.0)
    r_jun = residual(prob_jun, x, eps, 1.0e-5, 0.0)
    est = states_table(prob_mix, x)

    assert np.abs(r_mix).max() < 1.0e-4  # the mixer solved its own problem
    manifold_mach = np.abs(est[ES_M, :3]).max()
    dyn_head = float(est[ES_PT, 2] - est[ES_P, 2])  # dynamic head on the manifold outflow edge
    # The junction residual at the mixing solution equals that (small) outflow dynamic head.
    assert np.abs(r_jun).max() == pytest.approx(dyn_head, rel=1e-3)
    assert dyn_head / 1.0e5 < 2.0e-2  # the O(M^2) smallness at this low Mach
    assert manifold_mach < 0.15


def _distribution_network(manifold):
    """One inflow distributed to two pressure outlets (node 1 is the manifold)."""
    nodes = [
        cat.total_pressure_inlet(2.0e5, 300.0),
        manifold,
        cat.pressure_outlet(1.98e5),
        cat.pressure_outlet(1.97e5),
    ]
    edges = [(0, 1, 0.10), (1, 2, 0.03), (1, 3, 0.03)]
    return nefes.Network(_gas(), nodes, edges, p_ref=1.0e5, T_ref=300.0)


def test_high_recovery_distribution_is_near_isentropic():
    """Distributing a single feed, high recovery approaches the lossless splitter.

    A single inflow makes the minimum inflow total pressure its own, so its ideal loss is zero:
    recovery near 1 recovers essentially all the dynamic head (near-isentropic distribution, like
    the splitter), while recovery 0 dumps it.  The ideal is reached to the smoothing tolerance,
    so the match to the splitter is close, not bit-exact.

    Recovery is taken as 0.999 rather than 1 exactly: at recovery = 1 the mixer's mass-flow-dependent
    self-resistance vanishes and, with the continuation resistance driven to zero, its node block is
    singular in the split direction, so the converged flow is set at the smoothing noise floor.  A
    hair of loss restores a well-posed split without moving the physics off the near-isentropic limit.
    """
    hi = _distribution_network(cat.mixer(0.999)).solve()
    lo = _distribution_network(cat.mixer(0.0)).solve()
    spl = _distribution_network(cat.splitter()).solve()
    assert hi.converged and lo.converged and spl.converged

    # Same composition and temperature into every branch, so the only entropy is the pressure
    # loss.  The splitter is lossless; raising recovery moves the mixer toward it, so
    # recovery 1 dissipates less than the recovery 0 dump (and never violates the second law).
    s_hi = _node_entropy_production(hi, in_edges=(0,), out_edges=(1, 2))
    s_lo = _node_entropy_production(lo, in_edges=(0,), out_edges=(1, 2))
    s_spl = _node_entropy_production(spl, in_edges=(0,), out_edges=(1, 2))
    assert abs(s_spl) < 1e-6  # the lossless splitter is isentropic
    assert -1e-9 <= s_hi < s_lo  # recovery 1 dissipates less than the dump, staying second-law-safe

    # The recovery-1 split flows nearer the lossless splitter than the dump does.
    spl_mdot = spl.field("mdot")
    assert np.abs(hi.field("mdot") - spl_mdot).max() < np.abs(lo.field("mdot") - spl_mdot).max()


def _resisted_merge_network(manifold):
    """Two unequal feeds merging through ``manifold``, then a pipe to a pressure outlet.

    The downstream pipe carries the pressure drop, so the merge is well posed across the
    recovery range.  Node order: 0, 1 feeds; 2 manifold; 3 pipe; 4 outlet.
    """
    nodes = [
        cat.total_pressure_inlet(2.1e5, 400.0),
        cat.total_pressure_inlet(2.0e5, 300.0),
        manifold,
        cat.pipe(1.0, 0.15, 0.03),
        cat.pressure_outlet(1.5e5),
    ]
    edges = [(0, 2, 0.03), (1, 2, 0.03), (2, 3, 0.05), (3, 4, 0.05)]
    return nefes.Network(_gas(), nodes, edges, p_ref=1.0e5, T_ref=350.0)


def test_recovery_lowers_merge_entropy_toward_the_minimum():
    """Merging unequal streams, raising recovery lowers the entropy toward the minimum.

    Every recovery generates entropy (a merge is irreversible), but a higher recovery keeps more
    of the streams' total pressure, so the outlet leaves nearer the weakest feed and the entropy
    production falls monotonically toward the minimum-entropy limit.
    """
    sgen = []
    out_pt = []
    for recovery in (0.0, 0.5, 0.9):
        sol = _resisted_merge_network(cat.mixer(recovery)).solve()
        assert sol.converged, recovery
        assert np.abs(sol.field("M")).max() < 1.0
        s = _node_entropy_production(sol, in_edges=(0, 1), out_edges=(2,))
        assert s > 0.0  # a merge always generates entropy
        sgen.append(s)
        out_pt.append(sol.field("p_t")[2])
    assert sgen[0] > sgen[1] > sgen[2]  # entropy falls toward the minimum as recovery rises
    assert out_pt[0] < out_pt[1] < out_pt[2]  # the outlet keeps more total pressure


def test_general_merge_where_splitter_fails():
    """The mixer is the general merge element: it converges where the splitter cannot.

    Merging two streams of unequal total pressure is infeasible for the lossless splitter
    (which forces a single common total pressure); the mixer reconciles them through
    the mixing loss and converges, at the robust dump and at a low-loss recovery alike.
    """
    dump = _merge_network(cat.mixer(0.0), pt_hi=2.4e5, pt_lo=2.0e5).solve()
    lean = _resisted_merge_network(cat.mixer(0.9)).solve()
    with warnings.catch_warnings():  # the splitter deliberately fails to converge on the merge
        warnings.simplefilter("ignore")
        spl = _merge_network(cat.splitter(), pt_hi=2.4e5, pt_lo=2.0e5).solve()
    assert dump.converged and dump.verify() == []
    assert lean.converged  # a low-loss merge, which the lossless splitter cannot represent
    assert not spl.converged


def _pinned_merge_network(manifold, mdot_hi=5.0, mdot_lo=3.0, tt_hi=400.0, tt_lo=300.0, a_in=0.02, a_out=0.05):
    """Two unequal streams merging through ``manifold`` with both inflow rates prescribed.

    Prescribing the mass flows pins the split independently of the manifold, so even the
    resistance-free ``recovery = 1`` limit is well posed.  Node order: 0, 1 feeds; 2 manifold;
    3 outlet.  Edges e0 (0->2), e1 (1->2), e2 (2->3).
    """
    nodes = [
        cat.mass_flow_inlet(mdot_hi, tt_hi),
        cat.mass_flow_inlet(mdot_lo, tt_lo),
        manifold,
        cat.pressure_outlet(1.8e5),
    ]
    edges = [(0, 2, a_in), (1, 2, a_in), (2, 3, a_out)]
    return nefes.Network(_gas(), nodes, edges, p_ref=1.0e5, T_ref=350.0)


def test_full_recovery_merge_is_well_posed_when_flows_are_pinned():
    """At ``recovery = 1`` a merge is well posed when the network pins each inflow's rate.

    The ``recovery = 1`` limit adds no flow resistance of its own (total-pressure equalities only,
    as the splitter has none), so the flow split must be set by the network rather than the
    manifold.  With both inflow rates prescribed the split is pinned, the resistance-free limit
    converges, the outlet leaves at the weakest feed's total pressure, and the merge is the
    least dissipative at those rates.  Two bare total-pressure feeds do not pin the split, and the
    same limit is then under-determined and does not converge.
    """
    ideal = _pinned_merge_network(cat.mixer(1.0)).solve()
    assert ideal.converged, (ideal.residual_norm, ideal.print_residuals())
    assert ideal.verify() == []
    assert np.abs(ideal.field("M")).max() < 1.0

    s_ideal = _node_entropy_production(ideal, in_edges=(0, 1), out_edges=(2,))
    assert s_ideal > 0.0  # a merge is irreversible even at the least-dissipative limit

    pt = ideal.field("p_t")
    pt_min = min(pt[0], pt[1])
    assert pt[2] <= pt_min * (1.0 + 1e-6)  # never manufactures total pressure
    assert pt[2] >= pt_min * 0.99  # leaves at the weakest feed: the minimum-entropy limit

    # At the same prescribed rates the full dump generates more entropy than the ideal recovery.
    dump = _pinned_merge_network(cat.mixer(0.0)).solve()
    assert dump.converged
    assert _node_entropy_production(dump, in_edges=(0, 1), out_edges=(2,)) > s_ideal

    # Without pinning -- two bare total-pressure feeds on the node -- the split is under-determined
    # and the resistance-free limit does not converge (the splitter's own well-posedness need).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bare = _merge_network(cat.mixer(1.0)).solve()
    assert not bare.converged


def test_default_recovery_is_the_least_dissipative_ideal():
    """The default recovery is the least-dissipative ideal, ``1.0``."""
    assert cat.mixer().fparams[0] == 1.0


def test_diagnostic_flags_underpinned_high_recovery_merge():
    """A high-recovery merge with unpinned total-pressure feeds is flagged before it fails.

    Two total-pressure inlets attached straight to a ``recovery = 1`` mixer leave the
    flow split under-determined; the solve emits a warning naming the mixer and its feeds.
    """
    net = _merge_network(cat.mixer(1.0, name="mix"))
    messages = diagnose_mixers(net)
    assert len(messages) == 1 and "mix" in messages[0]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        net.solve(max_iter=40)
    assert any("mixer 'mix'" in str(w.message) for w in caught)


def test_diagnostic_silent_when_pinned_or_distributing_or_low_recovery():
    """The diagnostic stays silent when the split is pinned, distributing, or at low recovery."""
    # both inflow rates prescribed -> the split is pinned
    assert diagnose_mixers(_pinned_merge_network(cat.mixer(1.0))) == []
    # bare total-pressure feeds, but low recovery self-pins through the dump term
    assert diagnose_mixers(_merge_network(cat.mixer(0.9))) == []
    # a single inflow (distribution) is the lossless splitter, always well posed
    assert diagnose_mixers(_distribution_network(cat.mixer(1.0))) == []


def test_diagnostic_walks_through_lossless_pass_through_to_the_source():
    """The branch walk sees a fixed pressure source reached through a lossless duct."""
    nodes = [
        cat.total_pressure_inlet(2.2e5, 400.0),
        cat.duct(0.5),
        cat.total_pressure_inlet(2.0e5, 300.0),
        cat.mixer(1.0, name="mix"),
        cat.pressure_outlet(1.8e5),
    ]
    edges = [(0, 1, 0.02), (1, 3, 0.02), (2, 3, 0.02), (3, 4, 0.05)]
    net = nefes.Network(_gas(), nodes, edges, p_ref=1.0e5, T_ref=350.0)
    assert len(diagnose_mixers(net)) == 1  # the duct does not pin the feed


def test_recovery_parameter_addressing():
    """``recovery`` is a named parameter: readable, writable through with_params, validated."""
    net = _merge_network(cat.mixer(0.0, name="mix"))
    assert net.get("mix.recovery") == 0.0
    tuned = net.with_params({"mix.recovery": 0.5})
    assert tuned.get("mix.recovery") == 0.5
    assert net.get("mix.recovery") == 0.0  # base network is untouched

    with pytest.raises(ValueError):
        cat.mixer(1.5)
    with pytest.raises(ValueError):
        cat.mixer(-0.1)


def test_yaml_roundtrip_preserves_recovery(tmp_path):
    """A saved-and-reloaded case keeps the mixer and its recovery."""
    net = _merge_network(cat.mixer(0.4, name="mix"))
    path = str(tmp_path / "merge.yaml")
    net.save(path)
    back = nefes.load_case(path)
    assert back.get("mix.recovery") == 0.4
    sol = back.solve()
    assert sol.converged


def test_perturbation_operator_builds():
    """The acoustic layer accepts the mixer (auto-linearized, no storage stamp)."""
    net = nefes.Network(
        _gas(),
        nodes=[
            cat.total_pressure_inlet(2.0e5, 300.0, perturbation_bc=PerturbationBC.hard_wall()),
            cat.total_pressure_inlet(2.0e5, 300.0, perturbation_bc=PerturbationBC.hard_wall()),
            cat.mixer(0.0),
            cat.duct(0.5),
            cat.pressure_outlet(1.8e5, perturbation_bc=PerturbationBC.open_end()),
        ],
        edges=[(0, 2, 0.02), (1, 2, 0.02), (2, 3, 0.05), (3, 4, 0.05)],
    )
    sol = net.solve()
    assert sol.converged
    resp = perturbation_response(sol, np.array([200.0, 600.0]))
    tm = resp.transfer_matrix(2, 3)
    assert np.all(np.isfinite(tm))
