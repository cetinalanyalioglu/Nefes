"""The junction: a variable-port manifold that obeys the second law.

The junction ties every port to a common *effective* total pressure: each branch gives up a loss
on entering the mix (a combining branch) or leaving it (a dividing branch), so the node total
pressure never rises above the feeds and the mass-averaged outflow entropy never falls below the
feed mean.  Two closures set that loss:

* the geometry-free ``recovery`` in ``[0, 1]``, between the least-dissipative ideal (``1``, the
  default: the outlet at the minimum inflow total pressure, an isentropic split when distributing
  and the minimum-entropy limit when merging) and the full dump (``0``, the robust plenum that at
  low Mach ties the incident static pressures equal);
* per-branch loss coefficients ``K`` from tabulated junction data (see ``test_junction_loss``).

These tests pin the guarantees (non-negative entropy production, no manufactured total pressure),
the recovery limits (recovery lowers the merge entropy toward the minimum; high recovery
distributes near-isentropically), the well-posedness diagnostic for an under-pinned high-recovery
merge, the parameter addressing and YAML round-trip, the chamber-volume compliance, and that the
acoustic operator accepts the element.
"""

import warnings

import numpy as np
import pytest

import nefes
from nefes.elements import catalog as cat
from nefes.perturbation import PerturbationBC, perturbation_response
from nefes.shell.diagnostics import diagnose_junctions

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
    sol = _merge_network(cat.junction(recovery=0.0)).solve()
    assert sol.converged, (sol.residual_norm, sol.print_residuals())
    assert sol.verify() == []
    assert np.abs(sol.field("M")).max() < 1.0

    sgen = _node_entropy_production(sol, in_edges=(0, 1), out_edges=(2,))
    assert sgen > 0.0  # adiabatic mixing generates entropy

    # No manufactured total pressure: the merged stream leaves at or below every feed.
    pt = sol.field("p_t")
    assert pt[2] <= min(pt[0], pt[1]) * (1.0 + 1e-6)


def _distribution_network(manifold):
    """One inflow distributed to two pressure outlets (node 1 is the manifold).

    The outlet pressures give the two branches a moderate, well-separated Mach so the split is
    firmly conditioned (away from the near-singular, near-quiescent limit of a recovery = 1 split).
    """
    nodes = [
        cat.total_pressure_inlet(2.0e5, 300.0),
        manifold,
        cat.pressure_outlet(1.5e5),
        cat.pressure_outlet(1.55e5),
    ]
    edges = [(0, 1, 0.10), (1, 2, 0.03), (1, 3, 0.03)]
    return nefes.Network(_gas(), nodes, edges, p_ref=1.0e5, T_ref=300.0)


def test_recovery_lowers_distribution_entropy_toward_the_floor():
    """Distributing a single feed, raising recovery lowers the split entropy monotonically.

    The geometry-free recovery closure does not reach a bit-exact isentropic split (that is the
    ``K = 0`` closure, checked in ``test_junction_loss``); at recovery 1 the ideal-loss term keeps
    a small smoothing floor.  Within that, higher recovery keeps more total pressure and dissipates
    less, and every recovery stays second-law-safe.
    """
    hi = _distribution_network(cat.junction(recovery=1.0)).solve()
    mid = _distribution_network(cat.junction(recovery=0.5)).solve()
    lo = _distribution_network(cat.junction(recovery=0.0)).solve()
    assert hi.converged and mid.converged and lo.converged

    # Same composition and temperature into every branch, so the only entropy is the pressure loss.
    s_hi = _node_entropy_production(hi, in_edges=(0,), out_edges=(1, 2))
    s_mid = _node_entropy_production(mid, in_edges=(0,), out_edges=(1, 2))
    s_lo = _node_entropy_production(lo, in_edges=(0,), out_edges=(1, 2))
    assert -1e-9 <= s_hi < s_mid < s_lo  # recovery 1 dissipates least, staying second-law-safe

    # The exact lossless split (K = 0) is isentropic, below every recovery-closure value.
    exact = _distribution_network(cat.junction(K=0.0)).solve()
    assert exact.converged
    assert abs(_node_entropy_production(exact, in_edges=(0,), out_edges=(1, 2))) < 1e-6 * s_lo


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
    for recovery in (0.0, 0.4, 0.8):
        sol = _resisted_merge_network(cat.junction(recovery=recovery)).solve()
        assert sol.converged, recovery
        assert np.abs(sol.field("M")).max() < 1.0
        s = _node_entropy_production(sol, in_edges=(0, 1), out_edges=(2,))
        assert s > 0.0  # a merge always generates entropy
        sgen.append(s)
        out_pt.append(sol.field("p_t")[2])
    assert sgen[0] > sgen[1] > sgen[2]  # entropy falls toward the minimum as recovery rises
    assert out_pt[0] < out_pt[1] < out_pt[2]  # the outlet keeps more total pressure


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

    The ``recovery = 1`` limit adds no flow resistance of its own (total-pressure equalities only),
    so the flow split must be set by the network rather than the manifold.  With both inflow rates
    prescribed the split is pinned, the resistance-free limit converges, the outlet leaves at the
    weakest feed's total pressure, and the merge is the least dissipative at those rates.  Two bare
    total-pressure feeds do not pin the split, and the same limit is then under-determined.
    """
    ideal = _pinned_merge_network(cat.junction(recovery=1.0)).solve()
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
    dump = _pinned_merge_network(cat.junction(recovery=0.0)).solve()
    assert dump.converged
    assert _node_entropy_production(dump, in_edges=(0, 1), out_edges=(2,)) > s_ideal

    # Without pinning -- two bare total-pressure feeds on the node -- the split is under-determined
    # and the resistance-free limit does not converge.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bare = _merge_network(cat.junction(recovery=1.0)).solve()
    assert not bare.converged


def test_default_recovery_is_the_least_dissipative_ideal():
    """The default recovery is the least-dissipative ideal, ``1.0`` (fparams slot 1)."""
    assert cat.junction().fparams[1] == 1.0


def test_branch_recovery_broadcast_reproduces_the_scalar():
    """One factor per branch, all equal, is the same closure as the single broadcast factor.

    The per-branch form must be a strict generalization: repeating a factor across every port
    reproduces the scalar solve, so nothing about the closure changes when the vector form is used.
    """
    for sigma in (0.0, 0.5, 1.0):
        scalar = _pinned_merge_network(cat.junction(recovery=sigma)).solve()
        vector = _pinned_merge_network(cat.junction(recovery=[sigma] * 3)).solve()
        assert scalar.converged and vector.converged, sigma
        for field in ("mdot", "p", "p_t", "T"):
            assert np.allclose(scalar.field(field), vector.field(field), rtol=1e-10, atol=0.0), (sigma, field)


def test_branch_recovery_charges_the_branch_it_is_given_to():
    """Each factor acts on its own branch, in the order the branches are wired.

    Two streams that are identical apart from their recovery factor must lose differently: the
    branch on the full dump gives up its whole dynamic head, while the branch on the ideal gives up
    only its excess over the weakest feed.  Swapping the two factors swaps which stream is charged,
    which pins both the per-branch action and the port ordering of the vector.
    """
    equal = dict(mdot_hi=4.0, mdot_lo=4.0, tt_hi=300.0, tt_lo=300.0)
    first = _pinned_merge_network(cat.junction(recovery=[0.0, 1.0, 1.0]), **equal).solve()
    second = _pinned_merge_network(cat.junction(recovery=[1.0, 0.0, 1.0]), **equal).solve()
    assert first.converged and second.converged

    # the dumped branch arrives with a higher total pressure: it has to, since it is charged its
    # whole dynamic head to reach the same node
    pt_first, pt_second = first.field("p_t"), second.field("p_t")
    assert pt_first[0] > pt_first[1]
    assert pt_second[1] > pt_second[0]
    # the two solves are mirror images of each other (identical streams, swapped factors), to the
    # solver's convergence tolerance
    assert np.isclose(pt_first[0], pt_second[1], rtol=1e-6)
    assert np.isclose(pt_first[1], pt_second[0], rtol=1e-6)
    assert np.isclose(pt_first[2], pt_second[2], rtol=1e-6)


def test_branch_recovery_respects_the_second_law():
    """A mixed set of factors still produces entropy and never manufactures total pressure.

    The guarantee follows from each branch's loss being non-negative, whatever sets it, so mixing
    the factors across ports cannot break it.
    """
    sol = _pinned_merge_network(cat.junction(recovery=[0.0, 1.0, 0.3])).solve()
    assert sol.converged, (sol.residual_norm, sol.print_residuals())
    assert sol.verify() == []
    assert np.abs(sol.field("M")).max() < 1.0

    pt = sol.field("p_t")
    assert pt[2] <= min(pt[0], pt[1]) * (1.0 + 1e-6)  # node at or below the weakest feed
    assert _node_entropy_production(sol, in_edges=(0, 1), out_edges=(2,)) > 0.0

    # mixing in a dumped branch dissipates more than putting every branch on the ideal
    ideal = _pinned_merge_network(cat.junction(recovery=1.0)).solve()
    assert _node_entropy_production(sol, in_edges=(0, 1), out_edges=(2,)) > _node_entropy_production(
        ideal, in_edges=(0, 1), out_edges=(2,)
    )


def test_branch_recovery_is_inert_on_a_dividing_branch():
    """A factor acts only while its branch feeds the node.

    Under this closure a dividing branch carries no mixing loss, so its recovery factor has nothing
    to act on: changing the factors of the two outflows of a distribution leaves the mean flow
    untouched, while changing the single inflow's factor moves it.
    """
    base = _distribution_network(cat.junction(recovery=[1.0, 1.0, 1.0])).solve()
    outflows_changed = _distribution_network(cat.junction(recovery=[1.0, 0.0, 0.0])).solve()
    inflow_changed = _distribution_network(cat.junction(recovery=[0.0, 1.0, 1.0])).solve()
    assert base.converged and outflows_changed.converged and inflow_changed.converged

    assert np.allclose(base.field("mdot"), outflows_changed.field("mdot"), rtol=1e-9)
    assert not np.allclose(base.field("p_t"), inflow_changed.field("p_t"), rtol=1e-3)


def test_branch_recovery_validation():
    """The vector form is validated: range, port count, and exclusivity with ``K``."""
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        cat.junction(recovery=[0.5, 1.2])
    with pytest.raises(ValueError, match="one entry per port"):
        cat.junction(recovery=[0.5])
    with pytest.raises(ValueError, match="mutually exclusive"):
        cat.junction(recovery=[0.5, 0.5], K=0.3)
    with pytest.raises(ValueError, match="mutually exclusive"):
        cat.junction(recovery=[0.5, 0.5], static_pressure=True)
    # the list length is topology, so a mismatch against the wired port count is caught at build
    with pytest.raises(ValueError, match="one factor per port"):
        _merge_network(cat.junction(recovery=[0.5, 0.5])).solve()


def test_branch_recovery_parameter_addressing_and_roundtrip(tmp_path):
    """Per-branch recovery is a named vector parameter and survives a YAML round-trip."""
    net = _pinned_merge_network(cat.junction(recovery=[0.2, 0.6, 1.0], name="jn"))
    assert net.get("jn.recovery") == [0.2, 0.6, 1.0]
    assert "jn.K" not in net.parameters()  # the two closures are mutually exclusive

    tuned = net.with_params({"jn.recovery": [0.3, 0.3, 0.3]})
    assert tuned.get("jn.recovery") == [0.3, 0.3, 0.3]
    assert net.get("jn.recovery") == [0.2, 0.6, 1.0]  # base network is untouched

    path = str(tmp_path / "branch-recovery.yaml")
    net.save(path)
    back = nefes.load_case(path)
    assert back.get("jn.recovery") == [0.2, 0.6, 1.0]
    assert back.solve().converged


def test_diagnostic_reads_each_branch_own_recovery():
    """The under-pinned warning counts only the branches whose own factor is near the ideal."""
    # both feeds on the ideal and neither pinned by the network: under-determined
    assert len(diagnose_junctions(_merge_network(cat.junction(recovery=[1.0, 1.0, 1.0], name="jn")))) == 1
    # the second feed dumps its head, which pins it, leaving only one unpinned branch
    assert diagnose_junctions(_merge_network(cat.junction(recovery=[1.0, 0.0, 1.0]))) == []


def test_diagnostic_flags_underpinned_high_recovery_merge():
    """A high-recovery merge with unpinned total-pressure feeds is flagged before it fails.

    Two total-pressure inlets attached straight to a ``recovery = 1`` junction leave the flow
    split under-determined; the solve emits a warning naming the junction and its feeds.
    """
    net = _merge_network(cat.junction(recovery=1.0, name="jn"))
    messages = diagnose_junctions(net)
    assert len(messages) == 1 and "jn" in messages[0]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        net.solve(max_iter=40)
    assert any("junction 'jn'" in str(w.message) for w in caught)


def test_diagnostic_silent_when_pinned_or_distributing_or_low_recovery():
    """The diagnostic stays silent when the split is pinned, distributing, or at low recovery."""
    # both inflow rates prescribed -> the split is pinned
    assert diagnose_junctions(_pinned_merge_network(cat.junction(recovery=1.0))) == []
    # bare total-pressure feeds, but low recovery self-pins through the dump term
    assert diagnose_junctions(_merge_network(cat.junction(recovery=0.9))) == []
    # a single inflow (distribution) is the isentropic split, always well posed
    assert diagnose_junctions(_distribution_network(cat.junction(recovery=1.0))) == []
    # per-branch loss coefficients carry their own resistance, so the split is pinned
    assert diagnose_junctions(_merge_network(cat.junction(K=0.5))) == []


def test_diagnostic_walks_through_lossless_pass_through_to_the_source():
    """The branch walk sees a fixed pressure source reached through a lossless duct."""
    nodes = [
        cat.total_pressure_inlet(2.2e5, 400.0),
        cat.duct(0.5),
        cat.total_pressure_inlet(2.0e5, 300.0),
        cat.junction(recovery=1.0, name="jn"),
        cat.pressure_outlet(1.8e5),
    ]
    edges = [(0, 1, 0.02), (1, 3, 0.02), (2, 3, 0.02), (3, 4, 0.05)]
    net = nefes.Network(_gas(), nodes, edges, p_ref=1.0e5, T_ref=350.0)
    assert len(diagnose_junctions(net)) == 1  # the duct does not pin the feed


def test_recovery_parameter_addressing():
    """``recovery`` is a named parameter: readable, writable through with_params, validated."""
    net = _merge_network(cat.junction(recovery=0.0, name="jn"))
    assert net.get("jn.recovery") == 0.0
    tuned = net.with_params({"jn.recovery": 0.5})
    assert tuned.get("jn.recovery") == 0.5
    assert net.get("jn.recovery") == 0.0  # base network is untouched

    with pytest.raises(ValueError):
        cat.junction(recovery=1.5)
    with pytest.raises(ValueError):
        cat.junction(recovery=-0.1)


def test_yaml_roundtrip_preserves_recovery(tmp_path):
    """A saved-and-reloaded case keeps the junction and its recovery."""
    net = _merge_network(cat.junction(recovery=0.4, name="jn"))
    path = str(tmp_path / "merge.yaml")
    net.save(path)
    back = nefes.load_case(path)
    assert back.get("jn.recovery") == 0.4
    sol = back.solve()
    assert sol.converged


def test_chamber_volume_is_a_perturbation_only_compliance():
    """A junction ``volume`` leaves the mean flow untouched and enters only the acoustic storage."""
    lengthless = _distribution_network(cat.junction()).solve()
    plenum = _distribution_network(cat.junction(volume=2.0e-3)).solve()
    assert lengthless.converged and plenum.converged
    # volume is inert in the mean flow: identical steady mass flows
    assert np.allclose(lengthless.field("mdot"), plenum.field("mdot"), rtol=1e-9)
    # and it is a perturbation-layer parameter
    net = _distribution_network(cat.junction(volume=2.0e-3, name="jn"))
    assert net.parameters()["jn.volume"].layer == "perturbation"


def test_perturbation_operator_builds():
    """The acoustic layer accepts the junction (auto-linearized mean rows, chamber storage)."""
    net = nefes.Network(
        _gas(),
        nodes=[
            cat.total_pressure_inlet(2.0e5, 300.0, perturbation_bc=PerturbationBC.hard_wall()),
            cat.total_pressure_inlet(2.0e5, 300.0, perturbation_bc=PerturbationBC.hard_wall()),
            cat.junction(recovery=0.0),
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
