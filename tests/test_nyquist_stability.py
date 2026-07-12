"""Open-loop (Nyquist) stability driver -- validation against eigenmodes and the operator.

The Nyquist driver answers the stability question on the **real** frequency axis, so it
reaches the entropy / convected-wave regime where the contour eigensolver overflows.  The
tests anchor it three ways:

1. **Exact operator identities** -- the rank-1 source decomposition reproduces ``A - A_0``,
   the matrix-determinant lemma reproduces ``det A / det A_0``, and the return ratio scales
   linearly with the FTF gain.  These are round-off checks, independent of any tuning.
2. **Cross-check vs eigenmodes** on a purely acoustic (``isentropic``) flame tube where both
   drivers are valid: same unstable-mode count, matching crossing frequency.
3. **The headline** -- a choked-nozzle flame rig that is *stable* without the convected
   entropy wave and *unstable* with it: the indirect-noise (entropy -> nozzle -> acoustic)
   path is the sole destabilizer, which only the real-axis driver can resolve.

Run in the ``nefes`` env (numba).
"""

import warnings

import numpy as np
import pytest

from nefes.assembly.recover import ES_M
from nefes.elements import catalog as cat
from nefes.elements.dynamic_source import heat_release_response, n_tau_lowpass, tabulated
from nefes.perturbation import eigenmodes, nyquist_stability, open_loop_response
from nefes.perturbation.operator.boundary_bc import PerturbationBC
from nefes.perturbation.operator.operator import assemble_acoustic, build_acoustic_blocks
from nefes.perturbation.stability.nyquist import _passive_blocks, _rank1_terms
from nefes.shell.build import build_problem
from nefes.solver import solve
from nefes.solver.report import states_table
from nefes.thermo.configure import perfect_gas

R_AIR, GAMMA = 287.0, 1.4
CP = GAMMA * R_AIR / (GAMMA - 1.0)


# --------------------------------------------------------------------------
# Rigs
# --------------------------------------------------------------------------


def _acoustic_rijke(n, tau, fc, mdot=0.005, dT=400.0, area=0.01, L1=0.6, L2=0.4, tab=False):
    """Cold-air Rijke tube: hard-wall inlet -> duct -> low-pass n-tau flame -> duct -> open end.

    Purely acoustic (no choked nozzle / entropy generation to convert), so ``isentropic``
    Nyquist and ``eigenmodes`` are both valid and must agree.
    """
    tf = n_tau_lowpass(n, tau, fc)
    if tab:  # sample the closed form onto a real-frequency table (a "measured" FTF)
        fg = np.linspace(0.0, 2500.0, 400)
        tf = tabulated(fg, np.array([tf(f) for f in fg]))
    ds = heat_release_response(tf, ref_edge=1)
    els = [
        cat.mass_flow_inlet(mdot, 300.0, perturbation_bc=PerturbationBC.hard_wall()),
        cat.duct(L1),
        cat.heat_release_flame(mdot * CP * dT, dynamic_source=ds),
        cat.duct(L2),
        cat.pressure_outlet(1.0e5, perturbation_bc=PerturbationBC.open_end()),
    ]
    edges = [(0, 1, area), (1, 2, area), (2, 3, area), (3, 4, area)]
    prob = build_problem(perfect_gas(R_AIR, GAMMA), els, edges, mdot_ref=mdot, p_ref=1e5, h_ref=CP * 300.0)
    res = solve(prob)
    assert res.converged
    return prob, res.x


# Choked-nozzle flame rig (the entropy headline).  Low-pass n-tau so the direct acoustic
# path is stable; the convected entropy spot converts to sound at the nozzle and closes an
# indirect feedback loop that tips one mode unstable.
_CN = dict(AREA=0.02, A_STAR=0.012, MDOT=0.4, DT=750.0, L1=0.5, L2=0.8, N=0.6, TAU=3e-3, FC=300.0)


def _choked_rig(n=_CN["N"], tau=_CN["TAU"], fc=_CN["FC"], active=True):
    """inherited mass-flow inlet -> cold duct -> low-pass n-tau flame -> hot duct -> choked nozzle.

    The flame + choked nozzle do not converge from a cold guess, so the heat release is
    ramped (continuation).  ``active=False`` drops the FTF (the passive operator ``A_0``).
    """
    ds = heat_release_response(n_tau_lowpass(n, tau, fc), ref_edge=1) if active else None

    def mk(Qdot):
        els = [
            cat.mass_flow_inlet(_CN["MDOT"], 300.0, perturbation_bc=PerturbationBC.inherit()),
            cat.duct(_CN["L1"]),
            cat.heat_release_flame(Qdot, dynamic_source=ds),
            cat.duct(_CN["L2"]),
            cat.choked_nozzle_outlet(_CN["A_STAR"]),
        ]
        edges = [(0, 1, _CN["AREA"]), (1, 2, _CN["AREA"]), (2, 3, _CN["AREA"]), (3, 4, _CN["AREA"])]
        return build_problem(perfect_gas(R_AIR, GAMMA), els, edges, mdot_ref=_CN["MDOT"], p_ref=1e5, h_ref=CP * 300.0)

    dT = _CN["DT"]
    x0 = None
    for ddT in (1.0, 0.25 * dT, 0.5 * dT, 0.75 * dT, dT):
        prob = mk(_CN["MDOT"] * CP * ddT)
        res = solve(prob, x0=x0)
        assert res.converged, f"mean solve failed at dT={ddT}"
        x0 = res.x
    return prob, res.x


def _nyq(prob, x, fmax=1200.0, npts=900, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return open_loop_response(prob, x, np.linspace(0.0, fmax, npts), **kw)


def _eig_unstable(prob, x, band, iso=True):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = eigenmodes(prob, x, freq_band=band, growth_band=(-300.0, 300.0), isentropic=iso, residual_tol=1e-9)
    return [(f, g) for f, g in zip(r.freqs, r.growth_rates) if g > 0], r


# --------------------------------------------------------------------------
# 1. Exact operator identities (round-off; no tuning)
# --------------------------------------------------------------------------


def test_source_decomposition_reproduces_operator():
    """The rank-1 source terms rebuild ``A(omega) - A_0(omega) = sum_k F_k a_k b_k^T`` exactly."""
    prob, x = _acoustic_rijke(0.9, 3e-3, 250.0)
    blocks = build_acoustic_blocks(prob, x, isentropic=False)
    passive = _passive_blocks(blocks)
    terms = _rank1_terms(blocks)
    A_mat = np.column_stack([t.a for t in terms])
    B = np.column_stack([t.b for t in terms])
    for f in (90.0, 217.0, 350.0):
        w = 2.0 * np.pi * f
        A = assemble_acoustic(w, blocks, with_boundaries=True).toarray()
        A0 = assemble_acoustic(w, passive, with_boundaries=True).toarray()
        Fvec = np.array([complex(t.transfer(f)) for t in terms])
        S = (A_mat * Fvec[None, :]) @ B.T
        assert np.max(np.abs(A - A0 - S)) < 1e-8


def test_determinant_lemma_matches_direct_determinant():
    """``det(I_r + M) = det A / det A_0`` (the lemma the Nyquist count rests on)."""
    prob, x = _acoustic_rijke(0.9, 3e-3, 250.0)
    blocks = build_acoustic_blocks(prob, x, isentropic=False)
    passive = _passive_blocks(blocks)
    terms = _rank1_terms(blocks)
    A_mat = np.column_stack([t.a for t in terms])
    B = np.column_stack([t.b for t in terms])
    r = len(terms)
    for f in (120.0, 300.0):
        w = 2.0 * np.pi * f
        A = assemble_acoustic(w, blocks, with_boundaries=True).toarray()
        A0 = assemble_acoustic(w, passive, with_boundaries=True).toarray()
        Fvec = np.array([complex(t.transfer(f)) for t in terms])
        M = Fvec[:, None] * (B.T @ np.linalg.solve(A0, A_mat))
        D_lemma = np.linalg.det(np.eye(r) + M)
        D_direct = np.linalg.det(A) / np.linalg.det(A0)
        assert abs(D_lemma - D_direct) < 1e-6 * abs(D_direct)


def test_return_ratio_scales_linearly_with_gain():
    """``L(omega)`` is exactly proportional to the FTF gain (``A_0`` and injection are gain-free)."""
    freqs = np.linspace(0.0, 600.0, 200)
    p1, x1 = _acoustic_rijke(0.5, 3e-3, 250.0)
    p2, x2 = _acoustic_rijke(1.5, 3e-3, 250.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r1 = open_loop_response(p1, x1, freqs, isentropic=True, refine=False)
        r2 = open_loop_response(p2, x2, freqs, isentropic=True, refine=False)
    assert np.allclose(r1.freqs, r2.freqs)
    assert np.max(np.abs(r2.L - 3.0 * r1.L)) < 1e-9 * np.max(np.abs(r2.L))


# --------------------------------------------------------------------------
# 2. Cross-check against eigenmodes (purely acoustic, both drivers valid)
# --------------------------------------------------------------------------


def test_unstable_count_matches_eigenmodes():
    """Nyquist's encirclement count equals the number of growing modes eigenmodes finds."""
    prob, x = _acoustic_rijke(0.6, 4e-3, 180.0)
    nyq = _nyq(prob, x, isentropic=True)
    uns, _ = _eig_unstable(prob, x, band=(40.0, 1100.0))
    assert nyq.closed
    assert nyq.n_unstable == len(uns) >= 1


def test_crossing_frequency_matches_eigenmode():
    """A near-marginal mode: the Nyquist crossing frequency matches the eigenmode frequency."""
    prob, x = _acoustic_rijke(0.9, 3e-3, 250.0)
    nyq = _nyq(prob, x, isentropic=True)
    uns, _ = _eig_unstable(prob, x, band=(40.0, 1100.0))
    assert nyq.n_unstable == len(uns)
    # the lowest-frequency unstable mode is the near-marginal one the locus skims
    f_eig = min(f for f, g in uns)
    cross = sorted(nyq.crossings(tol=0.3), key=lambda c: c["freq_hz"])
    assert cross, "expected a near-critical crossing"
    assert cross[0]["freq_hz"] == pytest.approx(f_eig, rel=0.03)


def test_zero_gain_is_stable():
    """With no unsteady heat release the network is the passive (stable) resonator."""
    prob, x = _acoustic_rijke(1e-4, 3e-3, 250.0)
    nyq = _nyq(prob, x, isentropic=True)
    assert nyq.n_unstable == 0 and nyq.stable
    assert nyq.margin > 0.5  # far from the critical point


# --------------------------------------------------------------------------
# 3. The headline: entropy/indirect-noise coupling is the destabilizer
# --------------------------------------------------------------------------


def test_entropy_path_is_the_sole_destabilizer():
    """Stable as a pure-acoustic system, unstable once the convected entropy wave is kept.

    Dropping the entropy wave (``isentropic=True``) leaves only the direct acoustic feedback
    -- the rig is stable.  Keeping it (``isentropic=False``) activates the indirect-noise
    loop (entropy spot convects to the choked nozzle and converts back to sound), which tips
    a mode unstable.  This is exactly the regime the contour eigensolver cannot reach.
    """
    prob, x = _choked_rig()
    assert 0.0 < float(states_table(prob, x)[ES_M, 2]) < 1.0  # subsonic hot flow
    iso = _nyq(prob, x, fmax=1600.0, isentropic=True)
    full = _nyq(prob, x, fmax=1600.0, isentropic=False)
    assert iso.closed and full.closed
    assert iso.n_unstable == 0  # acoustic-only: stable
    assert full.n_unstable >= 1  # entropy retained: unstable
    # the instability sits at a finite frequency (the indirect-noise mode)
    assert any(50.0 < c["freq_hz"] < 1500.0 for c in full.crossings(tol=0.3))


def test_passive_operator_is_stable():
    """``A_0`` (the rig with the FTF removed) has no unstable mode -- the Nyquist count premise.

    The inherited mass-flow inlet and the choked-nozzle outlet are passive, so every passive
    resonance decays; the winding count then returns the absolute unstable-mode number.
    """
    prob0, x0 = _choked_rig(active=False)
    uns, r = _eig_unstable(prob0, x0, band=(50.0, 1600.0), iso=True)
    assert not uns, f"passive operator A_0 unexpectedly unstable: {uns}"
    assert int(np.sum(r.unstable)) == 0


# --------------------------------------------------------------------------
# 4. API behavior
# --------------------------------------------------------------------------


def test_nyquist_requires_a_dynamic_source():
    """A passive network has no return ratio -- the driver points the user to eigenmodes."""
    els = [
        cat.mass_flow_inlet(0.005, 300.0, perturbation_bc=PerturbationBC.hard_wall()),
        cat.duct(0.6),
        cat.pressure_outlet(1.0e5, perturbation_bc=PerturbationBC.open_end()),
    ]
    edges = [(0, 1, 0.01), (1, 2, 0.01)]
    prob = build_problem(perfect_gas(R_AIR, GAMMA), els, edges, mdot_ref=0.005, p_ref=1e5, h_ref=CP * 300.0)
    x = solve(prob).x
    with pytest.raises(ValueError, match="dynamic source"):
        open_loop_response(prob, x, np.linspace(0.0, 600.0, 50))


def test_tabulated_ftf_supported_on_the_real_axis():
    """A measured (tabulated) FTF -- which eigenmodes refuses -- drives the Nyquist test fine.

    The transfer functions are evaluated at *real* frequency, so a real-grid table is
    admissible; it reproduces the closed-form result and gives the same count.
    """
    prob_c, x_c = _acoustic_rijke(0.9, 3e-3, 250.0, tab=False)
    prob_t, x_t = _acoustic_rijke(0.9, 3e-3, 250.0, tab=True)
    nyq_c = _nyq(prob_c, x_c, isentropic=True)
    nyq_t = _nyq(prob_t, x_t, isentropic=True, npts=900)
    assert nyq_t.n_unstable == nyq_c.n_unstable >= 1
    # eigenmodes cannot continue a real-grid table into the complex plane
    with pytest.raises(ValueError):
        eigenmodes(prob_t, x_t, freq_band=(40.0, 300.0), isentropic=True)


def test_nyquist_stability_wrapper_returns_response():
    """``nyquist_stability`` is the convenience entry returning the same response object."""
    prob, x = _acoustic_rijke(0.6, 4e-3, 180.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = nyquist_stability(prob, x, np.linspace(0.0, 1200.0, 900), isentropic=True)
    assert r.summary()["n_unstable"] == r.n_unstable
    assert r.rank == 1


# --------------------------------------------------------------------------
# Off-axis mode estimates (rational fit) and the passive-premise check
# --------------------------------------------------------------------------


def test_mode_estimates_recover_eigenmodes():
    """The rational-fit mode estimates match the off-axis eigenmodes -- frequency and growth."""
    prob, x = _acoustic_rijke(0.6, 4e-3, 180.0)
    nyq = _nyq(prob, x, isentropic=True)
    uns, _ = _eig_unstable(prob, x, band=(40.0, 1100.0))
    ests = nyq.mode_estimates(unstable_only=True)
    assert len(ests) == len(uns) == nyq.n_unstable >= 1
    for f, g in uns:
        m = min(ests, key=lambda e: abs(e["freq_hz"] - f))
        assert m["freq_hz"] == pytest.approx(f, rel=0.02)  # frequency within 2 %
        assert m["growth_rate"] == pytest.approx(g, rel=0.15, abs=3.0)  # and the growth rate
        assert m["unstable"]


def test_mode_estimates_include_stable_modes_too():
    """Without the unstable filter, the fit also returns the stable (decaying) modes."""
    prob, x = _acoustic_rijke(0.9, 3e-3, 250.0)
    nyq = _nyq(prob, x, isentropic=True)
    allm = nyq.mode_estimates()
    assert any(not m["unstable"] for m in allm)  # some decaying modes
    assert sum(m["unstable"] for m in allm) == nyq.n_unstable


def test_passive_premise_holds_for_passive_terminations():
    """A_0 (FTF removed) is stable for the choked rig, so the absolute count equals the count."""
    prob, x = _choked_rig(active=True)
    nyq = _nyq(prob, x, fmax=1600.0, isentropic=False)
    assert nyq.passive_assumption_ok
    assert nyq.n_unstable_passive == 0
    assert nyq.n_unstable_absolute == nyq.n_unstable
    # the located passive resonances are all decaying (the count premise)
    assert all(not r["unstable"] for r in nyq.passive_resonances())


def test_passive_premise_holds_for_lossless_resonator():
    """Even the marginal modes of the lossless hard-wall/open-end A_0 are not flagged unstable."""
    prob, x = _acoustic_rijke(1e-4, 3e-3, 250.0)  # negligible gain: A ~ the passive resonator
    nyq = _nyq(prob, x, isentropic=True)
    assert nyq.n_unstable == 0
    assert nyq.passive_assumption_ok and nyq.n_unstable_passive == 0


# --------------------------------------------------------------------------
# Parameter sweep: the Nyquist stability map (bifurcation diagram)
# --------------------------------------------------------------------------


def _net_rijke(n, tau=3.0e-3):
    """Unsolved near-stagnant Rijke *network* with an n-tau flame -- the public sweep interface."""
    from nefes.elements.dynamic_source import n_tau_flame
    from nefes.shell import Network

    mdot, dT, area, L1, L2 = 0.001, 1000.0, 0.01, 0.25, 0.75
    net = Network(perfect_gas(R_AIR, GAMMA), p_ref=1.0e5, T_ref=300.0)
    inlet = net.add(cat.mass_flow_inlet(mdot, 300.0, perturbation_bc=PerturbationBC.open_end()))
    cold = net.add(cat.duct(L1))
    flame = net.add(cat.heat_release_flame(mdot * CP * dT))
    hot = net.add(cat.duct(L2))
    outlet = net.add(cat.pressure_outlet(1.0e5, perturbation_bc=PerturbationBC.open_end()))
    net.connect(inlet, cold, area=area)
    ref = net.connect(cold, flame, area=area)
    net.connect(flame, hot, area=area)
    net.connect(hot, outlet, area=area)
    net.set_dynamic_source(flame, n_tau_flame(n, tau, ref_edge=ref))
    return net


def test_stability_map_detects_the_ita_bifurcation():
    """Dialing the FTF gain to zero steps the unstable count down as the ITA mode restabilizes.

    The count is the robust integer the contour eigensolver sprays artifacts around at mid-gain;
    the step (3 -> 2) and the margin collapse pin the stability boundary and its onset frequency.
    """
    from nefes.perturbation import NyquistStabilityMap, nyquist_stability_map

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mp = nyquist_stability_map(
            build=_net_rijke,
            params=np.linspace(1.0, 0.05, 25),
            freqs=np.linspace(0.0, 1000.0, 801),
            isentropic=True,
            param_name="FTF gain n",
        )
    assert isinstance(mp, NyquistStabilityMap)
    assert mp.n_unstable[0] == 3 and mp.n_unstable[-1] == 2  # the ITA mode leaves the unstable set
    assert len(mp.onsets) == 1  # exactly one count change over the sweep
    _, _, delta = mp.onsets[0]
    assert delta == -1  # a single mode restabilizes
    assert float(np.min(mp.margin)) < 0.05  # the margin collapses where the mode crosses
    assert mp.all_closed  # every locus closed -> every count is a converged total


def test_stability_map_accepts_presolved_and_rejects_sourceless():
    """``build`` may return a solved solution; a network without a dynamic source is rejected."""
    from nefes.perturbation import nyquist_stability_map
    from nefes.shell import Network

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mp = nyquist_stability_map(
            build=lambda n: _net_rijke(n).solve(),  # a Solution, not a Network
            params=np.linspace(1.0, 0.7, 5),
            freqs=np.linspace(0.0, 1000.0, 401),
            isentropic=True,
        )
    assert mp.n_unstable.shape == (5,)

    def sourceless(_p):
        net = Network(perfect_gas(R_AIR, GAMMA), p_ref=1.0e5, T_ref=300.0)
        a = net.add(cat.mass_flow_inlet(0.005, 300.0, perturbation_bc=PerturbationBC.hard_wall()))
        d = net.add(cat.duct(0.6))
        b = net.add(cat.pressure_outlet(1.0e5, perturbation_bc=PerturbationBC.open_end()))
        net.connect(a, d, area=0.01)
        net.connect(d, b, area=0.01)
        return net

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="dynamic source"):
            nyquist_stability_map(build=sourceless, params=np.linspace(1.0, 0.5, 3), freqs=np.linspace(0.0, 600.0, 200))
