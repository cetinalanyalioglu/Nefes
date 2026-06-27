"""Eigenvalue trajectories: track the modes of ``det A(omega) = 0`` as a parameter varies.

A single eigenmode solve answers *where are the modes at this operating point*.  Many
questions are instead about *motion*: does this mode cross into instability as the flame
gain rises, does that resonance survive when the boundary is opened, is a given mode
intrinsic (ITA) or a cavity (acoustic) mode?  All of these are answered by varying one
parameter and watching each eigenvalue trace a curve in the complex plane.

The reliable way to draw those curves is **continuation**, not independent re-solves: seed
the spectrum once with :func:`eigenmodes`, then *march* each mode by a predictor-corrector
step -- linearly extrapolate where the eigenvalue is heading, then Newton-correct there
(:func:`_corrector`), seeded by the previous step's eigenvector.  Following one branch this
way sidesteps the assignment ambiguity of matching two independent spectra across a
near-degeneracy or an avoided crossing (where it is genuinely ill-posed).

The corrector converges on ``omega`` (the eigenvalue update ``|dw|``) rather than on a scaled
residual: the network operator can carry a hugely dominant boundary/regularizer entry, so
``||A x|| / max|A|`` reads "converged" for a stale eigenvalue long before it actually is --
an ``omega``-update test is immune to that.

The parameter can be *anything* about the setup -- a duct length, an area, a mass-flow, an
FTF gain or delay, a boundary reflection -- because the caller supplies a ``build`` callable
mapping the parameter value to a network.  Parameters that reshape the mean flow re-solve it
(warm-started from the previous step for continuity and speed); parameters that touch only
the perturbation operator (FTF gain/delay, boundary impedance) cost a near-trivial re-solve.

See :func:`eigenvalue_trajectory` for the entry point and :class:`TrajectoryResult` for the
output (with plotly views).
"""

import warnings
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse.linalg as spla

from .eigenmodes import build_operator, eigenmodes, EigenmodeResult


class TrajectoryWarning(UserWarning):
    """Diagnostic from continuation: an empty seed, a lost branch, a non-converged mean state."""


# Relative |omega| update below which the Newton corrector is considered converged onto an
# eigenvalue.  A scale-free test in omega -- deliberately NOT a residual ||A x|| / max|A|,
# which is blind to the eigenvalue when the operator carries a hugely dominant entry (a
# boundary/regularizer stamp can be ~1e9, swamping the residual long before omega settles).
_CONVERGE_RTOL = 1.0e-9
_CORRECTOR_MAXIT = 40

# A committed step that lands more than this multiple of the branch's recent stride away from
# where the predictor pointed is flagged as a possible branch swap / sharp turn.
_JUMP_FACTOR = 4.0
# ...but only once the absolute jump exceeds this floor (rad/s ~ a few Hz), so tiny steps near
# a stationary mode never trip the guard.
_JUMP_FLOOR = 2.0 * np.pi * 5.0

# Two live branches closer than this (relative to |omega|) are flagged as a near-collision
# (mode veering / a possible exceptional point), where the branch identity is ambiguous.
_COLLISION_RTOL = 8.0e-3


def _corrector(A_of, w0, v0, *, rtol=_CONVERGE_RTOL, maxit=_CORRECTOR_MAXIT):
    """Newton (residual-inverse iteration) to the eigenvalue of ``A`` nearest ``w0``.

    Iterates on ``omega`` until the update ``|dw|`` is negligible relative to ``|w|`` -- a
    scale-free convergence test.  The derivative action ``A'(omega) x`` is a central
    difference in ``omega`` (the operator is complex-analytic in it), so the step is
    source-agnostic.  Seeded by the previous continuation step's eigenvector for fast,
    identity-preserving convergence.

    Returns
    -------
    w : complex
        Converged eigenvalue.
    x : ndarray
        Unit-norm eigenvector.
    converged : bool
        Whether the ``|dw|`` test was met within ``maxit``.
    rel : float
        Achieved relative update ``|dw| / max(|w|, 1)`` (small == well converged).
    """
    w = complex(w0)
    x = np.asarray(v0, dtype=np.complex128)
    x = x / np.linalg.norm(x)
    rel = np.inf
    converged = False
    for _ in range(maxit):
        A = A_of(w)
        try:
            lu = spla.splu(A.tocsc())
        except RuntimeError:
            converged, rel = True, 0.0  # operator singular at w: w is an eigenvalue
            break
        h = 1e-6 * (abs(w) + 1.0)
        Ap_x = (A_of(w + h) @ x - A_of(w - h) @ x) / (2.0 * h)  # A'(omega) x
        y = lu.solve(Ap_x)
        denom = np.vdot(x, y)  # x^H y
        if denom == 0.0:
            break
        dw = -1.0 / denom
        x_new = -dw * y
        nrm = np.linalg.norm(x_new)
        if nrm == 0.0:
            break
        x = x_new / nrm
        w = w + dw
        rel = abs(dw) / max(abs(w), 1.0)
        if rel <= rtol:
            converged = True
            break
    return w, x, converged, float(rel)


@dataclass
class TrajectoryBranch:
    """One eigenvalue's path through the complex plane as the parameter is swept.

    A branch is born at the seed point and extended one parameter step at a time; it may be
    retired early if the corrector loses it (e.g. a mode racing out of the trackable region).
    Frequencies are in Hz (``Re(omega)/(2*pi)``) and growth rates in 1/s (``-Im(omega)``),
    matching :class:`fns.perturbation.EigenmodeResult`.

    Attributes
    ----------
    id : int
        Branch index (its seed-mode index; stable across the sweep).
    params : ndarray
        Parameter samples this branch is defined at (includes any adaptive sub-steps), in
        march order, shape ``(k,)``.
    omega : ndarray
        Complex modal angular frequencies (rad/s) at each sample, shape ``(k,)``.
    residuals : ndarray
        Per-sample corrector convergence -- the achieved relative ``omega``-update
        ``|dw| / max(|w|, 1)`` (small == well converged), shape ``(k,)``.  Index 0 holds the
        seed eigensolve's residual instead.
    modes : ndarray or None
        Unit-norm nodal eigenvectors at each sample, shape ``(k, n_col)`` (``None`` if
        ``store_modes=False`` was passed).
    events : list of (float, str)
        Flagged occurrences ``(parameter, message)`` -- near-collisions, weak corrections.
    alive : bool
        Whether the branch survived to the final parameter value.
    """

    id: int
    params: np.ndarray
    omega: np.ndarray
    residuals: np.ndarray
    modes: Optional[np.ndarray] = None
    events: List[Tuple[float, str]] = field(default_factory=list)
    alive: bool = True

    @property
    def freqs(self) -> np.ndarray:
        """Modal frequencies along the branch (Hz)."""
        return self.omega.real / (2.0 * np.pi)

    @property
    def growth(self) -> np.ndarray:
        """Growth rates along the branch (1/s); positive is unstable."""
        return -self.omega.imag

    @property
    def tangent(self) -> np.ndarray:
        """Sensitivity ``d(omega)/d(parameter)`` along the branch (rad/s per unit parameter).

        A finite-difference tangent of the tracked curve; ``-Im(tangent)`` is the growth-rate
        sensitivity, the headline number for "how fast does this mode move when I change the
        parameter".  Length matches :attr:`params`.
        """
        if self.params.size < 2:
            return np.zeros_like(self.omega)
        return np.gradient(self.omega, self.params)

    @property
    def start(self) -> complex:
        """Eigenvalue at the first parameter value (the seed point)."""
        return complex(self.omega[0])

    @property
    def end(self) -> complex:
        """Eigenvalue at the last parameter value reached."""
        return complex(self.omega[-1])

    def __repr__(self) -> str:
        p0, p1 = float(self.params[0]), float(self.params[-1])
        tail = "" if self.alive else ", retired early"
        flag = f", {len(self.events)} event(s)" if self.events else ""
        return (
            f"TrajectoryBranch {self.id}: {self.params.size} sample(s) over "
            f"[{p0:.4g}, {p1:.4g}]; f {self.freqs[0]:.4g}->{self.freqs[-1]:.4g} Hz, "
            f"growth {self.growth[0]:+.4g}->{self.growth[-1]:+.4g} 1/s{flag}{tail}"
        )


class _Tracker:
    """Mutable, growable backing for one branch during the march (frozen to a branch at the end)."""

    def __init__(self, bid, p0, w0, v0, r0, store_modes):
        self.id = int(bid)
        self.ps = [float(p0)]
        self.ws = [complex(w0)]
        self.rs = [float(r0)]
        self.vs = [np.asarray(v0, dtype=np.complex128).copy()] if store_modes else None
        self.last_v = np.asarray(v0, dtype=np.complex128).copy()
        self.events: List[Tuple[float, str]] = []
        self.alive = True
        self.store_modes = store_modes

    def predict(self, p_new):
        """Linear (secant) extrapolation of omega to ``p_new`` from the last two committed points."""
        if len(self.ps) >= 2 and self.ps[-1] != self.ps[-2]:
            w1, w0 = self.ws[-1], self.ws[-2]
            p1, p0 = self.ps[-1], self.ps[-2]
            return w1 + (w1 - w0) * ((p_new - p1) / (p1 - p0))
        return self.ws[-1]

    def commit(self, p, w, v, r):
        self.ps.append(float(p))
        self.ws.append(complex(w))
        self.rs.append(float(r))
        v = np.asarray(v, dtype=np.complex128)
        if self.store_modes:
            self.vs.append(v.copy())
        self.last_v = v.copy()

    def event(self, p, msg):
        self.events.append((float(p), msg))

    def freeze(self) -> TrajectoryBranch:
        return TrajectoryBranch(
            id=self.id,
            params=np.array(self.ps, dtype=float),
            omega=np.array(self.ws, dtype=np.complex128),
            residuals=np.array(self.rs, dtype=float),
            modes=(np.array(self.vs, dtype=np.complex128) if self.store_modes else None),
            events=list(self.events),
            alive=self.alive,
        )


def _solved_state(obj, x_warm, warm_start):
    """Coerce a ``build`` result into ``(prob, x)``.

    ``obj`` is either an unsolved network (has ``.solve`` -- we solve it, warm-started from the
    previous converged state) or an already-solved solution (has ``.problem`` and ``.x``).
    """
    if callable(getattr(obj, "solve", None)):
        if warm_start and x_warm is not None:
            try:
                sol = obj.solve(x0=x_warm)
            except Exception:
                # a stale warm start (e.g. the graph changed shape) -- restart cold so the
                # operator-size check downstream can report the topology change cleanly.
                sol = obj.solve(x0=None)
        else:
            sol = obj.solve(x0=None)
    else:
        sol = obj
    if not bool(getattr(sol, "converged", True)):
        raise RuntimeError(
            "the mean flow failed to converge at a parameter value -- the eigenvalue trajectory "
            "cannot continue through a non-converged state. Narrow the parameter range or its step."
        )
    return sol.problem, np.asarray(sol.x)


def eigenvalue_trajectory(
    build: Callable,
    params: Sequence[float],
    *,
    freq_band=None,
    growth_band=None,
    isentropic=False,
    eps=None,
    eps_fb=1e-6,
    u_floor=1e-8,
    residual_tol=1e-6,
    max_step_halvings=4,
    store_modes=True,
    warm_start=True,
    param_name="parameter",
    rng=None,
    seed_kwargs=None,
):
    r"""Track the perturbation network's eigenmodes as one setup parameter is varied.

    The spectrum is seeded once at ``params[0]`` with :func:`eigenmodes`; each mode is then
    continued through the remaining parameter values by a predictor-corrector march (linear
    extrapolation, then a Newton polish seeded by the previous eigenvector), so every mode
    traces a continuous branch in the complex plane.  This follows individual modes through
    near-degeneracies and avoided crossings far more reliably than matching independent
    spectra step to step.

    Parameters
    ----------
    build : callable
        ``build(p)`` returning the network at parameter value ``p``.  Return either an
        **unsolved** network (a :class:`fns.shell.Network`; it is solved here, warm-started
        from the previous step when ``warm_start``) or an already-solved solution object
        (anything exposing ``.problem``, ``.x``, ``.converged``).  The graph **topology must
        not change** across the sweep -- only parameter values may differ.
    params : array_like
        Parameter values to sweep, in march order (e.g. ``np.linspace(1.0, 0.0, 41)`` to dial
        a gain to zero).  The first value is the seed point.  May be increasing or decreasing.
    freq_band : tuple of float
        ``(f_lo, f_hi)`` seed search window in Hz (forwarded to :func:`eigenmodes`).
    growth_band : tuple of float, optional
        ``(g_lo, g_hi)`` seed growth-rate window in 1/s.  Only constrains the *seed*; a branch
        is then free to be continued outside it (e.g. an ITA mode diving in growth).
    isentropic : bool, optional
        Acoustic-only perturbations (pin the entropy wave), default False.
    eps, eps_fb, u_floor : float, optional
        Operator-assembly regularizers forwarded to :func:`build_operator`.
    residual_tol : float, optional
        Scaled-residual acceptance for a corrected eigenvalue (default 1e-6).
    max_step_halvings : int, optional
        Maximum adaptive interval bisections per parameter step (default 4).  When a corrector
        fails at the full step, the parameter interval is halved (re-solving the mean flow at
        the midpoint) and the ensemble re-marched, recursively, up to this depth.
    store_modes : bool, optional
        Keep the nodal eigenvector at every sample (default True) for downstream mode-shape or
        classification work; set False to save memory.
    warm_start : bool, optional
        Seed each mean-flow solve from the previous step's converged state (default True).
    param_name : str, optional
        Label for the swept parameter, used in reprs and plots (e.g. ``"FTF gain n"``).
    rng : numpy.random.Generator, optional
        Random source for the Beyn seed (default: a fixed, reproducible seed).
    seed_kwargs : dict, optional
        Extra keyword arguments forwarded to the seeding :func:`eigenmodes` call (e.g.
        ``n_nodes``, ``n_probe``, ``certify``).

    Returns
    -------
    TrajectoryResult
        The seed spectrum plus one :class:`TrajectoryBranch` per mode.

    Notes
    -----
    Continuation is reliable *along* a branch but cannot manufacture a crisp label where the
    physics is genuinely fuzzy: at an exact crossing/exceptional point two branches coalesce
    and identity is undefined.  Such events are flagged on the branches
    (:attr:`TrajectoryBranch.events`), not silently smoothed over.
    """
    params = np.asarray(params, dtype=float)
    if params.size < 2:
        raise ValueError("provide at least two parameter values to trace a trajectory")
    rng = np.random.default_rng(0) if rng is None else rng

    # --- seed the spectrum at the first parameter value ---------------------------------------
    prob0, x0 = _solved_state(build(params[0]), None, warm_start)
    seed = eigenmodes(
        prob0,
        x0,
        freq_band=freq_band,
        growth_band=growth_band,
        isentropic=isentropic,
        eps=eps,
        eps_fb=eps_fb,
        u_floor=u_floor,
        residual_tol=residual_tol,
        rng=rng,
        **(seed_kwargs or {}),
    )
    if seed.n_modes == 0:
        warnings.warn(
            "the seed spectrum is empty: no modes to trace. Widen freq_band/growth_band.",
            TrajectoryWarning,
            stacklevel=2,
        )
        return TrajectoryResult(branches=[], seed=seed, params=params, param_name=param_name, isentropic=isentropic)

    n_col = int(seed.modes.shape[1])
    trackers = [
        _Tracker(i, params[0], seed.omega[i], seed.modes[i], float(seed.residuals[i]), store_modes)
        for i in range(seed.n_modes)
    ]
    live = list(trackers)
    close_pairs: set = set()  # currently-flagged near-collision pairs (de-bounce repeats)

    def _correct_all(A_of, p_target):
        """Predict + Newton-correct every live branch at ``p_target``.

        Returns one ``(w, v, converged, rel, w_pred)`` per live branch.
        """
        out = []
        for b in live:
            w_pred = b.predict(p_target)
            w, v, conv, rel = _corrector(A_of, w_pred, b.last_v)
            out.append((complex(w), v, bool(conv), float(rel), complex(w_pred)))
        return out

    def _flag_collisions(p):
        """Flag pairs of live branches that have drawn within ``_COLLISION_RTOL`` (veering / EP)."""
        for a in range(len(live)):
            for b in range(a + 1, len(live)):
                wi, wj = live[a].ws[-1], live[b].ws[-1]
                scale = max(abs(wi), abs(wj), 1.0)
                key = (live[a].id, live[b].id)
                if abs(wi - wj) < _COLLISION_RTOL * scale:
                    if key not in close_pairs:
                        live[a].event(p, f"near-collision with branch {live[b].id} (veering / possible EP)")
                        live[b].event(p, f"near-collision with branch {live[a].id} (veering / possible EP)")
                        close_pairs.add(key)
                elif abs(wi - wj) > 2.0 * _COLLISION_RTOL * scale:
                    close_pairs.discard(key)

    def _jumped(b, w, w_pred):
        """Did branch ``b``'s corrected eigenvalue land anomalously far from the prediction?

        Only meaningful once a secant predictor exists (>=2 committed points); a large landing
        relative to the branch's own recent stride signals a swap onto a neighbouring mode or a
        sharp turn the step was too coarse to resolve.
        """
        if len(b.ws) < 2:
            return False
        recent = abs(b.ws[-1] - b.ws[-2])
        return abs(w - w_pred) > max(_JUMP_FACTOR * recent, _JUMP_FLOOR)

    def _commit(p_target, trials):
        """Commit a step: extend each branch, flag large jumps, retire non-converged ones."""
        retire = []
        for b, (w, v, conv, rel, w_pred) in zip(live, trials):
            jumped = _jumped(b, w, w_pred)
            b.commit(p_target, w, v, rel)
            if not conv:
                b.alive = False
                b.event(p_target, f"lost track (corrector did not converge, rel {rel:.1e}); branch retired")
                retire.append(b)
                continue
            if jumped:
                b.event(
                    p_target,
                    f"large step ({abs(w - w_pred) / (2 * np.pi):.0f} Hz vs predicted); "
                    "possible branch swap / sharp turn",
                )
        _flag_collisions(p_target)
        for b in retire:
            live.remove(b)

    def _advance(p_lo, p_hi, x_lo, depth):
        """March all live branches from ``p_lo`` to ``p_hi``, bisecting on corrector failure."""
        prob, x = _solved_state(build(p_hi), x_lo, warm_start)
        A_of, blocks, *_ = build_operator(prob, x, eps=eps, eps_fb=eps_fb, u_floor=u_floor, isentropic=isentropic)
        if int(blocks.J_alg.shape[0]) != n_col:
            raise ValueError(
                "the operator size changed across the sweep: the network topology must stay fixed "
                "(only parameter values may vary). Check that `build` keeps the same graph."
            )
        trials = _correct_all(A_of, p_hi)
        # A step is clean only if every branch converged AND none landed anomalously far from its
        # prediction (a swap onto a crowding neighbour) -- the latter is what subdivision fixes,
        # by handing the predictor a closer target so it stays on the fast-moving branch.
        clean = all(t[2] for t in trials) and not any(_jumped(b, t[0], t[4]) for b, t in zip(live, trials))
        if clean or depth >= max_step_halvings or not live:
            _commit(p_hi, trials)
            return x
        p_mid = 0.5 * (p_lo + p_hi)
        x_mid = _advance(p_lo, p_mid, x_lo, depth + 1)
        return _advance(p_mid, p_hi, x_mid, depth + 1)

    # --- march -------------------------------------------------------------------------------
    x_warm = x0
    for k in range(1, params.size):
        if not live:
            break
        x_warm = _advance(params[k - 1], params[k], x_warm, 0)

    branches = [t.freeze() for t in trackers]
    return TrajectoryResult(branches=branches, seed=seed, params=params, param_name=param_name, isentropic=isentropic)


@dataclass
class TrajectoryResult:
    """Eigenvalue trajectories of a perturbation network over a parameter sweep.

    Holds one :class:`TrajectoryBranch` per seeded mode plus the seed
    :class:`fns.perturbation.EigenmodeResult` (for the starting spectrum and its completeness
    certificate).  See :meth:`plot` (complex-plane paths) and :meth:`plot_vs_param` (frequency
    and growth versus the parameter).

    Attributes
    ----------
    branches : list of TrajectoryBranch
        One traced path per seed mode.
    seed : EigenmodeResult
        The spectrum at the first parameter value (the continuation seed).
    params : ndarray
        The requested parameter march values.
    param_name : str
        Label for the swept parameter.
    isentropic : bool
        Whether the operator was assembled acoustic-only.
    """

    branches: List[TrajectoryBranch]
    seed: EigenmodeResult
    params: np.ndarray
    param_name: str = "parameter"
    isentropic: bool = False

    def __len__(self) -> int:
        return len(self.branches)

    @property
    def n_branches(self) -> int:
        """Number of traced branches."""
        return len(self.branches)

    def __iter__(self):
        return iter(self.branches)

    def __getitem__(self, i) -> TrajectoryBranch:
        return self.branches[i]

    def _status(self) -> str:
        n = len(self.branches)
        retired = sum(0 if b.alive else 1 for b in self.branches)
        flagged = sum(1 for b in self.branches if b.events)
        bits = [f"{n} branch{'' if n == 1 else 'es'}"]
        if retired:
            bits.append(f"{retired} retired")
        if flagged:
            bits.append(f"{flagged} flagged")
        return ", ".join(bits)

    def __repr__(self) -> str:
        p = np.asarray(self.params, dtype=float)
        span = "empty" if p.size == 0 else f"{self.param_name} {p[0]:.4g} -> {p[-1]:.4g} ({p.size} steps)"
        lines = [f"TrajectoryResult: {self._status()} over {span}"]
        for b in self.branches:
            mark = " " if b.alive else "x"
            lines.append(
                f"  [{mark}] branch {b.id}: f {b.freqs[0]:7.4g}->{b.freqs[-1]:7.4g} Hz, "
                f"growth {b.growth[0]:+8.4g}->{b.growth[-1]:+8.4g} 1/s"
                + (f"  ({len(b.events)} event)" if b.events else "")
            )
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        p = np.asarray(self.params, dtype=float)
        span = "empty" if p.size == 0 else f"{self.param_name}: {p[0]:.4g} &rarr; {p[-1]:.4g} ({p.size} steps)"
        head = f"<b>TrajectoryResult</b> &mdash; {self._status()}<br><span style='color:#52606d'>{span}</span>"
        rows = [
            "<tr><th style='text-align:right;padding:2px 8px'>branch</th>"
            "<th style='text-align:right;padding:2px 8px'>f start &rarr; end [Hz]</th>"
            "<th style='text-align:right;padding:2px 8px'>growth start &rarr; end [1/s]</th>"
            "<th style='text-align:left;padding:2px 8px'>status</th></tr>"
        ]
        for b in self.branches:
            status = "alive" if b.alive else "retired"
            if b.events:
                status += f", {len(b.events)} event(s)"
            unst = b.growth[-1] > 0.0
            bg = "background:#fdecea;color:#611a15;" if unst else ""
            rows.append(
                f"<tr style='{bg}'>"
                f"<td style='text-align:right;padding:2px 8px'>{b.id}</td>"
                f"<td style='text-align:right;padding:2px 8px'>{b.freqs[0]:.4g} &rarr; {b.freqs[-1]:.4g}</td>"
                f"<td style='text-align:right;padding:2px 8px'>{b.growth[0]:+.4g} &rarr; {b.growth[-1]:+.4g}</td>"
                f"<td style='text-align:left;padding:2px 8px'>{status}</td></tr>"
            )
        table = "<table style='border-collapse:collapse;font-size:0.9em'>" + "".join(rows) + "</table>"
        return head + "<br>" + table

    def plot(self, *, title=None, show_markers=True, **layout):
        """Plot the eigenvalue paths in the (frequency, growth-rate) plane, one line per branch.

        Each branch is a coloured curve from its seed point (circle marker) to its final point
        (x marker); hovering a point reads off the parameter value.  The dashed line at
        ``growth = 0`` is the stability boundary.

        Parameters
        ----------
        title : str, optional
            Figure title.
        show_markers : bool, optional
            Draw a marker at every swept sample (default True), not just the endpoints.
        **layout
            Forwarded to ``Figure.update_layout``.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        import plotly.graph_objects as go

        from ..plotting.theme import COLORWAY, FNS_TEMPLATE_NAME

        fig = go.Figure()
        for b in self.branches:
            color = COLORWAY[b.id % len(COLORWAY)]
            fig.add_trace(
                go.Scatter(
                    x=b.freqs,
                    y=b.growth,
                    mode="lines+markers" if show_markers else "lines",
                    name=f"branch {b.id}",
                    line=dict(color=color, width=2),
                    marker=dict(size=5, color=color),
                    customdata=b.params,
                    hovertemplate=(
                        f"branch {b.id}<br>{self.param_name} = %{{customdata:.4g}}"
                        "<br>f = %{x:.5g} Hz<br>growth = %{y:.5g} 1/s<extra></extra>"
                    ),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[b.freqs[0]],
                    y=[b.growth[0]],
                    mode="markers",
                    marker=dict(size=11, color=color, symbol="circle", line=dict(width=1.5, color="white")),
                    showlegend=False,
                    hovertemplate=f"branch {b.id} start ({self.param_name}={b.params[0]:.4g})<extra></extra>",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[b.freqs[-1]],
                    y=[b.growth[-1]],
                    mode="markers",
                    marker=dict(size=11, color=color, symbol="x"),
                    showlegend=False,
                    hovertemplate=f"branch {b.id} end ({self.param_name}={b.params[-1]:.4g})<extra></extra>",
                )
            )
        fig.add_hline(y=0.0, line_dash="dash", line_color="#9aa5b1", line_width=1.4)
        fig.update_layout(
            template=FNS_TEMPLATE_NAME,
            title=title or f"Eigenvalue trajectories vs {self.param_name}",
            xaxis_title="frequency [Hz]",
            yaxis_title="growth rate −Im(ω) [1/s]",
            showlegend=True,
        )
        fig.update_layout(**layout)
        return fig

    def plot_vs_param(self, *, title=None, **layout):
        """Plot frequency (top) and growth rate (bottom) against the swept parameter.

        Two stacked panels sharing the parameter axis; clearer than the complex-plane view for
        reading a monotone trend (e.g. an ITA mode's growth diving as a gain is dialed to zero).

        Returns
        -------
        plotly.graph_objects.Figure
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        from ..plotting.theme import COLORWAY, FNS_TEMPLATE_NAME

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.07,
            subplot_titles=("frequency [Hz]", "growth rate −Im(ω) [1/s]"),
        )
        for b in self.branches:
            color = COLORWAY[b.id % len(COLORWAY)]
            name = f"branch {b.id}"
            fig.add_trace(
                go.Scatter(
                    x=b.params, y=b.freqs, mode="lines+markers", name=name, line=dict(color=color), legendgroup=name
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=b.params,
                    y=b.growth,
                    mode="lines+markers",
                    name=name,
                    line=dict(color=color),
                    legendgroup=name,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
        fig.add_hline(y=0.0, line_dash="dash", line_color="#9aa5b1", line_width=1.4, row=2, col=1)
        fig.update_xaxes(title_text=self.param_name, row=2, col=1)
        fig.update_layout(template=FNS_TEMPLATE_NAME, title=title or f"Eigenvalues vs {self.param_name}")
        fig.update_layout(**layout)
        return fig
