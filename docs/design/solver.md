# The solver

Newton's method converges rapidly near a solution and often not at all far from one, so iterating it alone on the network equations would fail when the solver must start from rest with no prior guess.
This document describes the safeguards that close that gap: scaling, step damping, and staged continuation that together find the operating point from rest on an arbitrary network, and the reuse of previous solutions that keeps later solves inexpensive.
It relies on the exact Jacobian from [the complex-step derivative](complex-step.qmd) and the assembled equations of [assembly](assembly.md), and it puts into practice the *discovery over prescription* principle of [the design philosophy](philosophy.md).

## Nondimensionalization {#sec-solver-scaling}

The network variables span many orders of magnitude — a mass flow of order unity beside a total enthalpy of order $10^5$ — so an unscaled residual makes a convergence tolerance meaningless and the linear algebra needlessly ill-conditioned.
The solver therefore works in nondimensional variables and residuals, scaling each unknown by a reference quantity and each residual by the reference of its own kind, given as:

$$
\widehat{\dot m} = \frac{\dot m}{\dot m_{\text{ref}}},
\quad
\widehat{p} = \frac{p}{p_{\text{ref}}},
\quad
\widehat{h}_t = \frac{h_t}{c_p T_{\text{ref}}},
$$

where the hatted quantities are the scaled unknowns and $\dot m_{\text{ref}}$, $p_{\text{ref}}$, $T_{\text{ref}}$ are the problem references.
With mass, pressure, and energy residuals each reduced to order unity, "small" means the same thing across every equation, and the convergence test and the linear solve both behave.
The references are a mix of user anchors and automatic estimates: $p_{\text{ref}}$ and $T_{\text{ref}}$ are set on the network (defaults $101325\,\mathrm{Pa}$ and $300\,\mathrm{K}$; see [reference/parameters](../reference/parameters.md)), $\dot m_{\text{ref}}$ and $h_{\text{ref}}$ are derived at build time from the boundary specification when not overridden (total specified inflow, an isentropic estimate from the largest boundary pressure drop, or a low-Mach fallback for a quiescent network; $h_{\text{ref}} = c_p T_{\text{ref}}$ for a perfect gas), and during the solve the mass and enthalpy scales are re-measured from the realized inlet flows at each continuation ($\kappa$) stage while $p_{\text{ref}}$ stays fixed, so the scaling tracks the flow once it establishes without collapsing at the quiescent start.

## The feed-mixing seed {#sec-solver-seed}

Newton converges from a guess that lies in the basin of the root, and the one quantity that can place a guess far outside it is the total enthalpy, which the network spreads across its edges while the pressure and the mass flow stay comparatively uniform.
A single uniform enthalpy is therefore an adequate start only when every edge sits near the same $h_t$; where it does not, the solver seeds each edge by propagating the feeds through the graph.
The mass flows are propagated first (inlets and sources inject, junctions sum, splitters divide), and the advected scalars, $h_t$ and the mixture fractions, are then blended mass-weighted by that flow.
Because conserved scalars mix linearly, each edge lands at its adiabatic-mixing state, from which the closure recovers the temperature.

Two kinds of network spread $h_t$ far enough to need this.
A reacting network carries absolute, formation-inclusive enthalpies, so an unburnt air edge and a burnt edge sit at very different $h_t$.
A network carrying a heat-release flame does the same for a different reason: the flame raises the total enthalpy of its outflow by

$$
\Delta h_t = \frac{\dot{Q}}{|\overline{\dot m}|},
$$

where $\dot{Q}$ is the prescribed power, so the edges downstream of it sit hundreds of kelvin above the feed.
The seed adds this rise on top of the mixing estimate, using the mass flow it has already propagated.

That divisor is exact wherever an inlet prescribes the flow, and an estimate wherever it does not.
A network fed through a total-pressure inlet leaves its mass flow to the solve, so the propagation has nothing to carry onto the flame's edge and falls back to $\dot m_{\text{ref}}$.
The estimate errs in a forgiving direction on its own: it is measured cold and cannot see that heat release throttles the flow, so it sits *below* the true value, which overstates the rise and seeds the flame hot.
That is the cheap side of the curve, and the solve absorbs it, taking more steps as the discrepancy grows.
The costly direction is a reference set well *above* the true flow, which understates the rise and seeds the flame cold on the steep side described below.
A solve therefore compares the flow its flame seed assumed against the one it reached, and reports a flame seeded from a mass flow an order of magnitude too large, naming $\dot m_{\text{ref}}$ as the quantity to reconsider (tests: `test_cold_flame_seed_is_reported`, `test_pt_inlet_flame_converges_from_default_seed`).
Leaving $\dot m_{\text{ref}}$ unset is the reliable choice: it is then derived from the boundary specification, where the error stays in the forgiving direction.

Seeding the burnt side cold is not a matter of a few wasted iterations.
The enthalpy rise varies as $1/\overline{\dot m}$, so the energy row's sensitivity to the mass flow, $\partial(\Delta h_t)/\partial\overline{\dot m} = -\dot{Q}/\overline{\dot m}^{2}$, steepens without bound as the flow falls.
A guess on the cold side of that wall sends the first step toward a smaller mass flow, which steepens the wall further; the line search then rejects every trial and the iteration stalls with the flow collapsed toward rest.
Anticipating the rise in the seed starts the iteration on the shallow side, and the solve converges in a handful of steps at any heat release (tests: `test_heat_release_flame_converges_from_default_seed`, `test_heat_release_flame_default_seed_matches_ramped_solve`).

## Damped Newton steps {#sec-solver-damping}

Each iteration solves the scaled Newton system, damped in the Levenberg–Marquardt manner so that a step remains well defined even where the Jacobian is momentarily singular, given as:

$$
\big(\overline{\mathbf{J}}^{\top}\overline{\mathbf{J}} + \lambda\mathbf{I}\big)\,\delta\mathbf{y} = -\,\overline{\mathbf{J}}^{\top}\widehat{\mathbf{R}},
$$

where $\overline{\mathbf{J}}$ is the scaled Jacobian, $\widehat{\mathbf{R}}$ the scaled residual, $\delta\mathbf{y}$ the update, and $\lambda$ the damping parameter.
For $\lambda \to 0$ the step is the pure Newton step, recovering quadratic convergence near the solution; for larger $\lambda$ it blends toward a cautious gradient-descent step that makes progress where the pure step would overshoot or where the Jacobian has lost rank — for instance at the undetermined split of a perfectly symmetric branching network at rest (see [well-posedness](../theory/well-posedness.md#sec-wellposed-physical-indeterminacy)).
The damping is adapted per iteration, raised when a step fails to reduce the residual and lowered as the iteration homes in, so the solver interpolates automatically between robustness far out and speed near the answer.

## The artificial-resistance continuation {#sec-solver-continuation}

The damped Newton step is still defeated by one structural trap: in a network driven only by pressure boundary conditions, the residuals have zero first-order sensitivity to the flows at the quiescent state, so the solver sees a flat landscape and cannot start the flow moving (see [well-posedness](../theory/well-posedness.md#sec-wellposed-zero-flow)).
The cure is a continuation in a physical parameter [@allgower_georg_1990]: a small fictitious friction $\kappa$ added to every pressure-type row.
This injects first-order flow sensitivity without changing the final answer.
With the friction active the network behaves like a resistive circuit, in which pressure differences push directly on the flows, and the solver locates the flow pattern readily.
The same zero-flow singularity is met by hydraulic pipe-network solvers, whose loss law is likewise quadratic in the flow: the Jacobian of the global gradient algorithm [@todini_pilati_1988] degenerates at rest, and @elhay_simpson_2011 prove it is singular whenever the zero-flow pipes form a loop or a path between fixed-head nodes, introducing the same linear-surrogate cure.
Of course, upon convergence, the artifical resistance should vanish.
Therefore, it is applied through a sequence of stages, given as:

$$
\kappa \in \{0.1,\ 0.01,\ 0\},
$$

each stage warm-started from the previous solution, with a smoothing width $\varepsilon = \max(0.3\kappa,\ 10^{-4})\,\dot m_{\text{ref}}$ that rounds the transport upwind switches and the other regularized primitives (see [transport](../theory/transport.qmd) and [the smoothness contract](smoothness-contract.md#sec-smooth-primitives)).
At $\kappa = 0.1$, $0.01$, and $0$ this gives $\varepsilon = 0.03$, $0.001$, and $10^{-4}$ times $\dot m_{\text{ref}}$ respectively, sharpening toward the exact, friction-free equations on the final stage.
An important remark is that this is a continuation in a physical parameter rather than a numerical fudge: every intermediate problem is a well-posed resistive network, and only the limit $\kappa \to 0$ restores the original equations, reached by a path that stays nonsingular throughout (tests: `test_quiescent_cold_start_converges`, `test_long_serial_chain_cold_start`, `test_many_parallel_branches_converge`).
This is also where the treatment here parts from the hydraulic-network practice it otherwise shares: there the linear surrogate is applied locally and kept permanently, which perturbs the solution the solver converges to, and @gorev_2022 report that the treatment in present use can converge to distinctly inaccurate results on networks containing low-resistance pipes.
Driving $\kappa$ to zero instead leaves no trace of the regularization in the converged state.

## Warm-start caches {#sec-solver-warmstart}

Two kinds of reuse keep repeated solves cheap.
Within a solve, each continuation stage begins from the converged state of the previous stage, so the friction is removed by a sequence of easy corrections rather than a single hard solve.
Across the reacting recovery, the equilibrium solve at each edge is warm-started from a cache of its previous composition and temperature, so the innermost thermodynamic iteration converges in a few steps rather than from a cold guess.
Because a warm start only supplies an initial iterate and never enters a residual, it changes the *cost* of a solve but not its *result* — the converged state is invariant to the warm start, a property checked directly so that the caches can never silently perturb the answer.

## The subsonic-scope backstop {#sec-solver-subsonic}

The steady equations admit, beside the physical subsonic root, a spurious supersonic one at over-critical operating points, and a cold seed can land on it; the present scope is subsonic (see [scope and limitations](../theory/limitations.md#sec-limits-mean-flow)), so a converged solve that carries a supersonic edge is checked before it is returned.
When it does, the solver re-solves once from a near-stagnation seed, which reliably reaches the subsonic branch when one exists, and keeps that recovery only if it lowers the peak Mach number.
What remains after the recovery is then judged in two bands.
A flow a hair past a sonic throat, an over-driven orifice that chokes and sits just supersonic on the isentropic relation, is kept with a warning as a near-choke state at the edge of the scope.
A flow running *far* past the speed of sound is instead the spurious or ill-posed branch, the signature of an over-critical demand or a resistance-free loop (see [the modeling guide](../reference/modeling-guide.md)), and is returned marked not converged, so a wildly supersonic result is never handed back as an accepted solution.
The guard is on by default and can be turned off (`nefes.config.enforce_subsonic = False`, or per solve) to accept the raw branch regardless of Mach number; genuine choking is untouched, since a real throat pins at Mach one and stays below the supersonic threshold (tests: `tests/test_subsonic_scope.py`).

With the mean-flow operating point found robustly, the acoustic operator is assembled about it and the analyses proceed; the practices that keep the result reproducible across environments are collected in [reproducibility](reproducibility.md).
