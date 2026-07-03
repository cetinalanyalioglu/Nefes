# The solver

Newton's method converges quadratically near a solution and not at all far from one, so a bare Newton iteration on the network equations would fail from the uninformed cold start the framework insists on supporting.
This document describes the globalization that closes that gap — the scaling, damping, and homotopy that turn the local method into one that finds the operating point from exact rest on an arbitrary network — and the warm-start caches that keep repeated solves cheap.
It builds on the exact Jacobian of [the complex-step derivative](complex-step.qmd) and the assembled system of [assembly](assembly.md), and it is the machinery that realizes the *discovery over prescription* principle of [the design philosophy](philosophy.md).

The presentation begins with the scaling that makes the system well-conditioned, then proceeds to the damped Newton step, building on this to the vanishing-friction homotopy that removes the zero-flow trap, and closes with the warm-start caches.

## Nondimensionalization

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
This single step accounts for much of the difference between a solver that crawls and one that converges cleanly.

## Damped Newton steps

Each iteration solves the scaled Newton system, damped in the Levenberg–Marquardt manner so that a step remains well defined even where the Jacobian is momentarily singular, given as:

$$
\big(\overline{\mathbf{J}}^{\top}\overline{\mathbf{J}} + \lambda\mathbf{I}\big)\,\delta\mathbf{y} = -\,\overline{\mathbf{J}}^{\top}\widehat{\mathbf{R}},
$$

where $\overline{\mathbf{J}}$ is the scaled Jacobian, $\widehat{\mathbf{R}}$ the scaled residual, $\delta\mathbf{y}$ the update, and $\lambda$ the damping parameter.
For $\lambda \to 0$ the step is the pure Newton step, recovering quadratic convergence near the solution; for larger $\lambda$ it blends toward a cautious gradient-descent step that makes progress where the pure step would overshoot or where the Jacobian has lost rank — for instance at the undetermined split of a perfectly symmetric branching network at rest (see [well-posedness](../theory/well-posedness.md)).
The damping is adapted per iteration, raised when a step fails to reduce the residual and lowered as the iteration homes in, so the solver interpolates automatically between robustness far out and speed near the answer.

## The vanishing-friction homotopy

The damped Newton step is still defeated by one structural trap: in a network driven only by pressure boundary conditions, the residuals have zero first-order sensitivity to the flows at the quiescent state, so the solver sees a flat landscape and cannot start the flow moving (see [well-posedness](../theory/well-posedness.md)).
The cure is a homotopy in a physical parameter — a small fictitious friction $\kappa$ added to every pressure-type row — that injects first-order flow sensitivity without changing the final answer.
With the friction active the network behaves like a resistive circuit, in which pressure differences push directly on the flows, and the solver locates the flow pattern readily; the friction is then reduced to zero over a short sequence of stages, given as:

$$
\kappa \in \{0.1,\ 0.01,\ 0\},
$$

each stage warm-started from the previous solution and using a smoothing width that shrinks with $\kappa$, so that the final stage solves the exact, friction-free equations.
An important remark is that this is a continuation in a physical parameter rather than a numerical fudge: every intermediate problem is a well-posed resistive network, and only the limit $\kappa \to 0$ restores the original equations, reached by a path that stays nonsingular throughout (tests: `test_quiescent_cold_start_converges`, `test_long_serial_chain_cold_start`, `test_many_parallel_branches_converge`).

## Warm-start caches

Two kinds of reuse keep repeated solves cheap.
Within a solve, each homotopy stage begins from the converged state of the previous stage, so the friction is removed by a sequence of easy corrections rather than a single hard solve.
Across the reacting recovery, the equilibrium solve at each edge is warm-started from a cache of its previous composition and temperature, so the innermost thermodynamic iteration converges in a few steps rather than from a cold guess.
Because a warm start only supplies an initial iterate and never enters a residual, it changes the *cost* of a solve but not its *result* — the converged state is invariant to the warm start, a property checked directly so that the caches can never silently perturb the answer.

With the mean-flow operating point found robustly, the acoustic operator is assembled about it and the analyses proceed; what remains of the design track is the practice that keeps all of this reproducible across environments, the subject of [reproducibility](reproducibility.md).
