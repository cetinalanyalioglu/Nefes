# Well-posedness of the mean-flow system

The variable choice of [state and recovery](state-and-recovery.qmd#sec-state-variable-set), the fixed equation split of [equation structure](equation-structure.md#sec-eqstruct-fixed-split), the smooth upwinding of [transport](transport.qmd#sec-transport-edge-equation), and the artificial-resistance term of [elements](elements.md#sec-elements-stabilization) are not matters of taste.
Each repairs a *structural* singularity that defeats a formulation which is physically correct yet numerically hopeless — a set of equations that state the right physics and still cannot be solved from a general starting point.
This document collects those failure modes, because each is a property of compressible-network equations rather than a tuning problem, and understanding them is what justifies the design decisions taken earlier.

The organizing object is the Jacobian, the matrix of first-order sensitivities on which Newton's method entirely relies (see [the solver](../design/solver.md)).
When two rows of the system respond identically to every unknown the Jacobian is singular, and the solver receives contradictory or empty guidance that no step-size control or damping can repair — the missing information is simply absent.

## What ill-posedness means here {#sec-wellposed-definition}

A steady operating point is found by Newton iteration on the residual vector $\mathbf{R}(\mathbf{x})$, which advances by solving $\mathbf{J}\,\Delta\mathbf{x} = -\mathbf{R}$ with $\mathbf{J} = \partial\mathbf{R}/\partial\mathbf{x}$ the Jacobian.
The step is well defined only where $\mathbf{J}$ is nonsingular, so the relevant question for every formulation is whether its Jacobian stays full-rank across the whole path the solver must travel — in particular through the quiescent cold start, where all flows vanish, and through flow reversal, where a flow changes sign.
It should be noted that two kinds of rank loss must be told apart.
A *formulation artifact* is a singularity introduced by how the equations are written, curable by rewriting them; a *physical indeterminacy* is a singularity of the problem itself, where the physics genuinely fails to fix the unknowns, and the only sound response is to regularize the question rather than pretend it has a unique answer.
The first two root causes below are artifacts, removed by construction in the earlier documents; the last two are physical, and are handled by continuation and damping.

## Root cause A: flux-form energy balances degenerate at zero flow {#sec-wellposed-flux-energy}

The most intuitive and tempting way to impose energy conservation on a two-port element is the flux form $R_E = \sum_i \sigma_i\,\dot m_i\,h_{t,i}$, summed over its ports.
Differentiating this row exhibits its defect, and its derivative is given as:

$$
\frac{\partial R_E}{\partial \mathbf{x}}
= \sum_i h_{t,i}\,\frac{\partial(\sigma_i\dot m_i)}{\partial \mathbf{x}}
\;+\; \sum_i \sigma_i\dot m_i\,\frac{\partial h_{t,i}}{\partial \mathbf{x}},
$$

where the first sum weights the mass-flux sensitivities by the port enthalpies and the second is proportional to the mass flows themselves.
As the flows vanish, $\dot m_i \to 0$, the second sum disappears; and if the enthalpy field is still uniform, $h_{t,i} = h_t$ — the natural state of a cold start — the first sum is exactly $h_t$ times the derivative of the mass balance.
The energy row is then $h_t$ times the mass row, the Jacobian is singular, and no information about the temperature field survives.

Intuitively, this degeneracy is physical and not merely algebraic: at zero flow the statement "energy carried in equals energy carried out" is satisfied by *any* temperature field, so the equation genuinely says nothing there.
The same collapse appears in characteristic variables, as it must, because the $M \to 0$ limits of the characteristic energy-flux coefficients are $\pm\varrho c^2/(\gamma-1) = \pm\varrho h$, again $h$ times the mass-flux coefficients (see [characteristics](characteristics.md)).
Every chain of isentropic elements started from rest sits on this singularity, which is precisely why a flux-form energy balance fails on multi-element chains.
The structural cure is to abandon the flux form for the *transport* form of [transport](transport.qmd): the edge equation $h_{t,e} = \theta H_{\text{tail}} + (1-\theta)H_{\text{head}}$ always carries a coefficient of $1$ on $h_{t,e}$ itself, so it never loses its diagonal and cannot collapse onto the mass balance (test: `test_long_serial_chain_cold_start`, solving a long chain from exactly zero flow).

## Root cause B: hard switches destroy smoothness and rank {#sec-wellposed-hard-switches}

A second temptation is to select the upstream side of a convected quantity, or the sign of a loss, with a hard switch such as $\operatorname{sign}(u)$ multiplying a residual row.
Such a factor fails in two independent ways.
It vanishes identically at $u = 0$, so the row it multiplies is null exactly at the quiescent state where the cold start lives, and the Jacobian loses a row there; and it jumps discontinuously across a flow reversal, whereas Newton's method presumes a differentiable residual and a jump halts it.
At a deeper structural level, the classical practice of *reassigning* which equation a port owns when the flow reverses changes the equation count on each side of the reversal, a discontinuity in the very shape of the system (see [equation structure](equation-structure.md#sec-eqstruct-information-flow)).
Both disappear under the constructions adopted earlier: the fixed per-edge split of [equation structure](equation-structure.md#sec-eqstruct-fixed-split) keeps the equation count invariant to flow direction, and the smooth upwind weight $\theta = \operatorname{sstep}(\dot m;\varepsilon)$ of [transport](transport.qmd#sec-transport-edge-equation) selects the upstream side continuously, so the residual and its Jacobian remain smooth through reversal.

## Root cause C: zero flow is a stationary point of pressure-driven networks {#sec-wellposed-zero-flow}

The third failure is the subtlest because the residuals are perfectly smooth and full-rank in count, yet the solver still stalls at rest.
The cause is that all pressure-type physics couples to the flow only through the dynamic head, whose leading behaviour is given as:

$$
p_t - p \;=\; \frac{\gamma}{2}\,p\,M^2 + \mathcal{O}(M^4),
\qquad
M^2 \propto \dot m^2,
$$

where $M$ is the Mach number and the dynamic head is quadratic in the mass flow.
Its sensitivity to the flow, $\partial(p_t - p)/\partial\dot m \propto \dot m$, therefore vanishes at $\dot m = 0$.
In a network driven *only* by pressure boundary conditions — with no mass-flow specification anywhere — the pressure residuals thus have zero first-order sensitivity to every flow unknown at the quiescent state.
Newton's method, which moves entirely on first-order sensitivities, sees a flat landscape and no reason to set the flows in motion: zero flow is a stationary point, and descent-type iterations stall on or near it.

The cure must inject first-order flow sensitivity without altering the converged answer, which is exactly what the artificial-resistance continuation of [elements](elements.md#sec-elements-stabilization) does.
With the fictitious friction $\kappa$ active, each pressure relation acquires a term linear in the flow, so the network behaves like a pipe-resistance circuit in which pressure differences push directly on the flows — as a voltage drives a current through a resistor — and the solver locates the correct flow pattern readily.
The friction is then reduced to zero over a short sequence of stages, the dimensionless schedule $\kappa_s$ stepping through values such as $(0.1,\ 0.01,\ 0)$ with each stage warm-started from the previous solution, so that the final stage solves the exact, friction-free equations (test: `test_quiescent_cold_start_converges`).
An important remark is that this is a continuation in a *physical* parameter, not a numerical fudge: every intermediate problem is a well-posed resistive network, and only the limit $\kappa \to 0$ restores the original equations, whose solution the continuation reaches by a path that is nonsingular throughout.

## A milder, physical indeterminacy {#sec-wellposed-physical-indeterminacy}

One further singularity is genuinely a property of the problem rather than of its formulation.
In a perfectly symmetric branching network at rest, the *split* of the flow between identical branches is undetermined at first order — any split satisfies the loop balance when nothing flows — so the Jacobian is singular there even when a mass-flow inlet fixes the total.
This is a physical indeterminacy: the symmetric problem truly has a one-parameter family of incipient splits, resolved only by whatever breaks the symmetry.
Rather than rewrite the equations, the solver regularizes the step through Levenberg–Marquardt damping (see [the solver](../design/solver.md#sec-solver-damping)), which replaces the singular $\mathbf{J}$ by $\mathbf{J} + \lambda\mathbf{I}$ near a rank deficiency and so takes a small, well-defined step that the following iterations refine (test: `test_many_parallel_branches_converge`).

Taken together, these repairs are why the mean-flow system is solvable from any admissible cold start: the transport form removes the energy degeneracy, the fixed split and smooth upwinding remove the switch discontinuities, the artificial-resistance continuation removes the pressure-driven stationarity, and the damping absorbs the residual symmetric indeterminacy.
The wave language of characteristic variables — the natural coordinates of the acoustics, and the setting in which the same operating-point Jacobian is reused — is developed in [characteristics](characteristics.md).
