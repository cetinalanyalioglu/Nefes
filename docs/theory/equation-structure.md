# Equation structure: counting and assignment

With the state vector fixed on every edge, the network has a definite number of unknowns, and the model must supply exactly that many independent equations.
The interesting question is not the total — that is settled by counting — but the *assignment*: which element or edge owns which equation.
This document shows why the natural, physics-based assignment fails the solver, and how a fixed assignment rule restores a square system whose very shape does not change as the flow is discovered.

The organizing concern is again robustness.
A Newton solver iterating from a cold start must be free to reverse the flow on any edge without the equation system changing size beneath it; a formulation whose structure flips mid-iteration cannot be made to converge by numerical care alone.

## Standing assumptions {#sec-eqstruct-assumptions}

This document inherits the [overview](overview.md) assumptions and adds none of its own; it is a counting argument over the graph of [framework](framework.md#sec-framework-graph), valid for any thermodynamic closure and any flow direction.

## The unknown count {#sec-eqstruct-unknown-count}

Each edge carries a state vector of $n_s$ unknowns: the triple $(\dot m, p, h_t)$ for a non-reacting gas, extended by one transported mixture fraction per injected feed stream for a reacting gas (see [state and recovery](state-and-recovery.qmd#sec-state-variable-set)).
A network of $E$ edges therefore has $n_s E$ unknowns, and a well-posed model must supply exactly $n_s E$ independent equations — no more, no fewer.
The whole of this document is the argument that a fixed assignment rule meets this count for every flow direction; we take $n_s = 3$ in the exposition and restore the transported scalars at the end, where the count is seen to close in the same way.

## The information-flow count and its flaw {#sec-eqstruct-information-flow}

The physically natural way to assign equations follows the direction in which information travels in the gas.
Small disturbances in a subsonic stream travel along three families of characteristics (see [characteristics](characteristics.md#sec-char-euler-decomposition)): pressure waves running downstream at speed $u + c$, pressure waves running upstream at $c - u$, and entropy and composition patterns simply convected with the flow at speed $u$.
Boundary-condition theory then requires an element to supply exactly as many conditions as there are wave families *leaving it* through its ports, which is given as:

$$
n_{\text{eq}}(P) = \underbrace{n}_{\text{one outgoing acoustic wave per port}} + \underbrace{n_{\text{out}}(P)}_{\text{one convected wave per outflow port}},
$$

where $P$ is an element, $n$ is its number of ports, and $n_{\text{out}}(P)$ is the number of its ports through which the flow leaves.
Summed over the network this count is globally correct: each of the two pressure-wave conditions is owned once per port, giving $2E$, and each edge is the outflow of exactly one of its two end elements, giving $E$ convected conditions, for a total of $3E$.
The familiar special cases recover the classical picture — a through-flow component supplies $2 + 1 = 3$ jump conditions, an inlet supplies $2$, an outlet $1$.

The flaw is that $n_{\text{out}}(P)$ changes discretely when a port reverses.
A three-port junction owes five equations with one inflow and two outflows, but only four with two inflows and one outflow.
A solver iterating across a flow reversal would therefore see the system change size in mid-flight — a structural discontinuity that no amount of damping or line-search can absorb, because it is a change in the shape of the problem rather than in its conditioning.
It should be emphasized that the offending conditions are exactly the *convected* ones: the two pressure-wave conditions per port are direction-independent, and only the carried condition migrates from port to port as the flow turns.

## The fixed split {#sec-eqstruct-fixed-split}

The resolution is to move the direction-dependent conditions off the elements and onto the edges.
Each element owns one equation per port, none of which references the flow direction, and each edge owns one transport equation for each quantity it carries, written in a smoothly upwinded form that *selects* its upstream side continuously rather than switching (see [transport](transport.qmd#sec-transport-edge-equation)).
The bookkeeping is then:

| owner | count per owner | content |
|---|---|---|
| interior element, $n$ ports | $n$ | one mass balance and $(n-1)$ pressure-type relations (total-pressure equality, momentum balance, static-pressure equality, loss correlation, …) |
| boundary element, $1$ port | $1$ | one specification ($\dot m$, $p$, or $p_t$) |
| edge | one per carried scalar | smoothly upwinded transport of the total enthalpy (and of each mixture fraction) |

Squareness is now unconditional.
Summing the element rows, every element contributes as many equations as it has ports, and since each edge presents exactly two ports to the network, the element rows total $2E$ regardless of the flow directions; adding the edge transport rows gives:

$$
\underbrace{\sum_P n_{\text{ports}}(P)}_{=\,2E} \;+\; \underbrace{E}_{\text{transport}} \;=\; 3E \;=\; \text{number of unknowns},
$$

where the sum runs over all elements and the second term is the one transport equation per edge.
Nothing of the physics of the information-flow count is lost: each edge still receives two pressure-wave conditions, one from each of its ends, and one convected condition; the convected condition simply lives on the edge and chooses its upstream side smoothly, rather than being reassigned discretely between ports.
This fixed one-equation-per-port rule is enforced at assembly time, so a malformed element cannot silently unbalance the system.

## Generalization to transported scalars {#sec-eqstruct-transported-scalars}

The count closes in the same way when the edges carry more than total enthalpy.
A reacting network carries, in addition to $(\dot m, p, h_t)$, one conserved mixture fraction per injected feed stream, so the state vector grows to $n_s = 3 + n_Z$ unknowns per edge, with $n_Z$ the number of distinct feed compositions.
The two pressure-and-mass conditions per edge are unchanged, so the element rows still total $2E$; each additional carried scalar contributes one further transport equation per edge, so the edge rows grow from $E$ to $(1 + n_Z)E$, and:

$$
\underbrace{2E}_{\text{element rows}} \;+\; \underbrace{(1 + n_Z)\,E}_{\text{transport rows: } h_t \text{ and each } Z} \;=\; (3 + n_Z)\,E \;=\; n_s E.
$$

The transported quantities — the total enthalpy and every mixture fraction — are precisely those owned by the edges, and each is carried by its own upwinded transport row; the mass flow and static pressure remain the province of the element balances.

## Direction discovery and sparsity {#sec-eqstruct-direction-sparsity}

The payoff of the fixed split is that the equation system has a shape independent of the solution.
Because the convected conditions live on the edges and select their upstream side by a smooth weight rather than a branch, **the solver may reverse the flow on any edge, or discover it from an exactly quiescent start**, without the residual vector or the Jacobian changing dimension — the direction is an output of the solve, not an input to the assembly (see [well-posedness](well-posedness.md)).

The same incidence that fixes the count also fixes the sparsity of the Jacobian.
An element row depends on the state of every edge incident to that element, and an edge transport row depends on its own edge and on every edge that shares one of its two endpoint elements, because the donated total enthalpy at an element is a mass-weighted mix over the streams meeting there (see [transport](transport.qmd#sec-transport-donor-enthalpy)).
This structure is exactly the block-sparsity pattern the assembly precomputes and reuses, and its construction is described in the [assembly](../design/assembly.md) design document.
With the equations counted and assigned, the remaining piece of the mean-flow formulation is the transport relation itself.
