# Overview

Nefes is a network solver for compressible, reacting flows and their thermoacoustics.
It represents a fluid system as a directed graph — components joined at their ports — and solves for the steady mean flow through that graph, then for the linear acoustic behaviour of small pulsations around that flow, without ever resolving the full three-dimensional field.
The purpose of this document is to state what the method computes, the assumptions under which it holds, and the four design decisions that shape everything downstream, and to serve as the map into the rest of the documentation.

The presentation begins with what a network solve produces and the regime in which it is valid, then proceeds to the four decisions the framework rests on — where the state lives, which variables carry it, how the equations are counted, and why the whole formulation is built around smoothness — building on these to the structural observation that unifies the mean flow and its acoustics, and culminating in a guide to how the three documentation tracks are organized.

## What the method computes

A *flow network* is the compressible-flow analogue of an electrical circuit: components such as nozzles, orifices, junctions, and plenums are joined at ports, and gas flows through them driven by pressure differences much as current is driven by voltage differences.
Given the network's connectivity, its element parameters, and its boundary data, a mean-flow solve answers the questions an engineer asks of such a system — how the flow splits between branches, what pressure, temperature, and Mach number each component sees, how much flow a supply delivers — in milliseconds rather than the hours a full computational-fluid-dynamics simulation would take.

Unlike an incompressible or electrical network, a compressible gas network carries three coupled quantities along every connection — mass flow, pressure, and energy — and the relations between them are nonlinear.
This admits genuinely compressible behaviour: the density varies along a path, hot and cold streams mix, and the mass flow *saturates* once the narrowest section reaches the speed of sound, the phenomenon known as *choking*.
On the reacting side, an element may add heat or undergo chemical equilibrium, so the network also resolves flame temperature rise, dilution, and the transport of each injected stream's composition.

The distinguishing aim of the project, however, lies one step beyond the mean flow.
Because every steady element relation is instantaneous and algebraic, its linearization about the converged operating point *is* the corresponding acoustic jump condition, so the same assembled operator that produced the mean flow, differentiated, becomes the frequency-domain acoustic network.
On that acoustic network the tool computes the forced response and scattering between chosen stations, the natural modes and their growth rates (thermoacoustic stability), the response of a compact flame as a dynamic source, and — inverting the forced problem — the identification of an unknown element's dynamic response from a measured network response.
It should be emphasized that this unity is structural rather than incidental: the mean flow and its acoustics are two evaluations of one operator, so no second model can drift from the first.

What the method does *not* compute is the spatial field inside a component.
An element is treated as a lumped relation between the states at its ports; the interior flow, the boundary layers, and the turbulence are represented through constitutive models (loss coefficients, area ratios, heat-release rates), not resolved.

## Standing assumptions and scope

The following assumptions hold throughout the mean-flow formulation, and each is revisited where a later document depends on it:

1. **Calorically perfect gas.** Each stream obeys $p = \varrho R T$ with constant specific heats, so $h = c_p T$ and $c^2 = \gamma R T$; this is a good approximation for air and combustion products over moderate temperature ranges, and the reacting layer relaxes it to mixture- and temperature-dependent properties where required (see [thermochemistry](thermochemistry.md)).
2. **Inviscid on the scale of a component.** The gas is governed by the Euler equations at the component scale; viscous and turbulent losses enter only through lumped constitutive terms in the element relations, not through a resolved shear field.
3. **Clean cross-section ports.** The flow crosses each port perpendicularly, so a single signed normal velocity carries all of the kinetic energy through it; this constrains the ports, not the interior flow, which may be fully multidimensional.
4. **A steady operating point exists.** The mean-flow solve seeks a steady state, at which the time-derivative terms vanish and the conservation laws reduce to algebraic balances between port states; steadiness alone — not any smallness of element volume — is what removes the transient term.
5. **Subsonic regime.** The present version is restricted to subsonic mean flows, whether flowing or quiescent; the throat may reach $M = 1$ and choke, but supersonic branches and shock seeding are deferred (see [limitations](limitations.md)).
6. **Linear acoustics.** The acoustic layer superposes on the mean flow fluctuations small enough that their squares are negligible, so the response is governed by the first-order (linearized) operator; amplitudes and nonlinear acoustic effects are outside the present scope.

The symbols, decorations, and sign conventions used here and in every other document are collected once in the [nomenclature](../nomenclature.md); the reader is referred there rather than to a redefinition in place.

## The four design decisions

The framework rests on four decisions, each derived in the document named beside it.
Together they are what let the solver *discover* the flow — its directions, its choke points, its backflow — from a cold start, rather than being told the answer in advance.

1. **All state lives on the connections, none in the components.** Edges carry the flow state; elements own only the physical relations between the states of the edges that meet at them (see [framework](framework.md)). An element is a control volume on which the governing balances are applied; it carries no state of its own, because the state lives on the edges and no volume-associated state is needed.
2. **Each edge carries the triple $(\dot m,\ p,\ h_t)$** — the mass flow rate, the static pressure, and the total enthalpy (see [state and recovery](state-and-recovery.qmd)). This choice has the property that every other flow quantity — density, velocity, temperature, Mach number, stagnation state — is recovered from the triple uniquely and smoothly, no matter how fast or in which direction the gas flows, a property that a more familiar triple such as $(p_t, T_t)$ does not share.
3. **A fixed bookkeeping rule makes the system square.** Every element contributes exactly as many equations as it has ports, and every edge contributes exactly one transport equation (see [equation structure](equation-structure.md) and [transport](transport.qmd)). The count is therefore square independently of which way the gas flows, which is precisely what allows flow directions to be discovered rather than prescribed.
4. **The formulation is built around smoothness.** Every residual is written without any branch, `abs`, `min`, or `max` on the flow state, so it is smooth and complex-analytic (see [well-posedness](well-posedness.md) and the [smoothness contract](../design/smoothness-contract.md)). Newton's method, suitably damped, then converges even from an exactly quiescent start, and the Jacobian is obtained *exactly* — to machine precision, free of subtractive cancellation — by complex-step differentiation [@martins_2003] (see [complex-step](../design/complex-step.qmd)).

The fourth decision has a consequence that reaches past the mean flow.
Because the Jacobian is assembled exactly at convergence, it doubles as the algebraic content of the acoustic problem: restoring the finite-volume storage terms dropped at steady state, the lossless-duct wave phases, and any unsteady flame source turns that same Jacobian $\overline{\mathbf{J}}$ into the frequency-domain operator $\mathbf{A}(\omega) = \overline{\mathbf{J}} + \mathrm{i}\omega\mathbf{M} + \mathbf{P}(\omega) + \mathbf{S}(\omega)$, which reduces to the steady operator $\mathbf{A}(0) = \overline{\mathbf{J}}$ at $\omega = 0$ (see [perturbation network](perturbation-network.md)).
This is the structural payoff of the design, and it is the reason a single tool serves both the steady and the acoustic question.

## How the documentation is organized

The documentation is arranged into three tracks over a shared foundation of [nomenclature](../nomenclature.md) and [references](../references.bib).

1. **Theory** — the physics and mathematics, from the graph model and governing balances through transport, elements, characteristics, and choking, then the reacting closures, the perturbation network, the dynamic flame sources, the forward acoustic analyses, and the inverse identification analysis. Each theory document opens with its own assumptions ledger and either cites or derives every claim it makes.
2. **Design philosophy** — why the code is shaped the way it is: the smoothness-over-branching principle, the complex-step derivative engine, the regularized-primitive library and its per-kernel safety contract, the assembly of the mean-flow and perturbation operators, the damped-Newton solver with its vanishing-friction homotopy, and the reproducibility discipline.
3. **Validation** — the evidence: a master map from every physical claim to the analytic or literature case that checks it, the internal consistency verifications (complex-step against finite difference, edge-direction-flip invariance, thermochemistry against an independent oracle), and the named literature benchmarks.

A reader seeking to reconstruct a result should follow the Theory track; a reader seeking to understand or extend the implementation should follow the Design track; a reviewer seeking to reproduce a figure should follow the Validation track to the notebook that generates it.
