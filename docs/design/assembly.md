# Assembly

Between the element kernels and the solver sits the *assembly* layer: the machinery that recovers each edge's full state, evaluates every residual row, and fills the sparse Jacobian by complex step — and then reuses the very same rows to stamp the acoustic operator.
This document describes that layer, because its structure is what makes the "one operator" reuse of the acoustics possible and what the identification de-embedding exploits for a one-factorization inverse.
It builds on the [kernel architecture](kernel-architecture.md) and the [complex-step derivative](complex-step.qmd), and feeds the globalization of [the solver](solver.md).


## Edge-state recovery precedes the residuals

A residual row consumes not the carried variables directly but the *recovered* edge state — density, velocity, temperature, sound speed, Mach number, stagnation quantities — so every residual evaluation is preceded by a recovery pass that fills a per-edge state table from the carried variables and the closure (see [state and recovery](../theory/state-and-recovery.qmd)).
This ordering is a small dependency graph: recovery reads the carried variables and writes the state table, and the residual rows read the state table.
Isolating recovery as a distinct pass keeps the closure confined to one place — the element rows never see the gas model, only the state it produced — and it lets the recovery apply its complex-step splice once per edge rather than inside every row that uses the result.

## The fixed row split

The assembled system is partitioned into two blocks whose sizes do not depend on the flow direction, which is the structural realization of *discovery over prescription* (see [equation structure](../theory/equation-structure.md)).
The *algebraic* block holds each element's mass balance and pressure-type relations — one mass row and its jump conditions per element — evaluated by the node kernels.
The *transport* block holds one smoothly-upwinded advection row per carried scalar per edge — total enthalpy always, plus any mixture fractions — evaluated by the donor-and-upwind kernel.
The split is fixed: an edge always owns its transport rows and an element always owns its algebraic rows, regardless of which way the gas flows, so the residual vector and the Jacobian keep their dimension as the solver reverses flows or discovers them from rest.
The count is square by construction — the algebraic rows plus the transport rows equal the unknowns — which is what lets Newton's method operate without the system ever becoming over- or under-determined.

## The sparse Jacobian by complex step

The Jacobian is sparse, because a residual row depends only on the edges local to it: an algebraic row on the edges incident to its element, and a transport row on its own edge and on every edge sharing one of its endpoint elements (through the mass-weighted donor mix).
The assembly captures this sparsity pattern once, from the network connectivity, and then fills the nonzero entries by complex step — seeding one column of unknowns at a time with an imaginary perturbation, evaluating the residual in the `complex128` specialization, and reading the derivative from the imaginary parts of the affected rows.
Because the pattern is fixed and the fill is exact, the Jacobian is assembled without any hand-derived derivative and without any wasted evaluation of structurally-zero entries.
This is the point at which the *exact derivatives* and *kernels over objects* principles meet the sparsity of a network: one seeded sweep per column group yields the exact sparse Jacobian.

## Acoustic stamps as a low-rank layer

The same assembled Jacobian is the base of the acoustic operator, and the unsteady physics is layered onto it rather than re-derived (see [the perturbation network](../theory/perturbation-network.md)).
Each acoustic block is a stamp on the base Jacobian $\overline{\mathbf{J}}$: the storage block adds finite-volume compliance and inertance through $\mathrm{i}\omega$, the propagation block overwrites duct-continuity rows with phase relations, the source block adds a flame's feedback, and the terminal closures overwrite boundary rows with reflection relations.
Two of these stamps — the dynamic source and the transfer-matrix element — enter as a *low-rank* modification of the operator, occupying only the few rows and columns their element spans.
An important consequence is that a low-rank update admits a one-factorization inverse: the known part of the operator is factored once per frequency, and the unknown element is recovered from a measured response by a small least-squares solve on top of that single factorization: the structure the identification of [identification](../theory/identification.md) exploits.
This is the payoff of assembling the acoustics as stamps on the mean-flow Jacobian rather than as a separate operator: the mean-flow work is done once, and both the forward acoustics and the inverse identification read it.

With the residual and Jacobian assembled and the acoustic stamps understood as a layer over them, what remains is the globalization that turns Newton's local method into a solver robust from a cold start — the subject of [the solver](solver.md).
