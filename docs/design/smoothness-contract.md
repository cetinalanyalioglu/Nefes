# The smoothness contract

The complex-step derivative is exact only when every residual is built from smooth operations, with no kinks or jumps tied to the flow state, so each element equation must obey a fixed set of rules (see [the complex-step derivative](complex-step.qmd)).
This document states those rules, introduces the smooth formulas that stand in for hard switches and bounds without changing the intended physics, quotes the approximation error those roundings introduce, and describes the automated check that verifies every element type against them.
These rules put into practice the *smoothness over branching* principle of [the design philosophy](philosophy.md).

## The contract {#sec-smooth-contract}

No residual may contain an `abs`, a `sign`, a `min`, a `max`, or a branch taken on a solution variable.
Each of these is a non-analytic operation, it has a kink or a jump where its argument changes sign or crosses a threshold, and a residual containing one is not differentiable there, so both Newton's method and the complex-step derivative fail exactly at the state the operation was meant to handle.
The contract does not forbid the *physics* of a switch: a flow reverses, a loss opposes the flow in either direction, a passage chooses between subsonic and choked, and a reacting edge is frozen or burnt.
It forbids only the *non-analytic realization* of those switches, requiring instead that each be expressed by a smooth function that reproduces the switch away from its transition and rounds its corner within a narrow, controlled band.

## The primitive library {#sec-smooth-primitives}

The framework supplies a small library of regularized primitives, each a $C^\infty$ replacement for a non-analytic operation, built so that its radicands stay strictly positive on the real axis and its complex continuation never approaches a branch cut.
The primitives, and the operations they replace, are as follows.

1. **Smoothed absolute value**, $\sqrt{x^2 + \delta^2}$ — replaces $|x|$; equals $\delta$ at the origin and tends to $|x|$ away from it.
2. **Smoothed positive part**, $\tfrac{1}{2}\big(x + \sqrt{x^2 + \delta^2}\big)$ — replaces $\max(x, 0)$; the inflow weight of the transport donor (see [transport](../theory/transport.qmd)).
3. **Smoothed step**, $\tfrac{1}{2}\big(1 + x/\sqrt{x^2 + \delta^2}\big)$ — replaces the Heaviside; the upwind weight that selects a convected quantity's upstream side.
4. **Marker gate** — a re-centred, normalized smoothed step with $g(0) = 0$ and $g(1) = 1$ exact, gating the frozen and burnt reacting closures (see [thermochemistry](../theory/thermochemistry.md#sec-thermo-marker-closures)).
5. **Smoothed signed square**, $x\sqrt{x^2 + \delta^2}$ — replaces $x|x|$; the direction-aware dynamic-head of a loss element, opposing the flow in both directions.
6. **Smoothed Fischer–Burmeister complementarity**, $a + b - \sqrt{a^2 + b^2 + \varepsilon^2}$ — encodes an either/or regime switch as a single residual; the choking complementarity (see [choking](../theory/choking.qmd#sec-choking-complementarity)).

Every residual switch in the framework is built from these, so that the whole assembled residual vector is analytic in a neighbourhood of the real axis by construction, and the complex-step Jacobian through it is exact.

## The error order {#sec-smooth-error}

Regularizing a switch is not free: it biases the solution by pinning the inactive side of a switch to a small nonzero value rather than exactly zero.
The bias is *quadratically* small in the smoothing width relative to the state, of order $\mathcal{O}(\delta^2/x^2)$ at a converged state with $|x| \gg \delta$, where $\delta$ is the regularization width and $x$ the relevant flow scale.
With the default widths — a smoothing of order $10^{-4}$ of a reference mass flow, and a fixed complementarity smoothing of order $10^{-5}$ — the imprint on the solution sits near the $10^{-8}$ relative level, far below any engineering tolerance and below the Newton convergence tolerance itself.
An important discipline follows for the choking complementarity: its smoothing must stay *small and fixed* rather than being relaxed along the solver's continuation, because its bias is a fictitious total-pressure loss that must remain far below the smallest driving pressure difference in the network, and widening it breaks weakly driven cases (see [the solver](solver.md#sec-solver-continuation)).

## The roll-call {#sec-smooth-rollcall}

A contract is only as good as its enforcement, and the smoothness contract is enforced element by element rather than by inspection.
Every element kernel must register a *probe* — a representative state for that element — in a central table, and a roll-call test fails if any element type in the catalogue has no probe, so a newly added element cannot silently escape the check (test: `test_every_element_kernel_is_swept`).
For each registered element the per-kernel sweep then verifies that the complex-step derivative agrees with a finite-difference derivative across the regimes where analyticity is most likely to break, e.g. forward flow, reverse flow, near-zero flow, and near the choking limit, so that a hidden branch is caught as a disagreement between the two derivatives (test: `test_kernel_complex_step_safe_across_regimes`).
Together the roll-call and the sweep make the contract self-policing: an element is admitted to the framework only once it is demonstrably smooth across the states the solver will drive it through.

The contract, the primitives, and the roll-call are what make the exact-derivative engine of [the complex-step derivative](complex-step.qmd) trustworthy in practice.
