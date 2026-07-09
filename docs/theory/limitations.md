# Scope and limitations

Stating the boundary of a method's validity is part of stating the method, and this document collects, in one place, the scope boundaries and approximations noted in passing throughout the theory.
Each limitation below is either a *scope boundary* — a regime the present version does not attempt — or a *bounded approximation* — a model that is accurate to a stated order and has a designed path to exactness.
None is a hidden assumption: the intent here is to draw the honest edge of the tool so that a result is trusted exactly as far as it should be.
The presentation groups the limitations by layer — mean flow, reacting flow, and acoustics — and closes by restating plainly what is *not* approximate.

## Mean-flow scope

The mean-flow model is restricted to subsonic operation, flowing or quiescent, up to a sonic throat.
Choking to $M = 1$ at a narrowest section is fully in scope and emerges from the element rows (see [choking](choking.qmd)); what is deferred is *supersonic flow inside the domain* and the structures that accompany it.
A normal shock standing in a diverging passage, and a declared supersonic exit, require an internal shock-position degree of freedom — the dual of the self-vacating row a supersonic inlet would carry — which is designed but not built.
It should be noted that an emergent supersonic converging–diverging nozzle is not impossible in principle; it is deferred because it needs that additional unknown, not because the formulation forbids it, and the same enrichment is what finite-frequency acoustics of shocked nozzles will require.

A second, milder bound is that the sudden-contraction and Borda-expansion pressure heads use the incompressible reduction of the momentum balance, accurate to $\mathcal{O}(M^2)$ (see [elements](elements.md)).
A dedicated contraction element that resolves the vena-contracta state, and so stays exact at higher Mach number, is planned but not present.

## Reacting-flow scope

The reacting closure is equilibrium-based: the frozen, equilibrium, and marker-gated closures of [thermochemistry](thermochemistry.md) are built, but finite-rate chemistry is not.
The forward-compatible path — promoting species to independent transported scalars with a well-stirred-reactor source and a Damköhler continuation — is designed into the thermochemistry library through its complex-analytic rate hooks, yet the current tool realizes the frozen/burnt distinction through the marker gate rather than through a residence-time balance.
The marker gate is exact wherever the flame is thin compared with the network's elements, which is the regime the lumped-element framework already assumes.

## Acoustic-model scope

Before the specific bounds, the premise beneath all of them: the acoustic layer is *linear*, and its perturbation is an infinitesimal one.
The consequence worth recording here is one the mean flow inherits.
The state the solver converges is the steady operating point, and it coincides with the temporal mean of the unsteady flow only to the order the linearization retains: averaging the exact balances leaves the correlation sources $\overline{\varrho' u'}$, $\overline{p' u'}$, and their kin, which are quadratic in the fluctuation and are therefore dropped (see [governing equations](governing-equations.md)).
At finite amplitude those sources bias the observed mean away from the steady solution, by an amount growing with the square of the amplitude and concentrating in the elements whose losses are quadratic in the velocity.
What the tool therefore predicts is whether a mode grows, not the amplitude at which its growth arrests, nor the mean state a saturated oscillation settles into; a measurement taken on a strongly oscillating rig is not the state this mean flow computes.
The forward-compatible path is the amplitude-dependent describing function: retaining the correlation sources and evaluating the dynamic sources of [dynamic sources](dynamic-sources.qmd) at a prescribed amplitude turns the same operator into a saturation condition, and the machinery to carry it — a nonlinear source stamp on an otherwise unchanged network — is already the shape of the source block.

Four further bounds apply to the acoustic layer, each already noted where it arises.
The compositional (indirect) noise coupling $R_\xi$ [@magri_2016] is retained everywhere the acoustic linearization is inherited from the mean-flow kernel, but it is dropped by the hand-written analytic terminal closures for a choked-nozzle or constant-mass-flow outlet, which carry the entropy coupling $R_s$ but no composition column; the solver raises a warning precisely when a reacting flow meets such a closure, so the gap is surfaced rather than silent (see [dynamic sources](dynamic-sources.qmd)).
The isentropic reduction removes the convected entropy wave altogether — exact for the acoustic spectrum of a flow without entropy sources, but by construction silent about indirect combustion noise — so a study of entropy-generated sound must use the full three-wave mode (see [identification](identification.md)).
The real-axis Nyquist stability count is taken relative to the passive operator $\mathbf{A}_0$ and equals the absolute count only when $\mathbf{A}_0$ is itself stable, a premise the driver checks with a rational fit; it is moreover a tally up to the swept band edge rather than an unqualified total (see [analyses](analyses.qmd)).
The automatic de-normalization of a flame's mean heat release is exact for the perfect-gas heat-release flame, which carries its power as a parameter, but for every other flame it uses the sensible-enthalpy rise $\dot m\,\overline{c}_p\,\Delta T$ and is accurate only to $\mathcal{O}(M^2)$; supplying the mean heat release explicitly removes the approximation (see [dynamic sources](dynamic-sources.qmd)).

A final modeling choice rather than a limitation: the storage block assembles the compact-limit reductions of finite-volume storage — a cavity compliance, an inline compliance and series inertance, a manifold compliance — rather than the full three-row storage of a spatially resolved control volume, which is the appropriate model for the lumped elements the framework uses.

## What is exact

Against these bounded approximations it is worth restating plainly what carries *no* approximation, so the edge is not mistaken for the whole.
The Jacobian is exact to machine precision, obtained by complex-step differentiation of analytic residuals rather than by finite differences (see [complex-step](../design/complex-step.qmd)).
The mean-flow mass balance is exactly linear and globally conservative, and the energy transport is exact up to the quadratically small smoothing of the upwind weights (see [transport](transport.qmd)).
The isentropic and choking rows reproduce the classical compressible-flow relations exactly in the subsonic regime, to the order of the fixed complementarity smoothing (see [choking](choking.qmd)).
The characteristic maps that turn the base Jacobian into the acoustic operator are the exact linearization of the state definitions, invertible at every physical state (see [characteristics](characteristics.md)).

The honest summary is therefore that the subsonic, equilibrium-reacting mean flow and the acoustics linearized about it form a complete and internally exact core, ringed by a small set of named, bounded approximations — each with a designed path to removal — and two firm scope boundaries: supersonic internal flow, and the finite-amplitude response beyond the linear one.
The evidence that the core performs as claimed is the subject of the validation track.
