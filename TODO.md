## To implement

- [ ] Proper estimation of friction coefficient for the pipe element, could be Moody charts or some other correlation.
- [ ] Full-nozzle composite carrying both distinct area reductions of De Domenico Fig. 3 in one element: the sub-throat vena contracta (`A_min = Gamma*A_T <= A_T`, sets the effective throat / choke) *and* the recovery jet plane (`A_j = beta*A_2 >= A_T`). Existing `lossy_nozzle` has only `A_j`; `sudden_contraction` only a downstream-referenced vena contracta. Stage `A_1 ->isen-> A_T ->isen-> Gamma*A_T ->isen-> A_j ->Borda-> A_2` with `Gamma`, `beta` inputs (later geometry-derived, see `scratch/nozzle-shape-loss-prediction.md`).
- [ ] Combined linear+quadratic (Forchheimer) flow resistance: `linear_resistance` alone only captures the viscous low-flow regime and undershoots real screens/dampers at higher mass flow, while `loss` (quadratic) alone loses all acoustic resistance at the quiescent (M->0) limit; add a combined element (`dpt = a*mdot + b*mdot*abs(mdot)`) or a composite chaining the two existing atoms so both regimes and the quiescent damping hold together.
- [ ] We should support Cantera YAML, and do not have our own format for species library / mechanism files. Only the "thermo.inp" style input for species library, and Cantera style input for species library and mechanisms (mechanisms not implented yet)

## To document

- [ ] We would prefer numbered, hierarchical sections for ease in referencing
- [ ] "Solution variables and state recovery", the section describes the thermo derivatives for the case of perfect gas, but we'd like to see here how the equilibrium solver gets into play through derivative (without going into internals of equilibrium solver)
- [ ] A proper flowchart for mean flow solver

## To verify

## To brainstorm

- [ ] We reject BC's when there is no absolute pressure reference from any of them. For such cases, would it work if we had an absolute pressure reference at some arbitrary edge state in the domain? If so, we could let the user enforce absolute pressure at exactly one edge.
- [ ] Balance/scale the perturbation operator `A(omega)` before factorization (two-sided diagonal balancing, or nondimensionalizing omega by a reference `c/L`) so the dominant boundary/regularizer entries no longer swamp the rest. This is the structural fix for the conditioning the stability code currently works around (the omega-update convergence test in the corrector, the max|A|-normalized residual); it would make residual-based tests meaningful again. Note: a similarity transform preserves eigenvalues, so it improves conditioning but does not thin the dense convected spectrum (that stays the Nyquist regime).

## To discuss

- [ ] "C_c" and "C_d" (discharge coefficient) distinction, we should made it clear. And if we not do so already, we should start supporting "C_d".
- [ ] We have a nice way of accessing and bundling a generic species library. How about viscosity and such transport properties? How can we handle this? How does NASA CEA and Cantera handle this?

## To test

## Issues

- [ ] Solving for mean flow on a 10000+ nodes case is remarkably fast, but doing a respons analysis to compute tranfer matrices takes too long. We should check whether we could improve this without going crazy lengths.
- [ ] Type hinting is pretty much missing from the entire codebase, and we'd prefer good type hinting to be present.
- [ ] The LaTeX rendering, especially inline rendering in Quarto docs still has a large font size, we need to reduce this a bit.
- [ ] In notebooks, we seem to have too many imports from our package, looking a bit intimidating. It is too much for the user, and perhaps some functionality could be offered as class methods? We should discuss.
- [ ] We don't have any mechanism to prevent connecting incompatible elements with each other - see the guardrails we put in the UI. Perhaps we were too strict, this is open to re-evalute.
- [ ] UI/solver parity: the solver's `CHECK_CONNECTED` rejects multiple disconnected sub-networks, but the UI validity pass only flags fully-isolated nodes. Add a connected-components check to the UI so a UI-valid model can't fail on solve.

## Deferred

- [ ] Frequency-dependent radiation / reservoir outlet BCs: a small library of named analytic radiation-impedance constructors returning `Z(omega)` callables for `PerturbationBC.impedance` (auto-converts to `R`). Unflanged open end (Levine–Schwinger low-`ka`: `Z/(rho c) ~= (ka)^2/4 + i*0.6133*ka`), flanged/baffled-piston (`(ka)^2/2 + i*0.82*ka`), and jet-pipe with mean flow (Munt/Rienstra, Mach/Strouhal-dependent). Keep each analytic in `omega` so it continues off the real axis and works with the contour eigensolver; tabulated/measured radiation data must go through the AAA/barycentric rational fit first (else Nyquist / forced-response only). Replaces the constant `open_end` / `mean_flow_open_end` reflections for realistic open outlets.
- [ ] Condensed-phase equilibrium *products* (soot/graphite, condensed oxides). v1 products are gas-only (`SpeciesLibrary.product_mask`); condensed species are feed-only (set elements + enthalpy, masked out of the burnt slate). Related edge case: an auto-slate feed element with no gas product (a metal) leaves the burnt solve unbalanced.
- [ ] Compositional ("indirect") noise -- one gap remains. It is already captured wherever the linearization is inherited: `J_alg` carries composition->acoustic at a flame, an area change, and the inherited `choked_nozzle_outlet` element (its critical-mass-flux row is complex-stepped through composition -> the `R_xi` column for free); and scalar *ports* now read out in the scattering matrices (`response.py`). Open: the hand-written analytic compact closures (`PerturbationBC.choked_nozzle`/`constant_mass_flow`) drop `R_xi` (flagged by `CompositionalNoiseWarning`) -- complex-step the closure's composition dependence to restore it. Theory: Magri JFM 2016.
- [ ] Non-compact / through-throat nozzle-response solver: integrate the linearized-Euler perturbation ODEs through the `M(x)` profile with the M=1 sonic-regularity (L'Hopital) condition (Stow-Dowling / Duran-Moreau). Handles the sonic throat the transit-time network model cannot (`tau_- = L/(c-u) -> inf`), and is the proper distributed-source compositional-noise route. The deferred supersonic / M=1 scope.
- [ ] Re-order algorithm - deferred because current solvers do not utilize this
- [ ] Composite-element follow-ups (the v1 build ships Part-I/II append-at-tail): (a) bandwidth-aware renumber of expanded internals into the Solve space (gated on the re-order algorithm above; `splu` re-permutes so append is harmless today -- `scratch/composite-elements.md` Part III); (b) composite YAML serialization (a composite currently raises on export -- emit the recipe + params, or the expanded atoms); (c) Class-3 sub-network composites (branching internals via the indexed port-wiring table -- `transfer_matrix` across a branch point already warns, use `multiport_scattering_matrix`).
- [ ] Heated finite-volume element's independent energy store: a finite-volume flame / heated plenum carrying its own energy DOF that couples the energy/transport row to `S(omega)` (the thermal capacitance the lumped flame neglects). Not a frozen-mean `M` stamp -- the energy "row" is the per-edge transport equation, and it needs finite-volume-flame mean geometry; gated on that design. See `scratch/inertance-end-correction-theory.md` s3.3 (the doc's own deferred item).
- [ ] `perturbation_response(freeze=...)`: a lossless frozen termination (closed stub) is ill-conditioned at its resonance (real-axis pole). Add optional auto-regularization (small wall loss / pole-skip).
