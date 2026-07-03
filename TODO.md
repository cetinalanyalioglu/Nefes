## To implement

- [ ] Proper estimation of friction coefficient for the pipe element, could be Moody charts or some other correlation.

## To verify

## To brainstorm

- [ ] We reject BC's when there is no absolute pressure reference from any of them. For such cases, would it work if we had an absolute pressure reference at some arbitrary edge state in the domain? If so, we could let the user enforce absolute pressure at exactly one edge.

## To discuss

- [ ] How does linear resistance element behave in mean flow?

## To test

## Issues

- [ ] In notebooks, we seem to have too many imports from our package, looking a bit intimidating. It is too much for the user, and perhaps some functionality could be offered as class methods? We should discuss.
- [ ] We don't have any mechanism to prevent connecting incompatible elements with each other - see the guardrails we put in the UI. Perhaps we were too strict, this is open to re-evalute.

## Deferred

- [ ] Frequency-dependent radiation / reservoir outlet BCs: a small library of named analytic radiation-impedance constructors returning `Z(omega)` callables for `PerturbationBC.impedance` (auto-converts to `R`). Unflanged open end (Levine–Schwinger low-`ka`: `Z/(rho c) ~= (ka)^2/4 + i*0.6133*ka`), flanged/baffled-piston (`(ka)^2/2 + i*0.82*ka`), and jet-pipe with mean flow (Munt/Rienstra, Mach/Strouhal-dependent). Keep each analytic in `omega` so it continues off the real axis and works with the contour eigensolver; tabulated/measured radiation data must go through the AAA/barycentric rational fit first (else Nyquist / forced-response only). Replaces the constant `open_end` / `mean_flow_open_end` reflections for realistic open outlets.
- [ ] Condensed-phase equilibrium *products* (soot/graphite, condensed oxides). v1 products are gas-only (`SpeciesLibrary.product_mask`); condensed species are feed-only (set elements + enthalpy, masked out of the burnt slate). Related edge case: an auto-slate feed element with no gas product (a metal) leaves the burnt solve unbalanced.
- [ ] Compositional ("indirect") noise -- one gap remains. It is already captured wherever the linearization is inherited: `J_alg` carries composition->acoustic at a flame, an area change, and the inherited `choked_nozzle_outlet` element (its critical-mass-flux row is complex-stepped through composition -> the `R_xi` column for free); and scalar *ports* now read out in the scattering matrices (`response.py`). Open: the hand-written analytic compact closures (`PerturbationBC.choked_nozzle`/`constant_mass_flow`) drop `R_xi` (flagged by `CompositionalNoiseWarning`) -- complex-step the closure's composition dependence to restore it. Theory: Magri JFM 2016.
- [ ] Non-compact / through-throat nozzle-response solver: integrate the linearized-Euler perturbation ODEs through the `M(x)` profile with the M=1 sonic-regularity (L'Hopital) condition (Stow-Dowling / Duran-Moreau). Handles the sonic throat the transit-time network model cannot (`tau_- = L/(c-u) -> inf`), and is the proper distributed-source compositional-noise route. The deferred supersonic / M=1 scope.
- [ ] Re-order algorithm - deferred because current solvers do not utilize this
- [ ] Composite-element follow-ups (the v1 build ships Part-I/II append-at-tail): (a) bandwidth-aware renumber of expanded internals into the Solve space (gated on the re-order algorithm above; `splu` re-permutes so append is harmless today -- `scratch/composite-elements.md` Part III); (b) composite YAML serialization (a composite currently raises on export -- emit the recipe + params, or the expanded atoms); (c) Class-3 sub-network composites (branching internals via the indexed port-wiring table -- `transfer_matrix` across a branch point already warns, use `multiport_scattering_matrix`).
- [ ] Heated finite-volume element's independent energy store: a finite-volume flame / heated plenum carrying its own energy DOF that couples the energy/transport row to `S(omega)` (the thermal capacitance the lumped flame neglects). Not a frozen-mean `M` stamp -- the energy "row" is the per-edge transport equation, and it needs finite-volume-flame mean geometry; gated on that design. See `scratch/inertance-end-correction-theory.md` s3.3 (the doc's own deferred item).
- [ ] `perturbation_response(freeze=...)`: a lossless frozen termination (closed stub) is ill-conditioned at its resonance (real-axis pole). Add optional auto-regularization (small wall loss / pole-skip).
