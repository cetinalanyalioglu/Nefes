## To implement

- [ ] Kinetic-energy coupling for the reacting closure: `EQ_KERNEL`/`EQ_FROZEN` currently drop `u^2/2` (h ~ h_t), O(M^2) at low Mach. Restore the outer KE fixed point (R-B2.2) with the 2-inner-solve IFT (closure ift, analogous to `perfect_gas._attach_density_imag`). Note the perturbation caloric map (`characteristics.edge_caloric`) drops the KE term (`m=0`) to match this closure; restore `m=u` there too when the closure carries KE.
- [ ] Element-dropping in the `@njit` equilibrium kernel (`fns/thermo/_chem.equilibrate_hp`) for burnt edges whose elemental abundance (`Z = xi @ Zfeed`) has a zero element (e.g. a parallel branch whose products lack carbon). Mirror thermolib's keep_el/keep_sp compaction (locate-on-real). Series injection (all elements present downstream) avoids it today.
- [ ] We need a robust and nice framework to create analytically continuous curves from the tabulated transfer function and reflection coefficient inputs for subsequent stability analysis.
- [ ] We are now considering "convenience" elements, that will transform into multiple elements when added to network. We should be careful to preserve proper numbering. I am not certain yet, maybe we re-run the re-numbering algorithm, or just insert the new "serial" portion as incremental numbers to the correct position and shift the rest? First user: a `helmholtz_resonator(V, neck_length, neck_area)` wrapper (tee + neck duct + cavity), now buildable by hand from the primitives.
- [ ] Storage `M`: neck inertance / end-correction (added-mass) terms via the `_STORAGE_BUILDERS` registry (`stamps.py`) -- area changes / losses / orifice necks stamp `i*omega*L` onto a momentum row, the inertance dual of the cavity compliance. Folds the geometric neck length's end corrections (delta_end ~ 0.85a flanged) into `l_eff`.
- [ ] Cavity energy-ledger hook (HR-plan Phase 4): generalize `_stored_energy` (currently duct-only) to add the cavity's lumped `0.5 * C * |p'_c|^2`, so `forced_power_balance` / `modal_energy_balance` close with a cavity present. Operator is already complete (`J_alg + i*omega*M + P + S`); this is the diagnostics side.
- [ ] Dedicated sudden-contraction element resolving the vena-contracta state (composite: isentropic to vena contracta + Borda re-expansion) for exact loss and minimum static pressure at higher Mach. The current `sudden_area_change` `cc`-loss uses the incompressible 1/2 rho u^2 head, accurate only to O(M^2).

## To verify

- [ ] Does "perturbation_response" properly cover entropy and scalar waves? Current docstring sounds incompatible.

## To brainstorm

- [ ] We reject BC's when there is no absolute pressure reference from any of them. For such cases, would it work if we had an absolute pressure reference at some arbitrary edge state in the domain? If so, we could let the user enforce absolute pressure at exactly one edge.

## To discuss

- [ ] We could add an element to force a split fraction at a splitter/junction - might be incompatible with the equation structure, we'd discuss.
- [ ] It could make sense to add a realistic frequency-dependent BC representing a open outlet or a reservoir, what would be the options here? What is available in the literature?
- [ ] Is Marble-Candel BC frequency dependent - if so why?
- [ ] Why do we need a "h_ref" in addition to "T_ref"?
- [ ] Verify residual normalization approach, it could be better if we normalize with the total stored quantities within the domain.
- [ ] We could make homotopy parameter dependent on the largest dP in the domain - for very small dP existing values seemed a bit high.
- [ ] How do we assign the "equilibrium" or "frozen" closure in the automatic mode currently? Still against solving a progress-variable like equation?

## To test

## Issues

- [ ] We don't have any mechanism to prevent connecting incompatible elements with each other - see the guardrails we put in the UI. Perhaps we were too strict, this is open to re-evalute.

## Deferred

- [ ] Compositional ("indirect") noise -- only the gaps remain. It is already captured wherever the linearization is inherited: `J_alg` carries composition->acoustic at a flame, an area change, and the inherited `choked_nozzle_outlet` element (its critical-mass-flux row is complex-stepped through composition -> the `R_xi` column for free). Open: (a) the hand-written analytic compact closures (`PerturbationBC.choked_nozzle`/`constant_mass_flow`) drop `R_xi` (flagged by `CompositionalNoiseWarning`) -- complex-step the closure's composition dependence to restore it; (b) scalar *ports* in the measurement scattering matrices (`response.py`) so `R_xi` reads out as a coefficient. Theory: Magri JFM 2016.
- [ ] Non-compact / through-throat nozzle-response solver: integrate the linearized-Euler perturbation ODEs through the `M(x)` profile with the M=1 sonic-regularity (L'Hopital) condition (Stow-Dowling / Duran-Moreau). Handles the sonic throat the transit-time network model cannot (`tau_- = L/(c-u) -> inf`), and is the proper distributed-source compositional-noise route. The deferred supersonic / M=1 scope.
- [ ] Re-order algorithm - deferred because current solvers do not utilize this
- [ ] `perturbation_response(freeze=...)`: a lossless frozen termination (closed stub) is ill-conditioned at its resonance (real-axis pole). Add optional auto-regularization (small wall loss / pole-skip).
