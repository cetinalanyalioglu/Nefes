## To implement

- [ ] Kinetic-energy coupling for the reacting closure: `EQ_KERNEL`/`EQ_FROZEN` currently drop `u^2/2` (h ~ h_t), O(M^2) at low Mach. Restore the outer KE fixed point (R-B2.2) with the 2-inner-solve IFT (closure ift, analogous to `perfect_gas._attach_density_imag`). Note the perturbation caloric map (`characteristics.edge_caloric`) drops the KE term (`m=0`) to match this closure; restore `m=u` there too when the closure carries KE.
- [ ] Element-dropping in the `@njit` equilibrium kernel (`fns/thermo/_chem.equilibrate_hp`) for burnt edges whose elemental abundance (`Z = xi @ Zfeed`) has a zero element (e.g. a parallel branch whose products lack carbon). Mirror thermolib's keep_el/keep_sp compaction (locate-on-real). Series injection (all elements present downstream) avoids it today.
- [x] Warm-start the per-edge equilibrium solve from the previous converged composition (and avoid the double solve in `closure_solve` + `thermo_state`) to cut the reacting-network cost.
- [ ] We need a robust and nice framework to create analytically continuous curves from the tabulated transfer function and reflection coefficient inputs for subsequent stability analysis.
- [x] Cut-on frequency analysis for ducts - we'd like to stay below that. We could add some utility tool somewhere to report cut-on frequencies.
- [x] Need some way to plot network topology and element indices/names in Jupyter environment, just for visual diagnostics.
- [ ] We are now considering "convenience" elements, that will transform into multiple elements when added to network. We should be careful to preserve proper numbering. I am not certain yet, maybe we re-run the re-numbering algorithm, or just insert the new "serial" portion as incremental numbers to the correct position and shift the rest?
- [ ] Dedicated sudden-contraction element resolving the vena-contracta state (composite: isentropic to vena contracta + Borda re-expansion) for exact loss and minimum static pressure at higher Mach. The current `sudden_area_change` `cc`-loss uses the incompressible 1/2 rho u^2 head, accurate only to O(M^2).
- [x] Extend acoustic-power diagnostics (`perturbation/power.py`): full-domain energy integral E (integrate the per-duct energy density along its length) to close `2 sigma E = net boundary power` *quantitatively* (now only sign-checked), plus a `boundary_power` for `ForcedResponse` (driven-case power balance) and an intensity-along-ducts field.
- [x] Nyquist driver: locate off-axis unstable-mode frequencies from the real-axis sweep (the `|D|` minima are onset/least-stable points, not the strongly-unstable mode frequencies); and a reliable `N(A0)` for the source-free passive operator so the encirclement count is absolute (not just `N(A)-N(A0)`) when the reacting/convective spectrum defeats the contour eigensolver.
- [x] Visualize search contour and found eigenvalues in the complex plane, if we have a method that already does this, update it.
- [x] Helper to easily generate mixtures, e.g. specify fuel, oxidizer and desired equivalence ratio - obtain mole/mass fractions. Update notebooks/examples that would make use of that.
- [x] Linear acoustic resistance element to model resistance in quiescent cases. Add a TODO item in the repo of UI to add the UI counterpart of this element after implementation.
- [x] Do we have any facility to pass the perturbation analysis related results to the UI save file? If not, we should figure out a way to do so.
- [x] Some central toggle to turn on/off LaTeX labels on plots - they don't always render properly in notebooks and user gets labels filled with "$", "\" etc.

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
- [x] We may not have completely internalized "thermo.inp". This file should be embedded in the project, installed with the python package as the default species library, and unless the user manually points to a new one, there should be no need to name it.

## Deferred

- [ ] Compositional ("indirect") noise -- only the gaps remain. It is already captured wherever the linearization is inherited: `J_alg` carries composition->acoustic at a flame, an area change, and the inherited `choked_nozzle_outlet` element (its critical-mass-flux row is complex-stepped through composition -> the `R_xi` column for free). Open: (a) the hand-written analytic compact closures (`PerturbationBC.choked_nozzle`/`constant_mass_flow`) drop `R_xi` (flagged by `CompositionalNoiseWarning`) -- complex-step the closure's composition dependence to restore it; (b) scalar *ports* in the measurement scattering matrices (`response.py`) so `R_xi` reads out as a coefficient. Theory: Magri JFM 2016.
- [ ] Non-compact / through-throat nozzle-response solver: integrate the linearized-Euler perturbation ODEs through the `M(x)` profile with the M=1 sonic-regularity (L'Hopital) condition (Stow-Dowling / Duran-Moreau). Handles the sonic throat the transit-time network model cannot (`tau_- = L/(c-u) -> inf`), and is the proper distributed-source compositional-noise route. The deferred supersonic / M=1 scope.
- [ ] Re-order algorithm - deferred because current solvers do not utilize this
- [ ] `perturbation_response(freeze=...)`: a lossless frozen termination (closed stub) is ill-conditioned at its resonance (real-axis pole). Add optional auto-regularization (small wall loss / pole-skip).
