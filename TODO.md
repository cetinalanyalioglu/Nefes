## To implement

- [ ] Kinetic-energy coupling for the reacting closure: `EQ_KERNEL`/`EQ_FROZEN` currently drop `u^2/2` (h ~ h_t), O(M^2) at low Mach. Restore the outer KE fixed point (R-B2.2) with the 2-inner-solve IFT (closure ift, analogous to `perfect_gas._attach_density_imag`). Note the perturbation caloric map (`characteristics.edge_caloric`) drops the KE term (`m=0`) to match this closure; restore `m=u` there too when the closure carries KE.
- [ ] Element-dropping in the `@njit` equilibrium kernel (`fns/thermo/_chem.equilibrate_hp`) for burnt edges whose elemental abundance (`Z = xi @ Zfeed`) has a zero element (e.g. a parallel branch whose products lack carbon). Mirror thermolib's keep_el/keep_sp compaction (locate-on-real). Series injection (all elements present downstream) avoids it today.
- [ ] Warm-start the per-edge equilibrium solve from the previous converged composition (and avoid the double solve in `closure_solve` + `thermo_state`) to cut the reacting-network cost.
- [ ] Cut-on frequency analysis for ducts - we'd like to stay below that
- [ ] Need some way to plot network topology and element indices in Jupyter environment
- [ ] We are now considering "convenience" elements, that will transform into multiple elements when added to network. We should be careful to preserve proper numbering. I am not certain yet, maybe we re-run the re-numbering algorithm, or just insert the new "serial" portion as incremental numbers to the correct position and shift the rest?
- [ ] Dedicated sudden-contraction element resolving the vena-contracta state (composite: isentropic to vena contracta + Borda re-expansion) for exact loss and minimum static pressure at higher Mach. The current `sudden_area_change` `cc`-loss uses the incompressible 1/2 rho u^2 head, accurate only to O(M^2).
- [ ] Extend acoustic-power diagnostics (`perturbation/power.py`): full-domain energy integral E (integrate the per-duct energy density along its length) to close `2 sigma E = net boundary power` *quantitatively* (now only sign-checked), plus a `boundary_power` for `ForcedResponse` (driven-case power balance) and an intensity-along-ducts field.

## To verify

## To brainstorm

- [ ] We reject BC's when there is no absolute pressure reference from any of them. For such cases, would it work if we had an absolute pressure reference at some arbitrary edge state in the domain? If so, we could let the user enforce absolute pressure at exactly one edge.

## To discuss

- [ ] outlet_boundaries.ipynb - the claim is that inlet needs to have a an R=.64 for neutrality. If I was setting this case up myself, I would probably put p'=0 for the reservoir inlet, but you choose R=0.8 arbitrarily. A boundary being neutral in terms of acoustic energy balance is *something*, is it commonly used as a boundary condition or as a special point in literature?

## To test

## Issues

- [ ] Mass flow inlet should not allow reverse flow.

## Deferred

- [ ] Re-order algorithm - deferred because current solvers do not utilize this
