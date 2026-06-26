## To implement

- [ ] Kinetic-energy coupling for the reacting closure: `EQ_KERNEL`/`EQ_FROZEN` currently drop `u^2/2` (h ~ h_t), O(M^2) at low Mach. Restore the outer KE fixed point (R-B2.2) with the 2-inner-solve IFT (closure ift, analogous to `perfect_gas._attach_density_imag`). Note the perturbation caloric map (`characteristics.edge_caloric`) drops the KE term (`m=0`) to match this closure; restore `m=u` there too when the closure carries KE.
- [ ] Element-dropping in the `@njit` equilibrium kernel (`fns/thermo/_chem.equilibrate_hp`) for burnt edges whose elemental abundance (`Z = xi @ Zfeed`) has a zero element (e.g. a parallel branch whose products lack carbon). Mirror thermolib's keep_el/keep_sp compaction (locate-on-real). Series injection (all elements present downstream) avoids it today.
- [ ] Warm-start the per-edge equilibrium solve from the previous converged composition (and avoid the double solve in `closure_solve` + `thermo_state`) to cut the reacting-network cost.
- [ ] We need a robust and nice framework to create analytically continuous curves from the tabulated transfer function and reflection coefficient inputs for subsequent stability analysis.
- [ ] Cut-on frequency analysis for ducts - we'd like to stay below that
- [ ] Need some way to plot network topology and element indices in Jupyter environment
- [ ] We are now considering "convenience" elements, that will transform into multiple elements when added to network. We should be careful to preserve proper numbering. I am not certain yet, maybe we re-run the re-numbering algorithm, or just insert the new "serial" portion as incremental numbers to the correct position and shift the rest?
- [ ] Dedicated sudden-contraction element resolving the vena-contracta state (composite: isentropic to vena contracta + Borda re-expansion) for exact loss and minimum static pressure at higher Mach. The current `sudden_area_change` `cc`-loss uses the incompressible 1/2 rho u^2 head, accurate only to O(M^2).
- [ ] Extend acoustic-power diagnostics (`perturbation/power.py`): full-domain energy integral E (integrate the per-duct energy density along its length) to close `2 sigma E = net boundary power` *quantitatively* (now only sign-checked), plus a `boundary_power` for `ForcedResponse` (driven-case power balance) and an intensity-along-ducts field.
- [ ] Nyquist driver: locate off-axis unstable-mode frequencies from the real-axis sweep (the `|D|` minima are onset/least-stable points, not the strongly-unstable mode frequencies); and a reliable `N(A0)` for the source-free passive operator so the encirclement count is absolute (not just `N(A)-N(A0)`) when the reacting/convective spectrum defeats the contour eigensolver.
- [ ] Print residuals equation-by-equation instead of a global residual value.
- [ ] Find a proper name for the parameter "stab" that reads well both in mathematical documentation and as a variable in the codebase. A greek letter can be considered here.

## To verify

- [ ] Make sure the default BC for mass flow inlet in the UI is "inherited"
- [ ] Make sure the default BC for total pressure inlet in the UI is "inherited"
- [ ] Total pressure inlet can allow reverse flow, the total pressure value would be used as static pressure - do we already support this?

## To brainstorm

- [ ] We reject BC's when there is no absolute pressure reference from any of them. For such cases, would it work if we had an absolute pressure reference at some arbitrary edge state in the domain? If so, we could let the user enforce absolute pressure at exactly one edge.

## To discuss

- [ ] It could make sense to add a realistic frequency-dependent BC representing a open outlet or a reservoir, what would be the options here? What is available in the literature?
- [ ] Is Marble-Candel BC frequency dependent - if so why?
- [ ] Why do we need a "h_ref" in addition to "T_ref"?

## To test

## Issues

## Deferred

- [ ] Compositional (scalar->acoustic) scattering -- "indirect noise" from composition inhomogeneities. Add the off-diagonal scalar->acoustic coefficient `R_xi` at non-uniform sections/terminals (compact nozzles, area changes), where an arriving scalar wave partially radiates as sound. Theory: Magri, JFM 2016, "Compositional inhomogeneities as a source of indirect combustion noise"; fits the existing `delta(reduced-mass-flow)=0` compact-nozzle route by complex-stepping the choked mass flux's composition dependence (reacting backend only -- a perfect gas has none). Needs reacting-thermo composition derivatives + a literature validation case. The same coefficient also unblocks scalar *ports* in the measurement scattering matrices (`response.py`). NB: scalar *driving* at an inflow (`PerturbationBC(driven=...)`), convection, and `ForcedResponse` surfacing are implemented and warn (`CompositionalNoiseWarning`); only the scattering/noise direction is missing.
- [ ] Re-order algorithm - deferred because current solvers do not utilize this
