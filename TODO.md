## To implement

- [ ] Dynamic source/flame response: n-tau / FTF source S(omega) in `stamp_sources`. The descriptor (`DynamicSource`, `CompiledProblem.node_dynamic_source`) and the steady `MASS_SOURCE` / `FLAME_*` elements now exist and are acoustically passive; wire S(omega) to consume `node_dynamic_source` (e.g. fuel mdot' responding to upstream u'). Lights up active/thermoacoustic modes in `eigenmodes()` (Beyn driver is source-agnostic); validate with a Rijke tube. Teach the fixed-pattern assembler (`operator._AssemblyPlan`) about S(omega)'s frequency dependence or fall back to `_assemble_reference` when a source is present, else its omega-dependence freezes at the pattern-probe frequency. Feedback sits on node rows so the isentropic option leaves it intact.
- [ ] Kinetic-energy coupling for the reacting closure: `EQ_KERNEL`/`EQ_FROZEN` currently drop `u^2/2` (h ~ h_t), O(M^2) at low Mach. Restore the outer KE fixed point (R-B2.2) with the 2-inner-solve IFT (closure ift, analogous to `perfect_gas._attach_density_imag`).
- [ ] Element-dropping in the `@njit` equilibrium kernel (`fns/thermo/_chem.equilibrate_hp`) for burnt edges whose elemental abundance (`Z = xi @ Zfeed`) has a zero element (e.g. a parallel branch whose products lack carbon). Mirror thermolib's keep_el/keep_sp compaction (locate-on-real). Series injection (all elements present downstream) avoids it today.
- [ ] Warm-start the per-edge equilibrium solve from the previous converged composition (and avoid the double solve in `closure_solve` + `thermo_state`) to cut the reacting-network cost.
- [ ] Cut-on frequency analysis for ducts - we'd like to stay below that
- [ ] Constant mass flow rate acoustic boundary condition. Linearize mass flow rate and directly use resultant expression as BC. 
- [ ] Need some way to plot network topology and element indices in Jupyter environment
- [ ] We are now considering "convenience" elements, that will transform into multiple elements when added to network. We should be careful to preserve proper numbering. I am not certain yet, maybe we re-run the re-numbering algorithm, or just insert the new "serial" portion as incremental numbers to the correct position and shift the rest?
- [ ] Dedicated sudden-contraction element resolving the vena-contracta state (composite: isentropic to vena contracta + Borda re-expansion) for exact loss and minimum static pressure at higher Mach. The current `sudden_area_change` `cc`-loss uses the incompressible 1/2 rho u^2 head, accurate only to O(M^2).

## To verify

## To brainstorm

## To test

## Deferred

- [ ] Re-order algorithm - deferred because current solvers do not utilize this
