## To implement

- [ ] Heat-release n-tau flame source S(omega) in `stamp_sources`: lights up active/thermoacoustic (growing) modes in `eigenmodes()` (the Beyn driver is already source-agnostic). Needs a steady heat-release element kernel + complex-step probe; validate with a Rijke tube. When added, teach the fixed-pattern assembler (`operator._AssemblyPlan`) about S(omega)'s frequency dependence (extra phase/boundary-style slots) or fall back to `_assemble_reference` when a flame is present, else its omega-dependence would freeze at the pattern-probe frequency. Its acoustic feedback must sit on node rows so the isentropic option leaves it intact (entropy generation on transport rows is dropped under isentropic, as intended).
- [ ] Cut-on frequency analysis for ducts - we'd like to stay below that
- [ ] Save data to YAML
- [ ] Constant mass flow rate acoustic boundary condition. Linearize mass flow rate and directly use resultant expression as BC. 
- [ ] Need some way to plot network topology and element indices in Jupyter environment
- [ ] We are now considering "convenience" elements, that will transform into multiple elements when added to network. We should be careful to preserve proper numbering. I am not certain yet, maybe we re-run the re-numbering algorithm, or just insert the new "serial" portion as incremental numbers to the correct position and shift the rest?
- [ ] Dedicated sudden-contraction element resolving the vena-contracta state (composite: isentropic to vena contracta + Borda re-expansion) for exact loss and minimum static pressure at higher Mach. The current `sudden_area_change` `cc`-loss uses the incompressible 1/2 rho u^2 head, accurate only to O(M^2).

## To verify

- [ ] Do we enforce unique element names? If not, we should. The external UI already takes care of that, manually built networks should also respect that. If we have a central "verification" procedure, this is something we should add there.
- [ ] For reversed flow in pressure outlets, do we use the same pressure value as static pressure when reversal happens, or do we use the prescribed static pressure value as backflow total pressure?

## To brainstorm

## To test

## Deferred

- [ ] Re-order algorithm - deferred because current solvers do not utilize this
