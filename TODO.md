## To implement

- [ ] Constant mass flow rate acoustic boundary condition. Linearize mass flow rate and directly use resultant expression as BC.
- [ ] Need some way to plot network topology and element indices in Jupyter environment
- [ ] We are now considering "convenience" elements, that will transform into multiple elements when added to network. We should be careful to preserve proper numbering. I am not certain yet, maybe we re-run the re-numbering algorithm, or just insert the new "serial" portion as incremental numbers to the correct position and shift the rest?

## To verify
- [ ] Do we enforce unique element names? If not, we should. The external UI already takes care of that, manually built networks should also respect that. If we have a central "verification" procedure, this is something we should add there.
- [ ] For reversed flow in pressure outlets, do we use the same pressure value as static pressure when reversal happens, or do we use the prescribed static pressure value as backflow total pressure?

## To brainstorm
- [ ] Area change elements may specify how area changes internally in a tabulated form or presets (e.g. linear), and we could discretize the element internally while solving or handle this at the element formulation level.

## Deferred

### Sudden-area-change switch biases the perturbation by O(eps)
- [ ] The momentum<->isentropic smooth switch leaks its loss residual into the frozen perturbation Jacobian; per-element `eps` is the current workaround. Proper fix: give the perturbation linearization its own sharp smoothing, decoupled from the mean-flow homotopy `eps`.
