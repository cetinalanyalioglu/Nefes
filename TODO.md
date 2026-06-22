## Major issues

- [ ]What are we missing in terms of network input verification? One example here is which elements allow area change across them, and which elements do not. We should ensure consistency of area changes.

- [ ] If network has more than 2 terminal nodes, the automatic forcing terminal selection in perturbation_response fails. We should think here, current ideas are: for case A force all inlets, for case B force all outlet or user should explicitly provide input.

## Minor issues

- [ ] Scaling issue in plots, the maximum limit is too harsh, a bit margin is required. Sometimes distance between subplots is also problematic.

- [ ] Plots need descriptive (but brief) titles.

- [ ] Some kwargs take frequency in Hz, some take in rad/s. We should unify all in Hz.

### entropy\_generator.ipynb
- [ ] The throat mach number should start from the quiescent case (zero Mach) or a very low Mach number to match the figures in the paper.

## To implement

- [ ] Constant mass flow rate acoustic boundary condition. Linearize mass flow rate and directly use resultant expression as BC.
- [ ] Need some way to plot network topology and element indices in Jupyter environment
- [ ] We are now considering "convenience" elements, that will transform into multiple elements when added to network. We should be careful to preserve proper numbering. I am not certain yet, maybe we re-run the re-numbering algorithm, or just insert the new "serial" portion as incremental numbers to the correct position and shift the rest?

## To test
- [ ] Verify when we perturbation response is *actually* linear by changing the amplitude of excitations

## To verify
- [ ] For reversed flow in pressure outlets, do we use the same pressure value as static pressure when reversal happens, or do we use the prescribed static pressure value as backflow total pressure?

## To brainstorm
- [ ] Area change elements may specify how area changes internally in a tabulated form or presets (e.g. linear), and we could discretize the element internally while solving or handle this at the element formulation level.

## Deferred

### Sudden-area-change switch biases the perturbation by O(eps)
- [ ] The momentum<->isentropic smooth switch leaks its loss residual into the frozen perturbation Jacobian; per-element `eps` is the current workaround. Proper fix: give the perturbation linearization its own sharp smoothing, decoupled from the mean-flow homotopy `eps`.

