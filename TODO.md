## Major issues

- What are we missing in terms of network input verification? One example here is which elements allow area change across them, and which elements do not. We should ensure consistency of area changes.

- If network has more than 2 terminal nodes, the automatic forcing terminal selection in perturbation_response fails. We should think here, current ideas are: for case A force all inlets, for case B force all outlet or user should explicitly provide input.

## Minor issues

### entropy_generator.ipynb
- The throat mach number should start from the quiescent case (zero Mach) or a very low Mach number to match the figures in the paper.

## To verify

## To brainstorm
- Area change elements may specify how area changes internally in a tabulated form or presets (e.g. linear), and we could discretize the element internally while solving or handle this at the element formulation level.

## Deferred

### Sudden-area-change switch biases the perturbation by O(eps)
The momentum<->isentropic smooth switch leaks its loss residual into the frozen perturbation Jacobian; per-element `eps` is the current workaround.
Proper fix: give the perturbation linearization its own sharp smoothing, decoupled from the mean-flow homotopy `eps`.

