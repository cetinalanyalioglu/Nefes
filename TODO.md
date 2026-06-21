## Major issues

## Minor issues

### entropy_generator.ipynb
- The throat mach number should start from the quiescent case (zero Mach) or a very low Mach number to match the figures in the paper.

### Sudden-area-change switch biases the perturbation by O(eps)
The momentum<->isentropic smooth switch leaks its loss residual into the frozen perturbation Jacobian; per-element `eps` is the current workaround.
Proper fix: give the perturbation linearization its own sharp smoothing, decoupled from the mean-flow homotopy `eps`.

## To implement
- Solver should print progress on a user-specified interval and verbosity level.

## To verify
- Quiescent analysis capability
