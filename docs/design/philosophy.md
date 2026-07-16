# Design philosophy

The theory documents establish *what* the framework computes; this track explains *why the software is shaped the way it is*, so that a contributor understands the load-bearing decisions before touching the code.
Four principles govern the implementation.
Each is a response to a specific failure mode of compressible-network solvers, and each is carried by a hard constraint that the rest of the codebase upholds without exception.

- **Smoothness over branching.**
  A residual never branches on the flow state: an upwind direction, a loss sign, a subsonic-or-choked switch is expressed by a smooth analytic weight, never by an `if`, an `abs`, a `min`, or a `max`.
  A branch is a kink that halts Newton's method exactly where a flow reverses or a passage chokes, and it is opaque to the derivative engine, which takes its decision on the real part and discards the imaginary seed.
  The constraint is that all residual mathematics is complex-step-safe, enforced kernel by kernel (see [the smoothness contract](smoothness-contract.md) and [the complex-step derivative](complex-step.qmd)).
- **Exact derivatives over approximate.**
  The Jacobian is obtained by complex-step differentiation, exact to machine precision, never hand-derived (a bug farm) nor finite-differenced (an irreducible cancellation error).
  A change to a residual then needs no matching derivative code, and no search direction is corrupted by a wrong or noisy Jacobian (see [the complex-step derivative](complex-step.qmd)).
- **Discovery over prescription.**
  The flow regime is an output of the solve, not an input: the solver is told neither the direction of flow on an edge, nor which passage chokes, nor where a stream reverses, and settles all of them from an uninformed cold start.
  The constraint is that no assembly step may consult the flow direction; direction-dependent behaviour enters only through smooth weights the solver is free to move (see [equation structure](../theory/equation-structure.md), [choking](../theory/choking.qmd), and [the solver](solver.md)).
- **Kernels over objects.**
  The heavy lifting is done by small, typed, compiled kernels dispatched on an integer identifier, not by a hierarchy of objects with virtual methods.
  The object shell builds networks, names elements, and presents results, but owns no numerics and never intrudes on the inner loop (see [kernel architecture](kernel-architecture.md)).

The four are a single stance read from four sides, and they compose as a chain: kernels make the dtype-generic dual compilation possible, dual compilation makes the exact complex-step derivatives cheap, the exact derivatives are trustworthy only because the residuals are smooth, and the smoothness is what lets the solver discover the regime without branching.
The remaining design documents develop each in turn, and [assembly](assembly.md) and [reproducibility](reproducibility.md) then show how they meet the network's sparsity and how the results stay reproducible in use.
