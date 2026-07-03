# Design philosophy

The theory documents establish *what* the framework computes; this track explains *why the software is shaped the way it is*, so that a contributor understands the load-bearing decisions before touching the code.
Four principles govern the design, and each is not a preference but a response to a specific failure mode of compressible-network solvers — each maps onto a hard constraint that the rest of the codebase upholds without exception.
This document states the four and the constraint each implies; the remaining design documents develop them in detail.

The principles are best read as a single stance: the solver should *discover* the physical state from an uninformed start, using *exact* information, over residuals that are *smooth* everywhere, evaluated by small *kernels* rather than mediated by objects.
The presentation takes them in the order in which they constrain the code, from the innermost numerical contract outward to the architectural one.

## Smoothness over branching

The first principle is that a residual must never branch on the flow state.
Every quantity a residual depends on — an upwind direction, a loss sign, a regime switch between subsonic and choked — is expressed by a smooth, analytic weighting rather than by an `if`, an `abs`, a `min`, or a `max`.
The reason is twofold and compounding.
A branch is a *kink*: Newton's method assumes a differentiable residual, and a discontinuous derivative halts it exactly where a flow reverses or a passage chokes — the states the solver most needs to pass through.
A branch is also *opaque to the derivative engine*: it takes its decision on the real part of a complex-stepped evaluation and discards the imaginary seed, silently corrupting the Jacobian (see [the complex-step derivative](complex-step.qmd)).
This principle is the hard constraint that all residual mathematics is complex-step-safe — smooth, complex-analytic, free of branches on the flow state — and it is enforced kernel by kernel by the roll-call of [the smoothness contract](smoothness-contract.md).

## Exact derivatives over approximate

The second principle is that the Jacobian is computed exactly, never approximated.
A hand-derived Jacobian is a bug farm — each residual change must be mirrored in a derivative by hand — and a finite-difference Jacobian carries an irreducible cancellation error that no fixed step escapes.
The framework instead obtains every derivative by complex-step differentiation, which is exact to machine precision and follows the residual automatically, so a change to a residual needs no matching derivative code.
The dividend is that the search directions are never corrupted by a wrong or noisy Jacobian, and the one place an iteration would defeat the seed — the implicit density solve — is handled by an exact implicit-function splice rather than a numerical fallback.
This principle presupposes the first: exact complex-step derivatives are available *only* because every residual is analytic, so the two principles are a single contract viewed from two sides.

## Discovery over prescription

The third principle is that the flow regime is an *output* of the solve, not an input to it.
The solver is not told which way the gas flows on each edge, which passage chokes, or where a stream reverses; it discovers all of these by settling the same square system of equations from an uninformed cold start.
This is what the fixed equation split, the smooth upwinding, and the emergent choking complementarity are all in service of: the residual vector and Jacobian keep their dimension and their smoothness regardless of the flow direction, so the solver may reverse a flow, choke a throat, or start from exact rest without any change to the assembled problem (see [equation structure](../theory/equation-structure.md) and [choking](../theory/choking.qmd)).
The hard constraint this implies is that no assembly step may consult the flow direction; direction-dependent behaviour enters only through smooth weights that the solver is free to move.

## Kernels over objects

The fourth principle is architectural: the heavy lifting is done by small, typed, compiled *kernels*, not by a hierarchy of objects.
Each element's residual is a plain function dispatched on an integer identifier and compiled by `numba`, dtype-generic so that one source yields both the real-valued residual and the complex-stepped Jacobian seed (see [kernel architecture](kernel-architecture.md)).
The object-oriented shell exists only at the edges of the system — for building networks, naming elements, and presenting results — and owns no numerics; the state lives on the edges and the equations in the kernels, so the performance-critical inner loop is a flat, cache-friendly sweep rather than a tree of virtual calls.
The constraint here is one of layering: numerics live in kernels that never import the shell, and the shell never intrudes on the inner loop.

## How the principles compose

The four are not independent choices but a chain, each enabling the next.
Kernels are what make dtype-generic dual compilation possible; dual compilation is what makes exact complex-step derivatives cheap; exact derivatives are trustworthy only because the residuals are smooth; and smoothness is what lets the solver discover the regime without branching.
The remaining design documents follow this chain: [the complex-step derivative](complex-step.qmd) and [the smoothness contract](smoothness-contract.md) develop the numerical core, [kernel architecture](kernel-architecture.md) and [assembly](assembly.md) the computational structure, [the solver](solver.md) the globalization that turns a local method into a robust one, and [reproducibility](reproducibility.md) the practices that keep the results trustworthy across environments.
