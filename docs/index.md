# Nefes documentation

Nefes models a compressible fluid system as a directed graph and solves for its steady mean flow, and for the linear perturbations (acoustics, entropy and scalar perturbations) behaviour around that flow, without resolving the full three-dimensional field.
Throughout the documentation we will casually refer to the whole set of these linearized perturbations as *acoustics*, and the associated network approach as *acoustic network*.
Depending on the context, the reader should be aware that the set of perturbation variables may include entropy waves and scalar (composition) waves in addition to the regular upstream and downstream acoustic waves.
This tree is the authoritative documentation: the physics and mathematics, the numerical and architectural philosophy, and the verification and validation evidence that backs every claim.

The pages are Markdown with `$…$` math, organized into three tracks plus a supporting layer.
A single symbol table ([`nomenclature.md`](nomenclature.md)) and a single bibliography ([`references.bib`](references.bib)) are shared by every document.

## Where to start

- **New to the method** — begin with the [theory overview](theory/overview.md): what Nefes computes, the four modeling decisions it rests on, and the scope of the current version.
- **Building a network** — the [element reference](#reference) below, then the [modeling guide](reference/modeling-guide.md) and the [annotated example index](reference/examples.md).
- **Contributing to the code** — the [design philosophy](design/philosophy.md) track, whose signature pieces are the complex-step derivative engine and the smoothness contract.

## Track I — Theory

The physics and mathematics, each document opening with its assumptions ledger and citing or deriving every claim.

- `theory/overview.md` — what the method computes, the four modeling decisions, the subsonic scope.
- `theory/framework.md` … `theory/choking.md` — the mean-flow formulation: graph model, governing balances, state and recovery, transport, elements, well-posedness, characteristics, and emergent choking.
- `theory/thermochemistry.md` — mixture thermodynamics, chemical equilibrium, and the reacting closures.
- `theory/perturbation-network.md` — linearization about the mean flow into the frequency-domain acoustic operator.
- `theory/dynamic-sources.md` — the flame response (flame transfer function / n–τ) and compositional noise.
- `theory/analyses.md` — forced response and scattering, eigenmodes, real-frequency stability, energy budget, and the frequency-domain transfer/scattering-matrix descriptors.
- `theory/identification.md` — the inverse analysis: de-embedding an element's dynamic response from a measured network transfer matrix.
- `theory/limitations.md` — scope boundaries and honest open ends.

## Track II — Design philosophy

Why the code is shaped the way it is (concepts and contracts, not the API, which is unstable until the tagged release).

- `design/philosophy.md`, `design/kernel-architecture.md`, `design/complex-step.md`, `design/smoothness-contract.md`, `design/assembly.md`, `design/solver.md`, `design/reproducibility.md`.

## Track III — Validation

- `validation/validation-map.md` — every physical claim mapped to the case and test that checks it.
- `validation/verification.md`, `validation/benchmarks.md` — internal consistency checks and named literature benchmarks.

## Reference

- [API reference](reference/api/index.qmd) — reference for the Python API.
- [Atomic elements](reference/atomic-elements.md) — the irreducible network elements (boundaries, area changes, losses, transport, reacting elements, manifolds), each with its residual and theory.
- [Composite elements](reference/composite-elements.md) — convenience elements that expand to a graph of atomics at build time (orifice, lossy nozzle, sudden contraction, Helmholtz resonator, Fanno pipe, tapered duct).
- `reference/parameters.md` — the named-parameter and modification API (user guide), with `reference/parameter-schema.md` for the implementation contract behind it.
- `reference/modeling-guide.md` — mapping real components to network elements.
- `reference/examples.md` — annotated index of the runnable notebooks under [`examples/`](../examples/).
- `reference/ui-case-format.md` — the case-exchange YAML format shared with the Nemo graphical companion.