# Reactive Flow Support & Standalone Thermochemistry Library — Requirements

Status: **draft for review** · Date: 2026-06-16 · Scope: design requirements, not an
implementation plan.

This document solidifies the requirements for adding **reactive-flow support** to the
`cbnflow` mean-flow network solver. Per the design decision below, the thermochemistry is
developed as a **separate, standalone library** with no dependency on `cbnflow`; the
network solver is one of its consumers. The immediate target is **chemical equilibrium
(HP) of arbitrary gaseous mixtures**; finite-rate / residence-time chemistry is a
**secondary target** the architecture must not preclude.

The document is in two parts:

- **Part A** — the standalone thermochemistry library (working name **`thermolib`**, TBD).
- **Part B** — the `cbnflow` integration that consumes it.

Requirement keywords: **shall** (mandatory), **should** (recommended), **may** (optional),
per RFC 2119. Requirement IDs are for traceability.


## 0. Architecture decision: two separate entities

- **AD-1.** The thermochemistry/equilibrium/kinetics functionality **shall** be developed
  as a **standalone library**, in its own repository, with its own tests, versioning, and
  release cycle, **not tied to `cbnflow`**. During development it is consumed via an
  **editable install from a sibling repository** (`pip install -e`); a published-package
  contract may follow once the API stabilizes (R-B6 / Q-7 resolved).
- **AD-2.** **Dependency direction is one-way:** `cbnflow` depends on the library; the
  library **shall not** import or assume anything from `cbnflow` (no notion of edges,
  ports, `(mdot, p, h_t)`, choking, or the network graph).
- **AD-3.** **Boundary of responsibility:**
  - The **library** owns: mechanism data, species & mixture thermodynamics, HP (and other)
    equilibrium, speed of sound, and (later) reaction-rate evaluation — all as functions of
    *thermodynamic* inputs `(composition, T or h, p)`.
  - **`cbnflow`** owns: the mapping from edge unknowns `(mdot, p, h_t)` to a thermodynamic
    point, the kinetic-energy fixed point, state recovery, choking, scalar transport, and
    the reactor/PSR *network* balances. These **consume** the library.
- **AD-4.** Separation is the enforcement mechanism for the "replaceable backend" goal:
  what was a coding discipline ("don't reference a concrete gas type") becomes a fact of
  the dependency graph.

> **Governing constraint (G-1).** `cbnflow` builds its Jacobian by **complex-step
> differentiation** ([`solver.complex_step_jacobian`](cbnflow/solver.py)), which requires
> every residual to be a **complex-analytic** function of the unknowns. Because the library
> is called *inside* that residual path, the library **shall** honor a differentiation
> contract (A.6): be callable with complex arguments, or return explicit analytic
> sensitivities. This is stated as a standalone property of the library (AD/complex-step
> friendliness), not as a `cbnflow`-specific coupling.


## 1. Objectives

- **O-1.** Simulate arbitrary gaseous mixtures of user-specified species.
- **O-2.** Provide chemical-equilibrium (HP) thermochemistry as the first reactive model.
- **O-3.** The library **shall** run **standalone** given species thermodynamic data and
  reaction-rate data, with **Cantera optional** (offline data source + validation oracle),
  never a runtime dependency.
- **O-4.** Make the thermochemistry **backend** (table vs. native kernel) replaceable
  behind one library API.
- **O-5.** Design so that **finite-rate / residence-time** chemistry is a later
  *relaxation of a limit*, not a rewrite.


## 2. Settled design decisions (from discussion)

- **D-1.** **Transport absolute total enthalpy** (including formation enthalpy) in
  `cbnflow`. HP equilibrium conserves enthalpy, so adiabatic constant-pressure reaction
  imposes **no source term** on the energy variable; the temperature rise emerges from the
  library's `T_eq(Z, h, p)`. The energy datum shifts from today's sensible `h_t = cp·T_t`.
- **D-2.** **Transport conserved scalars, slave species to equilibrium.** `cbnflow`
  transports composition as source-free conserved scalars and asks the library to
  reconstruct species. **Two descriptors are first-class, behind one composition
  abstraction:** (i) **elemental mass fractions** — general, any number of streams/fuels,
  ~4–5 scalars; and (ii) **single mixture fraction** — the 1-scalar two-stream special
  case. Library/closure dimensionality scales with the number of elements (~4–5) for (i) or
  is 1 for (ii), independent of the number of fuel species.
- **D-3.** **Two interchangeable thermochemistry backends inside the library; Backend D is
  built first** (decision 2026-06-16):
  - **Backend D — standalone native kernel** (NASA-style species polynomials +
    **element-potential / CEA-style** Gibbs minimization, branch-free so complex step
    propagates natively). The substrate that extends to finite rate. **Implemented first.**
  - **Backend A — offline table + analytic surrogate** (equilibrium states precomputed,
    optionally by Cantera; evaluated in-loop via complex-analytic interpolation). Optional
    speed/validation backend added later; validated for consistency against D.
  - *("B" — implicit-function-theorem sensitivities — is a differentiation technique used
    inside a live backend, not a separate backend. "C" — finite-differencing — is a
    debugging stopgap, excluded from the production design.)*
- **D-4.** In `cbnflow`, **generalize the single hard-wired `h_t` advection into a scalar
  registry**; `h_t` becomes the first registered scalar; the system stays square.
- **D-5.** **Reaction is localized to elements** in `cbnflow`, consistent with the existing
  "element = jump condition" architecture. Equilibrium-everywhere and a frozen→equilibrium
  reactor element are two configurations of the same machinery (§B.5).


---

# PART A — Standalone thermochemistry library (`thermolib`)

The library is self-contained and network-agnostic. Its inputs and outputs are purely
thermodynamic.

## A.1 Scope & independence
- **R-A1.1** The library **shall** have **no dependency** on `cbnflow` or any network
  concept (AD-2).
- **R-A1.2** The library **shall** function with **no Cantera import at runtime** once a
  mechanism is loaded (AD-3, O-3).
- **R-A1.3** Public types **shall** be expressed in thermodynamic terms only:
  composition (elemental or species), `T`/`h`, `p`, and derived properties.

## A.2 Mechanism & data ingestion
- **R-A2.1** The library **shall** load a mechanism comprising: element set, species list,
  per-species thermodynamic polynomial coefficients, and (for finite rate) reaction
  stoichiometry and rate parameters.
- **R-A2.2** The native mechanism file format **shall** be a **subset of Cantera's YAML**
  (elements; species with NASA-style coefficients; reactions with Arrhenius parameters), so
  it is human-readable and round-trips easily with Cantera.
- **R-A2.3** The library **should** provide an **offline** importer from a full Cantera
  mechanism file, producing the native representation of R-A2.2.

## A.3 Species & mixture thermodynamics
- **R-A3.1** The library **shall** evaluate per-species `cp(T)`, `h(T)`, `s(T)`, `g(T)`
  from the polynomial data, complex-analytically in `T`.
- **R-A3.2** Mixture properties **shall** be composable from species properties for an
  arbitrary composition vector.
- **R-A3.3** The same species-thermo evaluation **shall** serve both equilibrium (via Gibbs
  energies) and, later, finite-rate reverse rates (via `K_c(T)`), guaranteeing consistency
  (R-A5.2).
- **R-A3.4** The library **shall** provide **speed of sound**, distinguishing **frozen**
  and **equilibrium** sound speed, because the consumer's choking machinery depends on
  which is used (R-B3.1).

## A.4 Chemical equilibrium (HP and related)
- **R-A4.1** The library **shall** compute the HP-equilibrium state for a given elemental
  composition, enthalpy, and pressure, returning at least `T, rho, composition` and the
  derived properties of A.3. The equilibrium solve **shall** use the **element-potential
  (Lagrange-multiplier / CEA-style) formulation**: unknowns are element potentials plus
  species moles, with element conservation as the constraint — chosen because it maps
  directly onto the elemental composition descriptor (D-2) and yields a clean sensitivity
  block for the differentiation contract (A.6).
- **R-A4.1a (pressure handling).** Pressure **shall** be an ordinary input to the
  equilibrium and property calls (trivial for Backend D). For the **near-atmospheric,
  modest-variation** target, Backend A **may** fix a reference pressure initially and add a
  `p` axis later; Backend D needs no such restriction.
- **R-A4.2** Equilibrium **shall** be expressed so exact derivatives are available, by
  direct complex step (preferred) or by implicit-function-theorem sensitivity of the
  equilibrium conditions (A.6).
- **R-A4.3** Equilibrium **should** also be reachable in other variable pairs (e.g. TP) for
  validation and reuse, but HP is the only mandatory one for the MVP.
- **R-A4.4** The library **should** document an existence/robustness statement for the
  equilibrium solve (or the envelope where convergence is guaranteed).

## A.5 Finite-rate rate evaluation (forward compatibility — secondary target)
- **R-A5.1** The library **shall** be structured so net species production rates
  `ω̇(T, p, Y)` and their derivatives `∂ω̇/∂(Y,T)` can be added **complex-analytically**,
  so a consumer's complex-step machinery yields the stiff source Jacobian with no
  hand-coded derivatives.
- **R-A5.2** Reverse rates **shall** derive from equilibrium constants `K_c(T)` computed
  from the **same** species Gibbs energies used for equilibrium (detailed balance), so that
  the finite-rate result relaxes exactly to the equilibrium model as time → ∞.
- **R-A5.3** Rate evaluation **shall** remain pure thermochemistry: it takes `(T, p, Y)`
  and returns rates; it has no notion of residence time, reactors, or networks (those are
  the consumer's, §B.5).

## A.6 Differentiation contract
- **R-A6.1** Each backend **shall** declare one of two differentiation modes:
  1. *complex-transparent* — callable with a complex argument directly (Backend A; Backend
     D when written branch-free);
  2. *sensitivity-providing* — returns values plus an analytic Jacobian block for a live
     backend, to be spliced via the implicit-function theorem (the pattern already proven
     in [`state.solve_density`](cbnflow/state.py)).
- **R-A6.2** No in-library residual/property path **shall** use `abs`/`sign`/`max` or
  branch on a complex argument; smooth equivalents **shall** be used (mirroring
  [`cbnflow.smooth`](cbnflow/smooth.py), but the library **shall** carry its own copy — no
  `cbnflow` import).

## A.7 Backends
- **R-A7.1** The library **shall** expose a uniform thermochemistry API with a **selectable
  backend** (Backend A / Backend D), interchangeable from the consumer's perspective.
- **R-A7.2** Backend A's tabulation parametrization (e.g. `(Z, h)` at a fixed reference
  `p` initially, per R-A4.1a) **shall** be configurable; reduction choices are library
  concerns, not the consumer's.
- **R-A7.3** Backends A and D **shall** agree to a stated tolerance on a shared test set
  (R-A8.4).

## A.8 Non-functional (library)
- **R-A8.1 (standalone core).** The library **shall** follow the "standalone core solver"
  philosophy of [IDEAS.md](prototype/IDEAS.md): minimal object coupling, and **should** be
  written to stay JIT-friendly where this does not conflict with the differentiation
  contract (A.6). *(Complex-step and some JIT/AD paths are in tension; A.6 is the documented
  seam.)*
- **R-A8.2 (no silent Cantera dependence).** Absence of Cantera **shall not** disable any
  runtime feature given a loaded native mechanism; it shall only disable offline import and
  validation.
- **R-A8.3 (performance).** Backend A **shall** evaluate in `O(microseconds)` per call;
  Backend D **shall** be acceptable for prototype use, with equilibrium reusable as a warm
  start.
- **R-A8.4 (validation oracle).** A test mode **shall** compare library output to Cantera
  when available, and **shall** be skipped (not failed) when absent.

## A.9 Public API — sketch (non-normative)

Network-agnostic; thermodynamic inputs only.

```python
mech = Mechanism.from_native("fuel_air.yaml")        # or .from_cantera(...) offline
gas  = Thermo(mech, backend="kernel")                # or backend="table"

props = gas.properties(Y, T, p)        # cp, h, s, rho, a_frozen, a_equilibrium, ...
eq    = gas.equilibrate_HP(Z_elem, h, p)   # -> T, rho, Y, a_equilibrium, (sensitivities)
wdot  = gas.net_rates(Y, T, p)             # finite-rate (secondary); complex-analytic
```


---

# PART B — `cbnflow` integration (consumer of `thermolib`)

These requirements live in `cbnflow` and depend on Part A. They keep all network concepts
out of the library.

## B.1 Closure adapter
- **R-B1.1** `cbnflow` **shall** define a **closure adapter** that maps edge unknowns to a
  library call: from `(mdot, p, h_t, Z, area)` it forms the thermodynamic point and returns
  the full edge state (`rho, T, u, c, M, p_t, T_t`, entropy invariant).
- **R-B1.2** The solver **shall** access all thermochemistry **only** through this adapter;
  [`state.recover_state`](cbnflow/state.py) and the choking rows **shall not** reference a
  concrete gas/backend type.
- **R-B1.3** `PerfectGas` **shall** be refactored to satisfy the same adapter interface as a
  trivial native closure (no library call), preserving current behavior and tests
  (R-B6.1).

## B.2 State recovery & kinetic-energy coupling (theory)
- **R-B2.1** The density/state recovery **shall** be re-derived for variable composition:
  given `(p, h_t, Z, mdot, area)`, solve for `(T, rho)` through the library. An analog of
  THEORY.md's existence/uniqueness (monotonicity) argument **shall** be established, or the
  valid envelope stated.
- **R-B2.2** The fixed point between `h = h_t − u²/2`, `u = mdot/(ρA)`, and the library's
  `ρ(Z,h,p)` **shall** be solved with the same complex-safe locate-on-real / propagate-imag
  pattern used today in [`state.solve_density`](cbnflow/state.py).

## B.3 Choking consistency (theory)
- **R-B3.1** Choking complementarities and `M = u/c` **shall** use the **equilibrium sound
  speed** supplied by the library (R-A3.4). The emergent-choking behavior (THEORY.md §11)
  **shall** be re-validated for a reacting gas.

## B.4 Scalar transport framework
- **R-B4.1** The network **shall** support an ordered **registry of advected scalars**;
  `h_t` is registered first, reproducing today's behavior exactly.
- **R-B4.2** Each registered scalar **shall** add one unknown and one advection row per
  edge, preserving the square `(3 + N_s)·E` system.
- **R-B4.3** The default donor rule **shall** be the existing mass-weighted smooth-upwind
  mix ([`Element.donor_enthalpy`](cbnflow/elements.py) generalized); a passive scalar
  **shall** require no new element code.
- **R-B4.4** Elements **may** override a scalar's donor to inject boundary values (inlets)
  or reacted composition (reactor element).
- **R-B4.5** For conserved scalars the scheme **shall** preserve realizability — donor
  mixes are convex combinations — so transported mass fractions stay in `[0,1]` and sum to
  1 at convergence without clipping.
- **R-B4.6** Each scalar **shall** carry a variable/residual scale for the solver's
  nondimensionalization ([`network.variable_scales`/`residual_scales`](cbnflow/network.py)).
- **R-B4.7 (composition descriptor abstraction).** `cbnflow` **shall** provide one
  composition abstraction with two implementations: **elemental mass fractions** (general)
  and **single mixture fraction** (two-stream). The **single mixture fraction expands to an
  elemental composition** (linear interpolation between the two feed-stream elemental
  vectors) **before** any library call, so the library API only ever receives elemental (or
  species) composition — never a mixture fraction. This keeps the AD-3 boundary intact.

## B.5 Reaction models in the network
- **R-B5.1 (equilibrium-everywhere).** With an equilibrium closure, edge states are
  `f(Z, h, p)` and equilibrium holds at every edge with **no special element and no source
  term**; combustion occurs where streams of differing `Z` mix at junctions.
- **R-B5.2 (frozen→equilibrium reactor element).** A reactor element **shall** be
  expressible that takes frozen inflow and imposes equilibrium products as its jump
  condition, representing unburned (pre-ignition) regions without a transported reacting
  scalar.
- **R-B5.3 (finite-rate forward compatibility).** The design **shall** allow species to
  become independent transported scalars with a chemical source from a steady-PSR reactor
  element, conceptually
  `mdot·(Y_k,out − Y_k,in) = V·W_k·ω̇_k(T_out, p, Y_out)`, with residence time
  `τ = ρV/mdot`. The element residual is then another complex-analytic element equation,
  consuming the library's `net_rates` (R-A5.1).
- **R-B5.4** The frozen↔equilibrium transition **shall** emerge continuously from the
  Damköhler balance (`τ→0` frozen, `τ→∞` equilibrium); no 0/1 indicator is transported.
- **R-B5.5 (continuation).** A **Damköhler / chemistry-source homotopy** **should** be
  available, analogous to the vanishing-friction `stab` schedule
  ([`solver.solve`](cbnflow/solver.py)), using the equilibrium solution as the start
  iterate for stiff finite-rate solves.

## B.6 Non-functional (integration)
- **R-B6.1 (backward compatibility).** With `PerfectGas` selected, results and the `3·E`
  system **shall** be bit-for-bit unchanged; the existing test suite **shall** pass
  unmodified.
- **R-B6.2 (complex-analyticity).** All in-loop residual code **shall** remain
  complex-analytic (no `abs`/`sign`/`max`/branch-on-complex); reuse
  [`cbnflow.smooth`](cbnflow/smooth.py).
- **R-B6.3 (consistency).** The finite-rate `τ→∞` limit **shall** match the equilibrium
  model on a shared test set.


---

## 3. Suggested phasing (informative)

Library and integration can proceed largely in parallel once the API of A.9 is ratified.

**Library (`thermolib`)**
1. Mechanism ingestion + species/mixture thermo (A.2, A.3).
2. Backend D equilibrium (HP) + frozen/equilibrium sound speed (A.4, A.3.4); validate vs
   Cantera.
3. *(optional, later)* Backend A table/surrogate; prove A/D consistency (A.7.3).
4. Finite-rate `net_rates` with detailed balance (A.5).

**`cbnflow`**
1. Closure adapter; refactor `PerfectGas` behind it; tests stay green (B.1, B.6.1).
2. Scalar registry; validate with a passive tracer (B.4).
3. Wire equilibrium closure; re-derive recovery monotonicity & equilibrium choking
   (B.2, B.3); validate equilibrium-everywhere.
4. Reactor element (frozen→equilibrium) (B.5.2).
5. Finite-rate: species transport + PSR source + Damköhler continuation (B.5.3–B.5.5).


## 4. Decisions settled (2026-06-16 review)

- **Backend order:** Backend **D (native kernel) first**; Backend A optional/later (D-3).
- **Composition:** support **both** elemental mass fractions and single mixture fraction,
  with mixture fraction expanded to elemental before any library call (D-2, R-B4.7).
- **Pressure:** target is **near-atmospheric, modest variation** — `p` is an ordinary
  argument; Backend A may fix a reference `p` initially (R-A4.1a).
- **Finite rate:** **design hooks only**, build equilibrium now (O-5, A.5, B.5).
- **Equilibrium method:** **element-potential / CEA-style** (R-A4.1).
- **Mechanism format:** **Cantera-YAML subset**, Cantera importer optional (R-A2.2/2.3).
- **Packaging:** **editable install from a sibling repo** during development (AD-1).
- **Name:** keep working name **`thermolib`** for now.

## 5. Open questions / deferred

- **Q-6** Finite-rate mechanism size target (detailed vs. skeletal/reduced/global) — drives
  the species-transport count in B.5.3. Deferred with the finite-rate target.
- **Q-7** Version-pinning / release policy once the library API stabilizes and a published
  package replaces the editable install.
- **Q-8** Whether the elemental descriptor should additionally support frozen multi-species
  transport (for the finite-rate phase) under the same abstraction (R-B4.7).
```
