# Thermochemistry and reacting flow

The mean-flow framework is deliberately agnostic to the gas model: the element residuals and the choking rows see only the *recovered state* an edge carries, never the thermodynamic closure that produced it (see [state and recovery](state-and-recovery.qmd)).
This document describes the reacting closure that occupies the same interface as the perfect gas — a variable-composition, chemically reacting mixture — and shows how combustion, heat release, and the attendant temperature rise emerge from the machinery already built, with no new element for the reaction itself.

Two design decisions carry the whole construction, and both are chosen so that reaction changes the *equation of state* rather than the *balances*.
The first is to transport conserved mixture fractions and slave the species to chemical equilibrium, so that composition is carried by the same donor–upwind relations as any other scalar.
The second is to carry *absolute* total enthalpy, including the enthalpy of formation, so that an adiabatic reaction conserves the transported energy and needs no source term.
The presentation begins with the closure interface and the two entities it spans, then proceeds to the absolute-enthalpy datum and the transported mixture fractions, building on this to the equilibrium equation of state and its kinetic-energy coupling, and closing with the frozen/equilibrium/marker closures and what is deliberately deferred.

## The closure interface and its two entities

State recovery is a map from an edge's carried variables to the full thermodynamic and kinematic state that the residuals consume, and its structure is identical for every closure (see [state and recovery](state-and-recovery.qmd)).
For a reacting gas that map takes the carried variables $(\dot m,\ p,\ h_t)$ *together with* the edge's transported composition and area, and returns the density, temperature, sound speed, and derived stagnation state.
The solver reaches thermochemistry *only* through this adapter: neither the element rows nor the choking complementarity references a concrete gas model, so the perfect gas and the reacting mixture are interchangeable instances of one interface.

Concretely the work is split between two entities, and keeping the split clean is what lets each be tested on its own.
A standalone thermochemistry library owns the species data — NASA-style polynomials — and the chemical-equilibrium solve, and it is entirely network-agnostic: its inputs and outputs are purely thermodynamic (composition, enthalpy, pressure, and derived properties), with no notion of an edge or an element.
The network side is a thin adapter that forms the thermodynamic point from an edge's carried variables, calls the library, and packs the result into the edge-state table the assembly consumes.
The library's equilibrium solve uses an element-potential (CEA-style) Gibbs minimization, formulated to be branch-free so that the complex-step derivative propagates through it natively (see [complex-step](../design/complex-step.qmd)).

## Absolute enthalpy and the source-free energy balance

The reacting closure shifts the energy datum from the sensible enthalpy of the perfect-gas model to the *absolute* enthalpy, and this single choice is what removes the reaction source from the energy equation.
The transported quantity becomes the absolute total enthalpy, given as:

$$
h_t \;=\; h_{\text{abs}}(\mathbf{Y}_{\text{el}}, T) + \tfrac{1}{2}u^2,
$$

where $h_{\text{abs}}$ is the absolute specific enthalpy of the mixture — including the enthalpy of formation of its constituents — evaluated at the local composition and temperature, and $\tfrac{1}{2}u^2$ is the specific kinetic energy.
Because a constant-pressure adiabatic reaction conserves the absolute enthalpy exactly, the mixing and reaction of streams impose *no* source on the transported energy: the chemical energy released by recombination is already accounted for in the datum, and the temperature rise appears not as an added term but as the output of the equilibrium temperature $T_{\text{eq}}(\mathbf{Y}_{\text{el}}, h, p)$ at the conserved enthalpy.
Intuitively, the flame does not *add* energy to the balance; it *redistributes* enthalpy from chemical to thermal form at fixed total, and the equation of state reports the resulting temperature.
This is why the enthalpy transport of [transport](transport.qmd) needs no reaction term: the chemistry lives in the closure, not in the balance, and the per-edge transport equation is untouched.

## Transported mixture fractions

Composition is potentially high-dimensional — a detailed mechanism carries dozens of species — yet the quantity that is genuinely *conserved and convected* is low-dimensional, because chemical equilibrium fixes the species once the elemental make-up and the thermodynamic state are known.
The framework therefore transports conserved mixture fractions and reconstructs the species from equilibrium, rather than transporting the species themselves.
Two descriptors are provided behind one composition abstraction: elemental mass fractions for the general case of several streams or fuels, at roughly four to five scalars, and a single mixture fraction for the two-stream special case, at one scalar — the dimensionality scaling with the number of *elements*, not the number of species.

The transported mixture fractions are expanded to an elemental composition before any library call, given as:

$$
\mathbf{Y}_{\text{el}} \;=\; \sum_{i=1}^{K} Z_i\,\mathbf{Y}_{\text{el}}^{(i)},
$$

where $Z_i$ is the transported mixture fraction of feed stream $i$, $\mathbf{Y}_{\text{el}}^{(i)}$ is that feed stream's fixed elemental composition, and $K$ is the number of feed streams.
The expansion is linear, so the library only ever receives an elemental (or species) composition and never a network-specific mixture fraction — the separation that keeps the library network-agnostic.
An important property is that the donor mixes of [transport](transport.qmd) are convex combinations, so the transported fractions remain realizable — non-negative and summing to one — at convergence without any clipping, and each mixture fraction adds exactly one unknown and one transport row per edge, preserving the square $(3 + N_s)E$ system of [equation structure](equation-structure.md).
Combustion then requires no special element: it occurs wherever streams of differing mixture fraction meet and mix at a junction, and the equilibrium closure reports the burnt state there (tests: `test_passive_tracer_mixes_mass_weighted`, `test_problem_has_extra_scalar`, `test_branched_mixing_converges_from_seed`).

## The equilibrium equation of state and its kinetic-energy coupling

With the elemental composition in hand, the equation of state is the chemical-equilibrium solve, which returns the thermodynamic state from the conserved variables, given as the map:

$$
(\mathbf{Y}_{\text{el}},\ h,\ p) \;\longmapsto\; (T,\ \varrho,\ c_{\text{eq}},\ W),
$$

where $h$ is the static enthalpy, $T$ the equilibrium temperature, $\varrho$ the density, $c_{\text{eq}}$ the *equilibrium* sound speed, and $W$ the mixture molar mass.
The library distinguishes the equilibrium sound speed from the frozen one, and the choking machinery of [choking](choking.qmd) uses the equilibrium value, since the sonic condition of a reacting stream is set by the sound speed at which composition re-equilibrates.
The recovery has the same structure as in the perfect-gas case — a scalar consistency root enforcing kinematic–thermodynamic consistency — with only the equation of state replaced.

The reacting recovery carries the kinematic coupling exactly, which is what makes it valid at finite Mach number rather than only in the low-speed limit.
The static enthalpy left to the gas depends on the density through the velocity, and the consistency condition is a bracketed root on the static enthalpy, given as:

$$
h \;=\; h_t - \tfrac{1}{2}u^2 = h_t - \frac{m^2}{2\varrho^2},
\qquad
\varrho = \varrho_{\text{eq}}(\mathbf{Y}_{\text{el}}, h, p),
$$

where $m = \dot m/A$ is the mass flux density and $\varrho_{\text{eq}}$ is the equilibrium density at the given composition, static enthalpy, and pressure.
Solving this outer root returns the *exact* static state rather than the $\mathcal{O}(M^2)$ approximation $h \approx h_t$, and its derivatives are spliced by the implicit function theorem in exactly the manner of the perfect-gas recovery, so the complex-step Jacobian remains exact (tests: `test_ke_burnt_static_matches_oracle`, `test_ke_complex_step_matches_fd_warm_cache`, `test_ke_frozen_leg_self_consistent`).

## The frozen, equilibrium, and marker closures

Not every region of a reacting network is burnt: a pre-ignition passage carries unburnt reactants that must *not* be placed in chemical equilibrium, or the flame would be smeared across the whole domain.
Three closure flavors express this, and the network selects among them per edge.

1. **Frozen** — the unburnt reactant mixture, evaluated as a real gas of fixed composition with no reaction; the state upstream of a flame.
2. **Equilibrium** — the fully burnt state, in chemical equilibrium at the conserved $(\mathbf{Y}_{\text{el}}, h, p)$; the state downstream of a flame.
3. **Marker-gated** — a smooth blend of the two, selected by a transported *burnt marker*.

The burnt marker $b$ is itself a transported scalar, carried by the same donor–upwind relations, and it is bimodal at convergence — it settles to $b = 0$ on unburnt edges and $b = 1$ on burnt ones.
The blend is a smooth gate $g(b)$ built so that $g(0) = 0$ and $g(1) = 1$ hold to machine precision, so a frozen edge is *purely* frozen and a burnt edge *purely* equilibrium, with the gate active only in the transient while the solver discovers which edges are burnt.
An important remark is that the gate's slope $\mathrm{d}g/\mathrm{d}b$ is small at $b \in \{0, 1\}$, so the marker is nearly decoupled from the acoustics at the converged state — it selects the closure without contaminating the linearized operator.

The marker is set, not transported into existence: an equilibrium flame element writes $b = 1$ onto its genuinely downstream edge through its donor, which makes it an orientation-robust detector of "downstream of a flame" that does not depend on how the edges were wired, and whose linearization is zero, so the marker source is acoustically silent (tests: `test_auto_reacting_network_is_marker_gated`, `test_marker_self_corrects_any_seed`, `test_mean_flow_matches_hard_closure`, `test_marker_surfaced_and_species_blended`).
The two configurations of this machinery — equilibrium holding everywhere, and a frozen-to-equilibrium reactor element that imposes burnt products as its jump condition — are thus the same closure switched on different edges, not two separate mechanisms.

## What is deferred

The present closure is equilibrium-based, and finite-rate chemistry is designed for but not yet built.
The forward-compatible path is to promote the species to independent transported scalars, each with a chemical source supplied by a steady well-stirred-reactor element, so that the frozen-to-equilibrium transition emerges continuously from a Damköhler balance — frozen as the residence time tends to zero, equilibrium as it grows large — rather than from the marker gate.
The library is structured so that the net production rates and their derivatives can be added complex-analytically, and the reverse rates derive from the same species Gibbs energies used for equilibrium, so the finite-rate model relaxes exactly to the equilibrium one in the long-residence limit.
Until then, the marker gate is the current, equilibrium-based realization of the frozen/burnt distinction, and it is exact wherever the flame is thin compared with the network's elements.

With the reacting mean flow closed, the unsteady response of a flame's heat release enters the acoustic layer as a prescribed source, the subject of [dynamic sources](dynamic-sources.qmd); and the reacting closure's caloric derivatives feed the characteristic maps that turn the base Jacobian into the acoustic operator (see [perturbation network](perturbation-network.md)).
