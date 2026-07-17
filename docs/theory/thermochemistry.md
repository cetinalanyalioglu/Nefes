# Thermochemistry and reacting flow

The mean-flow framework is deliberately independent of the gas model: the element residuals and the choking rows see only the *recovered state* an edge carries, never the thermodynamic closure that produced it (see [state and recovery](state-and-recovery.qmd)).
This document describes the reacting closure that occupies the same interface as the perfect gas — a variable-composition, chemically reacting mixture — and shows how combustion, heat release, and the attendant temperature rise emerge from the machinery already built, with no new element for the reaction itself.

Two design decisions carry the whole construction, and both are chosen so that reaction changes the *equation of state* rather than the *balances*.
The first is to transport conserved mixture fractions and constrain the species to chemical equilibrium, so that composition is carried by the same donor–upwind relations as any other scalar.
The second is to carry *absolute* total enthalpy, including the enthalpy of formation, so that an adiabatic reaction conserves the transported energy and needs no source term.

## The closure interface and its two entities {#sec-thermo-closure-interface}

State recovery is a map from an edge's carried variables to the full thermodynamic and kinematic state that the residuals consume, and its structure is identical for every closure (see [state and recovery](state-and-recovery.qmd#sec-state-recovery-closure)).
For a reacting gas that map takes the carried variables $(\dot m,\ p,\ h_t)$ *together with* the edge's transported composition and area, and returns the density, temperature, sound speed, and derived stagnation state.
The solver reaches thermochemistry *only* through this adapter: neither the element rows nor the choking complementarity references a concrete gas model, so the perfect gas and the reacting mixture are interchangeable instances of one interface.

Concretely the work is split between two roles, and keeping the split clean is what lets each be tested on its own.
The thermochemistry proper owns the species data — NASA-style polynomials [@mcbride_2002] — and the chemical-equilibrium solve, and it is entirely independent of the network: its inputs and outputs are purely thermodynamic (composition, enthalpy, pressure, and derived properties), with no notion of an edge or an element.
The network side is a thin adapter that forms the thermodynamic point from an edge's carried variables, calls that engine, and packs the result into the edge-state table the assembly consumes.
The equilibrium solve uses an element-potential (CEA-style) Gibbs minimization [@gordon_mcbride_1994], formulated to be branch-free so that the complex-step derivative propagates through it natively (see [complex-step](../design/complex-step.qmd#sec-cs-analyticity)).

## Absolute enthalpy and the source-free energy balance {#sec-thermo-absolute-enthalpy}

The reacting closure shifts the energy datum from the sensible enthalpy of the perfect-gas model to the *absolute* enthalpy, and this single choice is what removes the reaction source from the energy equation.
The transported quantity becomes the absolute total enthalpy, given as:

$$
h_t \;=\; h_{\text{abs}}(\mathbf{Y}_{\text{el}}, T) + \tfrac{1}{2}u^2,
$$

where $h_{\text{abs}}$ is the absolute specific enthalpy of the mixture — including the enthalpy of formation of its constituents — evaluated at the local composition and temperature, and $\tfrac{1}{2}u^2$ is the specific kinetic energy.
Because a constant-pressure adiabatic reaction conserves the absolute enthalpy exactly, the mixing and reaction of streams impose *no* source on the transported energy: the chemical energy released by recombination is already accounted for in the datum, and the temperature rise appears not as an added term but as the output of the equilibrium temperature $T_{\text{eq}}(\mathbf{Y}_{\text{el}}, h, p)$ at the conserved enthalpy.
Intuitively, the flame does not *add* energy to the balance; it *redistributes* enthalpy from chemical to thermal form at fixed total, and the equation of state reports the resulting temperature.
This is why the enthalpy transport of [transport](transport.qmd) needs no reaction term: the chemistry lives in the closure, not in the balance, and the per-edge transport equation is untouched.

## Transported mixture fractions {#sec-thermo-mixture-fractions}

Composition is potentially high-dimensional, could refer to hundreds of species, yet the quantity that is genuinely *conserved and convected* is low-dimensional, because chemical equilibrium fixes the species once the **elemental make-up** and the thermodynamic state are known.
The framework therefore transports conserved mixture fractions and reconstructs the species from equilibrium, rather than transporting the species themselves.
Two descriptors are provided behind one composition abstraction: elemental mass fractions for the general case of several streams or fuels, at roughly four to five scalars, and a single mixture fraction for the two-stream special case, at one scalar — the dimensionality scaling with the number of *elements*, not the number of species.

The transported mixture fractions are expanded to an elemental composition before any species-set call, given as:

$$
\mathbf{Y}_{\text{el}} \;=\; \sum_{i=1}^{K} Z_i\,\mathbf{Y}_{\text{el}}^{(i)},
$$

where $Z_i$ is the transported mixture fraction of feed stream $i$, $\mathbf{Y}_{\text{el}}^{(i)}$ is that feed stream's fixed elemental composition, and $K$ is the number of feed streams.
The expansion is linear, so the species set only ever receives an elemental (or species) composition and never a network-specific mixture fraction — the separation that keeps the species set independent of the network.
An important property is that the donor mixes of [transport](transport.qmd#sec-transport-donor-enthalpy) are convex combinations, so the transported fractions remain realizable — non-negative and summing to one — at convergence without any clipping, and each mixture fraction adds exactly one unknown and one transport row per edge, preserving the square $(3 + N_s)E$ system of [equation structure](equation-structure.md#sec-eqstruct-transported-scalars).
Combustion then requires no special element: it occurs wherever streams of differing mixture fraction meet and mix at a junction, and the equilibrium closure reports the burnt state there (tests: `test_passive_tracer_mixes_mass_weighted`, `test_problem_has_extra_scalar`, `test_branched_mixing_converges_from_seed`).

### Auto-discovered and declared streams

Which streams the network transports is set in one of two modes.
By default (**auto**) the streams *are* the distinct injected compositions of the inlets, mass sources, and backflow-bearing outlets, auto-merged so that injecting the same composition in several places costs a single scalar; a feed names a species mixture and the stream set is discovered at build time.
Alternatively the streams may be **declared** up front, `equilibrium(streams={...}, mode="declared")`, which fixes a named, closed basis; each feed then states its composition as a blend over those streams (`composition={stream_label: amount}`) rather than a raw species mixture.

The declared mode decouples *which composition degrees of freedom exist* from *how many feeds there are*.
A single **premixed** inlet, given as a blend of two declared streams (a fuel and an oxidizer), keeps those streams separate, so its mixture fraction — the equivalence ratio — is a live degree of freedom that can fluctuate, even though there is one inlet and no in-line injector.
In auto mode the same premix would collapse to a single stream pinned to $Z = 1$, carrying no composition degree of freedom; declaring the two streams is what makes the equivalence ratio drivable.
The mean species composition is identical either way: keeping the streams separate is bookkeeping that changes only the transported degrees of freedom, not the physics (tests: `test_species_equal_the_single_composition_premix`, `test_premixed_inlet_carries_two_live_streams`).

### The product species set

The streams fix the transported *elements*; a separate choice fixes the *species* the equilibrium closure resolves those elements into — the product species set.
By default the species set is derived automatically, and the user names no species at all.
The feed compositions fix the reachable elements, and the candidate product slate is every gas-phase species the packaged NASA Glenn / CEA data can build from those elements.
That slate is broad (a hydrocarbon–air pool admits on the order of a hundred species), while only a few tens are non-trace at equilibrium, so when the candidate count is large the slate is reduced to the species that carry a non-negligible mole fraction across the lean-to-rich feed-mixing range, sampled at temperatures bracketing the burnt-gas state.
The reduction is a modelling economy, not a physical assumption: the discarded species are those whose equilibrium abundance is negligible over the operating envelope the feeds span.
Its aggressiveness is adjustable through five settings on `equilibrium()`: the reducer itself (`reducer`, with `"none"` keeping every candidate), the trace mole-fraction cutoff below which a species is dropped (`reduce_threshold`), the candidate count above which the reduction runs at all (`reduce_above`), a ceiling on the kept count (`max_species`), and species to keep regardless of abundance (`must_species`).
Left at their defaults these give a slate that is physically sufficient for a typical hydrocarbon–air flame; tightening the cutoff or lowering the gate trims it further, and disabling the reducer keeps the full candidate set.
The cutoff and the ceiling measure importance differently, and the difference matters when the question is *how large a set a case needs*: the cutoff ranks a species by its own peak mole fraction against a fixed bar, while the ceiling ranks all candidates against each other and keeps the highest few.
Sweeping `max_species` upward, with the cutoff loosened so it does not bind first, answers that question directly, where sweeping the cutoff answers it only indirectly.
The ceiling is a ceiling, not a target: it discards the lowest-ranked non-trace species but never pads the slate up to the ceiling with trace ones, and two keeps always survive it (and count against its budget) — the `must_species` and one carrier of every fed-in element, so the element-potential solve never loses a constituent it must balance.
The mole fractions the cutoff and the ranking read are always those of the *full* candidate equilibrium, not of any truncated set; once a set is truncated the surviving species re-equilibrate, so at an aggressive ceiling the kept set may not reproduce the full-slate state, which is exactly the effect a size sweep is meant to expose (the reduction warns when a ceiling discards species that clear the cutoff).

The derivation is deferred to the moment the network is built, because it needs the feeds, which are a property of the assembled network rather than of the gas model.
An explicit species set overrides this: passing `equilibrium(species_set=...)` — a subset of the packaged data, a custom `thermo.inp`, or a Cantera-subset mechanism — pins the species set and disables the automatic rewrite.
Because the automatic reduction policy can evolve, the explicit form is the one to use when a fixed, reproducible species set matters; the automatic form is the default for a concise model whose slate need only be physically sufficient.
Either way the resolved species set is inspectable once the network is built, through `net.gas.species_names` and the `reduction_report` recording which candidates were kept.
The same policy resolves the slate whether a case is loaded from file or a network is built in Python, so the two paths never diverge (tests: `test_deferred_library_reproduces_an_explicit_slate`, `test_python_build_matches_the_shared_policy`).

## The equilibrium equation of state and its kinetic-energy coupling {#sec-thermo-equilibrium-eos}

With the elemental composition in hand, the equation of state is the chemical-equilibrium solve, which returns the thermodynamic state from the conserved variables, given as the map:

$$
(\mathbf{Y}_{\text{el}},\ h,\ p) \;\longmapsto\; (T,\ \varrho,\ c_{\text{eq}},\ W),
$$

where $h$ is the static enthalpy, $T$ the equilibrium temperature, $\varrho$ the density, $c_{\text{eq}}$ the *equilibrium* sound speed, and $W$ the mixture molar mass.
The species set distinguishes the equilibrium sound speed from the frozen one, and the choking machinery of [choking](choking.qmd) uses the equilibrium value, since the sonic condition of a reacting stream is set by the sound speed at which composition re-equilibrates.
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

## Condensed-phase products {#sec-thermo-condensed-phase}

Rich combustion carries the equilibrium beyond the gas phase.
When the carbon-to-oxygen ratio of the burnt mixture exceeds unity there is not enough oxygen to bind every carbon atom as carbon monoxide, and the excess carbon condenses as solid soot; a purely gaseous species set cannot represent that state, and the element-potential system for it is singular.
The equilibrium solve therefore admits condensed species as an integral part of the same minimization.
A condensed species enters as a **pure phase at unit activity**, so its chemical potential is its standard Gibbs energy alone, with no mole-fraction or pressure term:

$$
\frac{\mu_c}{R T} \;=\; \frac{g_c(T)}{R T},
$$

where $g_c$ is the standard molar Gibbs energy of the condensed species and the absence of a $\ln(n_c/n)$ term is what distinguishes a pure phase from a gaseous one.
Its mole number is a direct unknown of the reduced Newton system, appended as one row and column carrying the element couplings $a_{ic}$ (and, in the enthalpy-constrained solve, the enthalpy coupling that ties it to the temperature correction), so a condensed phase adds to the element balance and to the energy balance without ever entering the gas total-mole constraint.

Which condensed phases are present is itself part of the solution, decided on the real state so the complex-step derivative is untouched.
A phase is admitted when forming it would lower the Gibbs energy, given as the condition that its potential undercuts the element-potential combination:

$$
\frac{g_c}{R T} \;<\; \sum_{i} a_{ic}\,\pi_i,
$$

where $\pi_i$ is the potential of element $i$ and $a_{ic}$ the number of atoms of element $i$ in the condensed species; a phase whose mole number is driven to zero during the iteration is dropped again.
At convergence the present phases satisfy the equality (their potential exactly matches the element combination) and the absent ones the strict inequality, which are the optimality conditions of the constrained minimization.
Because the phase set is a property of the converged real state, the implicit-function sensitivity is formed at fixed set and remains exact away from a phase-onset boundary, exactly as the active-element reduction already is.

Which species may condense is read from the data: a species carrying the CEA condensed-phase flag qualifies as a product only if its polynomial extends to combustion temperatures, so graphite (valid to several thousand kelvin) is a soot product while a liquid fuel, whose polynomial stops near its boiling point, remains a feed that sets elements and the enthalpy datum but never appears as a product.
The equilibrium sound speed re-equilibrates the gas at the acoustic frequency and holds the condensed phase frozen, since a solid does not re-form on the wave time scale; the condensed mass still loads the density through the reduced gas mole count.

## The frozen, equilibrium, and marker closures {#sec-thermo-marker-closures}

Not every region of a reacting network is burnt: a pre-ignition passage carries unburnt reactants that must *not* be placed in chemical equilibrium, or the flame would be smeared across the whole domain.
Three closure flavors express this, and the network selects among them per edge.

1. **Frozen** — the unburnt reactant mixture, evaluated as a real gas of fixed composition with no reaction; the state upstream of a flame.
2. **Equilibrium** — the fully burnt state, in chemical equilibrium at the conserved $(\mathbf{Y}_{\text{el}}, h, p)$; the state downstream of a flame.
3. **Marker-gated** — a smooth blend of the two, selected by a transported *burnt marker*.

The burnt marker $b$ is itself a transported scalar, but it answers a *reachability* question — is this edge downstream of a flame? — rather than conserving a physical quantity, so it is not mass-averaged like the mixture fractions.
Instead it rides a sticky noisy-OR of the node's ports:

$$
b = 1 - \prod_i \bigl(1 - \theta_i\, b_i\bigr),
$$

where the product runs over the ports of the node, $b_i$ is the marker on port $i$, and $\theta_i \in [0, 1]$ is the smooth upwind indicator the scalar transport already carries (unity on an inflow, zero on an outflow).
Any burnt inflow ($\theta_i b_i \to 1$) saturates $b$ to one, while an all-fresh node returns $b = 0$ exactly (no numerical creep off zero).
This is the behavior a mass-average cannot provide: where burnt products mix with a fresh oxidizer — the quench of a rich-quench-lean combustor, or exhaust-gas recirculation — an average would dilute the marker below the gate and wrongly freeze a zone that must re-equilibrate, whereas the OR keeps it burnt so the added air completes combustion.
It should be made clear here that the converged result never is a combination of fresh and burnt states, as the **the marker is bimodal at convergence**: it settles to $b = 0$ on unburnt edges and $b = 1$ on burnt ones.
The blending is required to ensure the analytical continuity that the complex-step differentiation requires.

The blend itself is a smooth gate $g(b)$ built so that $g(0) = 0$ and $g(1) = 1$ hold to machine precision, so a frozen edge is *purely* frozen and a burnt edge *purely* equilibrium, with the gate active only in the transient while the solver discovers which edges are burnt.
Because it is a blend, *both* legs are evaluated on *every* marker-gated edge and then weighted, so the equilibrium leg runs even where its weight $g(b)$ is zero — on an unburnt, possibly cold edge (a preheated-air feed near $400\,\mathrm{K}$, say).
The equilibrium solve must therefore stay finite there too: over a rich product slate most species have a vanishing abundance at such temperatures, and their moles underflow to zero, so the element-potential kernel floors the mole fraction inside $\ln(n_j/n_\text{tot})$ to keep the $n_j \ln(n_j/n_\text{tot})$ terms finite (an unfloored $0 \cdot \ln 0$ would seed the reduced Newton system with NaNs and abort the recovery; test: `test_kernel_cold_near_inert_mixture_stays_finite`).
An important remark is that the gate's slope $\mathrm{d}g/\mathrm{d}b$ is small at $b \in \{0, 1\}$, so the marker is nearly decoupled from the acoustics at the converged state — it selects the closure without contaminating the linearized operator, and this holds regardless of the (nonlinear, flow-dependent) transport law that sets $b$, since $b$ reaches the physical state only through the flat gate.

**The marker is set by an equilibrium flame:** it writes $b = 1$ onto its genuinely downstream edge through its donor, which makes it an orientation-robust detector of "downstream of a flame" that does not depend on how the edges were wired, and whose linearization is zero, so the marker source is acoustically silent (tests: `test_auto_reacting_network_is_marker_gated`, `test_marker_self_corrects_any_seed`, `test_mean_flow_matches_hard_closure`, `test_marker_surfaced_and_species_blended`).
The two configurations of this machinery — equilibrium holding everywhere, and a frozen-to-equilibrium reactor element that imposes burnt products as its jump condition — are thus the same closure switched on different edges, not two separate mechanisms.

## What is deferred {#sec-thermo-deferred}

The present closure is equilibrium-based, and finite-rate chemistry is designed for but not yet built.
The forward-compatible path is to promote the species to independent transported scalars, each with a chemical source supplied by a steady well-stirred-reactor element, so that the frozen-to-equilibrium transition emerges continuously from a Damköhler balance — frozen as the residence time tends to zero, equilibrium as it grows large — rather than from the marker gate.
The species set is structured so that the net production rates and their derivatives can be added complex-analytically, and the reverse rates derive from the same species Gibbs energies used for equilibrium, so the finite-rate model relaxes exactly to the equilibrium one in the long-residence limit.
Until then, the marker gate is the current, equilibrium-based realization of the frozen/burnt distinction, and it is exact wherever the flame is thin compared with the network's elements.

By design, the marker gate method requires equilibrium evaluations at every edge of the network, whereas upon convergence the edges that truly require the equilibrium values are the ones that are downstream of the flame elements or burnt inlets.
This introduced some computational overhead, and for large networks with a large percentage of unburnt states, the introduced overhead can consume a quite significant portion of the total solve time.
While this certainly not introduce a roadblock for practical networks (i.e. less than 500 edges), it may still become annoying.
A planned remedy for this is to introduce **tabulated equilibrium chemistry**.
This was deferred, because naively implementing this breaks down the capability to introduce arbitrary streams of different species into the network, which the present equilibrium approach offers.
However, an equivalent implementation that relies on table lookup instead of local equilibrium computations is the logical next step to improve computational efficiency.

The unsteady response of a flame's heat release enters the acoustic layer as a prescribed source (see [dynamic sources](dynamic-sources.qmd)), and the reacting closure's caloric derivatives feed the characteristic maps that turn the base Jacobian into the acoustic operator (see [perturbation network](perturbation-network.md#sec-perturb-one-operator)).
