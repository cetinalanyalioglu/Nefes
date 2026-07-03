# The network framework

The network framework is the abstraction on which everything else rests: a fluid system is represented as a directed graph, and the entire physical model is a set of relations attached to that graph.
This document fixes what the graph is, what the quantities carried on it represent, where the flow state lives, and the sign convention that lets the solver work without knowing in advance which way the gas flows.
It also states what the framework requires of a thermodynamic closure, without committing to a particular one.

The presentation begins with the graph model and the division of labour between elements and edges, then proceeds to the meaning of the edge quantities as area averages and the closure this entails, building on this to the orientation convention and the direction-independence it must satisfy, and culminating in the thermodynamic closure that the framework leaves open.

## Standing assumptions

This document inherits the standing assumptions of the [overview](overview.md); the two that are specific to the framework, and are examined in the sections that follow, are stated here:

1. **Port quantities are adequately described by their cross-sectional averages.** Each edge carries a single representative state standing for the average over its port; this is exact for a uniform profile and is corrected, where it matters, by a profile-shape factor.
2. **A thermodynamic closure supplies the state relations.** The framework requires a closure that returns the thermodynamic state, and its derivatives, from the carried variables; it is agnostic to whether that closure is a calorically perfect gas or a chemical-equilibrium mixture.

## The network as a directed graph

A *flow network* is a directed graph whose nodes are *elements* and whose edges represent the port cross-sections between them.
The two carry a strict division of labour, and it is the first design decision of the framework.

An *element* is a component — a nozzle, orifice, junction, plenum, boundary, or flame — modeled as a *control volume on which the governing conservation laws are applied*.
An element owns the *equations* those laws produce for it, but no *state* of its own.
The absence of element state is a consequence of the framework being *edge-based* rather than any statement about the element's size: the flow state is carried on the edges, and because the governing balances at an element reference only its port states, no separate volume-associated state is ever needed and none is introduced.
This is a different matter from the *zero-volume limit*, in which an element's internal volume is additionally taken to vanish so that its balances reduce to pure jump conditions between port states; that limit is a modeling choice applied to the compact elements for which it is appropriate — a thin flame, a sudden area change — and is derived, together with the source terms that survive it, in [governing equations](governing-equations.md).

An *edge* represents the planar port surface shared between two neighbouring elements, or between an element and the exterior, and it owns the *state vector* $\mathbf{x}_e$.
The state vector is the minimal set of variables from which every other flow quantity on the edge — density, velocity, temperature, sound speed, Mach number, stagnation state — can be recovered; the particular choice, and the recovery, are the subject of [state and recovery](state-and-recovery.qmd).
Intuitively, the elements are where the physics is imposed and the edges are where the gas is described, so the model reads as "components impose relations between the flow states of the connections that meet at them."
Placing the state on the edges keeps the unknown count fixed and independent of element type, a property exploited when the equations are counted (see [equation structure](equation-structure.md)).

A network definition therefore consists of three ingredients: the element types and their parameters, the edge (port) areas $A_e$, and the connectivity — which edges attach to which elements, and at which local ports.
Each element of degree $d$ numbers its incident ports $0, 1, \dots, d-1$, and the ordering is significant wherever an element treats its ports asymmetrically (for instance the reference port of a loss element or a junction), so it is preserved from the network definition through to the residual.
The connectivity is exactly a signed node–edge incidence: each edge is incident to two elements, entering one as an outgoing port and the other as an incoming port, and the sign of that incidence is the orientation factor introduced below.
The counting and assignment of equations on this incidence, and its storage as the sparsity pattern of the Jacobian, are treated in [equation structure](equation-structure.md).

## Edge quantities as section averages

The state on an edge is a single set of scalars, yet the port it represents is a two-dimensional surface over which the flow is in general non-uniform.
The reconciliation is that each edge quantity is a *cross-sectional average* over its port, and the framework's validity rests on that average being an adequate description of the port.
We write the section (area) average of a field $\phi$ over a port of area $A$ as:

$$
\langle \phi \rangle \equiv \frac{1}{A}\int_A \phi\,\mathrm{d}A,
$$

where the integral runs over the port cross-section.
The quantities that a balance actually needs are the *fluxes* through the port, and a convected flux is the section average of a product — the flux of a specific quantity $\psi$ is $\int_A m\,\psi\,\mathrm{d}A = A\,\langle m\,\psi\rangle$, with $m = \varrho u$ the mass flux density.
The lumped model, by contrast, forms this flux from the single representative edge state, as $\dot m\,\psi_e = A\,\langle m\rangle\,\psi_e$.
The two agree exactly only when the average of the product equals the product of the averages:

$$
\langle m\,\psi\rangle = \langle m\rangle\,\psi_e,
$$

which holds when the mass-flux and $\psi$ profiles are uncorrelated across the section — in particular for uniform profiles — and identifies the correct edge value $\psi_e = \langle m\,\psi\rangle/\langle m\rangle$ as the mass-flux-weighted section average.
Intuitively, the single-state closure is the statement that a port is well enough represented by one number that the transported total enthalpy of the stream is $\dot m\,h_{t}$ rather than $\langle \dot m\, h_t\rangle$; the two coincide when the profile is flat and differ when it is not.

Where the difference matters, it is absorbed by a *profile-shape factor*, defined as:

$$
\beta_\psi \equiv \frac{\langle m\,\psi\rangle}{\langle m\rangle\,\psi_e},
$$

where $\beta_\psi = 1$ for a uniform profile and departs from unity in proportion to the profile non-uniformity, in the manner of the kinetic-energy and momentum coefficients of classical hydraulics.
Supplying $\beta_\psi$ per element extends the valid regime of the single-state closure to developed or otherwise structured profiles without changing the framework, and none other than $\beta_\psi = 1$ is assumed in the present work.

A second source of section-scale fluctuation is turbulence, and the framework's stance on it is an assumption stated plainly: the turbulent fluxes are taken to cancel upon section averaging, so that the averaged balances are those of the mean profile.
The same assumption carries into the acoustic problem in a specific form.
There the fluctuation on a port is decomposed into an acoustic part and a turbulent part, and the turbulent part is assumed to vanish upon section averaging, leaving the plane-acoustic response as the quantity the perturbation network propagates (see [perturbation network](perturbation-network.md)).

## Orientation and the sign convention

Each edge $e$ is given a reference direction at build time, drawn from its *tail* element to its *head* element.
An essential point is that this arrow makes no physical claim: it does **not** assert that the gas flows from tail to head, but only fixes what "positive" means for the signed edge quantities — the mass flow rate $\dot m_e$, the velocity, and the Mach number.
It is the direct analogue of the outward surface-normal chosen for a face in a classical finite-volume method: an orientation fixed once, against which signed fluxes are measured, and whose particular choice does not affect the result.
If the converged solution returns a negative $\dot m_e$, the gas flows against the arrow, and the sign resolves itself as part of the solution rather than being prescribed.

To write an element's balance without reference to the global arrow directions, we introduce the orientation factor, defined as:

$$
\sigma_{P,e} =
\begin{cases}
+1 & \text{if } e \text{ points away from } P \text{ ($P$ is the tail)},\\
-1 & \text{if } e \text{ points towards } P \text{ ($P$ is the head)},
\end{cases}
$$

where $P$ is an element and $e$ an incident edge, so that $\sigma_{P,e}$ is precisely the sign of the signed node–edge incidence.
Then $\sigma_{P,e}\,\dot m_e$ is the mass flow *leaving* $P$ through $e$, whatever the global arrow conventions are, and it is convenient to name the two signed fluxes explicitly:

$$
\dot m^{\text{out}}_{P,e} = \sigma_{P,e}\,\dot m_e,
\qquad
\dot m^{\text{in}}_{P,e} = -\sigma_{P,e}\,\dot m_e,
$$

where $\dot m^{\text{out}}_{P,e}$ and $\dot m^{\text{in}}_{P,e}$ are the mass flows out of and into element $P$ through edge $e$.
The role of $\sigma_{P,e}$ is exactly that of the outward-normal sign in a finite-volume scheme: it fixes the sign with which each face contributes to the balance of its cell, so an edge shared by two elements enters their two balances with opposite signs, and mass is conserved across the shared face by construction.

A core requirement follows, and it is a requirement the framework must *satisfy*, not merely assume: the choice of edge arrows has no influence on the physical solution.
Reversing an edge's direction negates its $\dot m_e$ and flips every $\sigma_{P,e}$ that references it, and the two negations must cancel throughout, so that the recovered pressures, temperatures, and flow magnitudes are unchanged.
This *direction-flip invariance* is verified numerically rather than taken on faith (test: `test_edge_direction_invariance`).

## The thermodynamic closure

The framework is agnostic to the gas thermodynamics: it requires only that a *thermodynamic closure* be supplied, one that returns the thermodynamic state — and, for the exact Jacobian, its derivatives — from the carried variables and the composition.
Two closures are provided, and the element residuals see only the recovered state either produces, never the closure that produced it (see [state and recovery](state-and-recovery.qmd)).

The first, and the simplest, is the *calorically perfect gas*, for which the state relations are given as:

$$
p = \varrho R T,
\qquad
h = c_p T,
\qquad
c^2 = \gamma R T,
\qquad
\gamma = \frac{c_p}{c_v},
$$

where $p$ is the static pressure, $\varrho$ the density, $T$ the static temperature, $h$ the static specific enthalpy, $c$ the speed of sound, $R$ the specific gas constant, $c_p$ and $c_v$ the specific heats at constant pressure and volume, and $\gamma$ their ratio.
It is a good approximation for air and for combustion products over moderate temperature ranges, and it makes the framework concrete.
Throughout the documentation we use the caloric constant:

$$
\Gamma \equiv \frac{c_p}{R} = \frac{\gamma}{\gamma - 1}
\qquad (\approx 3.5 \text{ for air}),
$$

where $\Gamma$ groups the combination of specific heats that recurs in the state recovery and the isentropic relations, and it is the natural constant in which the total-enthalpy and stagnation relations are most compactly written.

The second is a *chemical-equilibrium mixture*, in which the specific heats and molar mass become temperature- and composition-dependent and are obtained from an equilibrium solve, and each reacting edge additionally carries a transported composition.
It enters the framework through the same interface — a recovered state and its derivatives — and its construction, together with the transported mixture fractions and the frozen/burnt closures, is deferred to [thermochemistry](thermochemistry.md).
The symbols used here and elsewhere are collected in the [nomenclature](../nomenclature.md).
