# Governing equations and network discretization

This document derives the equations a network solves, starting from the integral conservation laws of gas dynamics and reducing them, under the standing assumptions, to the algebraic jump conditions attached to each element.
It settles a distinction that is easy to conflate: *steadiness* is what removes the time-derivative term and leaves an algebraic balance, whereas the *zero-volume limit* is a separate modeling choice whose effect is only to discard the volumetric terms while the lumped ones survive.
The algebraic balances derived here are more than the mean-flow model: differentiated at the converged operating point, they *are* the content of the acoustic network, so that the same operator serves both problems — the reuse that is the through-line of the whole method (see [perturbation network](perturbation-network.md)).

The presentation begins with the integral laws and the flux vector, then proceeds to the steady balance between an element's ports, building on this to the zero-volume limit and the fate of source terms under it, and culminating in the single-state closure that models each port flux.

## Standing assumptions

The equations below are posed under the [overview](overview.md) assumptions, of which three are load-bearing here:

1. **Inviscid dynamics at the component scale.** The gas obeys the Euler equations over a component; viscous and turbulent losses enter only through lumped constitutive terms, not a resolved shear field.
2. **A steady operating point is sought.** The transient term vanishes because $\partial/\partial t \equiv 0$ at the operating point, independently of any element's internal volume.
3. **Port quantities are section averages.** Each port flux is formed from the single representative edge state, exact for a uniform profile and corrected otherwise by a profile-shape factor (see [framework](framework.md)).

## Integral conservation laws

We take the gas to be inviscid on the scale of a component, so its dynamics are governed by the Euler equations, written in integral form over a control volume $\mathcal{V}$ with bounding surface $\mathcal{S}$:

$$
\frac{\partial}{\partial t}\int_{\mathcal{V}} \widetilde{\mathbf{q}}\,\mathrm{d}\mathcal{V}
\;+\; \oint_{\mathcal{S}} \mathbf{f}_n \,\mathrm{d}\mathcal{S}
\;=\; \dot{\boldsymbol{\Omega}},
$$

where $\widetilde{\mathbf{q}}$ is the vector of conserved densities, $\mathbf{f}_n$ their flux through the boundary along the outward normal, and $\dot{\boldsymbol{\Omega}}$ collects any sources acting within $\mathcal{V}$.
It can be interpreted as the statement that the rate of change of a conserved quantity stored in $\mathcal{V}$, plus its net efflux through the boundary, equals its rate of production inside.
For mass, momentum, and energy the densities and fluxes are given as:

$$
\widetilde{\mathbf{q}} =
\begin{bmatrix} \varrho \\ \varrho u \\ \varrho e_t \end{bmatrix},
\qquad
\mathbf{f}_n =
\begin{bmatrix} \varrho u_n \\ \varrho u\, u_n + p\, n \\ \varrho u_n h_t \end{bmatrix},
$$

where $\varrho$ is the density, $u$ the velocity, $p$ the static pressure, $e_t$ the total energy per unit mass, $u_n$ the velocity component along the outward normal, $n$ the corresponding component of the surface normal, and $h_t = h + \tfrac{1}{2}u^2$ the *total enthalpy* — the sum of the static enthalpy $h = c_p T$ and the specific kinetic energy.
Intuitively, $h_t$ is the energy currency of a steady stream: it is exactly conserved along an adiabatic flow no matter how the gas accelerates, decelerates, or loses pressure, which is precisely why it, and not the temperature, is the quantity transported across the network (see [transport](transport.qmd)).

The surface integral over a port is a section average of the flux, in the sense of the [framework](framework.md): the boundary term $\oint_{\mathcal{S}} \mathbf{f}_n\,\mathrm{d}\mathcal{S}$ evaluated over a port of area $A$ is $A\,\langle \mathbf{f}_n\rangle$, and the single-state closure of the final section is exactly the statement that this average is adequately represented by the flux formed from the edge state.
Under that closure the turbulent contributions to the port flux are taken to cancel on averaging, so the balances below are those of the section-averaged profile.

## The steady balance and jump conditions

We seek a steady operating point, so the time-derivative term vanishes and the integral law reduces to a balance of surface fluxes against the source.
This reduction requires only steadiness: at the operating point the stored mass, momentum, and energy do not change in time, and the transient term is zero for an element of *any* internal volume.
For an element $P$ whose boundary consists of its port surfaces (the incident edges $e_i$) together with solid walls, the balance is given as:

$$
\sum_{e_i} \mathbf{F}_i \;=\; \dot{\boldsymbol{\Omega}}_P,
\qquad
\mathbf{F}_i \;=\; \sigma_{P,e_i}\,\mathbf{f}_i\, A_i,
$$

where $\mathbf{f}_i$ is the flux evaluated at the representative state of port $i$, $A_i$ is the port area, $\sigma_{P,e_i} \in \{+1,-1\}$ is the orientation factor (see [framework](framework.md)) that aligns each edge's arbitrary arrow with the outward normal of $P$, and $\dot{\boldsymbol{\Omega}}_P$ is the *net source* of $P$ — the integral of the volumetric source density $\dot{\boldsymbol{\omega}}$ over the element volume:

$$
\dot{\boldsymbol{\Omega}}_P \;=\; \int_{\mathcal{V}_P} \dot{\boldsymbol{\omega}}\,\mathrm{d}\mathcal{V}.
$$

The orientation factor plays the role of the outward-normal sign of a finite-volume scheme, so an edge shared by two elements enters their balances with opposite signs and the shared face conserves each quantity by construction.

No flux crosses the solid walls, so they contribute to none of the balances through convection.
They enter one balance only — the momentum balance — through the pressure force they exert on the gas, and it is precisely this wall-pressure term that *distinguishes* one element from another.
The mass and energy balances take the same universal form for every interior element, while the momentum/pressure relation encodes the geometry: it is what separates a nozzle from an orifice from a dump diffuser, each exerting a different wall force.
The catalogue of these relations is the subject of [elements](elements.md).

## The zero-volume limit and the fate of sources

The second simplification is the *zero-volume limit*, a modeling convenience applied to an element rather than a necessity — the steady balance above already holds for a finite volume.
Its effect is to discard whatever scales with the internal volume; the question worth settling is what the net source $\dot{\boldsymbol{\Omega}}_P = \int_{\mathcal{V}_P}\dot{\boldsymbol{\omega}}\,\mathrm{d}\mathcal{V}$ leaves behind as $\mathcal{V}_P \to 0$.
The answer depends on how the source density behaves in the limit, and it is cleanest to split it into a *regular* part, which stays bounded, and a *concentrated* part, whose magnitude grows as the volume shrinks so that its integral does not:

$$
\dot{\boldsymbol{\Omega}}_P
= \underbrace{\int_{\mathcal{V}_P}\dot{\boldsymbol{\omega}}_{\text{reg}}\,\mathrm{d}\mathcal{V}}_{\;=\,\mathcal{O}(\mathcal{V}_P)\,\to\,0\;}
\;+\;
\underbrace{\int_{\mathcal{V}_P}\dot{\boldsymbol{\omega}}_{\text{conc}}\,\mathrm{d}\mathcal{V}}_{\;\to\,\dot{\boldsymbol{\Omega}}_P^{\,\text{lumped}}\;},
$$

where $\dot{\boldsymbol{\omega}}_{\text{reg}}$ is a bounded density and $\dot{\boldsymbol{\omega}}_{\text{conc}}$ is a density that concentrates onto the vanishing element.
A bounded density integrates to a quantity of order $\mathcal{V}_P$, which vanishes with the volume, exactly as the transient term did.
A concentrated density is, in the limit, a Dirac source — formally $\dot{\boldsymbol{\omega}}_{\text{conc}} \to \dot{\boldsymbol{\Omega}}_P^{\,\text{lumped}}\,\delta(\mathbf{x} - \mathbf{x}_P)$ — whose integral is the finite rate $\dot{\boldsymbol{\Omega}}_P^{\,\text{lumped}}$ the element imposes regardless of its size.
The net source that survives the limit is therefore exactly this lumped rate, and it is the term that appears in the jump condition:

$$
\dot{\boldsymbol{\Omega}}_P^{\,\text{lumped}} \;=\; \lim_{\mathcal{V}_P \to 0}\int_{\mathcal{V}_P}\dot{\boldsymbol{\omega}}\,\mathrm{d}\mathcal{V}.
$$

Consequently the framework fully supports in-element sources, provided they are lumped: a heat-release element retains its full power $\dot{Q}_P$ in the energy balance, a fan or pump adds a finite shaft work or momentum source, and a reactor adds a finite species production rate.
A compact flame is thus a legitimate jump condition carrying a finite source, and mechanically such a source enters through the element's donor term without altering the per-edge transport equations or the equation count (see [transport](transport.qmd)).

A less obvious but equally important point is that the volume an element discards is not dynamically idle.
At steady state its accumulation rate is zero, but its *capacity* to store mass and energy reappears the moment the flow is unsteady: the transient term, restored under a harmonic perturbation, becomes the element's acoustic *storage* (see [perturbation network](perturbation-network.md)).
Taking an element to the zero-volume limit therefore has a consequence beyond dropping volumetric sources — it removes that element's storage contribution to the acoustic problem, which is why the limit is applied selectively: a component whose compliance or inertia shapes the acoustic response is kept at finite volume, while one whose internal volume is dynamically negligible, such as a thin flame, is idealized to zero volume.

## Edge-state closure

The steady balance references the flux $\mathbf{f}_i$ at each port, which is a section average over that port surface (see [framework](framework.md)).
We model this average by evaluating the flux formulas at the single representative state carried on the edge — the same closure a finite-volume method makes at a coarser scale, in which one cell-face state stands for the average over the face.
For a convected quantity $\psi$ this replaces the true port flux $A\,\langle m\,\psi\rangle$ by $\dot m\,\psi_e$, and the leading correction is the profile-shape factor:

$$
A\,\langle m\,\psi\rangle \;=\; \beta_\psi\,\dot m\,\psi_e,
\qquad
\beta_\psi = \frac{\langle m\,\psi\rangle}{\langle m\rangle\,\psi_e},
$$

where $\beta_\psi = 1$ for a uniform profile.
The present work takes $\beta_\psi = 1$ throughout, so the port flux is a function of the edge's own state alone; the element balances then couple only to the states of their incident edges, and the network's sparsity mirrors its connectivity.
A less obvious benefit of writing the correction explicitly is that it names the route to a wider validity: supplying a measured or modeled $\beta_\psi$ per element — a developed-profile kinetic-energy coefficient, say — restores the exact section flux without touching the equation structure, so the single-state closure can be extended to non-uniform profiles as an element-level refinement rather than a change of framework.

With the steady balances written and their port fluxes closed, what remains is to choose the variables that carry each edge's state — the choice on which the solver's robustness turns, and the subject of [state and recovery](state-and-recovery.qmd).
