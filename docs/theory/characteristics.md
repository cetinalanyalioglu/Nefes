# Characteristic variables

A disturbance in a flowing gas travels in three distinct ways: as a pressure (sound) wave running downstream, as a pressure wave running upstream, and as a temperature or composition pattern simply convected along with the gas.
The *characteristic variables* $f$, $g$, and $h$ are the amplitudes that book-keep these three carriers, and they are the natural language of duct acoustics and thermoacoustics.
This document derives the exact, invertible map between the wave amplitudes and the network's own unknowns, and then explores the question of whether solving the *mean flow* in wave variables (rather than in the network variables introduced in [state-and-recovery](state-and-recovery.qmd)) improves convergence.
Or, in a more general sense, this document shows whether solving the mean flow using any other linearly dependent variable set can affect the convergence behaviour.

The presentation begins with the decomposition of the one-dimensional Euler system and the definition of the amplitudes, then proceeds to the exact per-edge map to the network variables and the determinant that guarantees its invertibility, building on the "Newton-invarance", and closing with the three roles the characteristics actually play.

## Decomposition of the one-dimensional Euler system

Written in the primitive variables $\mathbf{q} = (\varrho, u, p)$, the one-dimensional Euler equations take the quasi-linear form $\partial_t \mathbf{q} + \mathbf{A}_q\,\partial_x \mathbf{q} = \mathbf{0}$, with the flux Jacobian given as:

$$
\mathbf{A}_q =
\begin{bmatrix}
u & \varrho & 0\\
0 & u & 1/\varrho\\
0 & \varrho c^2 & u
\end{bmatrix},
\qquad
\text{eigenvalues}\quad \lambda \in \{\,u + c,\; u - c,\; u\,\},
$$

where $u$ is the velocity, $\varrho$ the density, $c$ the sound speed, and the three eigenvalues are the propagation speeds of the three carriers — a downstream acoustic wave at $u + c$, an upstream acoustic wave at $u - c$, and a convected entropy pattern at $u$.
Diagonalizing the system in the corresponding eigenbasis defines the characteristic perturbation amplitudes, which in this project's convention are given as:

$$
\begin{aligned}
f &= \tfrac{1}{2}\Big(u' + \tfrac{p'}{\varrho c}\Big) &&\text{(downstream acoustic, } \lambda = u + c),\\[2pt]
g &= \tfrac{1}{2}\Big(-u' + \tfrac{p'}{\varrho c}\Big) &&\text{(upstream acoustic, } \lambda = u - c),\\[2pt]
h &= \varrho' - \tfrac{p'}{c^2} &&\text{(entropy / convected, } \lambda = u),
\end{aligned}
$$

where a prime denotes a fluctuation about the mean state, so that $u'$, $p'$, and $\varrho'$ are the velocity, pressure, and density fluctuations.
The relations invert to $u' = f - g$, $p' = \varrho c\,(f + g)$, and $\varrho' = h + p'/c^2$, which express the primitive fluctuations in terms of the wave amplitudes and are the form used to build the network map below (test: `test_characteristic_amplitude_relations`).
Intuitively, $f$ and $g$ carry the pressure–velocity content of the two sound waves, while $h$ isolates the density change that is *not* explained by the pressure change — the entropy spot that the flow merely transports.

## Exact maps to the network variables

The perturbation the acoustic problem actually manipulates is that of the edge state, $\widehat{\mathbf{x}}_e = (\widehat{\dot m}_e,\ \widehat{p}_e,\ \widehat{h}_{t,e})$, the complex amplitude of a time-harmonic fluctuation of the solver's own unknowns (see [nomenclature](../nomenclature.md)).
Relating it to the wave amplitudes requires only the definitions $\dot m = \varrho u A$ and $h_t = \Gamma p/\varrho + \tfrac{1}{2}u^2$, differentiated at the mean state, given as:

$$
\widehat{\dot m} = A\,(u\,\widehat{\varrho} + \varrho\,\widehat{u}),
\qquad
\widehat{h}_t = -\frac{\Gamma p}{\varrho^2}\,\widehat{\varrho} + u\,\widehat{u} + \frac{\Gamma}{\varrho}\,\widehat{p},
$$

where $A$ is the port area and $\Gamma = c_p/R$ the caloric constant, and the pressure amplitude passes through unchanged, $\widehat{p} = \widehat{p}$.
Composing these differentials with the inverse relations of the previous section gives the per-edge map $\widehat{\mathbf{x}}_e = \mathbf{T}_e\,\mathbf{w}$, with $\mathbf{w} = (f, g, h)^{\top}$ and the block $\mathbf{T}_e$ having the closed form:

| per unit amplitude of | $f$ | $g$ | $h$ |
|---|---|---|---|
| $\widehat{\dot m}$ | $A\varrho\,(M + 1)$ | $A\varrho\,(M - 1)$ | $A u$ |
| $\widehat{p}$ | $\varrho c$ | $\varrho c$ | $0$ |
| $\widehat{h}_t$ | $c + u$ | $c - u$ | $-h/\varrho$ |

where $M = u/c$ is the signed Mach number and $h = c_p T$ the static enthalpy that appears in the entropy column.
This form is not an approximation: it is the exact linearization of the state definitions, confirmed to machine precision against the shipped kernel (test: `test_characteristic_maps_are_inverse`).

The map is invertible at every recoverable state, and one determinant settles the point, given as:

$$
\det \mathbf{T}_e \;=\; -\,2\,A\,\varrho\,c\,\big(u^2 + h\big) \;<\; 0
\qquad\text{whenever}\quad p, T > 0,
$$

so the inverse $\mathbf{L}_e = \mathbf{T}_e^{-1}$, giving $\mathbf{w} = \mathbf{L}_e\,\widehat{\mathbf{x}}_e$, exists at $M = 0$, in reversed flow, and in supersonic flow alike — the change to and from wave amplitudes never breaks down on a physical state.
An important remark is that the entropy (third) row of the map carries the gas's *caloric coupling* through $\Gamma$, and for a reacting or variable-$\gamma$ edge this row is taken instead from a complex step of the converged closure, so that the wave maps remain consistent with the mean-flow Jacobian rather than assuming a perfect gas (see [perturbation network](perturbation-network.md)).

## The Newton-invariance 

With an invertible map in hand, one can ask whether posing the *mean-flow* Newton correction in wave amplitudes would change the iteration.
The claim is that it does not: solving the Newton system in characteristic amplitudes and mapping back yields the identical update as solving directly in the network variables.

The proof is immediate.
Let $\overline{\mathbf{J}}$ be the Jacobian of the residuals at the current iterate and $\mathbf{T}$ the invertible, block-diagonal (per-edge) map $\Delta\mathbf{x} = \mathbf{T}\,\mathbf{w}$ between a Newton increment $\Delta\mathbf{x}$ in network variables and its representation $\mathbf{w}$ in wave amplitudes.
Posed in wave amplitudes the Newton system is $(\overline{\mathbf{J}}\,\mathbf{T})\,\mathbf{w} = -\mathbf{R}$, and mapping the solution back yields the increment, given as:

$$
\Delta\mathbf{x} = \mathbf{T}\,(\overline{\mathbf{J}}\,\mathbf{T})^{-1}(-\mathbf{R})
= \mathbf{T}\,\mathbf{T}^{-1}\,\overline{\mathbf{J}}^{-1}(-\mathbf{R})
= -\,\overline{\mathbf{J}}^{-1}\mathbf{R},
$$

which is exactly the update obtained in the network variables.
Intuitively, a change of unknowns is a similarity transformation of the linear system, and Newton's method is invariant under it.
It should be noted that casting the mean-flow correction in characteristic variables therefore **cannot, by itself, improve convergence**.
What a change of variables *can* affect is the floating-point conditioning of the linear solve and the metric of any relaxation, both second-order effects; the genuine levers on robustness are the equation structure of [equation structure](equation-structure.md) and [transport](transport.qmd), the scaling, and the globalization of the solver (see [the solver](../design/solver.md)).

## What the characteristics are actually for

The wave language earns its place not in the mean-flow solve but in three other roles, each of which the map above makes exact.

1. **Counting and well-posedness.** The characteristic decomposition is what tells one how many boundary and jump conditions a well-posed problem needs, and on which side each belongs — an inflow carries conditions its outflow does not; the fixed per-edge split of [equation structure](equation-structure.md) is the smooth, direction-independent realization of that count.
2. **Diagnostics.** Any residual or Newton update can be read per edge as three wave amplitudes through $\mathbf{L}_e$, which makes the state of the iteration physically legible; at the converged mean state the residual-driven wave amplitudes vanish identically, the natural statement that no unbalanced waves remain.
3. **The acoustic network.** Linearizing the very same element equations at the converged mean state and transforming with $\mathbf{T}$ turns the base Jacobian $\overline{\mathbf{J}}$ into the acoustic scattering relations between the wave amplitudes — the reuse that is the consistency goal of the whole framework, and the subject of the [perturbation network](perturbation-network.md).

The third role is the through-line of the method: the map $\mathbf{T}$ derived here is precisely the bridge by which the operator assembled for the mean flow becomes, without re-derivation, the operator of the acoustics.
Before turning to that acoustic layer, one mean-flow phenomenon remains to be treated — the saturation of mass flow when a passage chokes, which the next document shows emerges from the element rows already written (see [choking](choking.qmd)).
