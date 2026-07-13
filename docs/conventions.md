# Conventions

This page collects, in one place, the conventions the project follows, so that they are visible before any derivation reads on them.
The symbols themselves are catalogued in [nomenclature](nomenclature.md); this page states the *choices* behind those symbols — the sign, the ordering, the orientation — several of which are arbitrary but are fixed once and held throughout.
Where a convention is developed at length elsewhere, the relevant document is linked rather than repeated.

## Time dependence and the sign of growth

A time-harmonic fluctuation is reconstructed from its complex amplitude with the convention

$$
X'(t) = \Re\{\widehat{X}\,e^{\mathrm{i}\omega t}\},
$$

that is, the $e^{+\mathrm{i}\omega t}$ sign, not $e^{-\mathrm{i}\omega t}$.
This choice is load-bearing, because it fixes the sign that reads growth from a complex mode frequency $\omega = \omega_r + \mathrm{i}\omega_i$: with $e^{+\mathrm{i}\omega t}$ the amplitude carries a factor $e^{-\omega_i t}$, so a mode *grows* when $\Im(\omega) < 0$.
The modal growth rate is therefore $\sigma = -\omega_i = -\Im(\omega)$, and a mode is unstable when $\sigma > 0$ (see [analyses](theory/analyses.qmd)).
The opposite convention $e^{-\mathrm{i}\omega t}$, common in parts of the acoustics literature, flips the sign of every imaginary part; a result stated under it is converted to ours by complex-conjugating each amplitude and reflection coefficient.

## Frequency over angular frequency

Frequency is reported and prescribed as the ordinary frequency $f$ in hertz, and this is the quantity that appears on every user input and graph axis.
The angular frequency $\omega = 2\pi f$ is used only inside derivations, and the crossing between the two is stated where it occurs.

## The imaginary unit and complex-step differentiation

The symbol $\mathrm{i}$ is reserved for the imaginary unit.
The same unit carries the complex-step derivative, in which one unknown is perturbed by a small imaginary step $x \leftarrow x + \mathrm{i}\,h_{\text{cs}}$ and the derivative is read from the imaginary part of a single evaluation, free of subtractive cancellation (see [complex-step](design/complex-step.qmd)).
For this to hold, all residual math is kept smooth and complex-analytic: no `abs`, `min`, `max`, or branch on the flow state, so that the Jacobian the acoustics need is the exact complex-step linearization of the same residuals the mean flow solves (see [smoothness contract](design/smoothness-contract.md)).

## Mean state, fluctuation, and amplitude

The decomposition of a field into its steady and unsteady parts uses three decorations, applied uniformly:

- **Mean (base) state** carries an overbar, $\overline{X}$, the converged steady state about which the acoustics are linearized (for example $\overline{\varrho}$, $\overline{c}$).
- **Fluctuation** carries a prime, $X' = X - \overline{X}$, the small organized departure the acoustic layer resolves; a stochastic turbulent departure, where it must be named alongside, carries a double prime $X''$ and is closed by the constitutive models rather than resolved.
- **Complex amplitude** carries a hat, $\widehat{X}$, the frequency-domain amplitude of a time-harmonic fluctuation as defined above.
- **Section (area) average** carries angle brackets, $\langle X\rangle$, the average of a field over a port cross-section.

Density is written $\varrho$, never $\rho$.
Numeric subscripts denote **port indices** ($p_0$, $p_{t,1}$), never the mean state; this is why the mean is an overbar and not a subscript zero, which would collide with the port index on an element's residual rows.

## Edge orientation and signs

Every edge carries an arbitrary reference arrow, fixed when the network is built.
The arrow defines the positive sense of all signed edge quantities and makes no claim about the flow direction, which the solver discovers: a negative mass flow $\dot m_e$ is simply flow running against the arrow.
An element reads its edges through an orientation factor $\sigma_{P,e}$, $+1$ when the element is the arrow's tail and $-1$ when it is the head.

## Characteristic wave amplitudes

The per-edge acoustic coordinates are the three characteristic amplitudes that diagonalize the linearized one-dimensional flow, ordered downstream-acoustic, upstream-acoustic, then entropy/convected:

$$
f = \tfrac{1}{2}\!\left(u' + p'/\varrho c\right)\ (\lambda = u + c),
\qquad
g = \tfrac{1}{2}\!\left(-u' + p'/\varrho c\right)\ (\lambda = u - c),
\qquad
h = \varrho' - p'/c^{2}\ (\lambda = u),
$$

with $\lambda$ the wave speed of each (see [characteristics](theory/characteristics.md)).
This order — $f$ first, $g$ second, $h$ third — is the order every wave vector, transfer matrix, and scattering matrix inherits below.

## Transfer-matrix convention

A transfer matrix relates the flow variables at an upstream station $a$ to those at a downstream station $b$ along their arrows, $\widehat{\mathbf{v}}_b = \mathbf{T}(f)\,\widehat{\mathbf{v}}_a$.
Its default variables are the characteristic amplitudes in their canonical order,

$$
\begin{bmatrix} \widehat{f} \\ \widehat{g} \\ \widehat{h} \end{bmatrix}_{b}
= \mathbf{T}(f)\,
\begin{bmatrix} \widehat{f} \\ \widehat{g} \\ \widehat{h} \end{bmatrix}_{a},
$$

and under the isentropic reduction the entropy row and column are dropped, leaving a two-port on $(\widehat{f}, \widehat{g})$ alone.
The other flavors keep this same ordering after their rescaling, so the primitive flavor reads $(\,\widehat{p}/\varrho c,\ \widehat{u},\ \widehat{\varrho}\,c/\varrho\,)$ with the pressure component first and the velocity second.
A transfer matrix is a frequency-domain constraint between the two stations and is not a causal input–output law (see [perturbation network](theory/perturbation-network.md)).

## Scattering-matrix convention

A scattering matrix relates the incoming waves at the two stations to the outgoing ones, $\widehat{\mathbf{w}}_{\text{out}} = \boldsymbol{\mathcal{S}}(f)\,\widehat{\mathbf{w}}_{\text{in}}$.
The waves are ordered by station and, within a station, by travel direction: the incoming waves are $a$'s downstream-running waves followed by $b$'s upstream-running waves, and the outgoing waves are $a$'s upstream-running waves followed by $b$'s downstream-running ones.
For the reduced $(\widehat{f}, \widehat{g})$ acoustics this reads

$$
\begin{bmatrix} \widehat{g}_a \\ \widehat{f}_b \end{bmatrix}
=
\begin{bmatrix} r_{+} & t_{-} \\ t_{+} & r_{-} \end{bmatrix}
\begin{bmatrix} \widehat{f}_a \\ \widehat{g}_b \end{bmatrix},
$$

so reflection at $a$ occupies the first row and transmission to $b$ the second, and the subscript on each coefficient records the travel direction of the *incoming* wave it acts on:
$r_{+}$ reflects the downstream-incoming wave $\widehat{f}_a$ back into $\widehat{g}_a$ and $t_{+}$ transmits it to $\widehat{f}_b$, while $r_{-}$ reflects the upstream-incoming wave $\widehat{g}_b$ back into $\widehat{f}_b$ and $t_{-}$ transmits it to $\widehat{g}_a$.
The classic duct-acoustics ordering, which places the transmitted wave $\widehat{f}_b$ in the first row and takes incoming $(\widehat{f}_a, \widehat{g}_b)$ to outgoing $(\widehat{f}_b, \widehat{g}_a)$, is kept only for round-trips with external two-port data and is not the default.

## Reflection coefficient and impedance

At a termination the acoustic reflection coefficient is the ratio of the returning to the incident wave, $R = \widehat{g}/\widehat{f}$, and the acoustic impedance $Z$ is the pressure-to-velocity ratio in the same convention.
The letters $R$ and $Z$ are reused for the specific gas constant and a mixture fraction in the mean-flow context; no document uses either letter in both roles at once (see [nomenclature](nomenclature.md)).

## Scope

The present model is subsonic, flowing or quiescent, up to a sonic throat.
Choking to $M = 1$ at a narrowest section is in scope and emerges from the element rows; what is deferred is supersonic flow inside the domain and the shock structures that accompany it (see [limitations](theory/limitations.md)).
The acoustics are linear and time-harmonic, and the disturbance on each edge is a plane wave, which assumes the frequencies of interest lie below the cut-on frequency of the ducts in the network.
