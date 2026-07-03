# Nomenclature

This page is the single source of notation for the Nefes documentation.
Every other document links here rather than redefining a symbol, so that a symbol carries one meaning throughout and the sign, orientation, and frequency conventions are stated once.
Where a symbol is genuinely reused in two roles, the two are listed together and the disambiguating context is named.

## Conventions

The notation follows a small set of rules, applied uniformly:

- **Mean (base) states** carry an overbar, $\overline{X}$, and denote the converged steady state about which the acoustics are linearized (for example the mean density $\overline{\varrho}$ and sound speed $\overline{c}$). A subscript zero is *not* used for the mean, as numeric subscripts denote port indices.
- **Fluctuations** carry a prime, $X' = X - \overline{X}$, and are the small unsteady departures from the mean.
- **Complex amplitudes** carry a hat, $\widehat{X}$, and are the frequency-domain amplitude of a time-harmonic fluctuation, $X'(t) = \Re\{\widehat{X}\,e^{\mathrm{i}\omega t}\}$.
- **Section (area) averages** carry angle brackets, $\langle X\rangle \equiv \tfrac{1}{A}\int_A X\,\mathrm{d}A$, and denote the average of a field over a port cross-section; the edge quantities are averages of this kind (see [framework](theory/framework.md)).
- **Vectors and matrices** are set in bold, $\mathbf{X}$; scalars are set in plain italic, $X$.
- **Frequency** is reported and prescribed as the ordinary frequency $f$ in hertz; the angular frequency $\omega = 2\pi f$ appears only inside derivations, and the crossing between the two is stated where it occurs.
- **Edge orientation.** Every edge carries an arbitrary reference arrow fixed at build time; it defines the positive sense of all signed edge quantities and makes no claim about the flow direction, which the solver discovers (a negative $\dot m_e$ is flow against the arrow).
- **The imaginary unit** is $\mathrm{i}$, reserved for it; a complex-step derivative uses the same unit with a real step $h_{\text{cs}}$ (see [complex-step](design/complex-step.qmd)).

## Roman symbols

| symbol | meaning |
|---|---|
| $\varrho,\ u,\ p,\ T$ | density, signed normal velocity, static pressure, static temperature |
| $h,\ h_t$ | static / total specific enthalpy, $h_t = h + \tfrac{1}{2}u^2$ |
| $p_t,\ T_t$ | total (stagnation) pressure / temperature |
| $c,\ M$ | speed of sound, signed Mach number $M = u/c$ |
| $s$ | entropy, in the invariant form $p/\varrho^{\gamma}$ |
| $\dot m,\ m$ | mass flow rate (signed along the edge arrow), mass flux density $m = \dot m / A$ |
| $A_e$ | edge (port) cross-sectional area |
| $R,\ c_p,\ c_v$ | specific gas constant, specific heats at constant pressure / volume |
| $H$ | static enthalpy implied by a trial density, $H = h_t - m^2 /(2\varrho^2)$ |
| $\dot Q$ | lumped heat-release rate of a flame element |
| $Z_i$ | conserved mixture fraction of feed stream $i$ (a transported scalar) |
| $\mathbf{Y}_{\text{el}}$ | elemental mass-fraction vector, expanded from the mixture fractions ($\mathbf Y_{\text{el}} = \sum_i Z_i\,\mathbf Y_{\text{el}}^{(i)}$) before any equilibrium call |
| $b$ | burnt-marker scalar (transported, gates the reacting closure); $g(b)$ its smooth gate |
| $\mathbf{x},\ \mathbf{R},\ \mathbf{J}$ | unknown vector, residual vector, Jacobian $\mathbf J = \partial\mathbf R/\partial\mathbf x$ |
| $\mathbf{x}_e$ | edge state vector, $\mathbf x_e = (\dot m_e,\ p_e,\ h_{t,e})$ (plus any transported scalars) |
| $E$ | number of edges in the network |
| $f,\ g,\ h$ | characteristic wave amplitudes (downstream-acoustic, upstream-acoustic, entropy) |
| $\mathbf{w}$ | characteristic amplitude vector, $\mathbf w = (f,\ g,\ h)^{\top}$ |
| $L$ | duct length; $L_{\text{eff}}$ its acoustic effective length (with end corrections) |
| $q$ | dynamic head of a stream, $q = \tfrac{1}{2}\varrho u^2$ (a signed, smoothed form is used in loss residuals) |
| $K_L$ | loss coefficient of a concentrated loss element, referenced to a dynamic head |
| $C_c$ | vena-contracta contraction coefficient of a sudden contraction ($C_c = 1$ is loss-free) |
| $f_0$ | resonant frequency [Hz] |
| $\mathbf{A}(\omega)$ | perturbation system matrix, $\mathbf A = \overline{\mathbf J} + \mathrm{i}\omega\mathbf M + \mathbf P(\omega) + \mathbf S(\omega)$ |
| $\overline{\mathbf{J}}$ | converged mean-flow (base) Jacobian; the algebraic block and the zero-frequency operator, $\mathbf A(0) = \overline{\mathbf J}$ |
| $\mathbf{M}$ | storage block: the finite-volume time-derivative terms (compliance and inertance), entering through $\mathrm{i}\omega$ |
| $\mathbf{P}(\omega)$ | propagation block: the lossless-duct phase relations $e^{-\mathrm{i}\omega\tau}$ |
| $\mathbf{S}(\omega)$ | source block: the prescribed unsteady feedback (a flame's heat-release response) |
| $\widehat{\mathbf{b}}$ | forcing amplitude of the perturbation system (zero for the stability problem) |
| $\mathbf{L}_e$ | per-edge change of basis from solution variables to characteristics, $\mathbf w = \mathbf L_e\,\widehat{\mathbf x}_e$ |
| $\mathbf{T}(f)$ | 2-port transfer matrix between two stations, $\mathbf v_{\text{down}} = \mathbf T\,\mathbf v_{\text{up}}$ |
| $\boldsymbol{\mathcal{S}}(f)$ | 2-port scattering matrix, mapping incoming to outgoing waves, $\mathbf w_{\text{out}} = \boldsymbol{\mathcal S}\,\mathbf w_{\text{in}}$ |
| $\mathbf{A}_0(\omega)$ | known (passive) perturbation operator, with the unknown element set to its reference |
| $n$ | interaction index of the $n$–$\tau$ flame response |
| $R$ | acoustic reflection coefficient at a termination, $R = \widehat g/\widehat f$ (context distinguishes it from the gas constant) |
| $Z$ | acoustic impedance at a termination (context distinguishes it from a mixture fraction) |
| $\operatorname{cond}(\cdot)$ | per-frequency condition number of the identification system (the identifiability diagnostic) |

## Greek symbols

| symbol | meaning |
|---|---|
| $\gamma$ | ratio of specific heats, $\gamma = c_p/c_v$ |
| $\Gamma$ | caloric constant, $\Gamma = c_p/R = \gamma/(\gamma-1)$ ($\approx 3.5$ for air) |
| $\sigma_{P,e}$ | orientation factor of edge $e$ at element $P$: $+1$ if $P$ is the tail, $-1$ if the head |
| $\beta_\psi$ | profile-shape factor of a convected scalar $\psi$ ($\beta_\psi = 1$ for a uniform profile) |
| $\varepsilon$ | smoothing width of a regularized primitive (mass-flow units) |
| $\theta,\ w,\ \xi$ | smooth upwind / donor / boundary-regime blending weights |
| $\kappa$ | vanishing-friction (stabilization) coefficient |
| $\varphi_\varepsilon$ | smoothed complementarity residual (Fischer–Burmeister), selecting the subsonic and choked regimes of a single row |
| $\lambda$ | Levenberg–Marquardt damping parameter |
| $\tau_+,\ \tau_-,\ \tau_u$ | duct transit times of the downstream-acoustic, upstream-acoustic, and convected paths |
| $\tau$ | flame time lag of the $n$–$\tau$ response |
| $\mathcal{F}(f)$ | dynamic-source transfer function; for the $n$–$\tau$ flame, $\mathcal F = n\,e^{-\mathrm{i}\omega\tau}$ |
| $\omega$ | angular frequency, $\omega = 2\pi f$ (derivations only); a mode's complex frequency is $\omega = \omega_r + \mathrm{i}\omega_i$ |
| $\sigma$ | modal growth rate, $\sigma = -\omega_i = -\Im(\omega)$; a mode is unstable when $\sigma > 0$ (convention $e^{+\mathrm{i}\omega t}$) |

An important remark on two reused letters: the operator **source block** $\mathbf{S}(\omega)$ and the frequency-domain **scattering matrix** $\boldsymbol{\mathcal{S}}(f)$ are distinct objects and are typeset differently for that reason; likewise $R$ and $Z$ denote a reflection coefficient and an impedance in the acoustic context and the gas constant and a mixture fraction in the mean-flow context, and no document uses a given letter in both roles at once.
A related remark on numeric subscripts: they denote **port indices** ($p_0$, $p_{t,1}$ for ports $0$ and $1$), never the mean state, which is why the mean/base state is written with an overbar ($\overline{\varrho}$, $\overline{c}$) rather than a subscript zero — the two would otherwise collide on the element residuals.

## Decorations, sub- and superscripts

| notation | meaning |
|---|---|
| $\overline{X}$ | converged mean / base value of $X$ |
| $X'$ | fluctuation of $X$ about the mean |
| $\widehat{X}$ | complex (frequency-domain) amplitude of $X'$ |
| $\langle X\rangle$ | section (area) average of $X$ over a port |
| $\dot X$ | a rate (per unit time) |
| $X_t$ | a total (stagnation) quantity |
| $X_e,\ X_P$ | a quantity carried on edge $e$ / owned by element $P$ |
| $\dot m^{\text{out}}_{P,e},\ \dot m^{\text{in}}_{P,e}$ | mass flow leaving / entering element $P$ through edge $e$, $\dot m^{\text{out}}_{P,e} = \sigma_{P,e}\dot m_e$ |

## Terms

| term | meaning |
|---|---|
| **element** | a network component (graph node): a control volume on which the governing equations are applied; it owns equations, not state (state lives on the edges) |
| **edge** | a port cross-section shared by two elements (or an element and the exterior); owns the state vector |
| **residual** | how far an equation is from being satisfied at the current guess; all zeros means solved |
| **Jacobian** | the matrix of sensitivities $\partial R_i/\partial x_j$ |
| **jump condition** | an algebraic relation between the states on the two sides of a compact element (one taken in the zero-volume limit) |
| **transport (edge) equation** | the donor/upwind relation that carries total enthalpy (and any scalar) along an edge |
| **choking** | mass-flow saturation when the narrowest section reaches $M = 1$ |
| **characteristic variables** | the wave amplitudes $(f, g, h)$ that diagonalize the linearized 1-D Euler system |
| **storage** | the finite-volume compliance and inertance restored to an element under an unsteady perturbation |
| **stamp** | a local contribution an element writes into the assembled (mean-flow or perturbation) operator |
| **transfer matrix** | a frequency-domain 2-port relating the flow variables at two stations along their arrows |
| **scattering matrix** | a frequency-domain 2-port relating the incoming waves at two stations to the outgoing ones |
| **flame transfer function (FTF)** | the linear frequency response of a flame's heat release to a reference fluctuation |
| **identification (de-embedding)** | recovering an unknown element's dynamic response from a measured network response, given a model of the rest |
| **complex-step derivative** | an exact derivative from a single imaginary-perturbed evaluation, free of subtractive cancellation |
| **well-posed** | having exactly as many independent conditions as unknowns — solvable and unambiguous |
