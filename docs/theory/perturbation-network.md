# The perturbation network

The consistency goal that motivates the entire framework is realized in this document: **the operator assembled for the mean flow, differentiated at the converged operating point, *is* the acoustic network.**
One operator serves both problems — the steady balance and the linear acoustics — so that the care taken to keep every mean-flow residual smooth and complex-analytic pays off a second time, as the exact linearization the acoustics require.
This document assembles that operator block by block and shows, for each block, whether it is *inherited* from the base Jacobian or *stamped* onto it.

The acoustic model is posed under three standing assumptions, stated once and relied on throughout.
The perturbations are *linear*, small departures about the converged mean state, so the operator is the Jacobian of the residuals at that state and nothing higher order.
The fields are *time-harmonic*, with the convention $X'(t) = \Re\{\widehat{X}\,e^{\mathrm{i}\omega t}\}$; this sign choice is load-bearing for stability, because a mode then grows when $\Im(\omega) < 0$ (see [analyses](analyses.qmd)).
The disturbances on each edge are *plane waves*, the one-dimensional acoustics of a duct [@dokumaci_2021], so that the wave amplitudes of [characteristics](characteristics.md) are the natural per-edge coordinates.
This implicitly assumes that the frequencies of interest are below the cut-on frequency of the ducts in the network.
Nefes offers tools to report the network wide cut-on frequencies to verify this.

The presentation begins with the single assembled operator and its reduction at zero frequency, then proceeds through the four blocks in turn — storage, propagation, source, and boundary closure — and closes with the ledger of what is inherited versus overwritten and the transfer- and scattering-matrix views that every downstream analysis consumes.

## The one operator

The base of the acoustic operator is the mean-flow Jacobian itself.
At the converged operating point the solver holds $\overline{\mathbf{J}}$, the exact complex-step Newton Jacobian of the residuals — the *algebraic block*, which the implementation names `J_alg` — and the per-edge characteristic maps of [characteristics](characteristics.md) turn its rows into the zero-frequency acoustic jump conditions without any re-derivation.
The full perturbation operator $\mathbf{A}(\omega)$ adds to this base the unsteady physics that a steady solve discards, and is given as:

$$
\mathbf{A}(\omega) \;=\; \overline{\mathbf{J}} \;+\; \mathrm{i}\omega\,\mathbf{M} \;+\; \mathbf{P}(\omega) \;+\; \mathbf{S}(\omega),
$$

where $\overline{\mathbf{J}}$ is the base (algebraic) Jacobian, $\mathbf{M}$ is the storage block of finite-volume compliance and inertance entering through $\mathrm{i}\omega$, $\mathbf{P}(\omega)$ is the propagation block of lossless-duct phase relations, and $\mathbf{S}(\omega)$ is the source block of prescribed unsteady feedback where thermoacoustics live. A boundary reflection closure, written onto the boundary rows, completes the assembly.
Each block is either *added* onto the rows $\overline{\mathbf{J}}$ already populates or *overwrites* a set of rows, and the sections below take them in turn.

The reduction at zero frequency is what makes the reuse exact.
As $\omega \to 0$ the storage term $\mathrm{i}\omega\mathbf{M}$ vanishes and every duct phase $e^{-\mathrm{i}\omega\tau} \to 1$, so the propagation rows collapse to plain wave-amplitude continuity and the operator returns the mean-flow Jacobian, given as:

$$
\mathbf{A}(0) \;=\; \overline{\mathbf{J}}
\qquad\text{(passive network)},
$$

which the assembly satisfies to the last bit (test: `test_zero_frequency_operator_equals_jacobian`).
An important qualification is that this identity holds for the *passive* network: an active flame contributes a nonzero direct-current gain $\mathbf{S}(0) \neq 0$ — the flame still responds to a steady perturbation — so with a source present $\mathbf{A}(0) = \overline{\mathbf{J}} + \mathbf{S}(0)$, and it is only the passive blocks that reduce to the bare Jacobian at $\omega = 0$.

## The storage block

A steady balance discards the time-derivative term of the conservation laws; the storage block restores exactly what that discarded, namely the compliance and inertance of an element's finite volume under an unsteady perturbation.
It enters the operator through $\mathrm{i}\omega$, so it is absent at zero frequency and grows linearly with it.

A lumped cavity is the archetype.
To the mean flow it is a wall — no mass crosses its face — but to the acoustics its volume stores mass compressibly, and its stamp is a compliance on the cavity's mass row, given as:

$$
\widehat{\dot m}' \;=\; -\mathrm{i}\omega\,C\,\overline{\varrho}\,\widehat{p}',
\qquad
C \;=\; \frac{V}{\overline{\varrho}\,\overline{c}^{\,2}},
$$

where $V$ is the cavity volume, $\overline{\varrho}$ and $\overline{c}$ the mean density and sound speed, and $C$ the acoustic compliance.
A length-bearing inline element contributes two kinds of stamp — a compliance $\ell A/\overline{c}^{\,2}$ per port, distributed over its up- and downstream lengths, and a series *inertance* $L_{\text{eff}}/A_{\text{ref}}$ on its pressure-drop row, referenced to the throat area — while a manifold chamber contributes the compliance $V/\overline{c}^{\,2}$ of its common volume.
The effective length $L_{\text{eff}} = \ell_{\text{up}} + \ell_{\text{down}} + \delta_{\text{end}}$ carries any end correction, so that the geometric length sets the propagation phase (below) while the end correction lives here, in the inertance (tests: `test_cavity_storage_is_the_compliance`, `test_inline_compliance_and_inertance_entries`, `test_side_branch_helmholtz_resonance_frequency`).

## The propagation block

A duct is transparent to the mean flow — it merely carries the total pressure continuously from one end to the other — but it is *not* compact to the acoustics: a wave takes a finite time to traverse it, and that transit imprints a phase.
The propagation block therefore *overwrites* the duct's continuity rows with the phase relations of its three characteristic waves.
The transit times of the downstream-acoustic, upstream-acoustic, and convected paths are given as:

$$
\tau_+ = \frac{L}{\overline{u} + \overline{c}},
\qquad
\tau_- = \frac{L}{\overline{c} - \overline{u}},
\qquad
\tau_u = \frac{L}{\overline{u}},
$$

where $L$ is the duct length and $\overline{u}$ the mean velocity, and the corresponding phase relations, diagonal in the wave amplitudes, are given as:

$$
\widehat{f}_1 = e^{-\mathrm{i}\omega\tau_+}\,\widehat{f}_0,
\qquad
\widehat{g}_0 = e^{-\mathrm{i}\omega\tau_-}\,\widehat{g}_1,
\qquad
\widehat{h}_1 = e^{-\mathrm{i}\omega\tau_u}\,\widehat{h}_0,
$$

where subscripts $0$ and $1$ denote the upstream and downstream faces and $\widehat{f}, \widehat{g}, \widehat{h}$ the downstream-acoustic, upstream-acoustic, and entropy amplitudes.
It can be interpreted directly: the downstream wave arrives at the far face delayed by its transit time and phase-shifted accordingly, and likewise for the upstream wave running back and the entropy spot convected along.
This is where the wave language of [characteristics](characteristics.md) earns its place — the propagation is diagonal in $(f, g, h)$ and would be dense in any other basis — and any transported composition scalar convects with the entropy wave at $\tau_u$.
At a quiescent mean state the entropy path is stationary and its phase decouples, and the block reduces to continuity at $\omega = 0$, consistent with the transparency of a duct to the mean flow (tests: `test_meanflow_duct_tau_plus_phase`, `test_duct_scattering_is_lossless_phase`, `test_duct_entropy_phase_and_decoupling`).

## The source block

The source block carries the one piece of physics that is genuinely *prescribed* rather than inherited: an element's unsteady feedback, of which a flame's heat-release response is the archetype.
Unlike the propagation and boundary blocks, it *adds* onto the rows the base Jacobian already populates rather than overwriting them, so it modifies — but does not replace — the element's mean-flow relation.
For a flame it lands on the downstream edge's total-enthalpy row with a factor $-\delta$, where $\delta = \overline{Q}/\dot m$ is the mean specific enthalpy rise, scaling the flame transfer function $\mathcal{F}(\omega)$ that maps a reference fluctuation to the heat-release response.
Because the transfer function not necessarily zero at $\omega = 0$, so is the source block's direct-current value, which is the qualification noted above; the full form of $\mathcal{F}$, its $n$–$\tau$ and tabulated instances, and the reference it responds to are the subject of [dynamic sources](dynamic-sources.qmd).

## The boundary closure

A terminal edge has only one neighbour, so it is closed not by a jump condition but by a *reflection* relation between its outgoing and incoming waves, which overwrites the terminal row.
In the acoustic amplitudes this reflection is given as:

$$
\widehat{g} \;=\; R(\omega)\,\widehat{f} \;+\; R_s\,\widehat{h},
$$

where $R$ is the reflection coefficient, $\widehat{f}$ and $\widehat{g}$ the incoming and outgoing acoustic amplitudes, and $R_s$ an entropy-to-acoustic coupling present at some terminations.
The coefficient depends on the termination and, through the mean Mach number $M$, on the mean flow; the standard cases are collected as follows.

1. **Hard wall**: $R = +1$ (the velocity fluctuation vanishes).
2. **Open end**: $R = -1$ (the pressure fluctuation vanishes), or $R = -(1 - M)/(1 + M)$ with a mean flow.
3. **Anechoic**: $R = 0$ (no reflection).
4. **Impedance $Z$**: $R = (Z - \overline{\varrho}\,\overline{c})/(Z + \overline{\varrho}\,\overline{c})$.
5. **Choked-nozzle outlet**: the compact Marble–Candel reflection [@marble_candel_1977], $R = (2 - (\gamma-1)M)/(2 + (\gamma-1)M)$.

The choked-nozzle termination is the one that also carries an entropy coupling, given as $R_s = (\overline{c}/\overline{\varrho})\,M/(2 + (\gamma-1)M)$, which converts an arriving entropy spot into a reflected acoustic wave — the compact-nozzle mechanism of indirect combustion noise, and the reason a reacting analysis cannot in general set $\widehat{h} = 0$ (see [identification](identification.md)).
In the limit $M \to 0$ the choked nozzle reduces to a hard wall, $R \to +1$ and $R_s \to 0$, as it must (tests: `test_terminated_duct_reflection`, `test_mean_flow_open_end`, `test_choked_nozzle_outlet_marble_candel`).

## What is inherited and what is overwritten

The economy of the method is best seen as a ledger of which rows the base Jacobian keeps and which the assembly replaces.

- **Inherited verbatim**: every element jump, conservation, and edge-advection row of $\overline{\mathbf{J}}$, reused unchanged as the algebraic content of the acoustics.
- **Added onto**: the storage block $\mathrm{i}\omega\mathbf{M}$ accumulates onto the conservation rows, and the source block $\mathbf{S}(\omega)$ onto the flame's energy row — neither disturbs the inherited entries.
- **Overwritten**: a duct's continuity rows become the propagation phases $\mathbf{P}(\omega)$; a terminal row becomes its reflection closure; a transfer-matrix element's rows become its prescribed two-port relation (see [elements](elements.md)); and under an isentropic reduction the entropy-transport rows are replaced by the pin $\widehat{h}_e = 0$.

The isentropic reduction deserves a remark, because it is the default for a purely acoustic study.
Pinning the entropy amplitude to zero on every edge, $\widehat{\varrho}' = \widehat{p}'/\overline{c}^{\,2}$, removes the convected wave and leaves a two-amplitude $(f, g)$ acoustics; this is *exact* for the acoustic spectrum of a flow with no entropy sources, and it is what one wants when comparing against classical duct-acoustic results.
It should be noted, however, that the reduction discards the entropy wave altogether, so it misses the indirect combustion noise that an entropy spot generates on reaching a compact nozzle — a distinction taken up quantitatively in [identification](identification.md) and [analyses](analyses.qmd).

## Transfer and scattering matrices

Between any two stations the operator induces a two-port relation, and two equivalent forms of it are used.
The *transfer matrix* $\mathbf{T}(f)$ relates the flow variables at the two stations along their arrows, $\mathbf{v}_{\text{down}} = \mathbf{T}\,\mathbf{v}_{\text{up}}$, while the *scattering matrix* $\boldsymbol{\mathcal{S}}(f)$ relates the incoming waves to the outgoing ones, $\mathbf{w}_{\text{out}} = \boldsymbol{\mathcal{S}}\,\mathbf{w}_{\text{in}}$; the two are inter-convertible closed-form rearrangements of the same information.
Both may be expressed in any of several *flavors* — the characteristic amplitudes $(f, g, h)$, the velocity-normalized primitives, the network variables $(\widehat{\dot m}, \widehat{p}, \widehat{h}_t)$, or the Riemann normalization — which are non-singular rescalings of the wave amplitudes, so a matrix converts between flavors by a similarity built from the per-edge basis blocks.
An important remark is that a transfer matrix is a frequency-domain constraint between the two stations and **should not be interpreted as a causal input–output law**; it relates the linearized states at a frequency, nothing more.
These two-port views are what the transfer-matrix element embeds (see [elements](elements.md)) and what the identification procedure recovers from data (tests: `test_transfer_scattering_round_trip`, `test_flavor_round_trip`, `test_quiescent_scattering_unitary`).

With the operator $\mathbf{A}(\omega)$ assembled, every acoustic question becomes a statement about it: its resonances are the frequencies where it is singular, a forcing drives $\mathbf{A}(\omega)\,\widehat{\mathbf{x}} = \widehat{\mathbf{b}}$, and its stability is read from where its determinant crosses the origin — the analyses of [analyses](analyses.qmd), fed by the source models of [dynamic sources](dynamic-sources.qmd) and, when an element is unknown, by the [identification](identification.md) of its two-port from data.
