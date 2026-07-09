# Identification of unknown elements

Every analysis of the [perturbation network](perturbation-network.md) presumes that each element in the operator is modeled.
Some are not: a component may be known only through measurement, or a flame's transfer function may be the very quantity one wishes to determine.
Identification — equivalently *de-embedding* — recovers such an unknown element's dynamic response from a measured or simulated network response, given a model of the rest of the network.
It is the inverse of the forward acoustic problem: rather than assembling $\mathbf{A}(\omega)$ from known parts to predict a response, one takes a response and a known *reference* operator and solves for the missing part.

The construction rests on a single structural fact — that an unknown element enters the operator as a low-rank update of a known reference — which makes the recovery a small, well-conditioned linear problem rather than a full inversion.
The presentation begins with that de-embedding principle, then proceeds to the recovery of a full two-port and of a flame transfer function, building on this to the isentropic-versus-full distinction and the condition-number diagnostic that guards identifiability, and closes with how the recovered data is continued into the complex plane for the stability analyses.

## The de-embedding principle

The unknown element enters the perturbation operator as a low-rank modification of a *reference* operator $\mathbf{A}_0(\omega)$, defined as the network with the unknown set to a passive default — an identity transfer matrix for an unknown two-port, or a silenced feedback for an unknown source.
Writing the modification in low-rank factors, the two cases are given as:

$$
\text{two-port:}\quad \mathbf{A}(\mathbf{X}) = \mathbf{A}_0 - \mathbf{P}\,(\mathbf{X} - \mathbf{I})\,\mathbf{Q}^{\top},
\qquad
\text{source:}\quad \mathbf{A}(\mathbf{G}) = \mathbf{A}_0 + \mathbf{P}\,\operatorname{diag}(\mathbf{G})\,\mathbf{Q}^{\top},
$$

where $\mathbf{X}$ is the unknown transfer matrix and $\mathbf{G}$ the unknown source gains, $\mathbf{P}$ selects the element's overwritten rows, and $\mathbf{Q}$ carries the upstream-face characteristic map.
Because the update is low-rank, the Woodbury identity [@hager_1989] recovers the unknown from the network response while re-using a *single* factorization of $\mathbf{A}_0$ per frequency, which serves the whole network whether it is a simple cascade or a branched topology — the branch simply folds into $\mathbf{A}_0$.
Intuitively, all the acoustics of the known part are captured once in $\mathbf{A}_0^{-1}$, and the measurement then only has to pin down the small block the unknown occupies.

## Recovering a two-port transfer matrix

The measurement is taken at two stations that straddle the unknown — an upstream edge $a$ and a downstream edge $b$ — where a set of excitations produces port-wave matrices $\mathbf{W}_a$ and $\mathbf{W}_b$, and the measured relation imposes $\mathbf{W}_b = \mathbf{M}_{\text{meas}}\,\mathbf{W}_a$.
Applying $\mathbf{A}_0^{-1}$ to the excitations and projecting onto the two stations expresses each port response as its reference value plus a contribution linear in the unknown, so the measured relation becomes a linear system for the unknown, given at the readable level as:

$$
\mathbf{K}_c \,\mathbf{m} = \mathbf{r},
\qquad
\mathbf{K}_c = \mathbf{G}_b - \mathbf{M}_{\text{meas}}\,\mathbf{G}_a,
\qquad
\mathbf{X} = \mathbf{I} + \mathbf{D}(\mathbf{m}),
$$

where $\mathbf{G}_a$ and $\mathbf{G}_b$ are the reference port responses to the low-rank probe, $\mathbf{r}$ the reference measurement residual, $\mathbf{m}$ the intermediate the Woodbury update solves for, and $\mathbf{X}$ the recovered transfer matrix reconstructed from it.
The system is solved in the least-squares sense, one small solve per frequency on top of the single $\mathbf{A}_0$ factorization, and the same procedure applies unchanged to a branched three-terminal network (tests: `test_identify_transfer_matrix_cascade`, `test_identify_transfer_matrix_branched_3_terminal`).

## Recovering a flame transfer function

When the unknown is a flame's response rather than a full two-port, the same de-embedding recovers the transfer-function gains of the source terms instead of a matrix block.
This is the rank-structured special case in which the unknown is a handful of scalar gains — one per reference quantity, such as a velocity and a pressure sensitivity — so the recovery solves a small system for those gains, given as:

$$
\mathbf{A}_{\text{sys}}\,\mathbf{G} = -\mathbf{R}_0,
$$

where $\mathbf{G}$ collects the per-term gains and $\mathbf{R}_0$ is the reference measurement residual.
Because there are far fewer unknowns than in a full matrix, a *single* measurement can separate several distinct sensitivities — for instance the velocity and pressure responses of the same flame — provided the excitation makes their reference fluctuations sufficiently independent (tests: `test_identify_single_input_ftf`, `test_identify_multi_input_ftf`).

## Isentropic versus full identification

The identification runs in one of two modes, distinguished by whether the convected entropy wave is retained, and the choice is not cosmetic.
The *full* mode carries all three waves $(f, g, h)$ and retains the entropy content, so the recovered response includes the coupling by which composition and temperature fluctuations generate sound.
The *isentropic* mode pins the entropy (and composition waves) amplitude to zero and recovers a purely acoustic two-port, which is the classical flame transfer matrix and the right object when the measurement itself is acoustic.

An important remark is that the isentropic mode **misses the indirect combustion noise**: the entropy-to-acoustic conversion that occurs when an entropy spot reaches a compact nozzle — the coupling $R_s$ of the [perturbation network](perturbation-network.md) — is invisible to an analysis that has set the entropy wave to zero.
Accordingly, an identification aimed at the acoustic two-port of a classical flame is run isentropically, while one that must account for entropy-generated (indirect) noise is run in the full mode (tests: `test_identify_acoustic_only_isentropic`, `test_isentropic_analysis_misses_the_indirect_noise`).

## Identifiability and its diagnostic

Recovery is a linear inverse problem, and like any inverse problem it can be well or ill posed depending on how informative the measurement is.
The diagnostic is the per-frequency condition number $\operatorname{cond}(\cdot)$ of the identification system, reported alongside the recovered response.
A large condition number at a frequency flags that the excitations there are too collinear to separate the unknown — for a multi-input flame, that the reference fluctuations of the several terms are nearly parallel, so their gains cannot be resolved independently — and the recovered value at such a frequency is not to be trusted.
The recovery degrades gracefully rather than failing catastrophically as the conditioning worsens, so the diagnostic is the honest reading of *where* the identification is reliable (tests: `test_identify_noise_degrades_gracefully`, `test_identify_multi_input_ftf`).

Two interpretive points deserve to be stated plainly.
First, transfer-matrix identification can target an interior element that also carries active flame feedback.
In the reference operator $\mathbf{A}_0$, that node's dynamic source is set acoustically silent and its perturbation acoustics are replaced by an identity transfer matrix, while the mean-flow jump is left unchanged.
The recovered matrix therefore includes any active feedback folded into the element's linear two-port, not its passive acoustics alone.
Second, a marker attached to an element declares *what* is unknown; until identification runs, that element is assembled with a passive default (an identity transfer matrix or a silent source).
The edges $a$ and $b$ supplied to the identification routine declare *where* the measured network response is read out; they need not coincide with the marked element's own faces, and the two choices should not be conflated.

## From data to the complex plane

The identification returns a real-frequency table of the recovered response and, by default, its analytic continuation.
The raw table lives only on the real axis, which suffices for a forced response and for the real-axis Nyquist stability driver of [analyses](analyses.qmd), both of which evaluate only there.
The stability eigensolver, however, searches the complex plane for the roots of $\det\mathbf{A}(\omega)$ and therefore needs an analytic function, so the table is continued by the barycentric-rational (AAA) fit [@nakatsukasa_2018] of [dynamic sources](dynamic-sources.qmd); the same continued object then drives both the real-axis Nyquist analysis and the complex-plane eigensolver, and the conversions between transfer and scattering forms preserve its analyticity (test: `test_identify_transfer_matrix_continuation_is_analytic`).

Identification thus closes the loop between measurement and model: a measured or simulated network response becomes a network-ready element — a transfer matrix or a flame transfer function — continued into the complex plane and ready to be placed back into the operator for the stability and response analyses of [analyses](analyses.qmd).
