---
title: 'Nefes: a network solver for reacting compressible flows and thermoacoustics'
tags:
- Python
- thermoacoustics
- combustion instability
- duct acoustics
- compressible flow
- network model
- gas turbines
authors:
- name: Çetin Ozan Alanyalıoğlu
  orcid: 0000-0002-8498-3088
  corresponding: true
  affiliation: 1
- name: Hendrik Nicolai
  orcid: 0000-0002-0355-2252
  affiliation: 2
affiliations:
- name: Department of Mechanical Engineering, Simulation of Reactive Thermo-Fluid Systems, Technical University of Darmstadt
  index: 1
- name: TBD
  index: 2
date: TBD
bibliography: paper.bib
---

# Summary

$\textsf{Nefes}$ is an open-source Python package that models a compressible
internal-flow system, such as a gas-turbine combustor or a duct network,
as a directed graph of connected elements instead of resolving the full
three-dimensional field. This graph first computes the steady-state mean flow, i.e., pressures, velocities, temperatures, and gas composition, applying chemical equilibrium at any point where streams burn or mix, and resolving smooth choking at the sonic throat. 
Starting from a quiescent state, the solver determines flow directions on its own; hence, the user only provides geometry and boundary conditions, not an initial guess of the solution.
On top of that converged mean flow, Nefes then computes the linear
perturbations superimposed on it — sound waves, convected hot spots
(entropy waves), and composition fluctuations — as a linearization of that same network model, ensuring the acoustic description stays fully consistent with the underlying flow.
These waves couple with unsteady combustion to drive thermoacoustic instability — a first-order design concern for low-emission gas turbines and rocket engines — and network models are the state-of-the-art fast-screening tool used at the design stage.
The same machinery
serves duct acoustics more broadly, including resonances, damping, and
scattering at area changes and junctions, with perturbations restricted
to flow-aligned (longitudinal) waves.
Within $\textsf{Nefes}$, these problems sit in one framework: the same network model carries the mean flow and the waves built on it, so thermoacoustics and noise in engineering systems can be studied without stitching separate tools together.

# Statement of Need

Thermoacoustic prediction on a ducted system is dual by construction: a steady mean flow must be known on the network, and the linear waves that ride on it must be consistent with that flow.
$\textsf{Nefes}$ treats both sides in one network model, so the wave problem is the exact linearization of the same equations that define the mean flow rather than a separate description.

Even obtaining that mean flow is already a network problem in its own right.
Many practical questions about a combustor or a ducted gas system are
questions about a network: how the mass flow divides among parallel
passages, what pressures and temperatures establish where streams merge
and burn, and whether a nozzle operates in the choked regime or not.
Open-source steady network solvers serve pipeline hydraulics and
thermodynamic cycle analysis well, but they do not cover combustors and
similar ducted gas systems, where the flow is momentum-resolved and
compressible, choking may emerge, and thermochemistry enters the
solve.
To the best of our knowledge, no open
tool combines momentum-resolved compressible duct flow, including Mach number
effects and a smooth, emergent treatment of choking, with
chemical-equilibrium thermochemistry evaluated inside the solve, on an
arbitrary graph whose flow directions the solver finds itself from a
quiescent start. $\textsf{Nefes}$ provides that combination; its mean-flow solver
is useful entirely on its own.

The same solver is also the foundation for the layer that motivated it
in the first place.
Predicting thermoacoustic instability [@juniper2018sensitivity] requires both the mean flow and the linear waves superimposed on it — and stability verdicts can hinge on mean-flow details that are easy to get slightly wrong in practice. 
In existing network tools the two layers are separate models,
and ensuring compatibility between them is often a user task. Because $\textsf{Nefes}$ solves the
mean flow, it constructs the perturbation problem, including acoustic,
entropy, and compositional waves [@magri2016compositional], as the
linearization of the same network model about the converged state, so
the two layers agree by construction. We are not aware of another
released tool that couples a solved reacting compressible mean flow to a
multi-wave perturbation network. A companion methods paper
[@alanyalioglu2026operator] presents the formulation and its
verification, and demonstrates that seemingly small inconsistencies
between the mean flow and the acoustic model can flip a stability
verdict.

There is a practical gap as well: the established open thermoacoustic
network tools are MATLAB-based — OSCILOS [@li2015oscilos] and taX
[@emmert2014tax], the latter also needing Simulink — so running them requires a commercial platform. $\textsf{Nefes}$ is
pure Python on the open numerical stack [@harris2020numpy;
@virtanen2020scipy; @lam2015numba], installable with pip, permissively
licensed (BSD-3-Clause), and scriptable end to end.

$\textsf{Nefes}$ is intended for researchers and engineers modeling compressible reacting flow using
networks, with or without the acoustic layer; for combustion and
thermoacoustics ones interested in screening combustor designs for instabilities; for
duct-acoustics work on transmission, reflection, and scattering of
longitudinal waves in ducts [@munjal2014ducts]; for teaching, where a
complete instability analysis fits in a short notebook; and as an
extensible base for design-stage analysis in industrial gas-turbine
practice.

# State of the Field

Steady flow-network solvers exist in several adjacent regimes. Pipeline
tools such as pandapipes [@lohmeier2020pandapipes] handle arbitrary
topology and genuinely compressible gases, including hydrogen, but in
the low-Mach, friction-dominated pipeline regime, without Mach number effects,
choking, or reaction. Cycle-analysis tools such as pyCycle
[@hendricks2019pycycle] do branch and mix streams and do carry
chemical-equilibrium thermochemistry, but they abstract the system as
zero-dimensional stations in a fixed cycle layout, with no wave layer.
TESPy [@witte2020tespy] models plant flowsheets with lumped components
and simplified combustion. GFSSP [@majumdar2013gfssp], the nearest
regime match with momentum-resolved branches and choking, is closed
source and treats reaction outside the solve. None of these tools
resolves the duct momentum balance through smooth choking, computes its
mean state with equilibrium chemistry, and determines flow directions on an
arbitrary graph rather than a prescribed pipeline or cycle; that
combination is the gap $\textsf{Nefes}$ fills.

Low-order thermoacoustic network models themselves are long established
[@dowling1995calculation; @dowling2003acoustic]. Among released
implementations, OSCILOS [@li2015oscilos] marches one-dimensional jump
relations along an essentially serial chain of modules for its mean
state, and taX [@emmert2014tax] propagates user-prescribed mean values
through local per-element relations, with branch flow splits supplied by
the user; LOTAN [@stow2001annular] is not public. These tools carry
entropy as well as acoustic waves, but none solves a coupled mean-flow
problem on the network or derives its acoustic elements from the same
equations as its mean state; among the released tools, none runs without
a commercial platform.

The closest methodological relative is the framework of
@merk2025jacobian, which derives acoustic, entropic, and compositional jump
conditions for compact elements as Jacobians of steady conservation
relations. $\textsf{Nefes}$ realizes the corresponding construction for the
assembled system — the network Jacobian evaluated at the converged mean
state — in released software, with the mean state solved rather than
supplied. We built a new tool from scratch rather than extending an
existing one because that consistency requires both layers to share one
set of element equations from the outset: it is an architectural property,
whereas in the established tools the two layers are separate models by
construction.

# Software Design

$\textsf{Nefes}$ is based on a fundamental modeling approach: every element, such
as a duct, an orifice, a flame, or a junction, contributes algebraic
residual equations in the flow states of its incident edges, and a
network is the assembly of these residuals on a directed graph. The
steady mean flow is the solution of the assembled system; the linear
perturbation problem is the exact linearization of the same residuals
about that solution, extended, in ducts, with a dedicated
wave-propagation model that is transparent to the mean-flow solve.
Consistency between the mean flow and the acoustics is therefore an
architectural property rather than a constraint the user has to enforce.

The exactness requirement of the linearization process imposes a
constraint on the element equations: every residual must be smooth in
the flow state, with no branches or switches, so that derivatives exact
to machine precision can be taken by complex-step differentiation
[@martins2013derivatives]. Some elements become less convenient to
write: choking, for example, enters as a smooth reformulation of the
sonic condition rather than a change of regime, and every function that
is not smooth in the flow state must be wrapped in a smooth
approximation.

![The Nefes architecture: the network description is frozen into an
immutable compiled problem; one compiled kernel returns residuals and
exact (complex-step) Jacobians to both the mean-flow solve and the
perturbation analysis.](figures/architecture.png)

In return, the Jacobian used by the solver, and with it the acoustic
operator built from it, cannot drift from the residuals, since both are
generated from the same source (Figure 1).

Assembly and user-facing layers are pure Python behind a small set of
objects and interfaces; the element kernels are compiled just-in-time (JIT)
with Numba [@lam2015numba] and are hidden from the user.
A typical mean-flow solution with its full stability analysis runs in seconds for tens of elements, or minutes for several hundred, on a laptop, making design-stage parameter sweeps practical.

# Features

The features of $\textsf{Nefes}$ are best described separately for each layer.

The mean-flow layer:

- Solves the steady compressible mean flow on an arbitrary network as
  one coupled system, on arbitrary edge directions from a quiescent
  start; smooth residuals allow the solver to reach hard operating
  points without hand-tuned initial guesses.
- Evaluates chemical-equilibrium thermochemistry (NASA-Glenn data)
  inside the solve: burnt edges use chemical equilibrium, unburnt edges stay
  frozen, and the solver decides which is which rather than requiring a
  per-edge prescription; verified against Cantera [@cantera].
- Treats choking as an emergent, smooth part of the solution rather than
  as a boundary condition selected by the user.
- Provides an element catalog of inlets and outlets, ducts and pipes
  with friction, orifices, area changes and nozzles, junctions and
  splitters, cavities, heat sources and equilibrium flames, plus
  composite elements (tapered ducts, Fanno pipes, Helmholtz resonators)
  that expand into subgraphs before the solve.

The perturbation layer, on the converged mean flow:

- Computes eigenmodes (with a count of modes in the searched band), a
  real-frequency Nyquist stability criterion, forced responses, and
  acoustic scattering and transfer matrices of any element or
  subnetwork.
- Transports acoustic, entropy, and compositional waves, including the
  entropy-to-sound conversion at accelerating and choked nozzles.
- Allows attaching prescribed dynamic responses (such as flame transfer
  functions) to supported elements, and supports the reverse operation:
  extracting an unknown element’s response from a measured response of
  the surrounding network.

Across both layers:

- Deterministic numerics.
- Cases that serialize to and from YAML.
- Documentation built from executable notebooks.
- A test suite (88 files) with a claim-to-test validation map.
- Interactive network building in [Nemo](https://github.com/cetinalanyalioglu/Nemo), a companion browser-based editor.

# Examples

The repository ships a gallery of runnable example notebooks, each
regenerating the figures it shows: steady reacting networks from a
single flame to a full gas-turbine combustor and can-annular
architectures; duct-acoustics cases including a helicopter exhaust muffler,
Helmholtz resonators, and frequency-dependent boundaries; and
thermoacoustic analyses spanning the Rijke tube, a gas-turbine
combustor, flame identification, and entropy- and composition-driven
instability, together with the validation cases 
cited below.

# Research Impact Statement

$\textsf{Nefes}$ is a modern open-source tool whose central novelty is the unification of reacting compressible mean-flow networks with their acoustic, entropy, and compositional linearization.
That foundation supports design screening, sensitivity and adjoint-based analysis, and optimization without separate mean-flow and acoustic models.
$\textsf{Nefes}$ is newly released software; its case for significance rests on
verification and validation rather than a prior publication record. The
shipped validation cases reproduce published results across all layers: branch
flows and nodal pressures of a compressed air pipe network
[@greyvenstein1994segregated], the transmission loss of a three-stage
helicopter exhaust muffler [@parrott1973], the compact-nozzle acoustic and entropy
response of @marble1977nozzle, and, on a published laboratory-combustor
case [@li2015oscilosreport], a thermoacoustic mode within one percent
of the OSCILOS result in frequency and growth rate. The examples
reproduce a published swirl-burner stability analysis [@emmert2017brs]
at figure level, down to its intrinsic-mode branch and
reflection-coefficient sweep. The novel numerical methodology is described in the
companion paper [@alanyalioglu2026operator], and the package is in
active use in the authors’ research activities. Every figure in the
documentation is generated by a runnable notebook shipped with the
repository.

# AI Usage Disclosure

Generative AI (Anthropic’s Claude) assisted with implementation and
documentation drafting. Development followed a specification-driven
workflow: the authors designed the architecture—data structures,
interfaces, and algorithms—and authored the corresponding specification
sheets, implementation examples where necessary, and acceptance tests
that define intended behavior; AI then produced the bulk of the source
code to satisfy those specifications. First drafts of the package
documentation were likewise AI-generated and subsequently reviewed and
edited by the authors. All AI-assisted output was reviewed, tested, and
edited by the authors; correctness is gated by the package’s test suite
and the validation benchmarks cited above.

# Acknowledgements

TBD.

# References
