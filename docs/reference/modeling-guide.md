# Modeling guide

This guide bridges the gap between a physical component — an orifice, a valve, a nozzle, a swirler — and the network element that represents it, so that a user who knows a component's catalogue data can choose the right element and supply the right numbers.
It is a user-facing companion to the [element reference](atomic-elements.md) and the [elements](../theory/elements.md) theory: the reference states each element's parameters, the theory derives its residual, and this guide maps real hardware onto them.
The emphasis throughout is on the two decisions that determine every mapping — *how much total pressure a restriction removes* and *at which pressure a boundary is set* — because getting those two right is most of correct modeling.

## The two questions a restriction answers

Every flow restriction in a network answers two questions, and a correct model answers both.
The first is *how much total pressure it removes* at a given flow — its loss characteristic — which sets where the flow settles.
The second is *whether it can choke* — whether the flow through it saturates when the pressure ratio is large enough — which sets its behaviour at high drop.
A component's catalogue data usually speak to the first question (a loss coefficient, a discharge coefficient, a flow factor) and are silent about the second; the framework supplies the choking behaviour automatically wherever a passage reaches a sonic throat (see [choking](../theory/choking.qmd)), so the modeling task is mainly to translate the catalogue loss into the right element and coefficient.

## Static versus total pressure

The single most useful picture to keep in mind is the distinction between static and total pressure across a restriction.
Total pressure is the quantity a loss consumes: an ideal, loss-free restriction conserves it, and a real one destroys an amount set by its loss characteristic.
Static pressure, by contrast, can *rise* through a restriction even as total pressure falls — most strikingly through a sudden expansion, where the flow decelerates and recovers static pressure while losing total pressure to turbulence.
The framework's loss elements are therefore written as total-pressure relations, and its boundary conditions are careful about which pressure they pin: an inlet drawn from a reservoir constrains total pressure, while an outlet discharging to the atmosphere constrains static pressure (see [elements](../theory/elements.md)).
Reading a catalogue number correctly means knowing which pressure it refers to and where it was measured, and the mapping below assumes total-pressure loss unless stated.

## Orifices, valves, and filters

A component characterized by a single loss coefficient referenced to a dynamic head — an orifice plate, a throttle valve, a filter, a screen — maps directly onto the concentrated **loss** element, whose residual removes $K_L$ dynamic heads of total pressure and opposes the flow in either direction (see [elements](../theory/elements.md#sec-elements-loss-family)).
The coefficient $K_L$ is the catalogue loss coefficient; where the datum is a *discharge coefficient* $C_d$ instead, the two are related through the effective area the jet contracts to, and an orifice that both contracts and dumps its jet downstream is more faithfully built as a **sudden contraction followed by a sudden expansion** — the composite that the [composite elements](composite-elements.md) reference provides ready-made.
A valve whose loss varies with its setting is the same element with a setting-dependent $K_L$, and a screen or perforate whose resistance is *linear* in the flow — significant at low speed and in the acoustics — is the **linear resistance** element rather than the quadratic loss, because a linear resistance stays active even at vanishing mean flow.

## Nozzles and area changes

A smooth contraction or diffuser that changes area with little loss is the **isentropic area change**, which conserves total pressure until its small port chokes and then admits the lumped normal-shock drop from the same row (see [choking](../theory/choking.qmd)).
An abrupt area change is the **sudden area change**, lossy in expansion by the Borda–Carnot amount that a momentum balance fixes without any empirical constant, and nearly loss-free in contraction up to a vena-contracta coefficient the user may set.
A converging nozzle discharging to a back pressure is modeled with a pressure outlet, which chokes at the critical pressure ratio on its own; a component known only by its throat area and assumed to run choked is the compact **choked-nozzle outlet**, which pins the outflow to the critical mass flux.
For acoustic work a length-bearing passage should be given its length, since the length sets the propagation phase even when it is inert to the mean flow (see [perturbation network](../theory/perturbation-network.md#sec-perturb-propagation-block)).

## Junctions, and the selection rule

Where several streams meet or split, the choice between a **static-pressure junction** and a **lossless splitter** is not cosmetic and is governed by a firm rule: use the static-pressure junction only where every port runs at low Mach number, and the splitter wherever a plenum feeds a fast branch.
A static-pressure junction feeding a fast port hands that branch more total pressure than the feed possesses — free energy that the network cannot dissipate, leaving it with no steady solution — so a plenum distributing to high-speed branches must be a splitter, while a low-speed header merging comparable streams may be a junction (see [elements](../theory/elements.md#sec-elements-junctions-splitters)).
Where streams *merge* and a port is not slow, use the **mixer**: it ties the ports to a common effective total pressure, charging each inflow a mixing loss, so it never manufactures total pressure and generates entropy at any port Mach number.
Its `recovery` sets that loss between the best and worst merges the streams allow: at $1$ (the default) it removes only each inflow's excess over the weakest feed, so the outlet leaves at the minimum inflow total pressure, the least the second law permits; at $0$ each inflow gives up its whole dynamic head, the full dump loss of a plenum, the most dissipative and best-conditioned.
At the default $\sigma = 1$ the element adds no flow resistance of its own (pressure equalities only, like the splitter), so the flow split must be pinned by the network: distributing one inflow this is the lossless splitter; merging, it is well posed only when every inflow's rate is pinned by a mass-flow inlet or a branch resistance, and two bare total-pressure feeds on the node leave the split under-determined (the splitter's own requirement, which the solve warns about).
Lower `recovery` toward $0$ for the robust dump when the feeds are not otherwise pinned: each inflow's dump term is then a self-supplied resistance, so it converges on any wiring and reduces to the junction at low Mach.
Unlike the splitter it converges on merges of unequal total pressure.
A divider whose split is imposed rather than discovered is the **forced splitter**, which fixes the branch fractions and keeps total-pressure continuity on the one free branch.

## Loops need resistance

When the topology closes a loop, every connection carrying flow around that loop must have a real resistance, a pipe or a loss, not a bare lossless link.
A ring of junctions (or splitters) tied to one another by resistance-free connections has no unique steady flow: a circulation can run around the loop without changing any node's pressure or upsetting any mass balance, so the mean-flow equations leave the size of that circulation undetermined.
The solver has nothing to pin it to, and can grow the circulation until the connecting passages run far past the speed of sound, a non-physical result that is a symptom of the missing resistance rather than a solver fault.
The remedy is to model the loop the way the hardware is built: an interconnector tube between combustor cans, a balance pipe between manifolds, or a bypass line is a **pipe** (friction and length) or at least a **loss**, so the flow around the loop is set by the small pressure differences that actually drive it.
A physical loop always carries resistance; leaving it out is the modeling error, and giving the connectors their resistance both restores a unique solution and keeps it comfortably subsonic.

## Where the orifice analogy ends

Some components resist the orifice picture, and it is worth knowing the limits.
An axial swirler loses total pressure not by a sudden-expansion mechanism but by turning the flow and shedding swirl, so its loss is not the Borda–Carnot amount and its two questions — loss and choking — decouple in a way a plain loss element does not capture; it is represented by a loss element with a measured coefficient, with the understanding that the model reproduces the pressure drop but not the swirl physics.
Fuel injectors and the coupling between a component's loss and its downstream dumping are similar: the framework represents their steady pressure drop faithfully and their unsteady response through a dynamic source (see [dynamic sources](../theory/dynamic-sources.qmd)), but not the internal aerodynamics that set those numbers, which remain a modeling input rather than a prediction.

## The practical recipe

The mapping from a catalogue datum to a network element reduces to a short recipe.
First, identify what the datum characterizes — a loss coefficient, a discharge coefficient, a flow factor, or a geometry — and which pressures it relates.
Second, choose the element whose residual matches that characterization: a loss element for a coefficient-based restriction, an area-change element for a geometric one, a boundary element for a termination, a junction or splitter for a meeting of streams.
Third, translate the datum into the element's parameter, converting a discharge coefficient or flow factor into the loss coefficient the element expects, and supply a length if the acoustics will use the component.
Finally, let the solver discover the flow directions, the choke points, and the operating state, checking the converged result against the component's expected behaviour rather than prescribing it — the discovery-over-prescription stance that the whole framework is built to support (see [the design philosophy](../design/philosophy.md)).
