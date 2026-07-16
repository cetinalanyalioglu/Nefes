# UI case format: result data

Nefes exchanges cases with its graphical companion Nemo through a single YAML document: the network definition, the canvas layout, and an optional results section that carries solver output for display on the graph.
The format is deliberately generic: Nemo colours the network from whatever the file declares, without needing to know how those numbers were produced.

## Connectivity: edges bind to ports by handle

Each flow edge names the two ports it plugs into as `sourceHandle` and `targetHandle`, written `"<node>-port-<ordinal>"`.
The node prefix of an edge's `sourceHandle` must be that edge's own `source`, and of its `targetHandle` its own `target`; a handle that names any other node leaves the edge attached to nothing, so Nemo drops it from the drawing without warning.
The loader therefore rejects such a file on read rather than admit an edge whose endpoint does not exist, and the writer never emits one: when a loaded case is edited so its live topology no longer matches the layout it was opened with, the stored handles are discarded and a fresh, self-consistent layout is synthesized in their place.

## Binding data to the network

A result set is a named collection of numeric series.
Each series is tied to the network by position: the *i*-th value belongs to the element with index *i* on edges or nodes, as declared for that series.
Elements are not referenced by name or type; that positional rule is the entire coupling between solver and UI.

Mean-flow quantities live on edges.
They can optionally be averaged onto the nodes for display; phase angles are never averaged in that way.

## Metadata alongside the fields

Each result set carries a display name and the per-element series.
Scalar conclusions (a modal frequency, a growth rate, a stability verdict) travel as labelled metadata entries that Nemo prints as plain text, so the file itself names what each number means.

## Animated result sets

A result set may vary along an axis (phase, frequency, or any swept parameter).
Each frame then holds one full network colouring for that parameter value.
Nemo keeps the colour scale fixed across frames so comparisons stay meaningful as the user steps through them.

Two built-in uses are typical.
An eigenmode can be shown over one phase cycle: the instantaneous perturbation field is sampled at evenly spaced phases through a period.
A forced-response sweep can be folded into one container with frequency as the frame variable, so the user can scrub through the band without opening many separate result sets.

## What each analysis contributes

1. **Mean flow**: edge fields such as mass flow, pressure, temperature, and Mach number; when the network transports composition, a companion set carries mixture fractions and species mole fractions.
2. **Forced response**: magnitude and phase of the requested perturbation quantities, either one frequency at a time or as the animated sweep above.
3. **Eigenmodes**: per-edge mode-shape magnitude and phase for each mode, with modal scalars in the metadata; optionally the phase animation.
4. **Nyquist stability**: a summary only: stability verdict, margins, and related scalars.