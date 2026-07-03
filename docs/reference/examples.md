# Example index

This is an annotated index of the runnable notebooks under `examples/`, each tagged with the theory and validation documents it demonstrates.
The notebooks are the executable companion to the documentation: where a theory document derives a relation and a validation document cites the test that checks it, an example *shows* it in a complete, runnable network.
They are grouped below by the layer they exercise — mean flow, reacting flow, acoustics, entropy and indirect noise, and identification — and a reader new to the tool is best served by starting with the first entry in each group.

## Mean flow and networks

| Notebook | Demonstrates | See |
|---|---|---|
| `converging_nozzle.ipynb` | Emergent choking and the operating map of a converging nozzle | [choking](../theory/choking.qmd) |
| `greyvenstein_laurie_network.ipynb` | The Greyvenstein & Laurie compressed-air benchmark network | [benchmarks](../validation/benchmarks.md) |
| `outlet_boundaries.ipynb` | The boundary elements: pressure, mass-flow, and choked-nozzle outlets | [elements](../theory/elements.md) |
| `composite_elements.ipynb` | Composite elements built from atomic ones (orifices, nozzles, resonators) | [composite elements](composite-elements.md) |
| `gas_turbine_combustor.ipynb` | A representative combustor network end to end | [framework](../theory/framework.md) |
| `gas_turbine_large.ipynb`, `huge_network_stress.ipynb` | Scaling and robustness on large networks | [the solver](../design/solver.md) |
| `save_load_demo.ipynb` | Serializing and round-tripping a network specification | [reproducibility](../design/reproducibility.md) |

## Reacting flow

| Notebook | Demonstrates | See |
|---|---|---|
| `reacting_flame.ipynb` | An equilibrium flame: frozen reactants in, burnt products out | [thermochemistry](../theory/thermochemistry.md) |
| `burnt_marker.ipynb` | The marker-gated frozen/equilibrium closure switch | [thermochemistry](../theory/thermochemistry.md) |
| `multiple_fuels.ipynb`, `multiple_fuel_manifold.ipynb` | Transported mixture fractions with several feed streams | [thermochemistry](../theory/thermochemistry.md) |
| `equivalence_ratio_instability.ipynb` | An equivalence-ratio fluctuation driving a thermoacoustic mode | [dynamic sources](../theory/dynamic-sources.qmd) |

## Acoustics

| Notebook | Demonstrates | See |
|---|---|---|
| `rijke_tube.ipynb` | The canonical Rijke-tube instability | [analyses](../theory/analyses.qmd), [benchmarks](../validation/benchmarks.md) |
| `eigenmode_analysis.ipynb` | Contour-integral modal stability and the completeness certificate | [analyses](../theory/analyses.qmd) |
| `mode_shape_animation.ipynb` | Reconstructing and animating a mode's spatial wave pattern | [characteristics](../theory/characteristics.md) |
| `helmholtz_resonator.ipynb`, `inertance_storage.ipynb` | Finite-volume storage: compliance and inertance | [perturbation network](../theory/perturbation-network.md) |
| `perturbation_boundary_conditions.ipynb`, `frequency_dependent_reflection.ipynb` | Terminal reflection closures, constant and frequency-dependent | [perturbation network](../theory/perturbation-network.md) |
| `dokumaci_expansion_chamber.ipynb` | An expansion-chamber acoustic two-port against a reference | [benchmarks](../validation/benchmarks.md) |

## Entropy and indirect noise

| Notebook | Demonstrates | See |
|---|---|---|
| `entropy_generator.ipynb`, `entropy_noise.ipynb` | Entropy generation and its convection to a nozzle | [dynamic sources](../theory/dynamic-sources.qmd) |
| `indirect_noise_instability.ipynb` | Indirect (entropy) noise closing a thermoacoustic loop | [analyses](../theory/analyses.qmd) |
| `compositional_noise.ipynb` | Composition-generated sound and the analytic-closure caveat | [limitations](../theory/limitations.md) |

## Identification

| Notebook | Demonstrates | See |
|---|---|---|
| `flame_identification.ipynb` | De-embedding a flame transfer function from a network response | [identification](../theory/identification.md) |
| `analytic_continuation.ipynb` | Continuing a tabulated response off the real axis for the eigensolver | [dynamic sources](../theory/dynamic-sources.qmd) |

Each notebook is self-contained and regenerable from the pinned environment described in [reproducibility](../design/reproducibility.md); the figures embedded in the theory documents are the same computations run inline, so a notebook and its companion document should always agree.
