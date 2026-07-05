# Example index

This is an annotated index of the runnable notebooks under `examples/`, each tagged with the theory and validation documents it demonstrates.
The notebooks are the executable companion to the documentation: where a theory document derives a relation and a validation document cites the test that checks it, an example *shows* it in a complete, runnable network.
They live in subfolders by the layer they exercise (`getting-started/`, `flow/`, `combustion/`, `acoustics/`, `thermoacoustics/`, `validation/`) and are grouped the same way below; a reader new to the tool is best served by starting with `getting-started/`.

## Getting started

| Notebook | Demonstrates | See |
|---|---|---|
| `getting-started/converging_nozzle.ipynb` | Emergent choking and the operating map of a converging nozzle | [choking](../theory/choking.qmd) |
| `getting-started/save_load_demo.ipynb` | Serializing and round-tripping a network specification | [reproducibility](../design/reproducibility.md) |

## Flow

| Notebook | Demonstrates | See |
|---|---|---|
| `flow/gas_turbine_large.ipynb`, `flow/huge_network_stress.ipynb` | Scaling and robustness on large networks | [the solver](../design/solver.md) |
| `flow/composite_elements.ipynb` | Composite elements built from atomic ones (orifices, nozzles, resonators) | [composite elements](composite-elements.md) |

## Combustion

| Notebook | Demonstrates | See |
|---|---|---|
| `combustion/reacting_flame.ipynb` | An equilibrium flame: frozen reactants in, burnt products out | [thermochemistry](../theory/thermochemistry.md) |
| `combustion/burnt_marker.ipynb` | The marker-gated frozen/equilibrium closure switch | [thermochemistry](../theory/thermochemistry.md) |
| `combustion/gas_turbine_combustor.ipynb` | A representative combustor network end to end | [framework](../theory/framework.md) |
| `combustion/rql_combustor.ipynb` | A staged rich-quench-lean combustor with the sticky burnt marker | [thermochemistry](../theory/thermochemistry.md) |
| `combustion/multiple_fuels.ipynb`, `combustion/multiple_fuel_manifold.ipynb` | Transported mixture fractions with several feed streams | [thermochemistry](../theory/thermochemistry.md) |

## Acoustics

| Notebook | Demonstrates | See |
|---|---|---|
| `acoustics/perturbation_boundary_conditions.ipynb`, `acoustics/frequency_dependent_reflection.ipynb` | Terminal reflection closures, constant and frequency-dependent | [perturbation network](../theory/perturbation-network.md) |
| `acoustics/outlet_boundaries.ipynb` | The boundary elements: pressure, mass-flow, and choked-nozzle outlets | [elements](../theory/elements.md) |
| `acoustics/helmholtz_resonator.ipynb`, `acoustics/inertance_storage.ipynb` | Finite-volume storage: compliance and inertance | [perturbation network](../theory/perturbation-network.md) |
| `acoustics/eigenmode_analysis.ipynb` | Contour-integral modal stability and the completeness certificate | [analyses](../theory/analyses.qmd) |
| `acoustics/mode_shape_animation.ipynb` | Reconstructing and animating a mode's spatial wave pattern | [characteristics](../theory/characteristics.md) |
| `acoustics/acoustic_refinement.ipynb` | When discretization matters for the scattering matrix | [smoothness](../design/smoothness-contract.md) |
| `acoustics/analytic_continuation.ipynb` | Continuing a tabulated response off the real axis for the eigensolver | [dynamic sources](../theory/dynamic-sources.qmd) |
| `acoustics/compositional_noise.ipynb` | Composition-generated sound and the analytic-closure caveat | [limitations](../theory/limitations.md) |

## Thermoacoustics

| Notebook | Demonstrates | See |
|---|---|---|
| `thermoacoustics/rijke_tube.ipynb` | The canonical Rijke-tube instability | [analyses](../theory/analyses.qmd), [benchmarks](../validation/benchmarks.md) |
| `thermoacoustics/equivalence_ratio_instability.ipynb` | An equivalence-ratio fluctuation driving a thermoacoustic mode | [dynamic sources](../theory/dynamic-sources.qmd) |
| `thermoacoustics/indirect_noise_instability.ipynb` | Indirect (entropy) noise closing a thermoacoustic loop | [analyses](../theory/analyses.qmd) |
| `thermoacoustics/entropy_noise.ipynb` | Entropy generation at a flame and its convection to a nozzle | [dynamic sources](../theory/dynamic-sources.qmd) |
| `thermoacoustics/flame_identification.ipynb` | De-embedding a flame transfer function from a network response | [identification](../theory/identification.md) |

## Validation

| Notebook | Demonstrates | See |
|---|---|---|
| `validation/greyvenstein_laurie_network.ipynb` | The Greyvenstein & Laurie compressed-air benchmark network | [benchmarks](../validation/benchmarks.md) |
| `validation/entropy_generator.ipynb` | The Cambridge Entropy Generator, nozzles with losses | [benchmarks](../validation/benchmarks.md) |
| `validation/dokumaci_expansion_chamber.ipynb` | An expansion-chamber acoustic two-port against a reference | [benchmarks](../validation/benchmarks.md) |

Each notebook is self-contained and regenerable from the pinned environment described in [reproducibility](../design/reproducibility.md); the figures embedded in the theory documents are the same computations run inline, so a notebook and its companion document should always agree.
