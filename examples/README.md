# Examples

The notebooks are grouped by the layer they exercise.
A reader new to the tool is best served by starting in **`getting-started/`** and then moving to whichever layer matches their interest.

- **`getting-started/`** — first contact: load a case, solve the mean flow, serialize a result.
- **`flow/`** — non-reacting steady mean flow on larger networks, plus element infrastructure.
- **`combustion/`** — steady reacting mean flow (equilibrium flames, markers, multi-fuel mixing); no acoustics.
- **`acoustics/`** — the perturbation network on a non-reacting mean flow (closures, storage, mode shapes, eigenmodes).
- **`thermoacoustics/`** — combustion–acoustic coupling: self-excited instability, indirect noise, flame identification.
- **`validation/`** — replications of literature benchmarks.

## `getting-started/`

- **`converging_nozzle.ipynb`** (+ **`converging_nozzle.yaml`**) — the canonical first
  example. Loads a network **saved from the FNetLibUI tool** (reservoir → feed pipe →
  isentropic contraction → tailpipe → back-pressure outlet), solves the steady mean flow,
  prints the converged edge states, sweeps the back pressure to show emergent choking
  (mass-flow saturation at `M = 1`), and runs the full `3 x 3` **perturbation** transfer /
  scattering analysis (two acoustic waves **plus the entropy wave**) on top of the
  converged mean flow. Plotly, Nefes theme.
- **`save_load_demo.ipynb`** — write a solved network back to the UI's native YAML (case
  **+** result data) and read it in again, round-tripping the mean-flow field.

## `flow/`

- **`gas_turbine_large.ipynb`** (+ **`gas_turbine_large.yaml`**) — the **large showcase**
  network (a gas-turbine **secondary-air / cooling** distribution). Two bleed feeds mix at a
  static-pressure junction, pass a contraction, and split across three sub-manifolds metering
  air to ~15 fixed-back-pressure sinks through orifices, dump nozzles and labyrinth seals. A
  cross-bridge makes the graph **not a tree**, and one sink sits above the local static
  pressure, producing **emergent backflow** (ingestion). 63 elements / 63 edges; converges
  in ~13 Newton steps, fully subsonic (max `M` ≈ 0.65). Tabulates the converged states,
  checks the global **mass balance**, draws the network as a **Sankey**, and runs the
  **multiport scattering** perturbation layer with per-terminal source attribution.
- **`huge_network_stress.ipynb`** — **generates** a 1000+ element network programmatically
  and solves its steady mean flow as a scale stress test, then extracts its end-to-end
  perturbation behaviour.
- **`composite_elements.ipynb`** — a **composite element** presents as one element but
  expands at build time into a small graph of atomic elements; the solver, Jacobian and
  perturbation layers never see the composite.
- **`parameter_studies.ipynb`** — the **named-parameter API**: the addressable inventory
  (`net.parameters()`), validated `get`/`set`/`update` writes (including a composite knob
  and the constant-area fan-out), the pristine-base `with_params` idiom, and
  `nefes.parameter_study` sweeps (1-D warm-chained operating line and a 2-D
  throat-area x inflow grid with `on_fail="continue"` at the choked corner). `plotly`.

## `combustion/`

- **`reacting_flame.ipynb`** — **reactive-flow fundamentals**. The `nefes.thermo`
  HP-equilibrium solver (adiabatic flame temperature vs equivalence ratio), the perfect-gas
  **heat-release flame** (`Qdot` total-enthalpy jump with the Rayleigh static-pressure drop),
  and the reacting **equilibrium flame** (`EQ_FROZEN` → `EQ_KERNEL`). Self-contained. `plotly`, Nefes theme.
- **`burnt_marker.ipynb`** — the **orientation-proof reacting closure**. A transported
  **burnt marker** scalar `b` gates a single `EQ_MARKER` blend of frozen (unburnt) and
  equilibrium (burnt) states; the flame stamps `b = 1` on whatever edge the flow actually
  leaves it by, so the split is correct no matter how the edges were drawn. Shows
  seed-independent self-correction and that the marker is acoustically passive. `plotly`.
- **`gas_turbine_combustor.ipynb`** — a **complete gas-turbine combustor**:
  compressor-discharge air → **fuel mass source** → **equilibrium flame** → **dilution-air
  mass source** → turbine-inlet outlet. Streams are named by **species**; the network
  transports one conserved **mixture fraction** per distinct injected composition. Sweeps
  fuel flow (equivalence ratio) and dilution air against flame / turbine-inlet temperatures.
- **`rql_combustor.ipynb`** — a **rich-quench-lean (staged) combustor**: a fuel-rich primary
  zone → **quench-air** mass source → a **lean** burnout zone that re-equilibrates and
  oxidizes leftover CO / H₂. Exercises the **sticky burnt marker** (a reachability label
  transported by a noisy-OR) and a fixed-overall air-split sweep of primary flame temperature
  and primary NO (the RQL low-NOx lever). `plotly`.
- **`multiple_fuels.ipynb`** — **two very different fuels at different positions**: n-octane
  (`C8H18`) in the primary zone, then hydrogen (`H2`) injected into the hot products as a
  **reheat** stage. Shows the chemistry plumbing is fuel-agnostic; each injected composition
  is its own transported **mixture fraction**.
- **`multiple_fuel_manifold.ipynb`** — burns **three different fuels in three parallel
  branches** off a single air supply, then mixes the hot products back into one outlet;
  stresses the reacting mean-flow solver on a branched (non-chain) topology.

## `acoustics/`

- **`perturbation_boundary_conditions.ipynb`** — exercises **every** named `PerturbationBC`
  closure on a single driven duct, checking each against its analytic value (diagonal
  reflections, the `excitation` source term, the entropy→acoustic coupling `R_s` of the
  `choked_nozzle` / `constant_mass_flow` outlets, and the default `inherit`).
- **`frequency_dependent_reflection.ipynb`** — a terminal's `PerturbationBC` carries a
  reflection coefficient `R` that is a **constant**, a **table** interpolated in frequency,
  or a **callable**.
- **`outlet_boundaries.ipynb`** — the two flow-fixing outflow boundaries beyond the
  static-pressure outlet: `mass_flow_outlet` (`ṁ' = 0`) and the compact `choked_nozzle_outlet`;
  with acoustic-power accounting.
- **`helmholtz_resonator.ipynb`** — the **storage block `M`** and its first producing element,
  the **`cavity`** (a wall to the mean flow, a compliance `V/c²` to acoustics). Composes a
  **Helmholtz resonator** from primitives and reproduces the analytic transmission-loss peak
  at `f₀ = c√(Aₙ/(V·l))/2π`.
- **`inertance_storage.ipynb`** — generalizes the storage block `M` to the **jump elements**:
  the **inertance** and the **manifold compliance** (`volume` on `junction`/`splitter`).
- **`mode_shape_animation.ipynb`** — Nefes's **spatially-resolved mode shapes**: the
  continuous perturbation field *inside* the ducts, animated over one oscillation cycle
  (the duct's own analytic phase relation, not an approximation).
- **`eigenmode_analysis.ipynb`** — **linear-stability** analysis: a network's free acoustic
  oscillations as the roots of `det A(ω) = 0` in the complex plane, via contour integration
  with a completeness certificate.
- **`acoustic_refinement.ipynb`** — **when discretization matters for acoustics.** A
  `tapered_duct` horn's mean flow is exact at any segment count `N`, but its scattering matrix
  is not; sweeps the inlet-outlet scattering matrix over frequency for several `N`, shows the
  ~`O(1/N)` convergence, and drives `grid_refine` / `auto_refine`.
- **`analytic_continuation.ipynb`** — continuing a **tabulated** transfer function / reflection
  coefficient off the real frequency axis, as the eigensolver requires.
- **`compositional_noise.ipynb`** — **compositional (indirect) noise** at a choked nozzle.
  Validates the inert acoustic limit against Marble–Candel and a resolved nozzle, then shows
  the inherited / resolved routes carry the composition → acoustic coupling `R_xi` that the
  hand-written closure silently drops.

## `thermoacoustics/`

- **`rijke_tube.ipynb`** — the fundamental thermoacoustic oscillator: a duct with a heat
  source that, under the right conditions, feeds energy into a duct mode and drives a
  **self-excited instability** (an `n-τ` active element in the acoustic field).
- **`equivalence_ratio_instability.ipynb`** — **fuel-supply** combustion instability: a
  chamber fluctuation modulates the **fuel flow rate** → local **equivalence ratio** → a
  mixture fluctuation that convects to the flame and burns into unsteady heat release, with
  the injector-to-flame time lag.
- **`indirect_noise_instability.ipynb`** — an **entropy-driven** thermoacoustic instability:
  a compact flame in a duct ending in a **choked nozzle** goes unstable through the entropy
  spot that convects down the hot duct and is partly converted **back into sound** at the
  nozzle — a path pure acoustics cannot see.
- **`entropy_noise.ipynb`** — **indirect combustion noise**: a flame's heat-release
  fluctuation sheds an entropy spot that is silent until accelerated through a downstream
  nozzle, where it radiates sound.
- **`flame_identification.ipynb`** — **identifying a flame's dynamic response** from a
  network-wide measurement. A branched combustor is characterized by its inlet→outlet
  **transfer matrix**, and the flame — buried inside the branches — is de-embedded with
  `nefes.perturbation.identify` as a transfer matrix and as a velocity `n-τ` FTF, with the
  de-embedding **conditioning** as the identifiability diagnostic.

## `validation/`

- **`greyvenstein_laurie_network.ipynb`** — verifies Nefes against Greyvenstein & Laurie
  (1994), Example 3: a **29-pipe compressed-air distribution network** (their only
  compressible case).
- **`entropy_generator.ipynb`** — replicates De Domenico, Rolland & Hochgreb (2019, *JSV* 440),
  "nozzles with losses", at the Cambridge Entropy Generator geometry: the mean-flow pressure
  rise (their Fig. 5) and the compact acoustic + entropic transfer functions (their Fig. 6).
- **`dokumaci_expansion_chamber.ipynb`** — validates the acoustic-network layer against
  Dokumacı (2021), Fig 5.15: a through-flow expansion-chamber muffler transmission loss.

## Running the notebooks

Each notebook prepends the repo root to `sys.path` (walking up until it finds the `nefes`
package), so no install of `nefes` is needed — just run it with a Python that has the project
dependencies (`numpy`, `scipy`, `numba`, `pyyaml`) plus the notebook stack and Plotly. Install
those with the `jupyter` extra (`pip install -e ".[jupyter]"`) or use the conda env, then:

```bash
conda activate nefes
jupyter lab examples/getting-started/converging_nozzle.ipynb
```

## Rendering into the documentation site

The Quarto docs render a **curated subset** of these notebooks as executed pages and list the rest
with a link to their source. Which is which is driven by a small block each notebook carries in its
own JSON metadata (`metadata.nefes`), so the gallery is generated straight from the notebooks with no
separate list to maintain:

```json
"metadata": {
  "nefes": {
    "title": "Self-excited Rijke tube",
    "description": "The fundamental thermoacoustic oscillator: a duct with a heat source that self-excites.",
    "category": "thermoacoustics",
    "render": "full"
  }
}
```

- `render: full` — executed at build time and shown as a full page.
- `render: list` — appears in the gallery table with its title and description, linking to its source
  on GitHub (not executed by the doc build).

The pre-render step `docs/_scripts/build_examples.py` walks every notebook, reads this block, and
generates the gallery. To promote a notebook from a listed link to a rendered page, flip its `render`
flag to `full`; to add a notebook, just drop it in with a `metadata.nefes` block. The sources stay
**output-free** on disk — the doc build executes copies, never the originals — so all plots must be
Plotly with the Nefes theme (`use_nefes_theme()`); matplotlib is not rendered.

Or solve a UI case in two lines:

```python
from nefes.io import load_case
sol = load_case("examples/getting-started/converging_nozzle.yaml").solve()
print(sol.edge(1))   # throat state: mdot, M, p, p_t, T, ...
```

## The UI case format

`load_case` reads the native YAML the **FNetLibUI** tool writes out for the
`fns-flow-network` model (defined in that repo under `public/models/`). The
relevant sections:

```yaml
model:
  id: fns-flow-network
  globalAttributes: {gasConstant: 287.0, heatCapacityRatio: 1.4,
                     referencePressure: 101325.0, referenceTemperature: 300.0,
                     referenceMassFlow: 5.0}      # 0 -> auto
  nodes:
    - {id: TotalPressureInlet_1, type: TotalPressureInlet,
       attributes: {label: reservoir, index: 0, totalPressure: 2.0e5, totalTemperature: 300.0}}
    - {id: IsentropicAreaChange_1, type: IsentropicAreaChange, attributes: {label: nozzle, index: 1}}
    - {id: PressureOutlet_1, type: PressureOutlet,
       attributes: {label: back-pressure, index: 2, pressure: 1.5e5, backflowTotalTemperature: 300.0}}
  edges:
    - {id: edge_1, source: TotalPressureInlet_1, target: IsentropicAreaChange_1,
       sourceHandle: TotalPressureInlet_1-port-0, targetHandle: IsentropicAreaChange_1-port-0,
       type: flow, attributes: {label: feed, index: 0, area: 0.020}}
    - {id: edge_2, source: IsentropicAreaChange_1, target: PressureOutlet_1,
       sourceHandle: IsentropicAreaChange_1-port-1, targetHandle: PressureOutlet_1-port-0,
       type: flow, attributes: {label: throat, index: 1, area: 0.010}}
```

**Ports matter and are preserved.** Each edge's `sourceHandle`/`targetHandle`
ends in `-port-<ordinal>`; the loader keeps those ordinals and densifies each
element's incident ports to `0..d-1`, so port-0 conventions (the LossElement
reference area, the junction/splitter reference port) match the canvas. Element
`type` names map to the Nefes catalog: `MassFlowInlet`, `TotalPressureInlet`,
`PressureOutlet`, `Wall`, `IsentropicAreaChange`, `SuddenAreaChange`, `LossElement`,
`Duct`, `JunctionStaticP`, `LosslessSplitter`. Supersonic boundaries are deferred
in v1 and raise a clear error.

## Perturbation boundary conditions

Each single-port element (`MassFlowInlet`, `TotalPressureInlet`, `PressureOutlet`,
`Wall`) carries an **acoustic closure** of the linear fluctuation problem
(theory.md §12.4): one reflection relation `w_incoming − R(ω)·w_outgoing = b(ω)`
written on the terminal, every flavor being a choice of `R` (and an excitation
forcing `b`).

**In the UI** the surface is deliberately small — each boundary node's *Acoustics*
group exposes a single **Acoustic boundary** dropdown (`boundaryType`) with three
choices: **Rigid** (a closed wall, `u'=0`), **Open** (an ideal pressure-release open
end, `p'=0`, `R=−1`), or **Impedance**, which reveals a **specific impedance** as
`impedanceMagnitude` (|Z|/ρc) and `impedancePhase` (degrees). Selecting one is
exclusive, so there is no precedence to resolve. Defaults: inlets/outlets default to
**open** (`p'=0`); the `Wall` node defaults to rigid (and offers only Rigid/Impedance).
The loader maps these to `PerturbationBC.hard_wall()`, `PerturbationBC.open_end()`, or
`PerturbationBC.impedance_polar(...)`. A boundary with no `boundaryType` keeps its
default closure (`inherit` for inlets/outlets — e.g. a pressure outlet → `p'=0`).

**In Python** the full `PerturbationBC` API is available (the richer closures are
Python-only):

| constructor | meaning |
| --- | --- |
| `PerturbationBC.inherit()` (default) | keep the linearized mean BC |
| `.hard_wall()` | rigid, `u'=0` (`R=+1`) |
| `.open_end()` | pressure-release, `p'=0` (`R=−1`) |
| `.mean_flow_open_end()` | convective open end, `R=−(1−M)/(1+M)` |
| `.anechoic()` | reflection-free (`R=0`) |
| `.reflection(R)` | prescribed `R` (constant, `(ω,values)` table, or callable) |
| `.impedance(Z, specific=…)` / `.impedance_polar(mag, phase_deg)` | `R=(Z−ρc)/(Z+ρc)` |
| `.excitation(amp, family=…)` | drive an incoming acoustic/entropy wave |
| `.choked_nozzle()` / `.compact_nozzle()` | compact choked outlet, `g=Rf+R_s·h` (Marble–Candel) |
| `.constant_mass_flow()` | outlet pinning `ṁ'=0`, `g=Rf+R_s·h` |

See **`acoustics/perturbation_boundary_conditions.ipynb`** for a worked demonstration of every
closure checked against its analytic value. The `Wall` element additionally blocks the
**mean** flow (`ṁ=0` on its edge). To force the response, attach an excitation (a
Python-only closure) and solve:

```python
import numpy as np
from nefes.elements import catalog as cat
from nefes.shell import build_problem
from nefes.perturbation import PerturbationBC, boundary_response
from nefes.solver import solve
from nefes.thermo.configure import perfect_gas

els = [
    cat.total_pressure_inlet(108000.0, 300.0, perturbation_bc=PerturbationBC.excitation(1.0)),  # drive
    cat.duct(0.5),
    cat.pressure_outlet(101325.0, 300.0, perturbation_bc=PerturbationBC.impedance_polar(2.0, 0.0)),
]
prob = build_problem(perfect_gas(287.0, 1.4), els, [(0, 1, 0.05), (1, 2, 0.05)], 5.0, 1e5, 1004.5 * 300.0)
res = solve(prob)
fr = boundary_response(prob, res.x, np.linspace(50.0, 3000.0, 200))
gamma_in = fr.reflection_at(0)   # input reflection g/f at the feed edge
```

The transfer/scattering-matrix analysis (`perturbation_response`) is unchanged and
boundary-condition agnostic; `boundary_response` instead solves the network as it is
*physically terminated*.
