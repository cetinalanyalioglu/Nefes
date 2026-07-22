# Examples

The notebooks are grouped by the layer they exercise.
A reader new to the tool is best served by starting in **`getting-started/`** and then moving to whichever layer matches their interest.
Each notebook opens with its network topology drawn by the **Nemo** UI, then builds and solves it entirely through the public `nefes` API (`Network`/`Solution`, `nefes.parameter_study`, and the acoustic analyses as `Solution` methods).

- **`getting-started/`** — first contact: load a case, solve the mean flow, serialize a result.
- **`flow/`** — non-reacting steady mean flow on larger networks, plus element infrastructure.
- **`combustion/`** — steady reacting mean flow (equilibrium flames, markers, multi-fuel mixing); no acoustics.
- **`acoustics/`** — the perturbation network on a non-reacting mean flow (closures, storage, mode shapes, eigenmodes).
- **`thermoacoustics/`** — combustion–acoustic coupling: self-excited instability, indirect noise, flame identification.
- **`validation/`** — replications of literature benchmarks.

## `getting-started/`

- **`converging_nozzle.ipynb`** (+ **`converging_nozzle.yaml`**) — the canonical first
  example. Loads a network **saved from the Nemo tool** (reservoir → feed pipe →
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
- **`can_annular_combustor.ipynb`** (+ **`can_annular_combustor.yaml`**) — the **reacting
  showcase**: a whole can-annular combustor **loaded from a saved case** and solved as one
  reacting network. One plenum (a lossless **splitter**) distributes compressor-discharge air in
  two stages, through per-can **distributor** splitters, to a **ring of eight reacting Jet-A(L)
  cans** (dome swirl air, plus liner-cooling and dilution air metered through the annulus and
  liner holes by **flow splitting**) and a turbine-cooling **bypass**; the delivery, feed, bypass,
  and **interconnector** lines are **pipes** (Darcy friction), and the cans are cross-linked by
  flame-tube interconnector tubes (a **non-tree** ring) and collected through a **choked NGV**
  throat that sets the combustor pressure. 191 elements / 229 edges, fully subsonic. The cans are
  **staged** (rich-to-lean around the ring), which drives an emergent **interconnector cross-flow**
  from the hot cans to the cool ones. Reports, as printed tables, the mass-conserving air budget,
  the per-can staging and cross-flow, an axial profile through one can, and the choked-throat
  pressure lever.
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
  **reheat** stage. Shows the chemistry plumbing does not depend on the fuel; each injected composition
  is its own transported **mixture fraction**.
- **`multiple_fuel_manifold.ipynb`** — burns **three different fuels in three parallel
  branches** off a single air supply, then mixes the hot products back into one outlet;
  stresses the reacting mean-flow solver on a branched (non-chain) topology.

## `acoustics/`

- **`perturbation_boundary_conditions.ipynb`** — exercises **every** named `PerturbationBC`
  closure on a single driven duct, checking each against its analytic value (diagonal
  reflections, the `driven=` source term, the entropy→acoustic coupling `R_s` of the
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
  coefficient off the real frequency axis, as the eigensolver requires: the impulse-response
  fit for finite-memory responses (the recommended route) and the rational fit for resonant ones.
- **`compositional_noise.ipynb`** — **compositional (indirect) noise** at a choked nozzle.
  Validates the inert acoustic limit against Marble–Candel and a resolved nozzle, then shows
  the inherited / resolved routes carry the composition → acoustic coupling `R_xi` that the
  hand-written closure silently drops.

## `thermoacoustics/`

- **`rijke_tube.ipynb`** — the fundamental thermoacoustic oscillator: a duct with a heat
  source that, under the right conditions, feeds energy into a duct mode and drives a
  **self-excited instability** (an `n-τ` active element in the acoustic field).
- **`eigenvalue_sensitivity.ipynb`** — **what moves each mode**: on the same self-excited
  Rijke tube, `modes.sensitivities()` ranks every setup parameter by how it shifts each
  mode's growth rate (cold-duct length stabilizes the fundamental; flame `n`/`τ`
  destabilize it; the second mode answers to different knobs). Uses the dedicated
  `sens.plot()` bar chart and confirms the top stabilizer with an
  `eigenvalue_trajectory` whose tangent matches the local slope.
- **`ita_and_cavity_modes.ipynb`** — the same tube, tuned so its spectrum carries **both
  families**: three **cavity** resonances set by the geometry and three **intrinsic
  thermoacoustic (ITA)** modes set by the flame lag alone. Separates them three ways —
  **anechoic ends** (`R = 0`) leave only the ITA ladder `f = (2k+1)/(2τ)`, verified against a
  closed-form growth rate `σ = ln(K/B)/τ`; **`eigenvalue_trajectory`** continued toward `n = 0`
  makes the ITA branches dive while cavity branches park on the passive resonances; and
  **`nyquist_stability_map`** recovers the same stability boundary as an integer count step,
  agreeing with the trajectory on both the onset gain and the onset frequency.
- **`equivalence_ratio_instability.ipynb`** — **fuel-supply** combustion instability: a
  chamber fluctuation modulates the **fuel flow rate** → local **equivalence ratio** → a
  mixture fluctuation that convects to the flame and burns into unsteady heat release, with
  the injector-to-flame time lag.
- **`fuel_transport_instability.ipynb`** — the same convective chain with **no dynamic
  source anywhere**: a *steady* fuel injector, a mixing duct, and a *static* equilibrium
  flame self-excite through the **passive** operator alone. The air-side fluctuation dilutes
  the mixture at the injector (exact relation `ξ' = −ξ̄·ṁ'/ṁ̄`, verified in-notebook), the
  equivalence-ratio wave convects silently to the flame, and the burnt hot spot converts to
  sound at the **choked outlet**. Certified unstable modes; per-family `convected=` surgery
  attributes the loop to the composition wave; boundary-power budget locates the energy
  entry at the nozzle; the growth rate bands in the transport lag.
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
- **`em2c_combustor.ipynb`** — a **cross-code check against OSCILOS** on a published case: the
  stable configuration of the EM2C swirl combustor (plenum → injection unit → chamber, compact
  flame with a second-order low-pass FTF). Nefes solves the mean flow on the network and pins the
  dominant mode at `153.4 Hz / -19.0 1/s` against the reported `152.6 Hz / -19.1 1/s`. Also
  separates the damping (the **dump plane**, not the flame, is the main sink), shows the shed
  **entropy wave is a spectator** at a pressure-release outlet, and maps the flame-lag stability
  band the operating point sits below.
- **`brs_combustor.ipynb`** — the **TUM BRS swirl burner** of Emmert et al., where the published
  spectrum is a *mix* of acoustic and **intrinsic (ITA)** modes. The flame response is published
  only as a figure, so it is digitized (`data/brs_ftf.csv`) and rebuilt as the **finite impulse
  response** it came from — entire, hence exact under analytic continuation, where a rational fit
  of the same samples would litter the search box with poles. Reproduces the reference's three
  quantitative figures against published eigenvalues extracted from their vector twins
  (`data/brs_published_*.csv`): the three-system spectrum (full / pure acoustic / pure ITA, with
  the dominant modes at `43.2 / 105.7 / 319.5 Hz` against `42.4 / 111.0 / 315.6`), the modulation
  sweep that assigns each full-system mode to its acoustic or intrinsic parent, and the reflection
  sweep ending in the **paradox**: making the exit *less* reflective drives the intrinsic mode
  unstable (`+23.7 Hz` growth at the anechoic end, published `+23.0`). Along the way it documents
  a reversed phase axis and a missing FTF normalization factor in the printed reference.

## `validation/`

- **`greyvenstein_laurie_network.ipynb`** — verifies Nefes against Greyvenstein & Laurie
  (1994), Example 3: a **29-pipe compressed-air distribution network** (their only
  compressible case).
- **`tfaws07_gfssp_compressible.ipynb`** — verifies the length-bearing elements against the
  classical one-dimensional compressible-flow theory, following the five GFSSP verification cases
  of Majumdar & Bandyopadhyay (TFAWS07-1016, 2007): **Fanno**, **Rayleigh**, the two combined, and
  both again in a converging–diverging passage. References are the closed-form Fanno relations and
  the generalized one-dimensional flow equation, integrated independently. Each case is run across
  segment counts, so what comes out is an **order of accuracy**, not just an overlay: Rayleigh flow
  is exact to solver tolerance (the compact flame's momentum jump *is* the Rayleigh relation), the
  momentum-closure pipe reaches the exact Fanno profile at second order while the Darcy–Weisbach
  closure settles on a different answer, and every case that splits two effects across a segment
  converges at first order.
- **`entropy_generator.ipynb`** — replicates De Domenico, Rolland & Hochgreb (2019, *JSV* 440),
  "nozzles with losses", at the Cambridge Entropy Generator geometry: the mean-flow pressure
  rise (their Fig. 5) and the compact acoustic + entropic transfer functions (their Fig. 6).
- **`parrott_helicopter_muffler.ipynb`** — validates the acoustic-network layer against
  Parrott, NASA TN D-7309 (1973): a flight-tested three-stage concentric extended-tube
  exhaust muffler. The Nefes network reproduces the classical transfer-matrix transmission
  loss to machine precision, with the measured field data overlaid, and additionally solves
  the mean flow the reference assumes.
- **`webster_horn.ipynb`** — validates the **tapered-duct composite** against an independent
  numerical solution of the classical horn (Webster) equations, for a 4:1 exponential horn.
  Establishes that the segment chain converges on the *true* horn rather than merely on itself,
  at the documented first-order rate in `1/N`, then shows the `grid_refine` / `auto_refine`
  helpers choosing `N` for a target tolerance — with their self-convergence estimate checked
  against the true error this reference makes measurable.

## Running the notebooks

The notebooks run against the **installed** `nefes` package (no `sys.path` bootstrapping): install
it in editable mode with the `jupyter` extra, or use the conda env, then launch Jupyter:

```bash
pip install -e ".[jupyter]"   # or: conda env create -f environment.yml
conda activate nefes
jupyter lab examples/getting-started/converging_nozzle.ipynb
```

Every notebook imports `nefes` and reaches only its public surface; state is read by name off the
`Solution`, and the acoustic analyses are `Solution` methods (`sol.eigenmodes(...)`,
`sol.forced_response(...)`, ...). The topology figure each notebook shows is a rendered PNG
committed next to it (`<name>_topology.png`), produced from the network by the Nemo UI.

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
Plotly with the Nefes theme, which the `nefes.plotting` figures carry by default and which `set_theme("light" | "dark")` switches; matplotlib is not rendered.

Or solve a UI case in two lines:

```python
import nefes
sol = nefes.load_case("examples/getting-started/converging_nozzle.yaml").solve()
print(sol.edge(1))   # throat state: mdot, M, p, p_t, T, ...
```

## The UI case format

`load_case` reads the native YAML the **Nemo** tool writes out for the
`nefes` model (defined in that repo under `public/models/`). The
relevant sections:

```yaml
model:
  id: nefes
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
| any constructor with `driven=(…)` | drive an incoming acoustic/entropy/scalar wave (e.g. `anechoic(driven=("acoustic",))`, scaled by `amplitudes=`) |
| `.choked_nozzle()` / `.compact_nozzle()` | compact choked outlet, `g=Rf+R_s·h` (Marble–Candel) |
| `.constant_mass_flow()` | outlet pinning `ṁ'=0`, `g=Rf+R_s·h` |

See **`acoustics/perturbation_boundary_conditions.ipynb`** for a worked demonstration of every
closure checked against its analytic value. The `Wall` element additionally blocks the
**mean** flow (`ṁ=0` on its edge). To force the response, mark a terminal's closure as
`driven` and solve:

```python
import numpy as np
import nefes
from nefes.elements import catalog as cat
from nefes.perturbation import PerturbationBC, forced_response

net = nefes.Network(
    nodes=[
        cat.total_pressure_inlet(108000.0, 300.0, perturbation_bc=PerturbationBC.anechoic(driven=("acoustic",))),  # drive
        cat.duct(0.5),
        cat.pressure_outlet(101325.0, 300.0, perturbation_bc=PerturbationBC.impedance_polar(2.0, 0.0)),
    ],
    edges=[(0, 1, 0.05), (1, 2, 0.05)],
)
sol = net.solve()
fr = forced_response(sol, np.linspace(50.0, 3000.0, 200))   # takes the Solution directly
gamma_in = fr.reflection_at(0)   # input reflection g/f at the feed edge
```

The transfer/scattering-matrix analysis (`perturbation_response`) is unchanged and
independent of the boundary conditions; `forced_response` instead solves the network as it is
*physically terminated*.
