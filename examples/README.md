# Examples

- **`converging_nozzle.yaml`** — a network **saved from the FNetLibUI tool** (the
  `fns-flow-network` model): reservoir → feed pipe → isentropic contraction →
  tailpipe → back-pressure outlet. The two ducts are inert in the mean flow but
  carry the wave phase used by the perturbation network.
- **`perturbation_boundary_conditions.ipynb`** — exercises **every** named
  `PerturbationBC` closure on a single driven duct, checking each against its analytic
  value: the diagonal reflections (`hard_wall`, `open_end`, `mean_flow_open_end`,
  `anechoic`, `reflection`, `impedance`/`impedance_polar`) read back as `g/f` at the
  termination; the `excitation` source term `b` (with `base_R` and acoustic/entropy
  `family`); the entropy→acoustic coupling `R_s` of the `choked_nozzle` /
  `constant_mass_flow` outlets (indirect noise, vs Marble–Candel); and the default
  `inherit`. All via `fns.perturbation.boundary_response`; Plotly, FNS theme.
- **`helmholtz_resonator.ipynb`** — demonstrates the **storage block `M`** and its first
  producing element, the **`cavity`**. Shows the cavity is a wall to the mean flow
  (`mdot = 0`) and a compliance `V/c²` to acoustics (its single `M` entry), then composes
  a **Helmholtz resonator** from primitives (tee + neck `duct` + `cavity`) and reproduces
  the analytic side-branch transmission-loss peak at `f₀ = c√(Aₙ/(V·l))/2π`, with the
  resonance tuned across a 16:1 cavity-volume sweep. Plotly, FNS theme.
- **`inertance_storage.ipynb`** — generalizes the storage block `M` to the **jump
  elements**: the **inertance** (`l_up`/`l_down`/`end_correction` on area changes, `loss`,
  `linear_resistance`) and the **manifold compliance** (`volume` on `junction`/`splitter`).
  Shows a neck modeled as an inline inertance (`iωM`) resonates at the same `f₀` as a neck
  `duct` (carried in `P(ω)`); that an `end_correction` lengthens `L_eff` and lowers `f₀`
  (≈20 % for a flanged `δ ≈ 0.85a`); and that a `junction(volume=V)` reproduces the
  `cavity(V)` compliance. Plotly, FNS theme.
- **`gas_turbine_large.yaml`** — the **large showcase** network (a gas-turbine
  **secondary-air / cooling** distribution), adapted from the preliminary-study
  prototype. Two bleed feeds — a `TotalPressureInlet` (HP) and a `MassFlowInlet`
  (LP) — mix at a static-pressure junction, pass a contraction, and split across
  three sub-manifolds metering air to ~15 fixed-back-pressure sinks through
  orifices (`IsentropicAreaChange`), dump nozzles (`+ SuddenAreaChange`) and
  labyrinth seals (series `LossElement`s). A `LossElement` **cross-bridge** links
  two sub-manifolds, so the graph is **not a tree**, and one sink sits above the
  local static pressure, producing **emergent backflow** (ingestion) closed by its
  `backflowTotalTemperature`. Six **constant-area `Duct`** sections sit on the
  transport runs (inlet feeds, main manifold pipe, sub-manifold feeds): inert in
  the mean flow, they carry the wave phase the perturbation network propagates. 63
  elements / 63 edges; converges in ~13 Newton steps, fully subsonic (max `M` ≈
  0.65).
- **`converging_nozzle.ipynb`** — loads the UI case, solves the steady mean flow,
  prints the converged edge states, sweeps the back pressure to show emergent
  choking (mass-flow saturation at `M = 1`), and runs the full `3 x 3`
  **perturbation** transfer / scattering analysis (two acoustic waves **plus the
  entropy wave**) on top of the converged mean flow. All figures are Plotly,
  styled with the shared FNS theme (`fns.plotting`).
- **`compositional_noise.ipynb`** — **compositional (indirect) noise** at a choked
  nozzle. Validates the inert acoustic limit (the inherited `choked_nozzle_outlet`
  element, the hand-written Marble–Candel closure, and a **resolved** convergent
  nozzle all agree on the reflection `R`), then shows that the inherited /
  resolved routes also carry the composition → acoustic coupling `R_xi` — the
  compositional noise — that the hand-written closure silently drops (the
  `CompositionalNoiseWarning`), all from the same complex step that gives `R` and
  the entropy noise `R_s`. Closes with the `M = 1` subsonic-scope note.
- **`entropy_generator.ipynb`** — a **validation** notebook replicating De Domenico,
  Rolland & Hochgreb (2019, *JSV* 440), "nozzles with losses", at the Cambridge
  Entropy Generator geometry. Builds the orifice-plate / isentropic / non-isentropic
  nozzles from `isentropic_area_change` + `sudden_area_change`, reproduces the
  mean-flow pressure rise (their Fig. 5) and the compact acoustic + entropic transfer
  functions (their Fig. 6), and checks the assembled compact scattering matrix (in the
  `riemann` = `(P+, P-, σ)` flavour) against an **independent** composition of the
  analytic jumps. The machine-precision version of that check lives in
  `tests/test_perturbation_dedomenico.py`.
- **`gas_turbine_large.ipynb`** — the companion notebook for
  `gas_turbine_large.yaml`. Loads and solves the secondary-air network, tabulates
  the converged edge states, checks the global **mass balance**, and charts the
  per-sink air distribution (the lone negative bar is the **backflow** sink). It
  then draws the whole network as a **Sankey** laid out on the UI canvas
  coordinates, and runs the **perturbation** layer: it shows why a 2-port transfer
  matrix is non-physical across the splitter/junctions (the `TransferMatrixWarning`)
  and instead uses the rigorous whole-network descriptors — the **multiport
  scattering matrix** and per-terminal **source attribution** at a chosen sink. All
  figures are Plotly, styled with the shared FNS theme (`fns.plotting`).

- **`reacting_flame.ipynb`** — **reactive-flow fundamentals**. The standalone
  `thermolib` HP-equilibrium solver (adiabatic flame temperature vs equivalence ratio
  for H2/air, straight from the NASA `thermo.inp` data), the perfect-gas **heat-release
  flame** (`Qdot` total-enthalpy jump with the Rayleigh static-pressure drop), and the
  reacting **equilibrium flame** (unburnt `EQ_FROZEN` approach → `EQ_KERNEL` products,
  "ignition" by a per-edge closure switch). The network flame T matches the standalone
  equilibrium across an equivalence-ratio sweep. Self-contained `matplotlib`.
- **`gas_turbine_combustor.ipynb`** — a **complete gas-turbine combustor**:
  compressor-discharge air → a **fuel mass source** (the injector) → an **equilibrium
  flame** → a **dilution-air mass source** (cooling to the turbine-inlet temperature) →
  the turbine-inlet outlet. Streams are named by **species** (`{"O2": 0.21, ...}`,
  `{"CH4": 1.0}`); the network transports one conserved **mixture fraction** per
  distinct injected composition, discovered automatically at build time. Tabulates the converged edge states, charts the axial temperature
  (flame jump then dilution cooling), and sweeps fuel flow (equivalence ratio) and
  dilution air against the flame / turbine-inlet temperatures. Tweak `mdot_fuel`,
  `mdot_dilution`, `Tair`, `p`.
- **`multiple_fuels.ipynb`** — **two very different fuels at different positions**:
  n-octane (`C8H18`) burned in the primary zone, then hydrogen (`H2`) injected into the
  hot products as a **reheat** stage (it re-equilibrates and releases more heat, no
  extra flame element). Shows the chemistry plumbing is fuel-agnostic — each injected
  composition is its own transported **mixture fraction**, reconstructed exactly by a
  forward blend (no element bookkeeping, no distinguishability restriction, so even two
  carbon-bearing fuels can co-mix unburnt). Prints the per-edge mixture-fraction flow
  and sweeps the H2 reheat.

## Running the notebook

The notebook adds the repo root to `sys.path`, so no install of `fns` is needed —
just run it with a Python that has the project dependencies (`numpy`, `scipy`,
`numba`, `pyyaml`) plus the notebook stack and Plotly. Install those with the
`jupyter` extra (`pip install -e ".[jupyter]"`) or use the conda env, then:

```bash
conda activate fns
jupyter lab examples/converging_nozzle.ipynb
```

Or solve a UI case in two lines:

```python
from fns.io import load_case
sol = load_case("examples/converging_nozzle.yaml").solve()
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
`type` names map to the FNS catalog: `MassFlowInlet`, `TotalPressureInlet`,
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

See **`perturbation_boundary_conditions.ipynb`** for a worked demonstration of every
closure checked against its analytic value. The `Wall` element additionally blocks the
**mean** flow (`ṁ=0` on its edge). To force the response, attach an excitation (a
Python-only closure) and solve:

```python
import numpy as np
from fns.elements import catalog as cat
from fns.perturbation import PerturbationBC, boundary_response
from fns.solver import solve
from fns.thermo.configure import perfect_gas

els = [
    cat.total_pressure_inlet(108000.0, 300.0, perturbation_bc=PerturbationBC.excitation(1.0)),  # drive
    cat.duct(0.5),
    cat.pressure_outlet(101325.0, 300.0, perturbation_bc=PerturbationBC.impedance_polar(2.0, 0.0)),
]
prob = cat.build_problem(perfect_gas(287.0, 1.4), els, [(0, 1, 0.05), (1, 2, 0.05)], 5.0, 1e5, 1004.5 * 300.0)
res = solve(prob)
fr = boundary_response(prob, res.x, np.linspace(50.0, 3000.0, 200))
gamma_in = fr.reflection_at(0)   # input reflection g/f at the feed edge
```

The transfer/scattering-matrix analysis (`perturbation_response`) is unchanged and
boundary-condition agnostic; `boundary_response` instead solves the network as it is
*physically terminated*.
