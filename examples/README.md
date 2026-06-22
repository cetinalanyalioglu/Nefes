# Examples

- **`converging_nozzle.yaml`** — a network **saved from the FNetLibUI tool** (the
  `fns-flow-network` model): reservoir → feed pipe → isentropic contraction →
  tailpipe → back-pressure outlet. The two ducts are inert in the mean flow but
  carry the wave phase used by the perturbation network.
- **`acoustic_terminations.yaml`** — a UI case demonstrating the **perturbation
  boundary conditions**: an acoustic **excitation** reservoir feeds a duct to an
  **impedance** liner outlet, with a side branch closed by a **wall** (a
  quarter-wave resonator with no mean flow). Load it, `solve()` the mean flow, then
  sweep `fns.perturbation.boundary_response(sol.problem, sol.x, omegas)`.
- **`converging_nozzle.ipynb`** — loads the UI case, solves the steady mean flow,
  prints the converged edge states, sweeps the back pressure to show emergent
  choking (mass-flow saturation at `M = 1`), and runs the full `3 x 3`
  **perturbation** transfer / scattering analysis (two acoustic waves **plus the
  entropy wave**) on top of the converged mean flow. All figures are Plotly,
  styled with the shared FNS theme (`fns.plotting`).
- **`entropy_generator.ipynb`** — a **validation** notebook replicating De Domenico,
  Rolland & Hochgreb (2019, *JSV* 440), "nozzles with losses", at the Cambridge
  Entropy Generator geometry. Builds the orifice-plate / isentropic / non-isentropic
  nozzles from `isentropic_area_change` + `sudden_area_change`, reproduces the
  mean-flow pressure rise (their Fig. 5) and the compact acoustic + entropic transfer
  functions (their Fig. 6), and checks the assembled compact scattering matrix (in the
  `riemann` = `(P+, P-, σ)` flavour) against an **independent** composition of the
  analytic jumps. The machine-precision version of that check lives in
  `tests/test_perturbation_dedomenico.py`.

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

The `Wall` element additionally blocks the **mean** flow (`ṁ=0` on its edge). To force
the response, attach an excitation (a Python-only closure) and solve:

```python
import numpy as np
from fns.io import load_case
from fns.perturbation import PerturbationBC, boundary_response

net = load_case("examples/acoustic_terminations.yaml")
net._elements[0].perturbation_bc = PerturbationBC.excitation(1.0)   # drive the reservoir
sol = net.solve()
fr = boundary_response(sol.problem, sol.x, np.linspace(50.0, 3000.0, 200))
gamma_in = fr.reflection_at(0)   # input reflection g/f at the feed edge
```

The transfer/scattering-matrix analysis (`perturbation_response`) is unchanged and
boundary-condition agnostic; `boundary_response` instead solves the network as it is
*physically terminated*.
