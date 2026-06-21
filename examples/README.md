# Examples

- **`converging_nozzle.yaml`** — a network **saved from the FNetLibUI tool** (the
  `fns-flow-network` model): reservoir → feed pipe → isentropic contraction →
  tailpipe → back-pressure outlet. The two ducts are inert in the mean flow but
  carry the wave phase used by the perturbation network.
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
`PressureOutlet`, `IsentropicAreaChange`, `SuddenAreaChange`, `LossElement`,
`Duct`, `JunctionStaticP`, `LosslessSplitter`. Supersonic boundaries are deferred
in v1 and raise a clear error.
