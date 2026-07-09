# Modifying parameters and running studies

A loaded case (from YAML or built in code) rarely stays fixed: parameter studies, calibration and design sweeps all need the same model solved at many operating points.
Nefes therefore gives every physical parameter a **name and a dotted address**, and one generic machinery to read and write it: `parameters()`, `get`, `set`, `update`, `with_params` and the sweep driver `parameter_study`.
Parameter writes never touch topology, so the compiled problem keeps its layout and a previous solution remains a valid warm start.
This page is the user guide; the internal architecture is described in [design/parameters](../design/parameters.md).

## Addresses

An address is a dotted `"name.parameter"` string; the head names an element or an edge by its display name, and the leaf names one of its declared parameters:

- `"inlet.mdot"` — a parameter of the element named `inlet`;
- `"orifice.throat_area"` — a composite's own knob (never its expanded internals);
- `"e3.area"` — an edge's flow area (areas live on edges, never on elements);
- `"p_ref"`, `"T_ref"` — the bare (dot-free) network-level references.

Element and edge names are assigned at construction (`name=` on the factories, `Network.connect(name=...)`) or by the UI; `Network.add` makes element names unique.
Note that a factory-default name is always numbered on add (`duct` becomes `duct-1`), so name the elements you intend to address.

Addressing is **fail-closed**: an unknown name or parameter raises immediately, with near-match suggestions, and a batch write resolves every address before anything is written.
A silent no-op is designed out.

## The inventory

`net.parameters()` returns every addressable parameter with its current value, SI unit and admissible range:

```python
>>> net.parameters()
address          value   unit   bounds
---------------  ------  -----  --------
inlet.mdot       0.3     kg/s   >= 0
inlet.Tt         700     K      > 0
orifice.throat_area  0.001  m^2    > 0
...
e0.area          0.005   m^2    > 0
p_ref            101325  Pa     > 0
```

The result is a list of `ParameterInfo` rows with dict-style access by address (`inv["inlet.mdot"].value`).
Advanced knobs that are usually left alone (the smoothing `eps`, the loss `ref_port`, the solver seed references) are hidden by default; pass `advanced=True` to include them.

## Reading and writing

`get` reads one address; `set` writes named parameters on one element; `update` batches writes by address:

```python
net.get("inlet.mdot")                      # 0.3
net.set("inlet", mdot=0.5, Tt=720.0)       # validated, in place
net.update({"orifice.throat_area": 1.2e-3, "e3.area": 0.01, "p_ref": 9.0e4})
```

Every write validates against the element's declared schema before anything is stored, exactly as the element's factory would:

```python
>>> net.set("inlet", mdot=-0.1)
ValueError: mdot must be >= 0 [kg/s] (got -0.1) on MassFlowInlet 'inlet'
```

Three write paths deserve a remark:

1. **Composites are rebuilt, never patched.**
   Setting `"orifice.throat_area"` re-runs the orifice factory with the merged parameters and swaps in the fresh spec, so the derived sub-elements and internal edges are regenerated consistently.
   Editing `sub_elements` or `internal_edges` by hand is exactly the drift this design removes.
2. **Constant-area elements fan out their area.**
   A duct, pipe, flame or mass source requires all incident edges to share one area, so `net.set("duct1", area=...)` writes that area to every incident edge (and a single-port boundary's `area` is its one edge).
   An area-change element carries genuinely per-edge areas; address them as `"e3.area"`.
3. **Object-valued fields go through the same door.**
   `perturbation_bc`, `dynamic_source`, `transfer_matrix`, `composition` (paired with `basis`), `marker` and `back_pressure` are set with the same `set`/`update`, validated by type instead of bounds; `set_perturbation_bc` and `set_dynamic_source` remain as named conveniences over it.
   On a reacting network a composition write additionally checks its species against the loaded library.

## Copies and studies: `with_params`

`net.copy()` deep-copies the specification (elements, edges, references), and `with_params` applies writes to such a copy:

```python
base = nefes.load_case("combustor.yaml")
net  = base.with_params({"inlet.mdot": 0.5})   # base stays pristine
```

**The functional `with_params` is the recommended idiom for studies**: the loaded base stays pristine, no state accumulates across sweep points, and each point is safe to solve independently.
In-place `set`/`update` is the low-level primitive underneath.

Because a parameter write never changes the edge count or order, warm starts chain across points:

```python
prev = None
for mdot in np.linspace(0.3, 0.7, 20):
    sol  = base.with_params({"inlet.mdot": mdot}).solve(x0=prev.x if prev else None)
    prev = sol
```

The one exception worth noting is a discretization composite's segment count (`fanno_pipe.n_segments`): it is addressable as a fidelity knob, but changing it re-discretizes the interior and therefore invalidates warm starts across that write.

## The sweep driver: `parameter_study`

`nefes.parameter_study` packages the loop above: an N-dimensional grid (or a zipped path) of addresses, one `with_params` copy per point, warm starts chained, and scalar probes collected into grid-shaped arrays:

```python
res = nefes.parameter_study(
    base,
    {"inlet.mdot": np.linspace(0.3, 0.7, 20), "outlet.p": [0.9e5, 1.0e5]},
    probe=lambda sol: {"M_max": float(sol.field("M").max())},
)
res.probes["M_max"].shape   # (20, 2)
res.converged               # bool mask, same shape
```

By default a non-converged point raises a pointed error; pass `on_fail="continue"` to record it (`converged=False`, probes `NaN`) and march on.
Pass `keep_solutions=False` on large sweeps to hold only the probed scalars.

For **eigenvalue continuation** over a parameter, the same base plugs into the existing driver through `Network.builder`, which returns the `build(p)` closure `eigenvalue_trajectory` and `nyquist` take:

```python
traj = eigenvalue_trajectory(
    base.builder("flame.Qdot"), np.linspace(1e3, 5e3, 21),
    freq_band=(50.0, 400.0), param_name="Qdot",
)
```

## Object-valued fields and their YAML round-trip

The object-valued fields are reachable by the same generic `set`, but not all of them have a YAML form; the table below states what a saved case can and cannot carry.

| Field | On | YAML round-trip | Note |
| --- | --- | --- | --- |
| `perturbation_bc` | boundary terminals | partial | rigid / open / constant specific impedance round-trip; anechoic, reflection, choked, driven and table/callable forms are code-only |
| `transfer_matrix` | `transfer_matrix_element` | none | the loader builds the node empty; attach post-load (`net.set(node, transfer_matrix=...)`); the `UnknownTransferMatrix` identification marker is accepted |
| `dynamic_source` | flames, mass source | none | attach post-load; `set_dynamic_source` is the named alias |
| `composition` + `basis` | inlets, outlet backflow, mass source | yes | the prime reacting study knob; validated against the species library |
| `marker` | inlets, outlet backflow, mass source | yes | burnt marker in $[0, 1]$; marker-gated networks only |
| `back_pressure` | choked nozzle outlet | yes | post-solve choke diagnostic |
| `eps` | sudden area change, loss | no | smoothing-width override (advanced) |

## What is deliberately outside this API

The network-level references `p_ref` and `T_ref` (and the advanced `mdot_ref` / `h_ref` seeds) are addressable: they are value knobs that preserve the state-vector dimension, so they are warm-start-safe.

The **gas model is not**: `thermoModel`, the gas constants, the species slate, the reducer and the mechanism file determine the gas library and the number of transported scalars.
Changing any of them reshapes the problem (`n_solve` changes) and invalidates every warm start, so it is a re-specification of the model rather than a parameter change, and it stays behind the explicit construction path (`Network(gas=...)`, the UI Model pane) that forces a cold solve.
