---
title: "Best practices"
---

This document is the single source of truth for **how Nefes is meant to be used**, for human users and for agentic AI collaborators.
Every code block here was checked against the actual signatures (line-cited to the source) and the runnable ones were executed.
When in doubt, prefer the patterns in this document over anything found in older notebooks: the notebooks predate the current public API and are inconsistent (see "Anti-patterns" at the end).

## The golden rule

Nefes exposes a small, deliberate public surface.
Use it, and do not reach into internal modules for things the surface already covers.

```python
import numpy as np
import nefes  # Network, Solution, parameter_study, load_case/save_case, cat, perfect_gas, equilibrium, PerturbationBC
```

The common workflow is reachable straight off `nefes`: `nefes.cat` (element factories), `nefes.perfect_gas` / `nefes.equilibrium` (gas models), and `nefes.PerturbationBC` (acoustic terminations).
The acoustic analyses (`eigenmodes`, `forced_response`, `perturbation_response`, ...) live in `nefes.perturbation`.
Sub-package imports (`from nefes.elements import catalog as cat`, `from nefes.perturbation import eigenmodes`) remain valid and are equivalent; the examples below use whichever reads clearest.

The whole workflow is: **build a `Network` -> `solve()` it to a `Solution` -> read named fields -> (optionally) run the perturbation layer on top.**
You almost never construct a `CompiledProblem`, call `solver.solve`, or index raw state vectors yourself.

## The six-step workflow at a glance

```python
import numpy as np, nefes
from nefes.elements import catalog as cat
from nefes.perturbation import PerturbationBC
from nefes.plotting import use_nefes_theme
use_nefes_theme()

# 1. BUILD
net = nefes.Network(
    nodes=[cat.total_pressure_inlet(2.0e5, 300.0), cat.isentropic_area_change(), cat.pressure_outlet(1.5e5, 300.0)],
    edges=[(0, 1, 0.020), (1, 2, 0.010)],
)

# 2. SOLVE  -> Solution (auto-verified)
sol = net.solve()

# 3. READ   (named fields, never raw indices)
sol.print_states()
throat_mach = sol.edge(1)["M"]
mach = sol.field("M")

# 4. PERTURB  (set closures on the network, run the acoustic layer on the solved mean flow)
net.set_perturbation_bc(0, PerturbationBC.hard_wall())
net.set_perturbation_bc(2, PerturbationBC.open_end())
sol = net.solve()
modes = sol.eigenmodes(freq_band=(100.0, 2000.0))  # acoustic analyses are methods on the Solution

# 5. SWEEP  (functional copies, warm-chained)
line = nefes.parameter_study(net, {"throat.area": np.linspace(0.008, 0.012, 20)})

# 6. SAVE / LOAD
sol.save("nozzle.yaml")
net2 = nefes.load_case("nozzle.yaml")
```

Everything below expands one step at a time.

---

## 1. Building a network

Three equivalent ways to get a `Network`; pick by situation.

**One-shot (preferred for programmatic builds).**
Pass `nodes` (element specs, in node order) and `edges` (`(tail, head, area)` tuples referencing node indices).
This form supersedes the lower-level `build_problem` entirely (`nefes/shell/network.py:55`).

```python
net = nefes.Network(
    nodes=[cat.mass_flow_inlet(0.5, 300.0), cat.duct(0.3), cat.orifice(1.5e-4), cat.pressure_outlet(101325.0)],
    edges=[(0, 1, 0.05), (1, 2, 0.05), (2, 3, 0.05)],
)
```

**Incremental (when edge ids matter, e.g. a flame's reference edge).**
`add` returns the node index; `connect` returns the edge id.
Capture the edge id when you will need it later (dynamic sources reference an edge).

```python
net = nefes.Network()
n_in    = net.add(cat.mass_flow_inlet(0.5, 300.0))
n_duct  = net.add(cat.duct(0.3))
n_out   = net.add(cat.pressure_outlet(101325.0))
e_feed  = net.connect(n_in, n_duct, 0.05)  # returns edge id 0
net.connect(n_duct, n_out, 0.05)
```

**Looking things up after building.** If you did not capture an index or edge id, recover it by name or endpoints, so you never hand-count nodes:

```python
i = net.element_index("flame")          # node index of the element named "flame"
name = net.element_name(i)              # the inverse: node index (or name) back to its label
spec = net.element("flame")             # the element itself, by name or index
e = net.edge_between("cold", "flame")   # the edge id joining two elements (e.g. a flame's ref_edge)
```

When only one endpoint is known, walk the topology instead of hand-counting: `edges_of` gives an element's incident edges (and `nodes_of` inverts it), so you never reach into a private edge list.
Both take a name or a node index, and the ids they return index the same edge table as `connect`, so they read straight off a solution (`sol.edge(e)`, `sol.field(name)[e]`).

```python
outs = net.edges_of("split", "out")     # edge ids leaving a splitter (it is the tail)
ins  = net.edges_of("split", "in")      # edge ids entering it (it is the head); "both" is the default
t, h = net.nodes_of(outs[0])            # the (tail, head) elements an edge connects
net.element_name(t), net.element_name(h)
```

`edges_of`'s `direction` is the wiring orientation (which endpoint an edge was attached to), not the solved flow direction, which may run either way along an edge.
`edge_between(a, b)` is the single-edge counterpart: it wants the pair, each a name or a node index, and ignores orientation (naming the two elements either way round returns the same edge id), raising only if no edge, or more than one, joins them.

Element names are unique (a factory default like `duct` is auto-numbered `duct-1`; an explicit `name=` is kept), so a name is a stable handle for lookups and for parameter addresses (§5).

**From a saved case (a UI export or a previous `save`).**

```python
net = nefes.load_case("case.yaml")  # == nefes.Network.from_yaml("case.yaml")
```

**Areas live on edges, not on elements.**
An area change reads the areas of its incident edges; you never pass an "outlet area" to the element factory.
Ports auto-assign in attachment order; pin them only when you must (`edges=[(t, h, area, tail_port, head_port)]`).

**Do not pass reference scales.**
`Network(...)` accepts `p_ref`, `T_ref`, `mdot_ref`, `h_ref`, but they are advanced overrides — leave them out.
The solver reads the operating pressure from the network's own boundaries, and re-measures the mass-flow and enthalpy scales from the realized inflow at solve time, so it self-scales to a 1-bar duct or a 200-bar reacting chamber alike.
Passing a scale is not just noise: a mismatched `mdot_ref` can nudge the cold start into the wrong basin (it is what put the converging-nozzle solve on the spurious supersonic branch before the subsonic guard existed).
Set one only in the rare case with no pressure boundary and no inflow to measure, and even then prefer letting it auto-derive first.

### Element catalog (`from nefes.elements import catalog as cat`)

Boundaries (single-port; the only factories that take `perturbation_bc=`):

| Factory | Purpose |
| :-- | :-- |
| `mass_flow_inlet(mdot, Tt, composition=None, basis="mole", name="inlet", perturbation_bc=None, marker=0.0)` | prescribed mass-flow inlet |
| `total_pressure_inlet(pt, Tt, composition=None, ...)` | prescribed total-pressure inlet |
| `pressure_outlet(p, Tt_backflow=300.0, composition=None, ...)` | static-pressure outlet (becomes a pt-inlet on backflow) |
| `mass_flow_outlet(mdot, ...)` | prescribed-mass-flow outlet (outflow only) |
| `choked_nozzle_outlet(throat_area, back_pressure=None, ...)` | compact sonic-throat outlet |
| `wall(name="wall", perturbation_bc=None)` | impermeable termination (mdot=0), rigid by default |

Internal (2-port and manifolds):

| Factory | Purpose |
| :-- | :-- |
| `isentropic_area_change(name="iac", l_up=0.0, l_down=0.0, end_correction=0.0)` | smooth lossless contraction/diffuser |
| `sudden_area_change(cc=1.0, ...)` | sudden step (Borda-Carnot / vena-contracta) |
| `loss(K, ref_port=0, ...)` | concentrated total-pressure loss `K * ½ρu²` |
| `linear_resistance(R, ...)` | linear loss `R * mdot` (survives zero mean flow) |
| `duct(length=0.0, name="duct")` | lossless constant-area duct (acoustic phase) |
| `pipe(length, diameter, friction_factor, formulation="darcy-weisbach", ...)` | length-bearing pipe; low-Mach Darcy head by default, compressible segment momentum when requested |
| `junction(name="junction", volume=0.0)` | static-pressure manifold (optional plenum); low-Mach ports only |
| `splitter(volume=0.0)` | total-pressure manifold |
| `mixer(recovery=1.0, ...)` | second-law merge for non-slow ports (never manufactures total pressure); default = least-dissipative ideal, pin each inflow or lower `recovery` |
| `forced_splitter(fractions, ...)` | one inflow split at prescribed mass fractions |
| `cavity(volume, ...)` | lumped volume: wall to mean flow, compliance to acoustics |

Reacting closures:

| Factory | Purpose |
| :-- | :-- |
| `equilibrium_flame(name="flame", dynamic_source=None)` | compact flame: frozen unburnt inflow -> HP-equilibrium products |
| `heat_release_flame(Qdot, ..., dynamic_source=None)` | perfect-gas heat addition `Qdot` [W] (no chemistry) |
| `mass_source(mdot, T, composition, u_inj=0.0, ..., marker=0.0)` | inline injection (fuel/dilution), no reaction |

Composites (present as one element, expand to a sub-graph at build time):

| Factory | Purpose |
| :-- | :-- |
| `orifice(throat_area, eps=None)` | isentropic contraction + Borda-Carnot loss |
| `lossy_nozzle(throat_area, beta, ...)` | general lossy nozzle (De Domenico) |
| `sudden_contraction(*, cc=0.62, ...)` | contraction resolving the vena-contracta state |
| `helmholtz_resonator(volume, neck_length, neck_area, ...)` | tee + neck duct + backing cavity |
| `fanno_pipe(length, diameter, friction_factor, n_segments, formulation="momentum", ...)` | distributed friction pipe converging to classical Fanno flow |
| `tapered_duct(area, length=None, n_segments=None, ...)` | horn / con-di nozzle from an (x, A) profile |
| `transfer_matrix_element(tm=None, ...)` | 2-port with a user-supplied acoustic transfer matrix |

Sizing helper: `cat.segments_for_frequency(length, sound_speed, f_max, points_per_wavelength=12) -> int`.

---

## 2. Gas model

The gas model is the first positional argument to `Network`; leave it out for the default.

**Perfect gas (default).**
`Network(gas=None)` resolves to dry-air perfect gas (`R=287, gamma=1.4`).
Override only to change the gas constant or ratio of specific heats:

```python
net = nefes.Network(nefes.perfect_gas(R=287.0, gamma=1.33), nodes=[...], edges=[...])
```

Heat can still be *added* to a perfect gas with `heat_release_flame(Qdot)`; you do not need the reacting model just to add heat.

**Reacting (HP-equilibrium chemistry).**
Declare the feed compositions, use `equilibrium()` with no arguments, and let an `equilibrium_flame` gate the reaction.
You never hand-curate a species list: the product slate is derived from the feeds over the packaged data when the network is built, and the transported scalars are the feed-stream mixture fractions (one per distinct injected composition), auto-discovered from the element compositions.

```python
from nefes.chem import equivalence_ratio_mixture

mix = equivalence_ratio_mixture({"CH4": 1.0}, {"O2": 0.21, "N2": 0.79}, phi=1.0)

net = nefes.Network(
    nefes.equilibrium(),  # automatic products from the feeds, over the bundled data
    nodes=[nefes.cat.mass_flow_inlet(1.0, 300.0, composition=mix, name="feed"),
           nefes.cat.equilibrium_flame(name="flame"),
           nefes.cat.pressure_outlet(101325.0, 300.0, name="out")],
    edges=[(0, 1, 0.05), (1, 2, 0.05)],
)
sol = net.solve()
# verified: converged, T 300 -> 2220 K, products (mole) N2~0.709 H2O~0.183 CO2~0.086 CO~0.009
```

`equivalence_ratio_mixture` and `equilibrium()` both fall back to the packaged NASA Glenn / CEA data when given no species set, so a premixed reacting teaser needs no imports beyond `nefes.chem`.
The `equilibrium_flame` with no `edge_models` gates the frozen/burnt split off the transported burnt marker, labelling the edges downstream of the flame automatically, so the recipe never pins a per-edge closure by hand.
The slate is resolved once, when the network is built, and stays inspectable afterwards:

```python
net.compile()
net.gas.species_names           # the resolved product species
net.gas.species_set.reduction_report  # which candidates were kept, and why
```

**Tuning the automatic slate.**
Five keyword dials on `equilibrium()` size the automatic set without hand-listing species.
`reducer="none"` keeps every candidate; `reduce_threshold` sets the trace mole-fraction cutoff (larger keeps fewer species, smaller keeps more); `reduce_above` sets the candidate count above which the reduction runs at all (lower it to trim a lean slate, raise it to keep a broad one whole); `max_species` caps the kept count; and `must_species` keeps named species regardless of abundance.

```python
gas = nefes.equilibrium(reduce_threshold=1e-4)  # trim harder: drop species below 1e-4 mole fraction
gas = nefes.equilibrium(reducer="none")          # keep the full candidate slate
gas = nefes.equilibrium(max_species=20)          # keep the 20 highest-peaking species
gas = nefes.equilibrium(must_species=["NO"])     # keep NO even though it is trace at equilibrium
```

`reduce_threshold` only bites once the candidate count exceeds `reduce_above`, so a small pool (a hydrogen/air case, say) is kept whole unless you also lower the gate; setting `max_species` runs the reduction regardless, since a cap has nothing to act on otherwise.
`max_species` is a ceiling, not a target: it keeps the highest-peaking species and only ever discards the lowest-ranked non-trace ones, so to sweep set size and see how large a slate a case needs, pair it with a loose `reduce_threshold` so the cap alone drives the count.

```python
for n in (5, 10, 20, 40):
    T = _flame_network(nefes.equilibrium(max_species=n, reduce_threshold=1e-12)).solve().edge(1)["T"]
    print(n, T)  # watch the flame temperature settle as the slate grows
```

The feed species and one carrier of every fed-in element always survive the cap (and count against it), so the equilibrium never loses a constituent it must balance; `max_species` cannot be combined with `reducer="none"`, and a `must_species` naming an element no feed supplies is rejected.
`must_species` accepts a high-temperature condensed product such as graphite `"C(gr)"` (add it to keep soot in a rich slate); it rejects an ion or a feed-only condensed species such as a liquid fuel, which never appears as an equilibrium product.
The same five settings exist on the case file (`speciesReducer`, `speciesReduceThreshold`, `speciesReduceAbove`, `speciesMax`, `speciesMust`).

**Advanced: pin the species and the closures.**
The automatic slate can evolve as the reduction policy improves, so pin it when you need a reproducible species set or a fixed per-edge closure.
Pass an explicit species set (a `SpeciesSet`) and pin the edges with `edge_models`.
A species set is drawn from a `SpeciesDatabase` (the master source) with `.select(...)`; the database is the packaged NASA Glenn / CEA `thermo.inp` by default, and `SpeciesDatabase.from_file(path)` accepts either another `thermo.inp`-format file or a Cantera-format YAML (only its species block is read).
`SpeciesSet.from_cantera(path)` is the one-call shortcut for a Cantera-subset species set:

```python
from nefes.thermo import SpeciesDatabase, EQ_FROZEN, EQ_KERNEL  # closure ids sit here beside perfect_gas/EQ_KERNEL

lib = SpeciesDatabase().select(["CH4", "O2", "N2", "CO2", "H2O", "CO", "OH", "H2", "H", "O", "NO"])  # bundled CEA data
net = nefes.Network(
    nefes.equilibrium(lib),
    nodes=[nefes.cat.mass_flow_inlet(1.0, 300.0, composition=mix, name="feed"),
           nefes.cat.equilibrium_flame(name="flame"),
           nefes.cat.pressure_outlet(101325.0, 300.0, name="out")],
    edges=[(0, 1, 0.05), (1, 2, 0.05)],
    edge_models=[EQ_FROZEN, EQ_KERNEL],  # unburnt | burnt closures, pinned by hand
)
```

`nefes.equilibrium` and the `EQ_*` closure ids are re-exported from `nefes.thermo` (alongside `perfect_gas` and `EQ_KERNEL`), so the reacting build needs no `nefes.thermo.api` / `nefes.thermo.configure` reach.
`SpeciesDatabase()` with no argument reads the packaged database (`nefes/thermo/database.py:51`, `default_thermo_inp()`); pass a path only for a custom mechanism.
`edge_models` pins each edge to a closure id from `nefes.thermo` (`EQ_FROZEN` unburnt, `EQ_KERNEL` burnt).

---

## 3. Solving

`Network.solve()` compiles, solves the steady mean flow, runs post-solve verification, and returns a `Solution` (`network.py:667`).

```python
sol = net.solve(
    x0=None,  # (3, E) initial state; default is a uniform co-directional guess
    tol=1e-10,  # convergence tolerance on the scaled residual 2-norm
    max_iter=80,  # Newton iterations per continuation stage
    kappa_stages=(0.1, 0.01, 0.0),  # artificial-resistance continuation schedule (warm-started in order)
    verbose=0,  # 0 silent; 1 per-stage line; 2 per-equation residual breakdown
)
```

For a related operating point, warm-start from a previous state to converge in a few iterations:

```python
sol2 = net.with_params({"feed.mdot": 0.6}).solve(x0=sol.x)
```

Never call `from nefes.solver import solve; solve(prob)` directly; the free function skips verification and returns no named access.

---

## 4. Reading a solution

All reads are by name; the raw state vector and the `ES_*` index constants are internal and never needed for reading.

```python
sol.converged  # bool
sol.iterations  # int
sol.residual_norm  # final scaled residual 2-norm

sol.field("M")  # one field across all edges  -> ndarray
sol.edge(1)  # every field on edge 1 -> {mdot, p, h_t, rho, u, T, c, M, p_t, area, W, cp}
sol.table()  # rows = fields, cols = edges (table(show_internal=False) hides composite internals)
sol.print_states()  # human table (HTML in a notebook, text otherwise)
sol.residuals()  # {equation label: scaled residual};  sol.print_residuals(top=5)
```

Field names: `mdot, p, h_t, rho, u, T, c, M, p_t, area`, plus `W` (mixture molar mass) and `cp`.

Reacting reads:

```python
sol.species(1)  # {species: fraction} — equilibrium products on a burnt edge (basis="mole" | "mass")
sol.mixture_fractions(1)  # {stream_label: xi} — transported feed-stream fractions
sol.marker(1)  # burnt marker 0 (fresh) -> 1 (burnt); meaningful under automatic gating
sol.heat_release()  # {flame_name: Q_watts} — each flame's heating power off the converged state
```

`heat_release()` reproduces the `Qdot` parameter of a `heat_release_flame` and, for an `equilibrium_flame` (whose power is an outcome of the equilibrium, not an input), evaluates the exact formation-enthalpy drop from frozen reactants to equilibrium products.
The same number de-normalizes an attached flame transfer function when its `q_mean` is not given explicitly.

Structural / diagnostic reads: `sol.composite(name)` (hidden interior of a composite), `sol.composites`, `sol.unchoked_nozzles()`, `sol.cuton_report()` (plane-wave validity ceiling), `sol.verify()`.

---

## 5. Changing parameters

Every element field and edge area is addressable by a dotted string `"<element-or-edge-name>.<field>"`.
Give an element an explicit `name=` so its address is stable and readable: an explicitly chosen name is kept verbatim (`cat.heat_release_flame(Q, name="flame")` → address `"flame.Qdot"`), while an element left at its factory default is auto-numbered (`cat.duct(0.3)` → `"duct-1"`), so the two ducts in a chain never collide.

**Discover** the addressable inventory:

```python
net.parameters()  # table of address / value / unit / bounds
net.parameters(advanced=True)  # adds seed refs (mdot_ref, h_ref) and smoothing knobs
net.get("throat.area")  # read one address
```

**Write** — two philosophies, and this is the distinction to get right:

| Call | Mutates? | Returns | Use for |
| :-- | :-- | :-- | :-- |
| `net.set("feed", mdot=0.6, Tt=720.0)` | in place | node index | a single element, several fields |
| `net.update({"throat.area": 0.011, "feed.mdot": 0.6})` | in place | `self` (chainable) | a batch, mixed elements/edges |
| `net.with_params({"throat.area": 0.011})` | **no — deep copy** | a new `Network` | parameter studies; base stays pristine |
| `net.copy()` | **no — deep copy** | a new `Network` | a clean branch with no writes |

```python
new_net = net.with_params({"throat.area": 0.011})  # net is untouched; new_net has the change
new_net.solve()
```

`with_params` is `copy().update(...)`; it is the recommended functional idiom because no state accumulates across sweep points.
Addresses are resolved and validated *before* any write, so a mistyped address leaves the network untouched.
Composite elements are rebuilt through their factory on a write, never patched.
Address forms: `"element.field"`, `"edge.area"`, and for a single-port or constant-area element `"element.area"` fans out to its incident edges; bare `"p_ref"`, `"T_ref"` (and advanced `"mdot_ref"`, `"h_ref"`) address the network references.

**Build once; vary with addresses.**
Construct the `Network` once, then change lengths, areas, feeds, dynamic-source knobs, and perturbation BCs with `with_params` / `set` / `set_perturbation_bc`.
Do not wrap construction in a parameterized `build(Lm=..., drive=...)` just to pass knobs — that is what the parameter API is for.
A second construction (or a custom `build(p)` for continuation) is warranted only when the change is structural: topology, element *kind*, or gas/species set (those reshape the problem and are outside this API).

**Nested addresses.** An object attached to an element can expose its own scalar knobs (the scalar-parameter protocol): an attached flame response exposes its gain and lag, a constant reflection/impedance boundary condition its magnitude and phase, an identified impulse response an overall `gain` and bulk `delay`.
These join the same address space and every write path:

```python
net.parameters(layer="perturbation")  # only the acoustic-layer rows (see below)
net.get("flame.dynamic_source.tau")  # a single-term source promotes its term's knobs
net.with_params({"flame.dynamic_source.gain": 0.5,  # blend the FTF halfway toward passive
                 "inlet.perturbation_bc.magnitude": 0.7})
# a multi-term source indexes its terms: "flame.dynamic_source.terms[1].gain"
```

**Layer tags.** Every inventory row carries `layer`: `"mean"` (reshapes the mean flow — feed conditions, lengths, areas) or `"perturbation"` (enters only the acoustic operator — storage volumes, inertance lengths, source and boundary knobs; the mean state is invariant to them by construction).
`net.parameters(layer=...)` filters on it, and the eigenvalue sensitivities use it to skip the mean-flow chain term where it vanishes identically.

---

## 6. Parameter studies (sweeps)

`nefes.parameter_study` drives `with_params` over a grid, warm-chaining each point from the last converged state.

```python
res = nefes.parameter_study(
    net,  # pristine base (never mutated)
    {"feed.mdot": np.linspace(0.18, 0.40, 12)},  # {address: 1-D values}
    probe=lambda s: {"dp": float(s.field("p_t")[0] - s.field("p_t")[-1]),
                     "M_max": float(s.field("M").max())},
    mode="grid",  # "grid" = outer product (one axis per address); "zip" = paired 1-D path
    warm_start=True,  # chain solve(x0=prev.x) from the last converged point
    keep_solutions=True,  # set False for large sweeps to drop the Solution objects
    on_fail="raise",  # "continue" records converged=False, probes NaN, and marches on
    progress=True,  # print a per-point status line (index, swept value, converged / iters / |R|)
)

res.probes["dp"]  # ndarray shaped like the grid (here (12,)); NaN where a point failed
res.converged  # bool ndarray
res.grid  # {address: ndarray} of the swept values
res.solutions  # list of Solution (or None if keep_solutions=False)
```

A 2-D grid is just two addresses; the probe arrays come back shaped `(len(a), len(b))`:

```python
grid = nefes.parameter_study(
    net,
    {"throat.area": np.linspace(0.9e-3, 1.6e-3, 8), "feed.mdot": np.linspace(0.18, 0.40, 12)},
    probe=lambda s: {"dp": float(s.field("p_t")[0] - s.field("p_t")[-1])},
    on_fail="continue",  # the choked corner may not converge; keep the rest
)
grid.probes["dp"].shape  # (8, 12)
```

For stability continuation, prefer `net.eigenvalue_trajectory("address", params, ...)` (or `net.nyquist_stability_map`); it uses `with_params` internally.
Reach for `net.builder("address")` or a hand-written `build(p)` only when the free drivers need a closure and the change is not a single address.

---

## 7. Perturbation and acoustics

The linear acoustic/entropy behaviour is solved *on top of a converged mean flow*.

**Call the analyses as methods on the `Solution`.**
The four everyday analyses are bound methods: `sol.eigenmodes(...)`, `sol.forced_response(freqs)`, `sol.perturbation_response(freqs)`, `sol.nyquist_stability(freqs)`.
Parameter-swept stability lives on the `Network`: `net.eigenvalue_trajectory(address, params, ...)` and `net.nyquist_stability_map(address, params, freqs)`.
The free functions in `nefes.perturbation` still exist and also accept a `Solution` as their first argument (`eigenmodes(sol, ...)`, `identify_transfer_function(sol, measured, ...)`, ...); use them for the analyses without a bound method (identification, terminals, verification). The lowest-level `(prob, x_bar)` pair keeps working too, but there is no reason to unpack a `Solution` yourself.

**Set the boundary closures first.**
Each single-port terminal carries a `PerturbationBC`; set it on the element factory or on the network.

```python
from nefes.perturbation import PerturbationBC
net.set_perturbation_bc(0, PerturbationBC.hard_wall())
net.set_perturbation_bc(2, PerturbationBC.open_end())
# or at construction: cat.total_pressure_inlet(pt, Tt, perturbation_bc=PerturbationBC.anechoic())
```

`PerturbationBC` constructors (`nefes/perturbation/operator/boundary_bc.py`):

| Constructor | Meaning |
| :-- | :-- |
| `inherit()` (default) | keep the linearized mean boundary condition |
| `hard_wall()` | rigid, `u'=0` (R=+1) |
| `open_end()` | pressure-release, `p'=0` (R=-1) |
| `mean_flow_open_end()` | convective open end, `R=-(1-M)/(1+M)` |
| `anechoic()` | reflection-free (R=0) |
| `reflection(R, entropy_coupling=None)` | prescribed R: constant, `(freqs, values)` table, callable, or a continuation fit |
| `impedance(Z, specific=False)` / `impedance_polar(magnitude, phase_deg=0.0, specific=True)` | `R=(Z-ρc)/(Z+ρc)` |
| `choked_nozzle()` / `compact_nozzle()` | compact choked outlet, couples arriving entropy -> reflected sound (Marble-Candel) |
| `constant_mass_flow()` | outlet pinning `mdot'=0` |

**Driving (forcing) an incoming wave** is NOT a separate constructor.
Add `driven=("acoustic",)` (or `("entropy",)`, or a scalar name) to any constructor, with optional `amplitudes={...}`:

```python
inlet_bc = PerturbationBC.anechoic(driven=("acoustic",))  # inject a unit incoming acoustic wave, no reflection
```

> There is **no** `PerturbationBC.excitation(...)`. Older `examples/README.md` shows one; it does not exist. Use `driven=`.

### 7a. Forced response — the network as physically terminated

Solve the field the declared boundary conditions actually produce.

```python
import numpy as np

net.set_perturbation_bc(0, PerturbationBC.anechoic(driven=("acoustic",)))
net.set_perturbation_bc(2, PerturbationBC.hard_wall())
sol = net.solve()

fr = sol.forced_response(np.linspace(50.0, 3000.0, 300))
fr.reflection_at(2)  # g/f at the outlet edge, shape (n_freq,)
fr.field(0, "primitive")  # p', u', ... at the inlet edge
fr.stored_energy()  # whole-domain Myers energy per frequency (resonance peaks)
```

### 7b. Transfer / scattering matrices — the network as a component

`perturbation_response` drives one unit wave per (terminal, family) and stacks the result, giving matrices *independent* of the physical terminations.

```python
resp = sol.perturbation_response(np.linspace(50.0, 3000.0, 300),
                                 excite=("acoustic", "entropy"))  # 3x3 block: two acoustics + entropy
T = resp.transfer_matrix(0, 2)  # (n_freq, n, n) between edges 0 and 2
S = resp.multiport_scattering_matrix()  # rigorous whole-network terminal scattering matrix
resp.plot_transfer_matrix(0, 2).show()
```

Use `multiport_scattering_matrix()` when terminals straddle branches; `transfer_matrix(a, b)` assumes a serial path (check `resp.transfer_residual(a, b) ~ 0`).

**Freezing terminals to fold in closed branches.**
By default `perturbation_response` neutralizes every 1-port terminal into a measurement port.
Pass `freeze=[node, ...]` to instead hold the listed terminals at their *declared* boundary condition and fold them into the operator, so an interior branch terminated by a wall — a closed tuning stub, a side resonator, a Helmholtz neck — drops out of the port list and the network reduces to its genuine open ports.
A frozen terminal keeps whatever closure it carries: a `wall` is rigid (`R=+1`) by default, or the explicit `PerturbationBC` set on it.
This is how a muffler with tuning stubs reads out as a clean inlet→outlet two-port, and it is also how a *lossy* branch enters — an absorptive or yielding end wall is just a wall whose declared reflection is `R<1`, frozen like any other, with no manual condensation of the multiport matrix.

```python
# closed tuning stubs terminated by walls -> clean inlet->outlet two-port
resp = sol.perturbation_response(freqs, freeze=[wall_a, wall_b])
S = resp.acoustic_scattering_matrix(inlet_edge, outlet_edge)  # (n_freq, 2, 2)

# an absorptive end wall: declare the loss on the wall (at build, or before solve), then freeze it
wall_a = net.add(cat.wall(perturbation_bc=PerturbationBC.reflection(0.8)))
```

The declared closure is read from the network at solve time, so set the wall's `PerturbationBC` at construction or with `net.set_perturbation_bc(node, bc)` *before* `net.solve()`; changing it on an already-solved network does not take effect.

### 7c. Eigenmodes — linear stability

Free oscillations are the roots of `det A(omega)=0`, found by contour integration with a completeness certificate.
Sign convention: **growth rate = -Im(omega); a mode is unstable iff its growth rate > 0.**

```python
modes = sol.eigenmodes(freq_band=(20.0, 1000.0), growth_band=(-600.0, 600.0))
modes.freqs  # Hz
modes.growth_rates  # 1/s  (>0 unstable)
modes.unstable  # bool mask
modes.certified  # True if the count found == the argument-principle expected count
modes.summary()  # per-mode dicts
modes.plot_spectrum().show()
modes.plot_mode(0).show()  # static mode shape
modes.animate_mode(0).show()  # animated over one cycle
modes.boundary_power(0)  # where energy enters/leaves for mode 0
```

**Convected-wave controls.** `isentropic=True` removes *both* convected families (entropy and composition) at once; `convected=` removes them one at a time, so a mode's damping or drive can be attributed to its carrier — temperature spots versus mixture-ratio spots:

```python
sol.eigenmodes(freq_band=..., convected="all")  # full operator (default)
sol.eigenmodes(freq_band=..., convected="entropy")  # entropy convects, composition frozen
sol.eigenmodes(freq_band=..., convected="composition")  # composition convects, entropy pinned
sol.eigenmodes(freq_band=..., convected="none")  # identical to isentropic=True
sol.nyquist_stability(freqs, convected="entropy")  # same knob on the real-frequency driver
```

Always check `modes.certified`: `True` means the number found matched the argument-principle count (a completeness guarantee); `False` means the count could not be confirmed and the solver **warns why** — it never silently returns a partial spectrum.
The common uncertified cases are a mode sitting on the search-region boundary (shift/widen `freq_band` or `growth_band`) and near-degenerate (repeated) modes.
One genuine limit: at **very low mean-flow Mach** the entropy characteristic ill-conditions the operator, so a choked or near-quiescent mean flow may yield an uncertified or empty result with a warning.
There the acoustic modes are still well-defined — recover them with `sol.eigenmodes(..., isentropic=True)` (drops the entropy wave, whose coupling is negligible at low Mach), or read the resonances off `sol.forced_response(freqs)` peaks.

### 7c-bis. Eigenvalue sensitivities — what moves each mode

Differentiate every found mode against the network's parameter inventory in one pass (one left eigenvector per mode, one operator re-assembly per parameter — no re-solve, no re-search).
Both routes are captured: the parameter's direct appearance in the operator and the mean-flow shift it causes.

```python
sens = modes.sensitivities()  # every scalar parameter; assembly settings carried over
sens = modes.sensitivities(include="*.length", exclude="*.mdot")  # glob narrowing
sens = modes.sensitivities(params=["Duct3.length", "plenum.volume"])  # explicit list
sens = modes.sensitivities(layer="perturbation")  # only acoustic-layer knobs (FTF gain/lag, BCs, volumes)
sens  # ranked table: growth-rate change per +1% of each parameter (positive = destabilizing)
sens["Duct3.length"]  # d omega / d p column for one address, one entry per mode
sens.dgrowth_dp, sens.dfreq_dp  # (n_modes, n_params) derivative matrices, per parameter unit
sens.top(5)  # most influential addresses
sens.failed  # parameters that could not be probed, with reasons
sens.plot().show()  # ranked bar chart (nefes.plotting.plot_sensitivities)
```

A zero-valued parameter (an unset `volume` or `end_correction`) is probed with a small absolute step and ranked by that step's effect, so untapped geometric features still show their leverage.
The derivatives are local slopes: confirm a finite design change with a trajectory (7d).
A near-degenerate pair warns — such modes respond to a parameter by splitting, and their individual slopes are ill-conditioned.

### 7d. Continuation of the spectrum in a parameter

Track each eigenvalue as a parameter varies, seeded once and predictor-corrector continued.

```python
traj = net.eigenvalue_trajectory("flame.Qdot", np.linspace(1e3, 5e3, 21),
                                 freq_band=(20.0, 1000.0), growth_band=(-600.0, 600.0))
traj.branches  # list of TrajectoryBranch; each has .freqs, .growth, .omega
traj.plot_vs_param().show()
```

`net.eigenvalue_trajectory` builds each swept point from a pristine copy (via `net.builder`) and labels the sweep by the address.
The free `eigenvalue_trajectory(build, ...)` accepts a custom `build(p)` only for structural changes that `with_params` cannot express (see §5).

### 7e. Nyquist stability map

Real-frequency return-ratio stability for the entropy / tabulated-FTF regime (needs at least one dynamic source).

```python
nq = sol.nyquist_stability(np.linspace(1.0, 1000.0, 500))
nq.n_unstable  # encirclement count
nq.stable  # bool
nq.plot().show()

sweep = net.nyquist_stability_map("flame.n", np.linspace(0.1, 2.0, 30), np.linspace(1.0, 1000.0, 500))
sweep.onsets  # bracketed intervals where the unstable count changes
```

---

## 8. Dynamic sources (flame / injector response)

A dynamic source is the frequency response `S(omega)` of a fluctuating source term (heat release, injected mass) that couples to a flow quantity read at a **reference edge**.
The mean flow ignores it; only the perturbation layer consumes it.
This feedback is what drives thermoacoustic instability.

Because the reference edge id is known only after `connect`, attach the source *after* wiring.

```python
from nefes.elements import n_tau_flame

net = nefes.Network()
n_cold  = net.add(cat.duct(0.3))
n_flame = net.add(cat.heat_release_flame(50e3))
e_ref   = net.connect(n_cold, n_flame, 0.05)  # velocity reference just upstream of the flame
net.connect(n_flame, net.add(cat.pressure_outlet(101325.0)), 0.05)

net.set_dynamic_source(n_flame, n_tau_flame(n=1.0, tau=2.5e-3, ref_edge=e_ref))
```

Builders (`nefes/elements/dynamic_source.py`), all attach via `set_dynamic_source` or an element's `dynamic_source=`:

| Builder | Response `F(f)` |
| :-- | :-- |
| `n_tau_flame(n, tau, ref_edge, quantity="u")` | n-tau heat-release flame `n·e^{-i2πfτ}` |
| `heat_release_response(transfer, ref_edge, ...)` | general heat-release FTF (any `TransferFunction`) |
| `mass_flow_response(transfer, ref_edge, ...)` | modulated mass injection (e.g. fuel feed) |
| `n_tau(n, tau)` / `n_tau_lowpass(n, tau, fc)` / `n_tau_lowpass2(n, tau, fc, zeta)` | bare / roll-off n-tau shapes |
| `finite_impulse_response(h, dt)` | sampled impulse response (pole-free, entire) |
| `constant(value)` / `tabulated(freqs, values)` | flat gain / real-axis table |

`quantity` is the referenced flow fluctuation (`"u"`, `"p"`, `"rho"`, `"mdot"`, or `"Z:<scalar>"`); `gain` must be real (put complex weighting inside the transfer function).

---

## 9. Analytic continuation of tabulated data

The eigensolver evaluates off the real frequency axis, so a *measured* flame transfer function or reflection coefficient (known only on a real grid) must be turned into an analytic model first.
A plain table (`tabulated`) is real-axis only and cannot go to the eigensolver.

Two routes; choose by the physics of the response:

**Impulse-response route (recommended default) — for finite-memory responses.**
A flame forgets a velocity disturbance after a convective time; compact elements have no internal resonator.
The fit is a finite sum of pure delays, so it has **no poles anywhere** and cannot put anything artificial in a stability window.

Both continuation routes live under `nefes.perturbation`:

```python
from nefes.perturbation import fit_impulse_response, rational_fit, continuation_warning
from nefes.elements import heat_release_response

fir = fit_impulse_response(f_meas, F_meas, duration=20e-3)  # F_meas complex, on real grid f_meas
net.set_dynamic_source(n_flame, heat_release_response(fir, ref_edge=e_ref))
```

**Rational route — for responses that ring** (cavity dampers, resonant end plates).

```python
fit = rational_fit(f_meas, F_meas, delay="auto")  # peel the pure delay before AAA fitting
continuation_warning(fit, freq_band=(5.0, 495.0), growth_band=(-340.0, 170.0))  # warn if poles fall in the window
```

A rational fit genuinely has poles; check `fit.poles_in_region(...)` / `continuation_warning(...)` before trusting the eigenmodes near them.
Both `fir` and `fit` are analytic and drop straight into `heat_release_response`, `mass_flow_response`, or `PerturbationBC.reflection`.

---

## 10. Identifying an unknown element response

De-embed a buried element (a flame inside a combustor) from a network-wide measured transfer matrix.

```python
from nefes.perturbation import identify_transfer_function

ident = identify_transfer_function(sol, measured,  # measured: a TransferMatrix between edges a, b
                                   node=n_flame, a=e_in, b=e_out)
ident.transfer_functions  # recovered FTF(s)
ident.conditioning  # identifiability diagnostic (large -> ill-posed)
```

`identify_transfer_matrix(...)` returns the matrix form; both take the same `node`, `a`, `b` addressing.

---

## 11. Composite elements and acoustic refinement

A composite presents as one element and expands into a small atomic graph at build time; the solver, Jacobian, and perturbation layers never see the composite.
For distributed elements (a horn, a con-di nozzle, a friction pipe), the mean flow is exact at any segment count but the *acoustic* response converges as `O(1/N)`.
Size the segmentation for the top frequency, or drive it automatically:

```python
from nefes.elements import auto_refine

n = cat.segments_for_frequency(length=0.5, sound_speed=340.0, f_max=3000.0)  # smallest N keeping segments compact
net = nefes.Network(nodes=[..., cat.tapered_duct(area_profile, length=0.5, n_segments=n), ...], edges=[...])

# or refine until a quantity of interest stops moving:
report = auto_refine(build=lambda N: my_network(N), n_start=4, probe=lambda sol: sol.field("M").max(), tol=1e-2)
```

---

## 12. Plotting

Plotly only, with the bundled Nefes theme, which comes in a light mode (on white) and a dark mode matching the documentation pages and the Nemo interface.
Every figure built by `nefes.plotting` carries the theme already, so nothing has to be called for a themed plot.
Switch modes with `set_theme`, which also makes the theme Plotly's default template, so figures assembled by hand match:

```python
from nefes.plotting import set_theme, palette
set_theme("dark")  # or "light" (the default)
```

For hand-built traces, `palette()` returns the colours of the active mode: `palette().colorway` (the categorical series colours, in draw order), `.accent`, `.ink`, `.muted`, `.rule`.
`COLORWAY` remains the light-mode series list for code that wants a fixed palette.

The analysis objects carry their own plotters (`modes.plot_spectrum()`, `resp.plot_transfer_matrix(a, b)`, `fr.plot_response()`, `traj.plot_vs_param()`, `fit.plot_fit()`).
The standalone helpers in `nefes.plotting` take raw arrays when you want a custom figure:

| Function | Consumes |
| :-- | :-- |
| `plot_network_topology(net, solution=sol, color_by="T")` | a network (+ optional solution) |
| `plot_spectrum(freqs, growth_rates)` | an eigenvalue solve |
| `plot_transfer_matrix(matrices, freqs)` / `plot_scattering_matrix(...)` | a response matrix |
| `plot_transfer_function(funcs, freqs, nyquist=False)` | a scalar FTF (Bode or Nyquist) |
| `plot_mode_shape(shape)` / `animate_mode_shape(series)` | a reconstructed mode field |
| `plot_pole_map(fit)` / `plot_fit(fit)` | a continuation fit |

`net.plot()` and `sol.plot()` are the topology diagram directly from the object.
For hand-built `go.Figure`s, pull series colors from `COLORWAY`; the theme supplies axes, grid, and fonts.
Do not inject offline-Plotly or MathJax `<script>` blocks into notebooks; `use_nefes_theme()` and the standard renderer are enough.

---

## 13. Saving and loading

```python
sol.save("run.yaml")  # topology + embedded mean-flow fields (alias of sol.to_yaml)
sol.save("run.yaml", fields=["mdot", "p", "M", "p_t"], node_data=True)

net2 = nefes.load_case("run.yaml")  # -> Network
sol2 = net2.solve()

sol3 = nefes.load_solution("run.yaml")  # -> Solution, restoring the embedded field (method="warm" re-verifies)
```

`Solution.to_yaml` appends a dataset when the file already holds the same network, so multiple operating points overlay in one file.
`net.to_yaml(path)` writes topology only.
Programmatic serialization uses `nefes.save_case` / `nefes.save_solution` / `nefes.io.dump_case`.

A reacting save also embeds a companion `"<dataset> chemistry"` dataset by default: the transported feed-stream mixture fractions (`xi:<stream>`), the per-edge burnt marker (`burnt`, `0` fresh / `1` burnt), and each species' mass fraction (`Y:<species>`).

---

## 14. Troubleshooting

Nefes fails loudly and rarely: it validates ill-posed setups before solving, the Newton solve is globalized (artificial-resistance continuation plus a Levenberg-Marquardt fallback), and the reacting recovery and boundary-pressure seed reach extreme operating points from a cold start.
When something does go wrong it is almost always one of a few categories, each with a specific tell.

**When `solve()` raises** — an ill-posed setup, caught before or early in the Newton loop, with a specific message: no absolute-pressure reference (two flow-fixing boundaries and nothing pinning pressure), a non-finite or non-positive edge area, a disconnected sub-network, an unknown element type, a supersonic boundary (deferred), or an unknown composition species.
Read the message; the fix is the network, not the solver.

**When `solve()` returns `converged=False`** — the solve ran but did not reach `tol`. Inspect rather than guess:

```python
sol.residual_norm  # how far off
sol.print_residuals()  # which equation (mass / pressure / energy / a scalar) is unmet, by node
sol.result.history  # the discriminator (below)
```

The residual `history` tells you which of two very different situations you are in:

- **still decreasing** → it was converging, just ran out of road: warm-start `x0=prev.x` from a nearby solved case (or `parameter_study(warm_start=True)`), or raise `max_iter`.
- **flat / stalled** → the demand is physically infeasible in subsonic flow (more mass flux than a throat can pass, a heat release the flow cannot absorb). No amount of iterating helps; change the setup — `print_residuals()` names the equation that cannot be met.

High-pressure reacting cold-starts now converge on their own (the seed reads the boundary pressures), so warm-starting is an accelerant, not a necessity.

**Subsonic scope is enforced automatically.** The steady rows admit a spurious *supersonic* isentropic branch beside the physical subsonic one, and a cold start can occasionally land on it. `solve()` guards against this: it detects a genuinely supersonic edge and re-solves once from a near-stagnation seed that reaches the subsonic branch, so a bare `net.solve()` returns the subsonic root with no hand-seeding (real choking is untouched — a throat still pins at M=1). It is on by default; opt out with `nefes.config.enforce_subsonic = False` (or `net.solve(enforce_subsonic=False)`) for the deferred supersonic work. If a case is *genuinely* supersonic in the mean, the re-solve cannot remove it and `solve()` warns rather than returning it silently.

**When an acoustic analysis raises or will not certify**

- *"non-subsonic / degenerate terminal"* or *"mean Mach >= 1"* — the mean flow is sonic somewhere and the plane-wave split is degenerate at M=1. Model a choked exit with the compact `choked_nozzle_outlet` (which never resolves the sonic point), not a resolved M=1 throat.
- `eigenmodes` returns `certified=False` with a warning — read it: a mode on the search-region boundary (shift/widen `freq_band`/`growth_band`), near-degenerate modes, or the low-Mach entropy limit (use `isentropic=True`, or read resonances off `forced_response` peaks). The modes found are still meaningful; only the completeness certificate is withheld.

**Analytic-continuation fits — check the fit, not just the call.** Both routes succeed silently even on unsuitable data, so verify quality:

- `fit_impulse_response(...).rms_misfit` — a high value means the memory `duration`/`dt` under-resolves the response (a sharply-delayed FTF needs a finer `dt`; a *ringing* response needs the rational route instead).
- `rational_fit(...)` on noisy data can scatter spurious poles into the search band — check `fit.poles_in_region(freq_band, growth_band)` (or `continuation_warning(...)`) before trusting eigenmodes near them.

**Results computed but flagged.** Some outcomes are valid but carry a warning worth heeding: an unchoked `choked_nozzle_outlet` (`sol.unchoked_nozzles()`), a plane-wave cut-on exceeded (`sol.cuton_report()`), an ill-conditioned identification (`ident.conditioning`). The result is correct for the model; the warning says the assumption is being stretched.

**The one rule.** Nefes does not silently return a wrong answer: a bad setup raises with a message, a hard problem returns an inspectable `converged=False`, and a stretched assumption warns. A number returned with no warning is trustworthy.

---

## Anti-patterns (what older notebooks do -> what to do instead)

| Found in notebooks | Use instead | Why |
| :-- | :-- | :-- |
| `from nefes.shell import build_problem` | `nefes.Network(gas, nodes, edges)` | the one-shot constructor supersedes it |
| parameterized `def build(Lm=..., drive=...)` around construction | build once; `with_params` / `set` / `eigenvalue_trajectory("address", ...)` | scalar and nested knobs (including perturbation-layer) are already addressable; reserve a custom `build(p)` for topology / element-kind / gas changes (§5) |
| `from nefes.solver import solve; solve(prob)` | `net.solve()` | Solution auto-verifies and gives named access |
| `from nefes.solver.report import states_table` + `ES_M`/`ES_P` indexing | `sol.field("M")`, `sol.edge(i)`, `sol.print_states()` | `ES_*` are internal state indices |
| `from nefes.perturbation.operator.boundary_bc import PerturbationBC` | `from nefes.perturbation import PerturbationBC` | it is re-exported one level up |
| `PerturbationBC.excitation(1.0)` | `PerturbationBC.anechoic(driven=("acoustic",))` | `excitation` does not exist |
| `sys.path.insert(...)` bootstrap + `import os, sys` | `pip install -e .` | run against the installed package |
| `plotly.offline` + `display(HTML(<mathjax script>))` | `use_nefes_theme()` | the offline/MathJax block clashes with the site and is stripped at doc build |
| `Solution(network, prob, res)` hand-construction | `net.solve()` | the constructor is internal plumbing |
