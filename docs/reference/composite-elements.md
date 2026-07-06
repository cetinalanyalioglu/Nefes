# Composite elements

A **composite element** presents to the user as a single element but expands, at
build time, into a small graph of [atomic elements](atomic-elements.md) joined by internal
edges. The expansion (`nefes.elements.composite.expand_composites`) is a pure graph
transformation run once at the top of `build_problem`, so the solver, the Jacobian
assembly and the perturbation layer never see a composite — at solve time an expanded graph
is indistinguishable from a hand-built one (no new kernels, no solver changes). See
`scratch/composite-elements.md` for the design theory.

Two classes ship:

- **Class 1 — fixed macro recipe.** A physical component that is exactly a short, fixed chain
  of atoms (orifice, lossy nozzle, sudden contraction, Helmholtz resonator).
- **Class 2 — discretization.** A continuously varying 1-D element resolved as an $N$-segment
  chain; $N$ is a fidelity knob and grid refinement (solve at $N$ and $2N$) *is* the
  verification (`fanno_pipe`, `tapered_duct`).

## The catalogue

| Composite | Class | Expands to (atomic sub-elements) | Internal edge(s) → area | Governing theory | Reads out |
| --- | --- | --- | --- | --- | --- |
| `orifice` | 1 | `isentropic_area_change`, `sudden_area_change` | iac→sac → $A_T$ | De Domenico (2019) orifice = max-loss limit: isentropic $A_1\to A_T$, then Borda–Carnot $A_T\to A_2$ | throat state |
| `lossy_nozzle` | 1 | `isentropic_area_change` ×2, `sudden_area_change` | iac0→iac1 → $A_T$; iac1→sac → $A_j$ | De Domenico nozzle $A_1\to A_T\to A_j\to A_2$; $\beta$ sweeps loss | throat, jet states |
| `sudden_contraction` | 1 | `isentropic_area_change`, `sudden_area_change` | contract→borda → $A_\text{vc}$ | Vena contracta: neck to $c_c A_2$ (min static $p$), then Borda loss | vena-contracta state |
| `helmholtz_resonator` | 1 (side-branch) | `junction`, `duct`, `cavity` | tee→neck → $A_n$; neck→cavity → $A_n$ | Neck inertance vs cavity compliance | neck, cavity states |
| `fanno_pipe` | 2 | $N$ × `pipe` | seg$_i$→seg$_{i+1}$ → $A=\pi D^2/4$ | Fanno flow: friction drives $M\to 1$; $N$-atom chain → distributed solution | per-segment states |
| `tapered_duct` | 2 | $N$ × (`isentropic_area_change`, `duct`) | see below | Area-varying horn / con-di from an $(x,A)$ table (or $A(x)$); chokes at true throat | per-segment states |

## Wave propagation inside a composite

Each internal edge is a genuine intermediate flow state, so in the perturbation problem it is
a real characteristic edge. Whether a composite **propagates acoustic waves** through its
interior depends on whether its atoms carry the phase stamp $\mathbf{P}(\omega)$:

- `duct` and `pipe` carry $\mathbf{P}(\omega)$ (wavenumber $k=\omega/c$, propagation $\propto e^{\pm ikL}$).
- `isentropic_area_change` / `sudden_area_change` are **compact** (lengthless) by default —
  they contribute only through $\overline{\mathbf{J}}$ (and $\mathbf{M}$ if given storage lengths).

So:

- **`tapered_duct` propagates waves** — each of its $N$ segments contains a length-$L/N$
  `duct`, so the horn's internal acoustics are resolved through $N$ real ducts (the
  area-change atoms handle the compact area jumps; the ducts carry the phase). This is exactly
  why it resolves horn / con-di acoustics that a single lumped area jump cannot.
- **`fanno_pipe` propagates waves** — its `pipe` atoms carry $\mathbf{P}(\omega)$ as well as friction.
- The Class-1 macros (`orifice`, `lossy_nozzle`, `sudden_contraction`) are built from compact
  atoms, so they are acoustically compact overall — their internal edge is a state, but no
  distributed phase accrues across it (the intended, correct behaviour for a compact orifice).
- `helmholtz_resonator` propagates along its neck `duct` (the inertance) into the cavity
  compliance.

## Recipes and parameters

Notation: $A_1$/$A_2$ = the external upstream/downstream **edge** areas (set when wiring the
composite's edges), `[e]` = an internal edge and its area. Every constructor also takes
`name`.

### `orifice(throat_area, eps=None)`
De Domenico maximum-loss plate — isentropic contraction to the throat, then a Borda
re-expansion.

$$A_1 \xrightarrow{\text{isen}} A_T \xrightarrow{\text{Borda}} A_2.$$

| Argument | Symbol | Meaning | Units | Default / constraint |
| --- | --- | --- | --- | --- |
| `throat_area` | $A_T$ | throat (vena-contracta plane) area | m² | required, $>0$ (and $<A_1,A_2$) |
| `eps` | $\varepsilon$ | sharpens the embedded Borda direction switch | kg/s | `None` |

### `lossy_nozzle(throat_area, beta, downstream_area, eps=None)`
The general De Domenico nozzle; `orifice` and the lossless nozzle are its endpoints.

$$A_1 \xrightarrow{\text{isen}} A_T \xrightarrow{\text{isen}} A_j=\beta A_2 \xrightarrow{\text{Borda}} A_2, \qquad \beta \in \left[\tfrac{A_T}{A_2},\, 1\right].$$

$\beta = A_T/A_2$ → orifice (max loss); $\beta = 1$ → lossless (the Borda becomes $A_2\to A_2$,
its loss vanishes).

| Argument | Symbol | Meaning | Units | Default / constraint |
| --- | --- | --- | --- | --- |
| `throat_area` | $A_T$ | throat area | m² | required, $>0$ |
| `beta` | $\beta$ | jet-to-downstream area ratio $A_j/A_2$ | — | required, in $[A_T/A_2,\,1]$ |
| `downstream_area` | $A_2$ | downstream edge area (must match the outflow edge) | m² | required, $>0$ |
| `eps` | $\varepsilon$ | sharpens the Borda switch | kg/s | `None` |

### `sudden_contraction(downstream_area, cc=0.62, eps=None)`
Resolves the vena contracta explicitly — the compressible upgrade to `sudden_area_change`'s
$c_c$-loss (exact loss **and** minimum static pressure at higher Mach).

$$A_1 \xrightarrow{\text{isen}} A_\text{vc}=c_c A_2 \xrightarrow{\text{Borda}} A_2.$$

Read the minimum static pressure off the throat edge (`solution.composite(name).throat_state`).

| Argument | Symbol | Meaning | Units | Default / constraint |
| --- | --- | --- | --- | --- |
| `downstream_area` | $A_2$ | downstream pipe area (must match the outflow edge) | m² | required, $>0$ |
| `cc` | $c_c$ | vena-contracta contraction coefficient | — | `0.62`, in $(0,1]$ |
| `eps` | $\varepsilon$ | sharpens the Borda switch | kg/s | `None` |

### `helmholtz_resonator(volume, neck_length, neck_area)`
Side-branch resonator (the only non-serial recipe): the main line passes straight through the
tee (`upstream_sub = downstream_sub = 0`); the neck and cavity hang off as an internal branch.
Resonates where the neck inertance meets the cavity compliance:

$$f_0 = \frac{c}{2\pi}\sqrt{\frac{A_n}{V\, l_n}}.$$

```
main line: --> (tee) -->
                  |  [An]
                (neck)
                  |  [An]
               (cavity)
```

| Argument | Symbol | Meaning | Units | Default / constraint |
| --- | --- | --- | --- | --- |
| `volume` | $V$ | backing cavity volume | m³ | required, $>0$ |
| `neck_length` | $l_n$ | neck length (acoustic inertance) | m | required, $>0$ |
| `neck_area` | $A_n$ | neck cross-sectional area | m² | required, $>0$ |

### `fanno_pipe(length, diameter, friction_factor, n_segments)`
$N$ equal `pipe` friction segments, each length $L/N$ and constant area $A=\pi D^2/4$:

$$(A) \;\text{--seg}_0\text{--}\; [A] \;\text{--seg}_1\text{--}\; [A] \;\cdots\; [A] \;\text{--seg}_{N-1}\text{--}\; (A).$$

Friction drives the subsonic flow toward $M=1$; as $N$ grows the chain converges to the true
Fanno solution and can approach exit choke. `n_segments = 1` degenerates to a single `pipe`
(not a composite).

| Argument | Symbol | Meaning | Units | Default / constraint |
| --- | --- | --- | --- | --- |
| `length` | $L$ | total pipe length | m | required, $>0$ |
| `diameter` | $D$ | hydraulic diameter | m | required, $>0$ |
| `friction_factor` | $f$ | Darcy friction factor | — | required, $\ge 0$ |
| `n_segments` | $N$ | segment count (fidelity knob) | — | required, $\ge 1$ |

### `tapered_duct(area, length, n_segments=None)`
An area-varying passage / horn / con-di nozzle from a profile $A(x)$ or a station table. Each
segment is an `isentropic_area_change` $A_i\to A_{i+1}$ **followed by** a length-$L/N$ `duct`
at the segment's downstream area (the catalog has no length-bearing area-change atom, so a
segment is two atoms). `upstream_sub = 0`, `downstream_sub = 2N-1`.

$$(A_0) \xrightarrow{\text{iac}_0} [A_1] \xrightarrow{\text{duct}_0} [A_1] \xrightarrow{\text{iac}_1} [A_2] \xrightarrow{\text{duct}_1} [A_2] \;\cdots\; (A_N).$$

Because each segment carries a real `duct` (spanning its own station interval), the taper
**propagates waves** (see above); a con-di profile chokes at its true min-area throat edge.

**Standard input is an $(x, A)$ table.** The axial positions $x_i$ [m] (strictly increasing)
set the station spacing, which **may be non-uniform** — cluster stations where the area varies
fastest (e.g. near a throat) and each segment's duct spans its own interval
$\Delta x_i = x_{i+1}-x_i$. The total length is **inferred** as $x_N - x_0$; a passed `length`
is only checked against that span. A callable $A(x)$ is also accepted — sampled at $N+1$
equispaced stations over $[0, L]$ (then `length` and `n_segments` are required).

| Argument | Symbol | Meaning | Units | Default / constraint |
| --- | --- | --- | --- | --- |
| `area` | $\{(x_i, A_i)\}$ or $A(x)$ | table of `(position, area)` pairs (x strictly increasing), or a callable | m, m² | required, all $A>0$; edges carry $A_0, A_N$ |
| `length` | $L$ | total axial length | m | **required for a callable**; for a table inferred as $x_N-x_0$ (checked if given) |
| `n_segments` | $N$ | segment count | — | **required for a callable**; for a table it is $\text{len(table)}-1$ |

## Sizing a Class-2 chain

- `segments_for_frequency(length, sound_speed, f_max, points_per_wavelength=12)` → the
  smallest $N$ that keeps each segment acoustically compact at $f_\text{max}$:

  $$N \ge P\, \frac{f_\text{max}\, L}{c},$$

  with $P$ points per wavelength (default 12).
- `grid_refine(build, N, probe)` solves at $N$ and $2N$ and reports the relative change in the
  quantities of interest — a converged refinement *is* the verification of the discretization.

## How composites interact with the rest of Nefes

**Centralized, element-agnostic machinery.** One expander (`expand_composites`) serves every
composite; it knows only their connectivity, never what the sub-elements are. Adding a new
composite means writing a constructor that returns a `CompositeElementSpec` — no solver,
Jacobian or perturbation code changes.

**Numbering — append, never insert.** A composite's first sub-element keeps the composite's own
node id; the remaining subs are appended at the tail, and internal edges are appended after all
user edges. So every *user* node and edge id keeps its exact meaning after expansion, and
because the user-facing API is edge-indexed (`states_table`, `transfer_matrix`,
`scattering_matrix`), captured indices stay valid for free. A `CompositeMap` on the compiled
problem hides the internals by default (`internal_nodes`, `internal_edges`) yet lets you read
intra-element states on demand (`solution.composite(name)`).

**Perturbation.** Because expansion happens before the perturbation layer, each internal edge
is a genuine intermediate flow state. A compact macro (orifice) recovers its compact scattering
matrix by composing its sub-elements through the internal edge (validated against an
independent Borda composition and the De Domenico / Cambridge references in
`tests/test_perturbation_dedomenico.py`); a Class-2 chain carries its $N$ internal edges — with
their $\mathbf{P}(\omega)$ ducts — so the acoustic response is the distributed (non-compact) one,
converged via `grid_refine`.

## UI (FNetLibUI) round trip

A composite serializes to the UI case format as the **single node the user specified**, never its
expanded internals: the writer (`yaml_out`) maps `CompositeElementSpec.kind` plus the factory
parameters retained on `CompositeElementSpec.params` to one UI node
(`Orifice`, `LossyNozzle`, `SuddenContraction`, `HelmholtzResonator`, `FannoPipe`, `TaperedDuct`),
and the loader (`yaml_in`) rebuilds it through the same catalog factory.
The UI's *Composite elements* palette category carries the same six elements, so a case authored
in either place round-trips.
Two conventions to know:

- a `fanno_pipe` with `n_segments = 1` short-circuits to a plain `pipe` atom and therefore
  serializes as a `Pipe` node;
- a `tapered_duct` built from a callable $A(x)$ serializes as its **resolved station table**
  (the `areaProfile` string), which reloads deterministically.

The port-explicit build path used by UI-loaded cases expands composites too: user port pins
survive at atomic endpoints, while the expansion re-derives flow-aligned ports on the rewired
sub-elements (`expand_composites(..., ports=...)`).
Round-trip coverage lives in `tests/test_composite_ui_io.py`.

## Known limitations (v1)

- **No nesting:** a composite's sub-elements must be atomic (`validate_composite` rejects a
  composite sub-element).
- **UI export needs a catalog kind:** a hand-built `CompositeElementSpec` with a bespoke `kind`
  (or without `params`) still raises on YAML serialization — only the catalog recipes above have
  a UI node type.
- **Serial + single side-branch only:** Class-3 branching sub-network composites are deferred.
- **Numbering is tail-append, not bandwidth-optimal:** SuperLU re-permutes internally so this
  costs nothing at solve time; a bandwidth-aware renumber is a deferred refinement.
