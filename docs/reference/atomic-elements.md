# Atomic elements

An **atomic element** is a single, irreducible network element with its own residual
kernel — as opposed to a [composite](composite-elements.md), which expands to a graph of
atomics at build time. (The codebase calls them *atomic*; *base* / *primitive* would be
synonyms.) Every element maps to one integer `residual_id` — the `@njit` dispatch key in
`nefes.elements.ids` — and is constructed from `nefes.elements.catalog`.

## How an element is defined

State lives on **edges** (intermediate flow states); an element is a **node** that imposes
residual equations coupling its incident edges. The band-1 unknowns per edge are the mass
flow $\dot m$, the static pressure $p$ and the total enthalpy $h_t$ (plus transported
scalars — mixture fractions, the burnt marker). An element of degree $d$ (its port count)
contributes:

- **1-port boundary/termination** — a single row: either a mass-flux row ($\dot m$ pinned) or
  an absolute-pressure row ($p$ or $p_t$ pinned).
- **interior element** ($d \ge 2$) — one **mass-balance** row plus $d-1$ **pressure-coupling**
  rows (the row-kind map is `nefes.elements.ids.row_kind_tags`, the single source of truth).

The **flow area** is a property of each *edge*, not the element: edges are wired as
`(tail_node, head_node, area)`, so an area-change element simply reads the differing areas of
its two incident edges. Every element also takes a `name` (a display label; sub-elements of a
composite are namespaced under it).

All residual math is **complex-step-safe** (smooth, complex-analytic, no `abs`/`min`/`max`/
branches on the flow state); Jacobians come from complex-step differentiation.

**Acoustic contribution.** In the perturbation problem the operator is

$$\mathbf{A}(\omega) = \overline{\mathbf{J}} + \mathrm{i}\omega\, \mathbf{M} + \mathbf{P} + \mathbf{S}.$$

Most elements contribute only through $\overline{\mathbf{J}}$ (the complex-step linearization of their
mean residual). Three add blocks beyond that default: `duct`/`pipe` add the phase-propagation
stamp $\mathbf{P}(\omega)$; `cavity` (and a plenum `junction`/`splitter`) add the storage block $\mathbf{M}$;
the flames carry an unsteady heat-release source $\mathbf{S}(\omega)$ on their `DynamicSource`
descriptor. Single-port boundaries can also carry an explicit `PerturbationBC`
(reflection/impedance); left at `None` they inherit the linearization of their own mean row.

## The catalogue

| Element | `residual_id` | Ports | Imposes (mean residual) | Acoustic contribution |
| --- | --- | --- | --- | --- |
| `mass_flow_inlet` | `MASS_FLOW_INLET` (0) | 1 | $\dot m = \dot m_\text{spec}\ge 0$; feeds $h_t(T_t)$, composition | inherited ($\dot m'=0$) |
| `total_pressure_inlet` | `PT_INLET` (1) | 1 | $p_t = p_{t,\text{spec}}$; feeds $h_t(T_t)$, composition | inherited |
| `pressure_outlet` | `P_OUTLET` (2) | 1 | $p = p_\text{spec}$; backflow stream on ingestion | inherited |
| `mass_flow_outlet` | `MASS_FLOW_OUTLET` (15) | 1 | $\dot m = \dot m_\text{spec}>0$; $p$ floats | inherited ($\dot m'=0$) |
| `choked_nozzle_outlet` | `CHOKED_NOZZLE_OUTLET` (16) | 1 | $\dot m = \dot m^{*}(p_t,T_t,A^{*})$ | inherited (Marble–Candel) |
| `wall` | `WALL` (11) | 1 | $\dot m = 0$ | hard wall $R=+1$ |
| `cavity` | `CAVITY` (18) | 1 | $\dot m = 0$ (wall to mean flow) | storage $\mathbf{M}$, $C=V/(\varrho c^2)$ |
| `isentropic_area_change` | `ISEN_AREA_CHANGE` (3) | 2 | mass + $p_{t,0}=p_{t,1}$ across an area change | default (+$\mathbf{M}$ if length-bearing) |
| `sudden_area_change` | `SUDDEN_AREA_CHANGE` (4) | 2 | mass + Borda–Carnot / vena-contracta loss | default |
| `loss` | `LOSS` (5) | 2 | mass + $\Delta p_t = K\cdot\tfrac12\varrho u^2$ | default (+$\mathbf{M}$ if length-bearing) |
| `linear_resistance` | `LINEAR_RESISTANCE` (17) | 2 | mass + $\Delta p_t = R\,\dot m$ | default (+$\mathbf{M}$ if length-bearing) |
| `duct` | `DUCT` (8) | 2 | mass + equal-area $p_t$ continuity | phase $\mathbf{P}(\omega)$ |
| `pipe` | `PIPE` (20) | 2 | mass + Darcy–Weisbach $K=fL/D$ | phase $\mathbf{P}(\omega)$ |
| `heat_release_flame` | `FLAME_HEAT_RELEASE` (12) | 2 | mass + $p_t$ continuity, $\Delta h_t=\dot Q/\dot m$ | default (+$\mathbf{S}(\omega)$ if dynamic) |
| `equilibrium_flame` | `FLAME_EQUILIBRIUM` (13) | 2 | mass + static-$p$ + $h_t$ + $Z$ conserved | default (+$\mathbf{S}(\omega)$ if dynamic) |
| `mass_source` | `MASS_SOURCE` (14) | 2 | mass/momentum/energy/composition injection | default |
| `junction` | `JUNCTION` (6) | variable | common **static** pressure $p_i=p_0$ | default (+$\mathbf{M}$ if plenum) |
| `splitter` | `SPLITTER` (7) | variable | common **total** pressure $p_{t,i}=p_{t,0}$ | default (+$\mathbf{M}$ if plenum) |
| `forced_splitter` | `FORCED_SPLITTER` (19) | variable | $\dot m_{\text{out},k}=\beta_k\dot m_\text{in}$ | default |
| *supersonic inlet/outlet* | (9 / 10) | 1 | **reserved — deferred** (v1 is subsonic) | — |

Below, the parameter tables list only each element's own arguments; every constructor also
accepts `name` (a label). Optional acoustic hooks — `perturbation_bc` (1-port boundaries) and
`dynamic_source` (flames, source) — are noted per element.

## Boundaries

### `mass_flow_inlet(mdot, Tt, composition=None, basis="mole", marker=0.0)`
Pins the edge mass rate and feeds a stream: an **inflow boundary** with $\dot m \ge 0$
(negative is rejected — use `pressure_outlet` for a reversing boundary).

$$\dot m = \dot m_\text{spec}, \qquad h_t = h_t(T_t,\ \text{composition}).$$

| Argument | Symbol | Meaning | Units | Default / constraint |
| --- | --- | --- | --- | --- |
| `mdot` | $\dot m_\text{spec}$ | prescribed inflow mass rate | kg/s | required, $\ge 0$ |
| `Tt` | $T_t$ | feed total temperature | K | required |
| `composition` | — | feed species mixture, e.g. `{"O2":0.21,"N2":0.79}` | — | `None` (perfect-gas) |
| `basis` | — | units of `composition` | `"mole"` / `"mass"` | `"mole"` |
| `marker` | $b$ | injected burnt-marker ($0$ fresh, $1$ burnt) | — | `0.0` |

### `total_pressure_inlet(pt, Tt, composition=None, basis="mole", marker=0.0)`
Pins the edge total pressure (the mass rate emerges from the interior) — the natural inlet
when a reservoir total pressure is known.

$$p_t = p_{t,\text{spec}}, \qquad h_t = h_t(T_t,\ \text{composition}).$$

| Argument | Symbol | Meaning | Units | Default / constraint |
| --- | --- | --- | --- | --- |
| `pt` | $p_{t,\text{spec}}$ | prescribed total pressure | Pa | required, $>0$ |
| `Tt` | $T_t$ | feed total temperature | K | required |
| `composition`, `basis`, `marker` | — | as `mass_flow_inlet` | — | as above |

### `pressure_outlet(p, Tt_backflow=300.0, composition=None, basis="mole", marker=0.0)`
Pins the edge **static** pressure. The only boundary that models **ingestion / backflow**: on
inflow it draws the backflow stream in at $T_{t,\text{back}}$. Its choked/unchoked behaviour is
emergent (complementarity against the prescribed back-pressure), so it is the boundary for a
flow that may reverse or un-choke.

$$p = p_\text{spec}.$$

| Argument | Symbol | Meaning | Units | Default / constraint |
| --- | --- | --- | --- | --- |
| `p` | $p_\text{spec}$ | prescribed static pressure | Pa | required, $>0$ |
| `Tt_backflow` | $T_{t,\text{back}}$ | total temperature of ingested backflow | K | `300.0` |
| `composition`, `basis`, `marker` | — | backflow stream (on ingestion) | — | as inlet |

### `mass_flow_outlet(mdot)`
Pins the outflow rate with the static pressure floating — a metered exhaust, or the mean-flow
partner of a downstream choked throat. Inherited acoustic termination $\dot m'=0$. **Outflow
only**.

$$\dot m = \dot m_\text{spec} > 0.$$

| Argument | Symbol | Meaning | Units | Default / constraint |
| --- | --- | --- | --- | --- |
| `mdot` | $\dot m_\text{spec}$ | prescribed outflow mass rate | kg/s | required, $>0$ |
| `perturbation_bc` | — | acoustic termination | — | `None` → $\dot m'=0$ |

### `choked_nozzle_outlet(throat_area)`
Asserts a sonic ($M=1$) throat of area $A^{*}$ just past the outlet plane: the outflow is the
**critical mass flux** for the interior stagnation state, so $p$ floats. For a perfect gas,

$$\dot m^{*} = A^{*}\, \frac{p_t}{\sqrt{T_t}}\, \sqrt{\frac{\gamma}{R}}\left(\frac{2}{\gamma+1}\right)^{\frac{\gamma+1}{2(\gamma-1)}}.$$

Because the nozzle is *compact* the application plane stays subsonic; the inherited
termination is the compact choked-nozzle (**Marble–Candel**) reflection with entropy→acoustic
coupling. $A^{*} < A_\text{outlet}$ (a contraction) is enforced.

| Argument | Symbol | Meaning | Units | Default / constraint |
| --- | --- | --- | --- | --- |
| `throat_area` | $A^{*}$ | sonic-throat area | m² | required, $>0$, $<A_\text{outlet}$ |
| `perturbation_bc` | — | acoustic termination | — | `None` → Marble–Candel reflection |

### `wall(perturbation_bc=None)`
$\dot m = 0$ on its single edge — the leg behind it is stagnant ($M=0$); its purpose is
acoustic. By default a rigid **hard wall** ($u'=0$, $R=+1$), identical at $M=0$ to the
inherited $\dot m'=0$. Pass `perturbation_bc` for a liner impedance. No physics parameters.

### `cavity(volume)`
A **wall to the mean flow** ($\dot m = 0$, no interior mean unknowns) but a **compliance to
acoustics**: the enclosed gas compresses isentropically, giving

$$C = \frac{V}{\varrho c^2}$$

which populates the storage block $\mathbf{M}$. Paired with a neck inertance (a short `duct` off a
`junction`) it forms a Helmholtz resonator, $f_0 = \dfrac{c}{2\pi}\sqrt{\dfrac{A_\text{neck}}{V\,l_\text{eff}}}$.

| Argument | Symbol | Meaning | Units | Default / constraint |
| --- | --- | --- | --- | --- |
| `volume` | $V$ | enclosed cavity volume | m³ | required, $>0$ |

## Area changes and losses

Every element here takes the optional **storage lengths** `l_up`, `l_down`, `end_correction`
(all default `0`, in metres): the passage half-lengths on each side give acoustic compliance
($\sim l_i A_i$ of stored gas) and inertance (series effective length
$l_\text{up}+l_\text{down}+\ell_\text{end}$), populating $\mathbf{M}$ while staying inert in the mean
flow.

### `isentropic_area_change(l_up=0.0, l_down=0.0, end_correction=0.0)`
A smooth, lossless area change: mass conservation plus **total-pressure continuity**
($p_{t,0}=p_{t,1}$) with the static↔dynamic conversion set by each port's edge area. By
default a lengthless (compact) jump.

| Argument | Symbol | Meaning | Units | Default |
| --- | --- | --- | --- | --- |
| `l_up`, `l_down` | $l_\text{up}, l_\text{down}$ | passage half-lengths (port 0 / port 1) | m | `0.0` |
| `end_correction` | $\ell_\text{end}$ | added-mass length (inertance only) | m | `0.0` |

### `sudden_area_change(cc=1.0, eps=None, l_up=0.0, l_down=0.0, end_correction=0.0)`
Mass conservation plus a direction-dependent loss. Forward flow (small → large) follows the
**Borda–Carnot** momentum balance; reverse flow (large → small) a **vena-contracta** loss

$$\Delta p_t = K_c\left(\tfrac12\varrho u^2\right)_\text{small}, \qquad K_c = \left(\frac{1}{c_c}-1\right)^2.$$

The incompressible head is accurate to $O(M^2)$; the [`sudden_contraction`
composite](composite-elements.md) is the compressible upgrade.

| Argument | Symbol | Meaning | Units | Default / constraint |
| --- | --- | --- | --- | --- |
| `cc` | $c_c$ | contraction coefficient (reverse flow) | — | `1.0`, in $(0,1]$ |
| `eps` | $\varepsilon$ | smoothing width for the direction switch | kg/s | `None` (global) |
| `l_up`, `l_down`, `end_correction` | — | storage lengths | m | `0.0` |

### `loss(K, ref_port=0, eps=None, l_up=0.0, l_down=0.0, end_correction=0.0)`
Conserves mass and drops total pressure by $K$ dynamic heads, the head signed by flow
direction:

$$p_{t,\text{in}} - p_{t,\text{out}} = K\left(\tfrac12\varrho u^2\right)_{\text{ref\_port}}.$$

The static state on each port is reconstructed from that port's own area, so the loss may
**straddle an area change**. With storage lengths it becomes an orifice impedance
$Z = R(u) + i\omega L_\text{eff}/A$.

| Argument | Symbol | Meaning | Units | Default / constraint |
| --- | --- | --- | --- | --- |
| `K` | $K$ | loss coefficient (dynamic heads) | — | required |
| `ref_port` | — | port whose $\tfrac12\varrho u^2$ head $K$ references | — | `0` (or `1`) |
| `eps` | $\varepsilon$ | smoothing width | kg/s | `None` |
| `l_up`, `l_down`, `end_correction` | — | storage lengths | m | `0.0` |

### `linear_resistance(R, l_up=0.0, l_down=0.0, end_correction=0.0)`
Drops total pressure **linearly** in the mass rate:

$$p_{t,\text{in}} - p_{t,\text{out}} = R\,\dot m.$$

Because it is linear (not the quadratic head), it survives the linearization with a non-zero
coefficient **even at zero mean flow** — the acoustic resistance of a screen/perforate/damper
in a quiescent network. With storage lengths, the quiescent orifice impedance
$Z = R + i\omega L_\text{eff}/A$.

| Argument | Symbol | Meaning | Units | Default / constraint |
| --- | --- | --- | --- | --- |
| `R` | $R$ | resistance (total-pressure drop per unit mass flow) | Pa·s/kg | required, $\ge 0$ |
| `l_up`, `l_down`, `end_correction` | — | storage lengths | m | `0.0` |

## Transport

### `duct(length=0.0)`
Equal-area total-pressure continuity in the mean (length-independent); the `length` is inert
in the steady residual and read only by the acoustic **phase stamp** $\mathbf{P}(\omega)$ (wavenumber
$k = \omega/c$), so it propagates waves $\propto e^{\pm i k L}$. Its two ports share one flow
area.

| Argument | Symbol | Meaning | Units | Default |
| --- | --- | --- | --- | --- |
| `length` | $L$ | acoustic propagation length | m | `0.0` |

### `pipe(length, diameter, friction_factor)`
The `DUCT + LOSS` unification (Greyvenstein–Laurie): one element that drops total pressure
with the Darcy–Weisbach coefficient

$$K = \frac{f\,L}{D}, \qquad p_{t,\text{in}} - p_{t,\text{out}} = K\left(\tfrac12\varrho u^2\right),$$

**and** carries its `length` for the acoustic phase stamp $\mathbf{P}(\omega)$. Constant area;
$D$ is the hydraulic diameter (friction term only). Chain several with the [`fanno_pipe`
composite](composite-elements.md) to resolve Fanno gradients.

| Argument | Symbol | Meaning | Units | Default / constraint |
| --- | --- | --- | --- | --- |
| `length` | $L$ | pipe length (friction and acoustic) | m | required, $>0$ |
| `diameter` | $D$ | hydraulic diameter | m | required, $>0$ |
| `friction_factor` | $f$ | Darcy friction factor | — | required, $\ge 0$ |

## Reacting elements and sources

### `heat_release_flame(Qdot, dynamic_source=None)`
A compact constant-area flame conserving mass and total pressure (a low-Mach compact-flame
idealization neglecting the $O(M^2)$ Rayleigh loss) while raising the total enthalpy:

$$\Delta h_t = \frac{\dot Q}{\dot m}, \qquad \Delta T_t = \frac{\dot Q}{\dot m\, c_p}.$$

With $\dot Q$ fixed the mean flame is acoustically **passive**; a `dynamic_source` (an $n$–$\tau$
flame transfer function) gives it the unsteady $\mathbf{S}(\omega)$ that drives thermoacoustic
instability.

| Argument | Symbol | Meaning | Units | Default / constraint |
| --- | --- | --- | --- | --- |
| `Qdot` | $\dot Q$ | heat-release rate | W | required ($>0$ heats) |
| `dynamic_source` | $\mathbf{S}(\omega)$ | unsteady heat-release response (FTF) | — | `None` (passive) |

### `equilibrium_flame(dynamic_source=None)`
The headline reacting flame: conserves mass, **static pressure** (low-Mach compact
idealization), total enthalpy (adiabatic) and elemental composition $Z$. "Ignition" is the
per-edge **closure switch** — the approach edge uses the frozen (`EQ_FROZEN`) closure, the
product edge the equilibrium (`EQ_KERNEL`) closure — so the temperature rise emerges from an
HP-equilibrium solve at the shared $(Z, h_t, p)$. Acoustically passive; a `dynamic_source`
adds the lagged $\mathbf{S}(\omega)$. No mean-flow parameters (the composition comes from the feeds).

| Argument | Symbol | Meaning | Units | Default |
| --- | --- | --- | --- | --- |
| `dynamic_source` | $\mathbf{S}(\omega)$ | unsteady heat-release response (FTF) | — | `None` (passive) |

### `mass_source(mdot, T, composition, u_inj=0.0, basis="mole", dynamic_source=None, marker=0.0)`
Injects a stream, conserving mass, momentum and energy with source terms:

$$\dot m_\text{out} = \dot m_\text{in} + \dot m, \qquad
\left[\varrho u^2 + p\right]_\text{out} = \left[\varrho u^2 + p\right]_\text{in} + \dot m\, u_\text{inj},$$

with total enthalpy and composition mixed in mass-weighted. A fuel injector is this element
with a fuel `composition`; it performs no reaction (ignition is the flame's job).

| Argument | Symbol | Meaning | Units | Default / constraint |
| --- | --- | --- | --- | --- |
| `mdot` | $\dot m$ | injected mass rate | kg/s | required, $>0$ |
| `T` | $T$ | injected stream total temperature | K | required |
| `composition` | — | injected species mixture | — | required |
| `u_inj` | $u_\text{inj}$ | axial injection velocity (momentum source) | m/s | `0.0` (transverse) |
| `basis` | — | units of `composition` | — | `"mole"` |
| `marker` | $b$ | burnt-marker of the injected stream | — | `0.0` |

## Manifolds (variable port count)

### `junction(volume=0.0)`
Ties all incident ports to a **common static pressure**: a mass balance $\sum_i \dot m_i = 0$
plus $d-1$ rows $p_i = p_0$. A plenum `volume` adds the compliance $C = V/(\varrho c^2)$ to $\mathbf{M}$
(inert in the mean flow) — a junction with a volume is a cavity with through-flow. A branch's
neck inertance is **not** a manifold parameter; model it as an explicit neck `duct` on that
branch (exactly what `helmholtz_resonator` assembles).

| Argument | Symbol | Meaning | Units | Default / constraint |
| --- | --- | --- | --- | --- |
| `volume` | $V$ | chamber volume (plenum compliance) | m³ | `0.0`, $\ge 0$ |

### `splitter(volume=0.0)`
As `junction`, but ties the ports to a **common total pressure** ($p_{t,i} = p_{t,0}$,
lossless) — the idealization when the manifold recovers dynamic head. Same optional `volume`.

### `forced_splitter(fractions)`
One inflow (port 0) split into $N$ outflows at **prescribed mass fractions**:

$$\dot m_{\text{out},k} = \beta_k\, \dot m_\text{in}, \qquad
\beta_N = 1 - \sum_{k} \beta_k .$$

You give $N-1$ fractions; the last (remainder) branch carries $\beta_N$ and keeps
total-pressure continuity with the inflow. The controlled branches float in pressure (a
control-valve / ideal flow-divider idealization); reverse flow is not modelled. Wire the
**inflow edge first** and the **remainder outflow last**.

| Argument | Symbol | Meaning | Units | Default / constraint |
| --- | --- | --- | --- | --- |
| `fractions` | $\{\beta_k\}$ | the $N-1$ controlled outflow fractions | — | each in $(0,1)$, $\sum<1$ |

## Reserved (deferred)

`SUPERSONIC_INLET` (9) and `SUPERSONIC_OUTLET` (10) are reserved dispatch ids for supersonic
boundaries. v1 scope is **subsonic** (flowing or quiescent); supersonic / shock-seeding
boundaries are deferred.
