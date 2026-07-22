# Element constitutive relations

An element is the network's unit of physics: a control volume on which the governing balances are applied, contributing the equations that tie the states on its incident edges together (see [network abstraction](abstraction.md)).
This document is the constitutive library — it derives, element class by element class, the residual rows each one writes into the system, and the reasoning that makes those rows smooth, direction-safe, and thermodynamically admissible.
The same rows are more than a mean-flow model: differentiated at the operating point they *are* the acoustic element stamps (see [perturbation network](perturbation-network.md)), so the care taken here to keep every residual complex-analytic is what later lets one operator serve both problems.

Every interior element shares one structural template — a single mass balance and one pressure-type relation per additional port.
The parameter-level catalogue of constructors is the subject of the [atomic-elements](../reference/atomic-elements.md) reference; the concern here is the physics each residual encodes.

## The common template {#sec-elements-common-template}

Two conventions hold for every residual below.
The ports of an element are numbered in connection order, and each residual is written as a smooth, complex-analytic function of the flow state, using the regularized primitives of the [complex-step](../design/complex-step.qmd) design note in place of any branch, absolute value, or min/max.
Under these conventions two devices recur in every element and are stated once here rather than repeated.

**The mass balance (row 1).**
Every interior element conserves mass, and this is its first residual row, given as:

$$
R_{\text{mass}} \;=\; \sum_{e} \sigma_{P,e}\,\dot m_e \;=\; 0,
$$

where the sum runs over the edges incident to element $P$, $\dot m_e$ is the signed mass flow along edge $e$, and $\sigma_{P,e} = \pm 1$ orients that edge's arrow at $P$ ($+1$ tail, $-1$ head).
This row is exact, linear, and independent of the edge-arrow convention: flipping an edge's arrow flips both $\sigma_{P,e}$ and the sign of $\dot m_e$ at the solution, so the physical outflow $\sigma_{P,e}\dot m_e$ — and hence the balance — is unchanged.

**The stabilization term.**
Every interior *pressure-type* row additionally carries a small linear resistance $-\,\kappa\,\dot m^{\text{out}}_{P,\text{port}}$, with $\kappa$ the artificial-resistance coefficient and $\dot m^{\text{out}}_{P,\text{port}} = \sigma_{P,e}\dot m_e$ the outflow at that port.
Its role is to regularize the early solver stages, and it is driven to exactly zero before convergence, so that the equations actually satisfied at the operating point are the exact ones.
The term is therefore written but not repeated in the discussion of each element; its purpose and its harmlessness are the subject of [well-posedness](well-posedness.md#sec-wellposed-zero-flow), and it is denoted $\kappa$-term below.

An interior element with $n$ ports thus supplies one mass row and $n-1$ pressure-type rows, the latter carrying the element's geometry.
Energy continuity does not appear among these rows: the total enthalpy is delivered onto each edge by the transport relations through the element's donor (see [transport](transport.qmd)), so an interior element writes no energy row of its own.

## Isentropic area change {#sec-elements-isentropic-area}

The simplest two-port is a smooth, internally monotone contraction or diffuser that changes the flow area without loss — until its small port chokes.
Beyond the shared mass row it supplies one pressure relation, given as:

$$
R_2 \;=\; \varphi_\varepsilon\!\Big(1 - M^{\text{in}}_{\text{small}},\;
\frac{p_{t,\text{small}} - p_{t,\text{large}}}{p_{t,\text{small}}}\Big)\,p_{t,\text{small}} \;-\; \kappa\text{-term},
$$

where $M^{\text{in}}_{\text{small}}$ is the Mach number at the smaller port oriented into the element, $p_{t,\text{small}}$ and $p_{t,\text{large}}$ are the total pressures at the small and large ports, and $\varphi_\varepsilon$ is the smoothed Fischer–Burmeister complementarity residual (see [complex-step](../design/complex-step.qmd)).
The complementarity encodes two regimes in a single smooth row: while the small port is subsonic — in either flow direction — the row reduces to total-pressure equality $p_{t,\text{small}} = p_{t,\text{large}}$, the classical isentropic element; when the small port reaches $M = 1$ in the diverging direction the element chokes and a total-pressure drop, the lumped internal normal shock, becomes admissible.
The choked branch and the operating map it produces are the subject of [choking](choking.qmd); here it suffices that both regimes issue from the one row.

Intuitively, this two-equation element reproduces the classical isentropic jump.
Energy continuity $h_{t,0} = h_{t,1}$ is delivered by the edge transport, and entropy continuity in the lossless regime then follows from the entropy lemma of [state and recovery](state-and-recovery.qmd) — continuous $p_t$ and $T_t$ imply continuous $s$ — so mass, energy, and constant entropy hold across the element in subsonic operation, valid for either flow direction and regular at $\dot m = 0$ (tests: `test_subsonic_nozzle_matches_isentropic` against the analytic relations, `test_long_serial_chain_cold_start` for a chain solved from rest).

## Sudden area change {#sec-elements-sudden-area}

A sudden, rather than smooth, change of area is lossy in one direction and nearly loss-free in the other, and its residual blends the two according to which way the gas flows.

**Expansion — the Borda–Carnot analysis.**
When a jet leaves a small pipe into a larger one it cannot follow the abrupt corner: it separates and mixes back out to the full area downstream, with turbulent loss.
The magnitude of that loss needs no empirical constant, because a momentum balance fixes it — the separated dead-water corner holds the small-pipe static pressure $p_s$ against the annular back wall.
Steady momentum on the control volume between the small section $A_s$ and the large section $A_l$ is given as:

$$
\underbrace{\dot m\,u_l - \dot m\,u_s}_{\text{momentum change}}
\;=\;
\underbrace{p_s A_s}_{\text{inlet}} \;+\; \underbrace{p_s (A_l - A_s)}_{\text{back wall}} \;-\; \underbrace{p_l A_l}_{\text{outlet}},
$$

where $u_s$ and $u_l$ are the small- and large-port velocities and $p_l$ the large-port static pressure, which rearranges to $\dot m\,(u_l - u_s) + A_l\,(p_l - p_s) = 0$.
An important remark is that the *static* pressure rises through a sudden expansion while the *total* pressure drops; the entropy production comes out of the momentum balance rather than being inserted by hand, and in the low-speed limit the relation reduces to the familiar loss $\Delta p_t = \tfrac{1}{2}\varrho(u_s - u_l)^2$ (test: `test_expansion_unaffected_by_cc`, which exercises the Borda momentum branch).

**Contraction — the vena contracta.**
The same momentum algebra applied to a contraction would predict an entropy *decrease*, which is physically impossible; a real sudden contraction is instead nearly loss-free up to a vena contracta and loses total pressure only in the re-expansion that follows.
That loss is referenced to the small-port dynamic head through a contraction coefficient, and the contraction residual is given as:

$$
R^{\text{contr}}_2 \;=\; \big(p_{t,0} - p_{t,1}\big) \;-\; \operatorname{sgn}_{\text{dir}}\, K_c\, q_{\text{small}},
\qquad
K_c = \Big(\frac{1}{C_c} - 1\Big)^{\!2},
\qquad
q_{\text{small}} = \tfrac{1}{2}\varrho_{\text{small}}\,u_{\text{small}}^2,
$$

where $C_c$ is the vena-contracta contraction coefficient, $K_c$ the resulting loss coefficient, $q_{\text{small}}$ the small-port dynamic head, and $\operatorname{sgn}_{\text{dir}}$ orients the total-pressure drop onto the small port.
The default $C_c = 1$ recovers exact total-pressure continuity, the loss-free contraction; a smaller $C_c$ introduces the measured contraction loss, and this incompressible-head form is accurate to $\mathcal{O}(M^2)$ (tests: `test_contraction_lossless_default_conserves_pt`, `test_contraction_loss_matches_Kc`, `test_contraction_loss_grows_as_cc_drops`).

**The direction-invariant blend, and a subtlety.**
The two regimes are combined by a smooth weight $\xi = \operatorname{sstep}(\dot m^{\text{in}}_{\text{small}};\varepsilon)$, which tends to $1$ when the flow enters through the small port (expansion) and to $0$ otherwise, giving the residual as:

$$
R_2 \;=\; \xi\,R^{\text{mom}}_2 \;+\; (1 - \xi)\,R^{\text{contr}}_2 \;-\; \kappa\text{-term},
$$

where $R^{\text{mom}}_2$ is the expansion momentum residual scaled to pressure units and $R^{\text{contr}}_2$ the contraction residual above, both sign-normalized so that near $\dot m = 0$ their pressure content is the same $(p_0 - p_1)$ and the two halves of the blend cannot cancel there and leave the element without an effective equation.
A subtlety worth recording is that the convective momentum flux $\dot m\,u = \dot m^2/(\varrho A)$ is *even* under an edge-arrow flip — both factors change sign together — so no $\sigma$ may multiply it; writing the momentum balance as $\sum \sigma(\dot m u + pA)$ by analogy with the mass and energy balances silently breaks the element's arrow-independence, a mistake made transiently during development and caught by `test_edge_direction_invariance`.

## The loss family {#sec-elements-loss-family}

A number of elements share one residual shape — a total-pressure drop proportional to a head that depends on the through-flow — and differ only in how that head is formed.

**Concentrated loss.**
A valve, orifice, filter, or any device characterized by a loss coefficient $K_L$ referenced to a dynamic head contributes the pressure relation, given as:

$$
R_2 \;=\; p_{t,0} - p_{t,1} - K_L\, q_{\text{signed}} \;-\; \kappa\text{-term},
\qquad
q_{\text{signed}} = \tfrac{1}{2}\,\varrho_{\text{avg}}\;u_{\text{ref}}\sqrt{u_{\text{ref}}^2 + u_\varepsilon^2},
$$

where $K_L$ is the loss coefficient, $\varrho_{\text{avg}}$ the port-average density, $u_{\text{ref}} = \dot m_{\text{through}}/(\varrho_{\text{avg}}\,A_{\text{ref}})$ the reference velocity formed from the through-flow and the reference-port area, and $u_\varepsilon$ a small regularizing velocity.
The signed head $q_{\text{signed}}$ is a smooth form of $\tfrac{1}{2}\varrho_{\text{avg}}\,u|u|$, so the loss always *opposes* the flow whichever way it runs — the second law holds in both directions — and it passes smoothly through $u = 0$.

**Length-bearing and lossless variants.**
Three siblings carry a length and are named here for completeness:

1. **Lossless duct**: a length-bearing but loss-free segment enforcing total-pressure continuity, $R_2 = p_{t,0} - p_{t,1} - \kappa\text{-term}$; the length it carries matters only to the acoustics, where it supplies the propagation phase.
2. **Friction pipe**: a constant-area segment with a selectable mean-flow closure and the duct acoustic phase described below.
3. **Linear resistance**: a screen, perforate, or damper whose drop is *linear* in the through-flow, $R_2 = p_{t,0} - p_{t,1} - r_{\text{lin}}\,\dot m_{\text{through}} - \kappa\text{-term}$; unlike the quadratic head, this term does not vanish with the mean dynamic head and so remains active in the linearized problem even at zero mean flow, the resistance a quiescent network still presents to an acoustic wave.

The pipe's default `darcy-weisbach` formulation is the mean-flow and acoustic unification of duct and concentrated loss following Greyvenstein & Laurie, with $K_L=fL/D$ in the total-pressure relation above.
This closure is exact in the incompressible limit and is retained for low-Mach hydraulic networks and their literature comparisons.
The alternative `momentum` formulation integrates the static-pressure and axial-momentum balance over the segment:

$$
R_2
=
\left(p+\varrho u^2\right)_0
-
\left(p+\varrho u^2\right)_1
-
\frac{fL}{D}\,q_{w,\text{signed}}
-
\kappa\text{-term},
\qquad
q_{w,\text{signed}}
=
\frac{1}{2}
\left[
\left(\tfrac12\varrho u|u|\right)_0
+
\left(\tfrac12\varrho u|u|\right)_1
\right],
$$

where the absolute value is the same analytic regularization used by the loss family, the velocity is oriented from port 0 to port 1, and $q_{w,\text{signed}}$ is the endpoint (trapezoidal) approximation to the wall head distributed along the segment.
Together with constant mass flow and total enthalpy, refinement of this formulation converges to the classical perfect-gas Fanno relations; the `fanno_pipe` composite uses it by default.
Both formulations become the ordinary Darcy pressure drop at low Mach number, but they are not interchangeable near a sonic state.

## Junctions and splitters {#sec-elements-junctions-splitters}

A multi-port node that merges or distributes streams supplies one mass balance and $n - 1$ pressure couplings of its remaining ports against port $0$.
Two couplings are available, given as:

$$
\text{static-pressure junction:}\quad R_{1+i} = p_0 - p_i,
\qquad
\text{lossless splitter:}\quad R_{1+i} = p_{t,0} - p_{t,i},
\qquad i = 1,\dots,n-1,
$$

where the junction ties all ports to a common *static* pressure and the splitter to a common *total* pressure (each row also carrying its $\kappa$-term).
The static-pressure junction is the classical header or manifold node, appropriate where every port runs at low Mach number so that the kinetic terms it ignores are negligible; enthalpy mixing of several inflows is automatic through the donor mechanism of [transport](transport.qmd#sec-transport-donor-enthalpy).
The lossless splitter is an isentropic distribution plenum: with $h_t$ delivered by the edge transport and $p_t$ common, entropy is continuous into every outflow branch, reproducing the classical lossless splitter of mass, energy, and constant entropy.

**A selection rule that is not cosmetic.**
The static-pressure junction must be used *only* where every port runs at low Mach number.
At a fast port, equal static pressure plus the port's velocity head hands the branch a total pressure $p_t \approx p + \tfrac{1}{2}\varrho u^2$ — more total pressure than the feed possesses — which is free energy and a second-law violation.
It should be noted that the consequence is not merely a small error: the surplus must be destroyed somewhere downstream, and if no element can do so the network has *no steady solution at all* and the solver can only stall.
The rule of thumb is therefore that a plenum feeding fast branches takes a splitter (common $p_t$), while a low-speed header collecting comparable streams takes a static-pressure junction (common $p$); a merge with a port that is not slow takes the mixer below, which lifts the low-Mach restriction by charging the mixing loss instead of ignoring it.

**Mixer.**
The mixer is the general merge that respects the second law at any port Mach number.
It ties every port to a common *effective* total pressure,

$$
R_{1+i} = p_{t,i}^{\mathrm{eff}} - p_{t,0}^{\mathrm{eff}} - \kappa\text{-term},
\qquad
p_{t,k}^{\mathrm{eff}} = p_{t,k} - \ell_k,
\qquad i = 1,\dots,n-1,
$$

where the inflow loss interpolates between two limits set by the recovery $\sigma \in [0,1]$:

$$
\ell_k = \chi_k\Big[(1-\sigma)\,\underbrace{(p_{t,k}-p_k)}_{\text{dynamic head}} + \sigma\,\underbrace{(p_{t,k}-p_t^{\min})}_{\text{excess over weakest feed}}\Big],
\qquad
p_t^{\min} = \min_{j\,\in\,\text{inflows}} p_{t,j},
$$

with $\chi_k$ the smooth inflow indicator (one on an inflow port, zero on an outflow) and $p_t^{\min}$ the smooth minimum over the inflow total pressures.
The two limits are the worst and best merges the state allows.
At $\sigma = 0$ each inflow surrenders its whole dynamic head $p_{t,k}-p_k$, the full dump loss of a plenum: the most dissipative merge, and the best conditioned because the loss is a private per-port quantity.
At $\sigma = 1$ each inflow surrenders only its excess over the weakest feed $p_{t,k}-p_t^{\min}$, so every port leaves at the minimum inflow total pressure, the least dissipation the second law permits for the given streams.
An outflow ($\chi_k \to 0$) takes no loss and leaves at the common node total pressure.
For any $\sigma$ the couplings hold the node total pressure at or below every inflow's, $p_t^{\mathrm{node}} \le p_{t,i}$, so no branch ever gains total pressure.
With the total enthalpy and composition mass-averaged by the same donor as the junction, and specific entropy decreasing in total pressure (at fixed enthalpy and composition) and concave in enthalpy, the mass-averaged outflow entropy is at or above the feed mean: the entropy production $\dot S_{\mathrm{gen}} = \dot m\, s^{\mathrm{node}} - \sum_{\text{in}} \dot m_i\, s_i \ge 0$ by construction, whatever the port Mach numbers.

At the recovery limit $\sigma = 1$ the dump term vanishes and the couplings become pure total-pressure equalities, $p_{t,k} - \ell_k = p_t^{\mathrm{node}}$ with $p_t^{\mathrm{node}} = p_t^{\min}$, carrying no relation between a port's mass flow and its pressure drop.
The element then adds no flow resistance of its own, exactly as the splitter adds none, so the flow split is left to the rest of the network.
Distributing a single inflow, that is automatic: the minimum runs over the one feed, its loss vanishes, and every outflow leaves at the inflow's own total pressure, which is exactly the lossless (isentropic) splitter.
Merging several streams, $\sigma = 1$ is the least-dissipative limit, and it is well posed precisely when the network pins each inflow's rate independently of the manifold, through a prescribed inflow (a mass-flow inlet) or a real resistance in its branch (a loss, an orifice, a pipe).
Two total-pressure reservoirs attached directly to the node do not meet that condition: the weakest feed then sees no pressure drop and its flow rate is undetermined, so the merge has no unique steady state, exactly the requirement the splitter already carries and not a property of the element.
For $\sigma < 1$ the dump term $(1-\sigma)(p_{t,k}-p_k)$ rises with each inflow's own dynamic head, a self-supplied resistance that pins the split without help from the network, so lower recovery converges on any topology; the conditioning degrades smoothly as $\sigma \to 1$ and that self-resistance fades.
The default is $\sigma = 1$, the least-dissipative ideal, which asks that each inflow be pinned by the network; the solve carries a topological check that warns when a mixer near this limit is reached by two or more total-pressure inlets through no resistance, the under-pinned merge above.
Lowering $\sigma$ toward $0$ gives the robust full dump, well posed for any wiring, which at low Mach reduces to the static-pressure junction.
Because it charges a fast inflow its mixing loss instead of manufacturing free energy, the mixer is the general merge element, converging on merges of unequal total pressure that the lossless splitter cannot represent.

**Forced splitter.**
A flow divider whose split is imposed rather than discovered is a variant of the splitter: with one inflow at port $0$, the first $n - 2$ outflow ports each carry a fixed fraction $\beta_i$ of the inflow rate, and the last outflow port carries the remainder while keeping total-pressure continuity with the inflow.
Because reverse flow is disallowed, no upwind switch is needed and every row is linear in the flow state, so the complex-step Jacobian is exact without smoothing.

## Boundary elements {#sec-elements-boundary}

A boundary element terminates a single edge and supplies exactly one equation; its donor enthalpy becomes active only if the flow actually enters the network there (see [transport](transport.qmd#sec-transport-donor-enthalpy)).

**Mass-flow inlet.**
A prescribed inflow rate pins the outflow into the domain, given as $R = \sigma_{P,e}\dot m_e - \dot m^{\text{spec}}$, with the donor $H_P = c_p T_t^{\text{spec}}$ supplying the specified stagnation enthalpy.

**Total-pressure inlet (reservoir).**
A reservoir is drawn from losslessly, so the natural condition is on total pressure; but if the network turns around and discharges into the reservoir that condition becomes impossible, and the correct condition is then on static pressure.
The residual blends the two according to the flow direction, given as:

$$
R \;=\; \xi\,\big(p_t - p_t^{\text{spec}}\big) \;+\; (1 - \xi)\,\big(p - p_t^{\text{spec}}\big),
\qquad
\xi = \operatorname{sstep}\!\big(\dot m^{\text{out}}_{P,e};\varepsilon\big),
$$

where $\xi \to 1$ on outflow into the domain (draw) and $\xi \to 0$ on ingestion (discharge into the reservoir).
The blend is a necessity rather than a convenience: an arriving stream carrying surplus total pressure cannot shed it losslessly, so demanding $p_t = p_t^{\text{spec}}$ on ingestion would leave *no steady solution*; physically the jet dumps its velocity head into the reservoir by turbulent mixing outside the network, and the static-pressure branch is the correct one.

**Pressure outlet.**
A static-pressure outlet matches the exit static pressure while subsonic, admits a choked branch at the discharge limit, and accepts backflow, given as:

$$
R \;=\; \xi\,\varphi_\varepsilon\!\Big(1 - M^{\text{in}},\;\frac{p - p^{\text{spec}}}{p^{\text{spec}}}\Big)\,p^{\text{spec}}
\;+\; (1 - \xi)\,\big(p_t - p^{\text{spec}}\big),
\qquad
\xi = \operatorname{sstep}\!\big(\dot m^{\text{in}}_{P,e};\varepsilon\big),
$$

with the donor $H_P = c_p T_t^{\text{backflow}}$.
Discharging subsonically the complementarity reduces to $p = p^{\text{spec}}$; at the choking limit the exit pins at $M = 1$ and the exit pressure detaches upward from the specification, the underexpanded choked-orifice discharge of [choking](choking.qmd); on backflow the specification acts as the total pressure of the returning stream, which carries the prescribed backflow temperature (test: `test_reverse_flow_reverses_drop`).

**Prescribed-outflow and choked-nozzle outlets.**
Two further terminations complete the set.
A mass-flow outlet pins the outflow rate, $R = -\sigma_{P,e}\dot m_e - \dot m^{\text{spec}}$, and inherits a constant-mass-flow acoustic termination.
A choked-nozzle outlet lumps a compact sonic throat of area $A^\ast$ just downstream and sets the outflow to the critical mass flux of the interior stagnation state, given as:

$$
\dot m^{\text{out}} \;=\; \varrho_t\, c_t\, A^\ast \left(\frac{2}{\gamma + 1}\right)^{\!\frac{\gamma + 1}{2(\gamma - 1)}},
$$

where $\varrho_t$ and $c_t$ are the stagnation density and sound speed recovered from the local state and $\gamma$ the local isentropic exponent.
Because the sonic point sits in the lumped throat rather than in the domain, the application plane stays subsonic and the inherited acoustic operator is the compact choked-nozzle (Marble–Candel) reflection [@marble_candel_1977], entropy coupling included.

**Wall.**
An impermeable termination sets $R = \sigma_{P,e}\dot m_e = 0$, admitting no mass across the face.
A finite cavity shares this mean-flow residual — it is a wall to the steady flow — and differs only acoustically, where its volume enters the storage block as a compliance (see [perturbation network](perturbation-network.md#sec-perturb-storage-block)).

## The transfer-matrix element {#sec-elements-transfer-matrix}

Some components are known not by a constitutive law but by a *measured or prescribed* two-port frequency response, and the transfer-matrix element is the vehicle for embedding such a component in the network.
To the mean flow it is passive — its steady residual is identical to that of an isentropic area change, conserving mass and energy and remaining isentropic — so it perturbs the operating point no more than a lossless duct would.
Its distinctive behaviour is confined to the perturbation layer, where its acoustic rows are overwritten by the user-supplied transfer matrix instead of the linearized jump; that stamp, and the identification procedure that can supply the matrix from data, are the subject of the [perturbation network](perturbation-network.md) and [identification](identification.md#sec-ident-two-port).
An important remark is that the matrix is a frequency-domain relation between the two stations and **should not be interpreted as a causal input–output law**; it constrains the linearized states at a frequency, nothing more.

## The stabilization term {#sec-elements-stabilization}

During the early solver stages only, every interior pressure-type row carries the artificial-resistance term noted in the common template, given as:

$$
R_{1+i} \;\mathrel{-}=\; \kappa\,\dot m^{\text{out}}_{P,\text{port }i},
\qquad
\kappa = \kappa_s\, r_{\text{art}},
\qquad
r_{\text{art}} = \min\!\left(\frac{\Delta p_{\max}}{\dot m_{\text{ref}}},\, 1\right),
$$

where $\kappa$ is the artificial-resistance coefficient actually stamped into the row, $\kappa_s$ the dimensionless continuation schedule, and $r_{\text{art}}$ a resistance scale built from $\Delta p_{\max}$, the span between the network's highest and lowest prescribed absolute pressure.
Scaling by $r_{\text{art}}$ sizes the fictitious pressure drop at the reference flow as a fixed fraction $\kappa_s$ of the real driving drop; the cap at unity means it only ever *softens* the friction, for a network whose driving drop is small against its flow, and leaves a healthy one untouched at $r_{\text{art}} = 1$.
Where no driving drop is known a priori, as in a purely mass-driven network, the scale falls back to unity.
It is a small fictitious friction between port $0$ and port $i$, signed as the second law dictates, and its necessity (it removes a zero-flow degeneracy that would otherwise strand the solver) is argued in [well-posedness](well-posedness.md#sec-wellposed-zero-flow).
It is harmless because the final solver stage sets $\kappa_s = 0$, so the equations satisfied at convergence are the exact constitutive relations of this document rather than their stabilized surrogates (test: `test_long_serial_chain_cold_start`, converging from rest through the staged continuation to $\kappa = 0$).

Why these particular forms are chosen over the more obvious flux-form or hard-switch alternatives is argued in [well-posedness](well-posedness.md).
