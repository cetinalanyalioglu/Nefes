## To implement

- [ ] Kinetic-energy coupling for the reacting closure: `EQ_KERNEL`/`EQ_FROZEN` currently drop `u^2/2` (h ~ h_t), O(M^2) at low Mach. Restore the outer KE fixed point (R-B2.2) with the 2-inner-solve IFT (closure ift, analogous to `perfect_gas._attach_density_imag`). Note the perturbation caloric map (`characteristics.edge_caloric`) drops the KE term (`m=0`) to match this closure; restore `m=u` there too when the closure carries KE.
- [ ] Element-dropping in the `@njit` equilibrium kernel (`fns/thermo/_chem.equilibrate_hp`) for burnt edges whose elemental abundance (`Z = xi @ Zfeed`) has a zero element (e.g. a parallel branch whose products lack carbon). Mirror thermolib's keep_el/keep_sp compaction (locate-on-real). Series injection (all elements present downstream) avoids it today.
- [ ] We are now considering "convenience" elements, that will transform into multiple elements when added to network. We should be careful to preserve proper numbering. I am not certain yet, maybe we re-run the re-numbering algorithm, or just insert the new "serial" portion as incremental numbers to the correct position and shift the rest? First user: a `helmholtz_resonator(V, neck_length, neck_area)` wrapper (tee + neck duct + cavity), now buildable by hand from the primitives. We call this concept "composite elements".
- [ ] Storage `M` remainder -- per-branch manifold inertance: a neck length on each junction/splitter branch's `p0-pi` row, stamping `-l_eff_i/A_i` at its `mdot_i` column (the generic per-port compliance + series inertance + manifold volume are done -- `_inline_storage`/`_manifold_storage`). Lock sign/orientation against a flipped-edge probe. See `scratch/inertance-end-correction-theory.md` s3.2/s5.
- [ ] Storage energy-ledger hook (HR-plan Phase 4): generalize `_stored_energy` (currently duct-only) to add the lumped storage energy -- cavity/plenum `0.5 * C * |p'|^2` and inertance `0.5 * (L_eff/A) * |mdot'|^2` -- so `forced_power_balance` / `modal_energy_balance` close with storage present. Operator is already complete (`J_alg + i*omega*M + P + S`); this is the diagnostics side.
- [ ] Dedicated sudden-contraction element resolving the vena-contracta state (composite: isentropic to vena contracta + Borda re-expansion) for exact loss and minimum static pressure at higher Mach. The current `sudden_area_change` `cc`-loss uses the incompressible 1/2 rho u^2 head, accurate only to O(M^2).

## To verify

## To brainstorm

- [ ] We reject BC's when there is no absolute pressure reference from any of them. For such cases, would it work if we had an absolute pressure reference at some arbitrary edge state in the domain? If so, we could let the user enforce absolute pressure at exactly one edge.

## To discuss

- [ ] How does linear resistance element behave in mean flow?

## To test

## Issues

- [ ] We don't have any mechanism to prevent connecting incompatible elements with each other - see the guardrails we put in the UI. Perhaps we were too strict, this is open to re-evalute.

## Deferred

- [ ] Frequency-dependent radiation / reservoir outlet BCs: a small library of named analytic radiation-impedance constructors returning `Z(omega)` callables for `PerturbationBC.impedance` (auto-converts to `R`). Unflanged open end (Levine–Schwinger low-`ka`: `Z/(rho c) ~= (ka)^2/4 + i*0.6133*ka`), flanged/baffled-piston (`(ka)^2/2 + i*0.82*ka`), and jet-pipe with mean flow (Munt/Rienstra, Mach/Strouhal-dependent). Keep each analytic in `omega` so it continues off the real axis and works with the contour eigensolver; tabulated/measured radiation data must go through the AAA/barycentric rational fit first (else Nyquist / forced-response only). Replaces the constant `open_end` / `mean_flow_open_end` reflections for realistic open outlets.
- [ ] Condensed-phase equilibrium *products* (soot/graphite, condensed oxides). v1 products are gas-only (`SpeciesLibrary.product_mask`); condensed species are feed-only (set elements + enthalpy, masked out of the burnt slate). Related edge case: an auto-slate feed element with no gas product (a metal) leaves the burnt solve unbalanced.
- [ ] Compositional ("indirect") noise -- one gap remains. It is already captured wherever the linearization is inherited: `J_alg` carries composition->acoustic at a flame, an area change, and the inherited `choked_nozzle_outlet` element (its critical-mass-flux row is complex-stepped through composition -> the `R_xi` column for free); and scalar *ports* now read out in the scattering matrices (`response.py`). Open: the hand-written analytic compact closures (`PerturbationBC.choked_nozzle`/`constant_mass_flow`) drop `R_xi` (flagged by `CompositionalNoiseWarning`) -- complex-step the closure's composition dependence to restore it. Theory: Magri JFM 2016.
- [ ] Non-compact / through-throat nozzle-response solver: integrate the linearized-Euler perturbation ODEs through the `M(x)` profile with the M=1 sonic-regularity (L'Hopital) condition (Stow-Dowling / Duran-Moreau). Handles the sonic throat the transit-time network model cannot (`tau_- = L/(c-u) -> inf`), and is the proper distributed-source compositional-noise route. The deferred supersonic / M=1 scope.
- [ ] Re-order algorithm - deferred because current solvers do not utilize this
- [ ] Heated finite-volume element's independent energy store: a finite-volume flame / heated plenum carrying its own energy DOF that couples the energy/transport row to `S(omega)` (the thermal capacitance the lumped flame neglects). Not a frozen-mean `M` stamp -- the energy "row" is the per-edge transport equation, and it needs finite-volume-flame mean geometry; gated on that design. See `scratch/inertance-end-correction-theory.md` s3.3 (the doc's own deferred item).
- [ ] `perturbation_response(freeze=...)`: a lossless frozen termination (closed stub) is ill-conditioned at its resonance (real-axis pole). Add optional auto-regularization (small wall loss / pole-skip).
