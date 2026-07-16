# Benchmarks

Where [verification](verification.md) checks that the framework solves its equations correctly, this document checks that those equations reproduce established results: named cases from the literature and from analytic theory that a compressible-network and thermoacoustic solver is expected to recover.
Each benchmark states the case, the reference it is compared against, and the quantitative agreement obtained, with the test that performs the comparison.

## Greyvenstein & Laurie compressed-air network {#sec-bench-greyvenstein-laurie}

The steady mean-flow solver is benchmarked against Example 3 of [@greyvenstein_laurie_1994], a compressed-air pipe network solved in that work by an independent method and reported as a table of branch mass flows and nodal pressures.

The network is assembled as a Nefes model with the corresponding pipe, junction, and boundary elements, solved from a cold start, and compared entry by entry against the published table.
The converged branch mass flows and nodal pressures match the reference table, and the flow is confirmed to sit in the low-Mach regime the original analysis assumes, so the comparison is like-for-like (tests: `test_converges`, `test_mass_flows_match_table_iv`, `test_node_pressures_match_table_iv`, `test_low_mach_regime`).

This case exercises the mean-flow machinery end to end (friction-pipe losses, many junction couplings, boundary conditions, and the continuation that finds the operating point) against an external result rather than an internal consistency check.

## Rijke-tube thermoacoustic stability {#sec-bench-rijke-tube}

The acoustic and dynamic-source layers are benchmarked against the Rijke tube, the canonical thermoacoustic oscillator: a duct with a compact heat source whose unsteady heat release couples to the acoustic field and, for a destabilizing time lag, drives a self-excited instability.
This benchmark was chosen as it has an analytic counterpart: the dispersion relation of a two-region duct with a compact flame, whose roots give the modal frequencies and growth rates in closed form, so the framework's eigenmodes can be compared against exact values rather than against another solver.

With no unsteady heat release the tube is a passive resonator and the computed modal frequencies match the analytic resonances (test: `test_passive_flame_matches_analytic_frequencies`); with an $n$–$\tau$ flame a mode acquires a positive growth rate and both its frequency and growth match the analytic root, and the same flame is stabilizing at one time lag and destabilizing at another, reproducing the $n$–$\tau$ stability band (tests: `test_n_tau_flame_drives_self_excited_instability`, `test_n_tau_lag_sets_stability_band`).

The reacting counterpart, with an equilibrium flame in place of the prescribed heat source, ignites and matches the same analytic construction, confirming that the reacting closure and the acoustic source compose correctly (tests: `test_reacting_flame_ignites_and_matches_analytic`, `test_reacting_n_tau_flame_matches_analytic_instability`).

## EM2C combustor, cross-checked against OSCILOS {#sec-bench-em2c-oscilos}

Where the Rijke tube compares Nefes against an analytic dispersion relation, this case compares it against an independent code on a published laboratory configuration: the stable setting of the EM2C swirl-stabilized combustor of [@palies_2011], as it is set up in Sec. 5.4.1 of the OSCILOS technical report [@li_oscilos_2017], where the combustor is reduced to a plenum, an injection unit, and a chamber with a compact flame at the chamber inlet.
This case was chosen because it is stated entirely in text: lengths, radii, the mean velocity at the injector exit, the reactant and burnt-gas temperatures, the two reflection coefficients, and a closed-form second-order low-pass flame transfer function [@dowling_1997] of the kind [dynamic sources](../theory/dynamic-sources.qmd) supplies, so no quantity has to be read off a figure.
OSCILOS reports the dominant mode at $152.6\;\mathrm{Hz}$ with growth rate $-19.1\;\mathrm{s^{-1}}$.

Nefes solves the mean flow on the network from the mass flow and the heat power alone, rather than marching the sections as the reference does, and evaluates the perturbation operator at that converged state.
The dominant mode comes out at $153.4\;\mathrm{Hz}$ and $-19.0\;\mathrm{s^{-1}}$, agreeing to $0.5\,\%$ in frequency and $0.7\,\%$ in growth rate and landing on the same, stable, side of the axis (tests: `test_mean_flow_matches_the_reported_operating_point`, `test_dominant_mode_matches_oscilos`).
The residual is bounded by an ambiguity in the reference itself, whose cold-tube case states $300\;\mathrm{K}$ while quoting the sound speed of air at $293.15\;\mathrm{K}$; running this case at both temperatures brackets the published pair (test: `test_inlet_temperature_ambiguity_brackets_the_published_mode`).

Two by-products of the same run are checked alongside the eigenvalue.
The abrupt area increase at the chamber inlet, not the flame, supplies most of the damping: replacing it with a lossless area change leaves an essentially neutral passive network (test: `test_the_dump_plane_carries_the_passive_damping`), and the flame is both resistive and reactive, adding damping while pulling the modal frequency up by ten hertz (test: `test_the_flame_pulls_the_frequency_up_and_adds_damping`).
The entropy wave the flame sheds leaves through the pressure-release outlet without being converted back into sound, so the full operator and its isentropic reduction share the eigenvalue (test: `test_the_entropy_wave_is_a_spectator_at_an_open_end`); this is the negative control matching the indirect-noise benchmark below, where a choked outlet makes the same wave decisive.

The comparison is like-for-like only once two conventions of [@li_oscilos_2017] are matched: its calorically perfect gas with $\gamma = 1.4$ held on both sides of the flame, and its treatment of the flame plane as a Borda–Carnot expansion followed by constant-area heat addition.
Both are stated in its source, and nothing else about the Nefes model is fitted to it.
The example notebook `examples/thermoacoustics/em2c_combustor.ipynb` walks the case through.

## BRS combustor: acoustic and intrinsic thermoacoustic modes {#sec-bench-brs-combustor}

The two thermoacoustic cases above check frequencies and growth rates against a flame whose response is prescribed in closed form.
This case checks something the others cannot: that the framework recovers the *kind* of each mode, separating the acoustic resonances of the geometry from the intrinsic (ITA) modes sustained by a feedback loop closing through the flame alone [@bomberg_2015].
The configuration is the perfectly premixed swirl-stabilized BRS test rig of [@komarek_polifke_2010], analysed as a network by [@emmert_2017], who report three dominant modes near $42$, $111$ and $315\;\mathrm{Hz}$, of which two are acoustic and one is intrinsic, together with the result that lowering the reflection at the combustor exit stabilizes the acoustic modes while *destabilizing* the intrinsic one.

The reference's network table fixes the mean state completely, and it is over-determined in a way that confirms the reading: for a calorically perfect gas its quoted impedance ratio is $\xi = \sqrt{1 + \theta}$, its area ratios and plenum radius imply the rig's $90 \times 90\;\mathrm{mm}$ chamber, and its density and sound speed put the rig at one atmosphere.
Nefes solves the network from the mass flow and the heat power and returns $\theta = 5.5898$ against the stated $5.59$ and an inlet Mach number of $0.00110$ against $0.0011$ (test: `test_mean_flow_reproduces_the_reported_operating_point`).

The one quantity the reference does not state in numbers is the flame transfer function, which it publishes only as a figure.
The curves of that figure are traced and shipped as `examples/thermoacoustics/data/brs_ftf.csv`, and the response is reconstructed as the finite impulse response the reference says it is, using the model of [dynamic sources](../theory/dynamic-sources.qmd#sec-source-impulse-response).
The recovery is confirmed against an independent record of the same quantity: the reference's dissertation prints the impulse response itself as a vector-graphics stem plot, whose exact sample values ship as `examples/thermoacoustics/data/brs_impulse_response.csv`; the two records agree in gain to an rms of $0.006$ and in phase within two printed pixels, both show the positive axial lobe followed by the negative swirl lobe that [@komarek_polifke_2010] describe for this burner, and the response peaks at the $4.8\;\mathrm{ms}$ convective delay the dissertation itself quotes (tests: `test_the_impulse_response_has_the_shape_of_a_swirl_flame`, `test_the_reconstructed_response_reproduces_the_digitized_curve`, `test_the_digitized_curve_agrees_with_the_reference_impulse_response`).

The comparison targets are the eigenvalues of the reference's own figures, read from their vector twins in the dissertation, where the coordinates come from the drawing commands rather than from pixels; they ship as `examples/thermoacoustics/data/brs_published_*.csv`.

With the flame passive the network has exactly the two acoustic modes the reference expects — the Helmholtz mode of plenum and swirler tube and the quarter wave of the hot chamber — at $54.4$ and $320.4\;\mathrm{Hz}$ against the published $54.5$ and $320.5$, both near-neutral as ideal lossless modes must be (test: `test_the_passive_network_matches_the_published_acoustic_modes`).
Activating the flame damps both and adds a mode near $106\;\mathrm{Hz}$ with no passive counterpart anywhere near it: the intrinsic mode (test: `test_the_flame_adds_one_mode_that_the_passive_network_does_not_have`).
The three dominant modes come out at $43.2$, $105.7$ and $319.5\;\mathrm{Hz}$ against the published $42.4$, $111.0$ and $315.6$, all stable, with the intrinsic mode the least damped and marginally so; the Helmholtz and intrinsic growth rates land within $2.5\;\mathrm{Hz}$ of the published values, and the near-degenerate pair at $315\;\mathrm{Hz}$ — an avoided crossing whose splitting is hypersensitive to the flame response — is compared through its robust mean (tests: `test_three_dominant_modes_match_the_published_frequencies`, `test_the_robust_growth_rates_match_the_published_values`, `test_the_near_degenerate_pair_matches_in_its_mean`, `test_the_intrinsic_mode_is_the_least_damped`).

The reference's pure intrinsic system — the burner mouth and flame between anechoic ends — is solved as a network of its own, and its dominant eigenvalue lands on the published square at $105.2\;\mathrm{Hz}$, $-22.1\;\mathrm{Hz}$ growth within a few hertz (test: `test_the_pure_ita_network_matches_the_dispersion_relation`).
Its scalar dispersion relation surfaced an inconsistency in the printed reference worth recording: solved as printed with the paper's own flame response, the relation predicts an *unstable* pure intrinsic mode, contradicting the paper's own all-stable figure.
The scalar equations hold for a heat release normalized by the flame-side velocity, while the published FTF is normalized at the burner mouth; the area ratio $\alpha_2$ bridges the two, and with it the published intrinsic spectrum is recovered (test: `test_the_pure_ita_relation_needs_the_normalization_bridge`).
Nefes assembles the jump conditions from the conservation laws, so the consistent coupling is automatic.

The residual is attributed rather than absorbed.
Reading the published figure to one pixel leaves $0.135\;\mathrm{rad}$ of phase uncertainty, which moves the intrinsic mode by about $\pm 4\;\mathrm{Hz}$; re-running with the dissertation's exact impulse response puts it at $110.8\;\mathrm{Hz}$, $-2.1\;\mathrm{Hz}$ against the published $111.0$, $-2.0$, so the visible offset is the figure-reading of the FTF and nothing else.
Shrinking the mean flow twentyfold, which removes the Mach-number terms the reference omits, moves every mode by less than a hertz (test: `test_the_mach_number_terms_do_not_carry_the_comparison`).

The reference's closing paradox is reproduced quantitatively: sweeping the outlet reflection from a perfectly reflecting open end toward an anechoic termination stabilizes the acoustic modes while driving the intrinsic mode unstable, with the neutral crossing near $|R| = 0.92$ and the anechoic endpoint at $+23.7\;\mathrm{Hz}$ growth against the published $+23.0$ (tests: `test_reducing_the_outlet_reflection_destabilizes_the_intrinsic_mode`, `test_reducing_the_outlet_reflection_stabilizes_the_acoustic_mode`, `test_the_anechoic_outlet_endpoint_matches_the_published_track`).

The example notebook `examples/thermoacoustics/brs_combustor.ipynb` reproduces the reference's three quantitative figures — the three-system spectrum, the modulation sweep that assigns each full-system mode to its acoustic or intrinsic parent, and the reflection sweep — overlaid on the published eigenvalues, including the digitization and its uncertainty.

## Indirect combustion noise {#sec-bench-indirect-noise}

The entropy-coupling machinery is benchmarked against indirect noise: an entropy fluctuation generated at a flame is convected downstream and converted to sound at a compact nozzle, the mechanism by which combustion generates noise without a direct acoustic source.
The benchmark follows the established chain: an unsteady heat release generates an entropy fluctuation, the fluctuation convects to the nozzle with the duct's entropy phase, and the nozzle converts it to a reflected acoustic wave through the Marble–Candel entropy coupling.
The framework reproduces each step (generation at the flame, convection to the nozzle, and conversion to sound) and, as a deliberate negative control, confirms that an isentropic analysis with the entropy wave suppressed misses the indirect noise entirely.
That is the documented limitation, stated here as a checked result rather than an unsupported claim (tests: `test_entropy_converts_to_indirect_noise_at_nozzle`, `test_isentropic_analysis_misses_the_indirect_noise`).
The compact-orifice and nozzle two-ports underlying this chain are additionally checked against the De Domenico normalization and an independent Borda composition, confirming the scattering-matrix construction in the presence of a mean flow and composition (tests: `test_orifice_matches_independent_borda_composition`, `test_nonisentropic_nozzle_matches_composition`, `test_scattering_riemann_equals_dedomenico_normalisation`).

## Canonical acoustic two-ports {#sec-bench-canonical-two-ports}

The transfer-matrix layer is benchmarked against cases with a known closed-form two-port.
A lossless isentropic nozzle in a quiescent duct must transmit an acoustic wave with unit magnitude and reflect nothing, which the computed scattering matrix reproduces (test: `test_isentropic_nozzle_unit_transmission_zero_reflection`), and a straight duct must contribute exactly the lossless propagation phase $e^{-\mathrm{i}\omega L/\overline{c}}$, which the transmission coefficient matches (test: `test_duct_scattering_is_lossless_phase`).
These small cases anchor the two-port machinery to analytic values before it is trusted on the composite elements and networks of the larger benchmarks.

Taken together, the benchmarks confirm that the framework reproduces a steady compressible network, a self-excited thermoacoustic instability with its analytic frequencies and growth rates, the indirect-noise chain with its correct negative control, and the canonical acoustic two-ports.
This is the external evidence that complements the internal [verification](verification.md) and that the [validation map](validation-map.md) indexes.
Cases where no benchmark has been run because the physics is deferred are noted in [limitations](../theory/limitations.md).
