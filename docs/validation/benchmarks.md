# Benchmarks

Where [verification](verification.md) checks that the framework solves its equations correctly, this document checks that those equations reproduce established results: named cases from the literature and from analytic theory that a compressible-network and thermoacoustic solver is expected to recover.
Each benchmark states the case, the reference it is compared against, and the quantitative agreement obtained, with the test that performs the comparison.
The cases span the framework's main claims: a steady compressible network, a self-excited thermoacoustic instability, indirect combustion noise, and a canonical acoustic two-port.

## Greyvenstein & Laurie compressed-air network

The steady mean-flow solver is benchmarked against Example 3 of [@greyvenstein_laurie_1994], a compressed-air pipe network solved in that work by an independent method and reported as a table of branch mass flows and nodal pressures.

The network is assembled as a Nefes model with the corresponding pipe, junction, and boundary elements, solved from a cold start, and compared entry by entry against the published table.
The converged branch mass flows and nodal pressures match the reference table, and the flow is confirmed to sit in the low-Mach regime the original analysis assumes, so the comparison is like-for-like (tests: `test_converges`, `test_mass_flows_match_table_iv`, `test_node_pressures_match_table_iv`, `test_low_mach_regime`).

This case exercises the mean-flow machinery end to end (friction-pipe losses, many junction couplings, boundary conditions, and the continuation that finds the operating point) against an external result rather than an internal consistency check.

## Rijke-tube thermoacoustic stability

The acoustic and dynamic-source layers are benchmarked against the Rijke tube, the canonical thermoacoustic oscillator: a duct with a compact heat source whose unsteady heat release couples to the acoustic field and, for a destabilizing time lag, drives a self-excited instability.
This benchmark was chosen as it has an analytic counterpart: the dispersion relation of a two-region duct with a compact flame, whose roots give the modal frequencies and growth rates in closed form, so the framework's eigenmodes can be compared against exact values rather than against another solver.

With no unsteady heat release the tube is a passive resonator and the computed modal frequencies match the analytic resonances (test: `test_passive_flame_matches_analytic_frequencies`); with an $n$–$\tau$ flame a mode acquires a positive growth rate and both its frequency and growth match the analytic root, and the same flame is stabilizing at one time lag and destabilizing at another, reproducing the $n$–$\tau$ stability band (tests: `test_n_tau_flame_drives_self_excited_instability`, `test_n_tau_lag_sets_stability_band`).

The reacting counterpart, with an equilibrium flame in place of the prescribed heat source, ignites and matches the same analytic construction, confirming that the reacting closure and the acoustic source compose correctly (tests: `test_reacting_flame_ignites_and_matches_analytic`, `test_reacting_n_tau_flame_matches_analytic_instability`).

## EM2C combustor, cross-checked against OSCILOS

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

## Indirect combustion noise

The entropy-coupling machinery is benchmarked against indirect noise: an entropy fluctuation generated at a flame is convected downstream and converted to sound at a compact nozzle, the mechanism by which combustion generates noise without a direct acoustic source.
The benchmark follows the established chain: an unsteady heat release generates an entropy fluctuation, the fluctuation convects to the nozzle with the duct's entropy phase, and the nozzle converts it to a reflected acoustic wave through the Marble–Candel entropy coupling.
The framework reproduces each step (generation at the flame, convection to the nozzle, and conversion to sound) and, as a deliberate negative control, confirms that an isentropic analysis with the entropy wave suppressed misses the indirect noise entirely.
That is the documented limitation, stated here as a checked result rather than an unsupported claim (tests: `test_entropy_converts_to_indirect_noise_at_nozzle`, `test_isentropic_analysis_misses_the_indirect_noise`).
The compact-orifice and nozzle two-ports underlying this chain are additionally checked against the De Domenico normalization and an independent Borda composition, confirming the scattering-matrix construction in the presence of a mean flow and composition (tests: `test_orifice_matches_independent_borda_composition`, `test_nonisentropic_nozzle_matches_composition`, `test_scattering_riemann_equals_dedomenico_normalisation`).

## Canonical acoustic two-ports

The transfer-matrix layer is benchmarked against cases with a known closed-form two-port.
A lossless isentropic nozzle in a quiescent duct must transmit an acoustic wave with unit magnitude and reflect nothing, which the computed scattering matrix reproduces (test: `test_isentropic_nozzle_unit_transmission_zero_reflection`), and a straight duct must contribute exactly the lossless propagation phase $e^{-\mathrm{i}\omega L/\overline{c}}$, which the transmission coefficient matches (test: `test_duct_scattering_is_lossless_phase`).
These small cases anchor the two-port machinery to analytic values before it is trusted on the composite elements and networks of the larger benchmarks.

Taken together, the benchmarks confirm that the framework reproduces a steady compressible network, a self-excited thermoacoustic instability with its analytic frequencies and growth rates, the indirect-noise chain with its correct negative control, and the canonical acoustic two-ports.
This is the external evidence that complements the internal [verification](verification.md) and that the [validation map](validation-map.md) indexes.
Cases where no benchmark has been run because the physics is deferred are noted in [limitations](../theory/limitations.md).
