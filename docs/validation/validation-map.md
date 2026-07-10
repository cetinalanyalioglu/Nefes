# Validation map

This document records how physical claims in the theory track are checked.
Each row names the claim, the independent reference it is compared against (an analytic relation, an equilibrium calculation, or a published case), and the test or example that runs the comparison.
The tables are grouped by mean flow, thermochemistry, and acoustics with its inverse; where no check exists, the gap is marked openly rather than left unstated.
Broader consistency evidence and named literature cases appear in [verification](verification.md) and [benchmarks](benchmarks.md).

The test names are the routines in the codebase and can be run as written.

## Mean-flow claims

| Claim | Reference | Check |
|---|---|---|
| Density recovery exists and is unique for any flow state with $p, h_t > 0$ | analytic monotonicity of $F(\varrho)$ | `test_real_root_matches_brentq`, `test_round_trip_physical_state` |
| The recovered state is direction-independent where it must be | $p_t, T_t$ depend on $M^2$ | `test_round_trip_physical_state` |
| An isentropic area change reproduces the classical jump | analytic compressible-flow relations | `test_subsonic_nozzle_matches_isentropic` |
| A sudden expansion loses the Borda–Carnot total pressure | momentum balance | `test_expansion_unaffected_by_cc` |
| A sudden contraction loss follows $K_c = (1/C_c - 1)^2$ | vena-contracta model | `test_contraction_loss_matches_Kc`, `test_contraction_lossless_default_conserves_pt` |
| Element residuals are invariant to an edge-arrow flip | direction-convention algebra | `test_edge_direction_invariance` |
| A long chain converges from exactly zero flow | continuation well-posedness | `test_long_serial_chain_cold_start` |
| A pressure-driven quiescent network converges | artificial-resistance continuation | `test_quiescent_cold_start_converges` |
| A symmetric branching network resolves its split | Levenberg–Marquardt damping | `test_many_parallel_branches_converge` |
| Choking saturates the mass flow at the sonic value | critical mass flux | `test_choked_nozzle_saturates_mass_flow` |
| The critical pressure ratio is the knee of the operating map | $p^\ast/p_t \approx 0.528$ | `test_critical_pressure_ratio_is_the_knee` |
| A choked orifice discharge detaches its exit pressure upward | underexpanded discharge | `test_choked_nozzle_outlet_critical_mass_flux` |

## Thermochemistry claims

| Claim | Reference | Check |
|---|---|---|
| The equilibrium engine matches an independent equilibrium solver | Cantera oracle (needs a Cantera + numba env) | `test_cantera_validation`; public/solver packings agree via `test_public_and_solver_paths_agree`, `test_frozen_from_xi_matches_properties` |
| The kinetic-energy-coupled reacting state is exact | equilibrium oracle at the KE-coupled enthalpy | `test_ke_burnt_static_matches_oracle` |
| A transported passive scalar mixes as the mass-weighted donor | convex-combination mixing | `test_passive_tracer_mixes_mass_weighted` |
| A passive scalar does not perturb the mean flow | scalar-registry squareness | `test_passive_tracer_does_not_perturb_mean_flow` |
| A carried scalar stays realizable in $[0,1]$ | convexity of the donor mix | `test_passive_tracer_realizable` |
| The marker gate selects the frozen/burnt closure and self-corrects | bimodal-marker convergence | `test_auto_reacting_network_is_marker_gated`, `test_marker_self_corrects_any_seed` |
| The marker-blended mean flow matches the hard closure | frozen/equilibrium limits | `test_mean_flow_matches_hard_closure` |

## Acoustic and identification claims

| Claim | Reference | Check |
|---|---|---|
| The operator reduces to the base Jacobian at zero frequency (passive) | $\mathbf{A}(0) = \overline{\mathbf{J}}$ | `test_zero_frequency_operator_equals_jacobian` |
| A duct carries the lossless propagation phase | $e^{-\mathrm{i}\omega\tau_+}$ | `test_meanflow_duct_tau_plus_phase`, `test_duct_scattering_is_lossless_phase` |
| A cavity stamps the finite-volume compliance | $C = V/\overline{\varrho}\,\overline{c}^{\,2}$ | `test_cavity_storage_is_the_compliance` |
| A choked-nozzle outlet reflects with the Marble–Candel coefficient | compact-nozzle reflection + entropy coupling | `test_choked_nozzle_outlet_marble_candel` |
| The characteristic maps are exact and invertible | linearized state definitions | `test_characteristic_maps_are_inverse`, `test_characteristic_amplitude_relations` |
| Transfer and scattering matrices round-trip, and across flavors | closed-form conversions | `test_transfer_scattering_round_trip`, `test_flavor_round_trip` |
| An $n$–$\tau$ flame drives a self-excited instability | analytic dispersion root | `test_n_tau_flame_drives_self_excited_instability` |
| The modal frequency and growth rate of a published combustor are reproduced | OSCILOS [@li_oscilos_2017] on the EM2C combustor [@palies_2011] | `test_dominant_mode_matches_oscilos`, `test_mean_flow_matches_the_reported_operating_point` |
| A shed entropy wave is inert at a pressure-release outlet | no entropy-to-acoustic conversion without acceleration | `test_the_entropy_wave_is_a_spectator_at_an_open_end` |
| The eigensolver's mode count is certified complete | argument-principle winding | `test_eigenmodes_certified_count_matches` |
| An eigenvalue-free region yields no modes, whatever the band | argument-principle winding as the Beyn rank | `test_beyn_moment_rank_is_ambiguous_on_an_empty_contour`, `test_no_modes_survive_an_eigenvalue_free_region`, `test_eigenmodes_are_insensitive_to_the_band_edge` |
| A mode is told from an arbitrary frequency on an ill-conditioned operator | residual on the equilibrated $\mathbf{D}_r\mathbf{A}\mathbf{D}_c$ | `test_equilibrated_residual_separates_a_mode_from_an_arbitrary_point` |
| The search sub-contours cover the region they certify | elliptical tiling geometry | `test_subcontours_cover_the_counted_region` |
| The growth-rate sign matches the boundary energy budget | Myers acoustic energy | `test_boundary_power_sign_matches_growth_every_mode` |
| The Nyquist count agrees with the eigensolver | matrix-determinant lemma | `test_unstable_count_matches_eigenmodes` |
| A de-embedded element reproduces its measured response | Woodbury identity | `test_identify_transfer_matrix_cascade`, `test_identify_single_input_ftf` |
| The isentropic analysis omits indirect noise (as documented) | entropy-to-acoustic coupling | `test_isentropic_analysis_misses_the_indirect_noise` |

## Coverage remarks

The entries above name one check per claim, not every check that backs it: many claims are tested several times, and the complete list lives in the tests themselves; this table is only a guide.
The items left open in [limitations](../theory/limitations.md) (finite-rate chemistry, supersonic internal flow, and the compositional-noise gap at analytic terminal closures) are deliberately omitted here, because they mark what the present version does not yet cover or only approximates; where a partial check exists (for example a warning when compositional noise is dropped at a closure), it is noted in the limitations document, not counted as a proven result.

Basic consistency checks that support many of these entries are gathered in [verification](verification.md); the named literature cases and their quantitative agreement appear in [benchmarks](benchmarks.md).
