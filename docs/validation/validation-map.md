# Validation map

This document is the traceability table that binds the framework's claims to their evidence: for each falsifiable physical claim, the reference it is checked against — an analytic relation, a chemical-equilibrium oracle, or a literature case — and the test or example that performs the check.
It is the single place a reviewer can consult to confirm that no claim in the theory track stands on assertion alone, and it is the index the [verification](verification.md) and [benchmarks](benchmarks.md) documents expand.
A claim that carries no check is, by the project's own standard, flagged as such rather than presented as established; the tables below therefore aim to be honest about coverage, not merely to list passes.

The entries are grouped by layer — mean flow, thermochemistry, and acoustics with its inverse — and each names the claim, the reference it is measured against, and the test that measures it.
The test names are the actual functions in the suite, so an entry can be run directly.

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
| The eigensolver's mode count is certified complete | argument-principle winding | `test_eigenmodes_certified_count_matches` |
| An eigenvalue-free region yields no modes, whatever the band | argument-principle winding as the Beyn rank | `test_beyn_moment_rank_is_ambiguous_on_an_empty_contour`, `test_no_modes_survive_an_eigenvalue_free_region`, `test_eigenmodes_are_insensitive_to_the_band_edge` |
| A mode is told from an arbitrary frequency on an ill-conditioned operator | residual on the equilibrated $\mathbf{D}_r\mathbf{A}\mathbf{D}_c$ | `test_equilibrated_residual_separates_a_mode_from_an_arbitrary_point` |
| The search sub-contours cover the region they certify | elliptical tiling geometry | `test_subcontours_cover_the_counted_region` |
| The growth-rate sign matches the boundary energy budget | Myers acoustic energy | `test_boundary_power_sign_matches_growth_every_mode` |
| The Nyquist count agrees with the eigensolver | matrix-determinant lemma | `test_unstable_count_matches_eigenmodes` |
| A de-embedded element reproduces its measured response | Woodbury identity | `test_identify_transfer_matrix_cascade`, `test_identify_single_input_ftf` |
| The isentropic analysis omits indirect noise (as documented) | entropy-to-acoustic coupling | `test_isentropic_analysis_misses_the_indirect_noise` |

## Coverage remarks

Two points of honesty about the table belong here.
First, the entries above are the *representative* checks for each claim, not the whole suite: many claims carry several tests, and the full set is the test suite itself, of which this table is the navigable index.
Second, the claims deferred in [limitations](../theory/limitations.md) — finite-rate chemistry, supersonic internal flow, and the compositional-noise gap at analytic terminal closures — are *not* listed as validated, because they are scope boundaries or known approximations rather than established results; where a partial check exists (for instance the warning that fires at a compositional-noise-dropping closure), it is recorded against the limitation, not as a validated capability.

The internal consistency checks that underlie many of these entries — the derivative, invariance, and round-trip verifications — are collected in [verification](verification.md), and the named literature cases are presented with their quantitative agreement in [benchmarks](benchmarks.md).
