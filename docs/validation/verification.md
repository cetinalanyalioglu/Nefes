# Verification

Verification asks whether the framework solves its equations correctly: whether the numerics faithfully realize the mathematics.
Validation, by contrast, asks whether the equations describe reality, which [benchmarks](benchmarks.md) addresses.
This document collects the internal consistency checks that answer the verification question: derivatives, structural invariances, the reacting closure, and checks that forward and inverse acoustic operations undo each other.
Each check compares the implementation against something it can compute independently, so a passing result confirms the mathematics is implemented correctly, not merely that the code runs.

## The derivative engine {#sec-verif-derivative-engine}

The most fundamental verification is that the Jacobian is exact, since every Newton step and every acoustic operator depends on it.
The complex-step Jacobian is checked against a finite-difference Jacobian to tolerance, both at the operating point and away from it, so agreement in a well-conditioned region does not mask a discrepancy elsewhere (tests: `test_cs_jacobian_matches_finite_difference`, `test_cs_jacobian_matches_fd_off_operating_point`).
Each element kernel is also checked on its own: every element must supply a probe, and each kernel is tested so that its complex-step and finite-difference derivatives agree across forward, reverse, near-zero, and near-choke states, the regimes where a hidden branch would surface (tests: `test_every_element_kernel_is_swept`, `test_kernel_complex_step_safe_across_regimes`).
A companion check confirms that the residual is continuously differentiable through the zero-flow state on every edge, the single point most prone to a concealed kink (test: `test_residual_c1_through_zero_flow_all_edges`).

## Structural invariances {#sec-verif-structural-invariances}

Two invariances that the theory requires are verified numerically rather than taken on trust.
The first is edge-direction-flip invariance: an element's physics must not depend on the arbitrary reference arrow assigned to an edge at build time, so flipping an arrow must leave the converged state unchanged.
This is checked directly, because the invariance is easy to break; writing a convective momentum flux with a sign factor silently violates it (test: `test_edge_direction_invariance`).
The second is characteristic Newton-invariance: because the per-edge map between network variables and wave amplitudes is an invertible change of coordinates, solving the Newton system in either representation must yield the identical update.
The map's exactness and invertibility are verified as the premise of that result (tests: `test_characteristic_maps_are_inverse`, `test_characteristic_amplitude_relations`), and the update itself follows as an algebraic identity rather than a numerical check (see [characteristics](../theory/characteristics.md#sec-char-newton-invariance)).

## Fanno-flow convergence {#sec-verif-fanno-flow}

The momentum formulation of a pipe segment is checked against the independent classical Fanno relation

$$
\mathcal F(M)
=
\frac{1-M^2}{\gamma M^2}
+
\frac{\gamma+1}{2\gamma}
\ln\!\left[
\frac{(\gamma+1)M^2}{2+(\gamma-1)M^2}
\right],
\qquad
\frac{f_D\Delta x}{D}
=
\mathcal F(M_1)-\mathcal F(M_2).
$$

SciPy's bracketing root solver inverts this relation at every station without calling Nefes, and the pressure and temperature profiles use the exact starred-state ratios.
Grid refinement from 8 to 64 segments reduces the worst relative profile error monotonically for a moderate subsonic case, reaching below $10^{-3}$ at 64 segments with an observed order above 1.8.
A separate near-choke case ends at $M=0.95$ and reaches a worst relative error below $3\times10^{-3}$ at 64 segments.
The same oracle is exercised in reverse flow, and zero friction reduces both pipe formulations to the lossless relation (tests: `test_momentum_fanno_converges_to_analytic_profile`, `test_momentum_fanno_near_choke_converges_to_analytic_profile`, `test_momentum_fanno_is_orientation_safe_in_reverse_flow`, `test_zero_friction_pipe_is_lossless`).

## The thermochemistry closure {#sec-verif-thermochemistry-closure}

The reacting closure is verified against an independent chemical-kinetics package (Cantera), run in an environment that carries both Cantera and numba (see [reproducibility](../design/reproducibility.md#sec-repro-environments)).
The compiled equilibrium engine's temperature, density, and composition are checked against Cantera's equilibrium solve at matched conditions (test: `test_cantera_validation`), and the solver and the public entry point are checked to reach that same engine identically: the burnt state and the frozen state reconstructed from transported mixture fractions agree whether they are assembled for a standalone call or inside the network (tests: `test_public_and_solver_paths_agree`, `test_frozen_from_xi_matches_properties`).
The kinetic-energy-coupled recovery, the outer root that returns the exact static state rather than the low-Mach approximation, is verified against Cantera evaluated at the KE-coupled static enthalpy, so that the coupling as well as the equilibrium is confirmed (test: `test_ke_burnt_static_matches_oracle`).

## Forward and inverse consistency {#sec-verif-forward-inverse}

The acoustic layer is verified by checking that operations which should undo each other do.
The transfer-matrix and scattering-matrix representations of a two-port are converted back and forth, and across the several variable flavors, with the requirement that the result returns to the original to tolerance (tests: `test_transfer_scattering_round_trip`, `test_flavor_round_trip`).
The strongest check is the identification round-trip: a known element is placed in a network, its acoustic response is computed forward, and identification is then asked to recover the element from that response given a model of the rest; the recovered transfer matrix or flame transfer function must match the element it was measured from (tests: `test_identify_transfer_matrix_cascade`, `test_identify_single_input_ftf`).
Where a measurement cannot separate the unknown, the reported condition number rises and the recovery degrades gracefully rather than returning a confident wrong answer (test: `test_identify_noise_degrades_gracefully`).

Together these checks verify that the implementation realizes the mathematics: exact derivatives, honoured invariances, a faithful closure, and consistent forward and inverse acoustics.
The literature comparisons in [benchmarks](benchmarks.md) examine whether the equations match physical reality.
