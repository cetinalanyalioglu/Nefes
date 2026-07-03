# Verification

Verification asks whether the framework solves its equations *correctly* — whether the numerics faithfully realize the mathematics — as distinct from validation, which asks whether the equations describe reality (the subject of [benchmarks](benchmarks.md)).
This document collects the internal consistency checks that answer the verification question: the derivative check, the invariance checks, the closure oracle, and the round-trip checks that confirm the forward and inverse operations are mutual inverses.
Each is a check the framework performs on *itself*, against a reference it can compute independently, so that a passing suite certifies the implementation rather than merely exercising it.

The presentation groups the checks by what they verify — the derivative engine, the structural invariances, the thermochemistry closure, and the acoustic forward/inverse consistency.

## The derivative engine

The most fundamental verification is that the Jacobian is exact, since every Newton step and every acoustic operator depends on it.
The assembled complex-step Jacobian is checked against a finite-difference Jacobian to tolerance, both at the operating point and away from it, so that agreement in a well-conditioned region does not mask a discrepancy elsewhere (tests: `test_cs_jacobian_matches_finite_difference`, `test_cs_jacobian_matches_fd_off_operating_point`).
Beyond the assembled Jacobian, each element kernel is verified analytic in its own right: a roll-call requires every element to register a probe, and a per-kernel sweep confirms that its complex-step and finite-difference derivatives agree across forward, reverse, near-zero, and near-choke states — the regimes where a hidden branch would surface (tests: `test_every_element_kernel_is_swept`, `test_kernel_complex_step_safe_across_regimes`).
A companion check confirms that the residual is continuously differentiable through the zero-flow state on every edge, the single point most prone to a concealed kink (test: `test_residual_c1_through_zero_flow_all_edges`).

## Structural invariances

Two invariances that the theory requires are verified numerically rather than taken on trust.
The first is *edge-direction-flip invariance*: an element's physics must not depend on the arbitrary reference arrow assigned to an edge at build time, so flipping an arrow must leave the converged state unchanged.
This is checked directly, and the check exists because the invariance is easy to break — writing a convective momentum flux with a sign factor silently violates it (test: `test_edge_direction_invariance`).
The second is the *characteristic Newton-invariance*: because the per-edge map between network variables and wave amplitudes is an invertible change of coordinates, solving the Newton system in either representation must yield the identical update.
The map's exactness and invertibility are verified as the premise of that theorem (tests: `test_characteristic_maps_are_inverse`, `test_characteristic_amplitude_relations`), and the theorem itself follows as an algebraic identity rather than a numerical result (see [characteristics](../theory/characteristics.md)).

## The thermochemistry closure

The reacting closure is verified against an independent chemical-kinetics package used as an oracle, run in the separate environment that carries it (see [reproducibility](../design/reproducibility.md)).
The equilibrium kernel's temperature, density, and composition are checked against the oracle's equilibrium solve at matched conditions, and the frozen state reconstructed from transported mixture fractions is checked against the oracle's frozen mixture (tests: `test_kernel_matches_thermolib`, `test_frozen_from_xi_matches_thermolib`).
The kinetic-energy-coupled recovery — the outer root that returns the exact static state rather than the low-Mach approximation — is verified against the oracle evaluated at the KE-coupled static enthalpy, so that the coupling as well as the equilibrium is confirmed (test: `test_ke_burnt_static_matches_oracle`).

## Forward and inverse consistency

The acoustic layer is verified by checking that operations which should be mutual inverses are.
The transfer-matrix and scattering-matrix representations of a two-port are converted back and forth, and across the several variable flavors, with the requirement that the round-trip returns the original to tolerance — a check that the closed-form conversions are correct and analyticity-preserving (tests: `test_transfer_scattering_round_trip`, `test_flavor_round_trip`).
The strongest forward/inverse check is the *identification round-trip*: a known element is placed in a network, its acoustic response is computed forward, and the identification is then asked to recover the element from that response given a model of the rest — the recovered transfer matrix or flame transfer function must match the element it was measured from (tests: `test_identify_transfer_matrix_cascade`, `test_identify_single_input_ftf`).
An important part of this check is that the *conditioning* diagnostic behaves correctly: where a measurement cannot separate the unknown, the reported condition number rises and the recovery degrades gracefully rather than returning a confident wrong answer (test: `test_identify_noise_degrades_gracefully`).

Together these checks verify that the implementation realizes the mathematics — exact derivatives, honoured invariances, a faithful closure, and consistent forward and inverse acoustics — leaving the separate question of whether the equations match physical reality to the literature comparisons of [benchmarks](benchmarks.md).
