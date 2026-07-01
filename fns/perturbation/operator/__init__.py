"""Operator-assembly layer for the perturbation network.

The pieces that build the operator ``A(omega) = J_alg + i*omega*M + P + S``:
characteristic transforms (:mod:`characteristics`), transfer/scattering-matrix
algebra (:mod:`matrices`), terminal discovery (:mod:`terminals`), operator
verification (:mod:`verify`), perturbation boundary conditions
(:mod:`boundary_bc`), the storage/source stamps (:mod:`stamps`), and the
assembler itself (:mod:`operator`).

Namespace only -- this ``__init__`` imports no submodule, so the parent
:mod:`fns.perturbation` package controls the (dependency-ordered) re-export.
"""
