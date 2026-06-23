"""Perturbation network built on a converged mean flow (theory.md s12).

A second analysis over the same compiled network and converged state -- not a
second solver.  It reuses the connectivity, the complex-step Jacobian (which is
the zero-frequency perturbation operator), and the frozen mean thermo state.

The 1-D Euler system carries **three** perturbation characteristics -- two
acoustic (``f``, ``g``) and one entropy/convected wave (``h``) -- so the
perturbation network has the **same variable count as the mean flow**.  Its
transfer and scattering matrices are therefore ``N x N`` (``N = 3`` for inert
flow, larger with reacting scalars), each entry complex (magnitude + phase).
This is more than acoustics: it is the linear-response twin of the mean-flow
network, sharing its equations exactly.

All Python/SciPy, no new @njit kernel.  v1 implements the transfer / scattering
matrix analysis (theory s12.7 (i), :func:`perturbation_response`) and the linear
stability analysis (s12.7 (ii), :func:`eigenmodes` -- the nonlinear eigenproblem
``det A(omega) = 0`` by Beyn's contour-integral method).  Both operate on the
*same* assembled operator ``A(omega)``; the storage ``M`` and source ``S`` faces
are wired but inert (no producing element yet), and drop into both analyses with no
change once an element produces them.
"""

from .characteristics import (
    char_to_dx,
    dx_to_char,
    edge_transforms,
    basis_matrix,
    basis_block_from_state,
    BASIS_LABELS,
)
from .operator import build_acoustic_blocks, assemble_acoustic, AcousticBlocks
from .verify import verify_acoustic
from .boundary_bc import PerturbationBC
from .terminals import Terminal, find_terminals
from .response import (
    perturbation_response,
    PerturbationResponse,
    excite_perturbation,
    PerturbationField,
    TransferMatrixWarning,
    acoustic_response,
    AcousticResponse,
)
from .forced import boundary_response, ForcedResponse
from .stamps import boundary_forcing
from . import matrices
from .matrices import (
    tm_in_basis,
    tm_to_sm,
    sm_to_tm,
    partition,
    scattering_labels,
    wave_speeds,
    wave_signs,
)
from .duct import duct_modes, DuctAcoustics
from .drivers import modes_from_det, scattering_2port
from .contour import Contour, ellipse_contour, circle_contour, beyn, winding_count, lu_logdet_phase
from .eigenmodes import eigenmodes, EigenmodeResult, EigenmodeWarning

# perturbation-network primary names (thin aliases over the original spellings)
build_blocks = build_acoustic_blocks
assemble_operator = assemble_acoustic
PerturbationBlocks = AcousticBlocks
verify_perturbation = verify_acoustic

__all__ = [
    # characteristic maps + flavors
    "char_to_dx",
    "dx_to_char",
    "edge_transforms",
    "basis_matrix",
    "basis_block_from_state",
    "BASIS_LABELS",
    # operator
    "build_blocks",
    "assemble_operator",
    "PerturbationBlocks",
    "verify_perturbation",
    "build_acoustic_blocks",
    "assemble_acoustic",
    "AcousticBlocks",
    "verify_acoustic",
    # boundary conditions + forced response
    "PerturbationBC",
    "boundary_response",
    "ForcedResponse",
    "boundary_forcing",
    # response + matrices
    "perturbation_response",
    "PerturbationResponse",
    "excite_perturbation",
    "PerturbationField",
    "TransferMatrixWarning",
    "find_terminals",
    "Terminal",
    "acoustic_response",
    "AcousticResponse",
    "matrices",
    "tm_in_basis",
    "tm_to_sm",
    "sm_to_tm",
    "partition",
    "scattering_labels",
    "wave_speeds",
    "wave_signs",
    # ducts / drivers
    "duct_modes",
    "DuctAcoustics",
    "modes_from_det",
    "scattering_2port",
    # stability / eigenmodes (nonlinear eigenproblem det A(omega) = 0)
    "eigenmodes",
    "EigenmodeResult",
    "EigenmodeWarning",
    "Contour",
    "ellipse_contour",
    "circle_contour",
    "beyn",
    "winding_count",
    "lu_logdet_phase",
]
