"""Perturbation network built on a converged mean flow.

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

All Python/SciPy, no new @njit kernel.  Implements the transfer / scattering
matrix analysis (:func:`perturbation_response`) and the linear
stability analysis (:func:`eigenmodes` -- the nonlinear eigenproblem
``det A(omega) = 0`` by Beyn's contour-integral method).  Both operate on the
*same* assembled operator ``A(omega) = J_alg + i*omega*M + P + S``: the storage
``M`` (a finite-volume :func:`~nefes.elements.catalog.cavity`) and the dynamic source
``S`` (a flame / mass source) drop into both analyses unchanged when their producing
element is present.
"""

from .continuation import (
    FiniteImpulseResponse,
    RationalFit,
    continuation_warning,
    finite_impulse_response,
    fit_impulse_response,
    rational_fit,
)
from .fields.cuton import (
    ALPHA_CIRCULAR,
    CutOnReport,
    DuctCutOn,
    cuton_frequency,
    duct_cuton_frequencies,
)
from .fields.duct import DuctAcoustics, scattering_2port
from .fields.modeshape import DuctSegment, NetworkGeometry, PathField, build_geometry, reconstruct_field
from .fields.power import (
    BoundaryPower,
    ForcedPowerBalance,
    ModalEnergyBalance,
    acoustic_energy_density,
    acoustic_flux_spectrum,
    acoustic_intensity,
    boundary_power,
    compact_power_spectrum,
    duct_energy_spectrum,
    forced_power_balance,
    intensity_along_network,
    modal_energy_balance,
    passive_reflection_bound,
)
from .identify import (
    TransferFunctionIdentification,
    TransferMatrixIdentification,
    UnknownTransferMatrix,
    identify_transfer_function,
    identify_transfer_matrix,
    unknown_dynamic_source,
)
from .matrix import FreqMatrix, PortState, ScatteringMatrix, TransferMatrix
from .operator import matrices
from .operator.boundary_bc import PerturbationBC
from .operator.characteristics import (
    BASIS_LABELS,
    basis_block_from_state,
    basis_matrix,
    char_to_dq,
    char_to_dx,
    dx_to_char,
    edge_caloric,
    edge_transforms,
)
from .operator.matrices import (
    partition,
    scattering_labels,
    sm_to_tm,
    tm_in_basis,
    tm_to_sm,
    wave_signs,
    wave_speeds,
)
from .operator.operator import AcousticBlocks, assemble_acoustic, build_acoustic_blocks
from .operator.stamps import boundary_forcing
from .operator.terminals import Terminal, find_terminals
from .operator.verify import verify_acoustic
from .response.forced import CompositionalNoiseWarning, ForcedResponse, forced_response
from .response.response import (
    PerturbationField,
    PerturbationResponse,
    TransferMatrixWarning,
    excite_perturbation,
    perturbation_response,
)
from .stability.contour import Contour, beyn, circle_contour, ellipse_contour, lu_logdet_phase, winding_count
from .stability.eigenmodes import EigenmodeResult, EigenmodeWarning, build_operator, eigenmodes
from .stability.nyquist import (
    NyquistResponse,
    NyquistStabilityMap,
    NyquistWarning,
    nyquist_stability,
    nyquist_stability_map,
    open_loop_response,
)
from .stability.sensitivity import EigenmodeSensitivityResult, SensitivityWarning, eigenvalue_sensitivities
from .stability.trajectory import TrajectoryBranch, TrajectoryResult, TrajectoryWarning, eigenvalue_trajectory

# perturbation-network primary names (thin aliases over the original spellings)
build_blocks = build_acoustic_blocks
assemble_operator = assemble_acoustic
PerturbationBlocks = AcousticBlocks
verify_perturbation = verify_acoustic

__all__ = [
    # characteristic maps + flavors
    "char_to_dx",
    "char_to_dq",
    "dx_to_char",
    "edge_caloric",
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
    "forced_response",
    "ForcedResponse",
    "CompositionalNoiseWarning",
    "boundary_forcing",
    # response + matrices
    "perturbation_response",
    "PerturbationResponse",
    "excite_perturbation",
    "PerturbationField",
    "TransferMatrixWarning",
    "find_terminals",
    "Terminal",
    "matrices",
    "tm_in_basis",
    "tm_to_sm",
    "sm_to_tm",
    "partition",
    "scattering_labels",
    "wave_speeds",
    "wave_signs",
    # frequency-domain complex-matrix descriptors
    "TransferMatrix",
    "ScatteringMatrix",
    "PortState",
    "FreqMatrix",
    # identification (recover an element's response from a measured transfer matrix)
    "identify_transfer_matrix",
    "identify_transfer_function",
    "TransferMatrixIdentification",
    "TransferFunctionIdentification",
    "UnknownTransferMatrix",
    "unknown_dynamic_source",
    # duct acoustics oracle
    "DuctAcoustics",
    "scattering_2port",
    # plane-wave validity (higher-order-mode cut-on)
    "cuton_frequency",
    "duct_cuton_frequencies",
    "CutOnReport",
    "DuctCutOn",
    "ALPHA_CIRCULAR",
    # analytic continuation of tabulated transfer functions / reflection coefficients
    "RationalFit",
    "rational_fit",
    "continuation_warning",
    "fit_impulse_response",
    "finite_impulse_response",
    "FiniteImpulseResponse",
    # acoustic-power diagnostics
    "acoustic_intensity",
    "acoustic_energy_density",
    "passive_reflection_bound",
    "boundary_power",
    "BoundaryPower",
    "acoustic_flux_spectrum",
    "compact_power_spectrum",
    "intensity_along_network",
    "duct_energy_spectrum",
    "forced_power_balance",
    "ForcedPowerBalance",
    "modal_energy_balance",
    "ModalEnergyBalance",
    # stability / eigenmodes (nonlinear eigenproblem det A(omega) = 0)
    "eigenmodes",
    "EigenmodeResult",
    "EigenmodeSensitivityResult",
    "SensitivityWarning",
    "eigenvalue_sensitivities",
    "EigenmodeWarning",
    "build_operator",
    # eigenvalue trajectories (parameter continuation of the spectrum)
    "eigenvalue_trajectory",
    "TrajectoryResult",
    "TrajectoryBranch",
    "TrajectoryWarning",
    # spatial mode-shape reconstruction (analytic intra-duct field)
    "build_geometry",
    "reconstruct_field",
    "NetworkGeometry",
    "PathField",
    "DuctSegment",
    "Contour",
    "ellipse_contour",
    "circle_contour",
    "beyn",
    "winding_count",
    "lu_logdet_phase",
    # Nyquist open-loop stability (real-frequency sweep; entropy/reacting regime)
    "open_loop_response",
    "nyquist_stability",
    "nyquist_stability_map",
    "NyquistResponse",
    "NyquistStabilityMap",
    "NyquistWarning",
]
