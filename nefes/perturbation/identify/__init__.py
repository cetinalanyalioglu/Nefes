"""Identify an element's dynamic response from a measured network transfer matrix.

Two unknowns, one method: a blackbox 2-port's transfer matrix, or the
transfer function(s) of a flame / mass-source feedback.  Mark the unknown on its element,
supply a transfer matrix measured between two of its edges, and de-embed -- the rest of the
network being known -- by a per-frequency linear solve over the perturbation operator.

See :func:`identify_transfer_matrix`, :func:`identify_transfer_function`, and the markers
:class:`UnknownTransferMatrix` / :func:`unknown_dynamic_source`.
"""

from .core import (
    identify_transfer_matrix,
    identify_transfer_function,
    TransferMatrixIdentification,
    TransferFunctionIdentification,
)
from .markers import UnknownTransferMatrix, unknown_dynamic_source

__all__ = [
    "identify_transfer_matrix",
    "identify_transfer_function",
    "TransferMatrixIdentification",
    "TransferFunctionIdentification",
    "UnknownTransferMatrix",
    "unknown_dynamic_source",
]
