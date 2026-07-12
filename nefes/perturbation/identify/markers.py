"""Markers for the quantities an identification recovers.

Attach one to an element to declare *what is unknown*, set up the network, and hand the
problem to :mod:`nefes.perturbation.identify`.  The mean flow ignores them; a forward
perturbation run before identification treats the element as its passive default (an
isentropic area change for a transfer-matrix element, a silent source for a flame).

Public: :class:`UnknownTransferMatrix`, :func:`unknown_dynamic_source`.
"""

from __future__ import annotations

from typing import List, Sequence

from ...elements.dynamic_source import Constant, DynamicResponseTerm, DynamicSource


class UnknownTransferMatrix:
    """Marker: the 2-port transfer matrix of a
    :func:`~nefes.elements.catalog.transfer_matrix_element` is to be identified.

    Parameters
    ----------
    n : int, optional
        Matrix dimension to recover: ``2`` (acoustic ``(f, g)``) or ``3`` (adding the entropy
        wave ``h``, the default).  This sets how many independent measurement channels the
        de-embed needs.

    Notes
    -----
    The entropy channel (``n = 3``) is a genuine part of the recovery, not a placeholder.
    Transported reacting scalars (composition waves) are not among the recoverable channels
    here: the marker resolves only the acoustic and, at ``n = 3``, the entropy 2-port.
    """

    is_unknown = True

    def __init__(self, n: int = 3):
        if int(n) not in (2, 3):
            raise ValueError(f"an unknown transfer matrix must be 2x2 or 3x3; got n={n}")
        self.n = int(n)

    def __repr__(self):
        return f"UnknownTransferMatrix(n={self.n})"


def unknown_dynamic_source(terms: Sequence, *, target: str = "Qdot", q_mean=None) -> DynamicSource:
    """A :class:`~nefes.elements.dynamic_source.DynamicSource` whose transfer functions are unknown.

    Placeholder ``Constant(0)`` transfers make the source acoustically silent for any forward
    run; :func:`~nefes.perturbation.identify.identify_transfer_function` reads the terms'
    reference edges and quantities and recovers one transfer function per term.

    Parameters
    ----------
    terms : sequence
        One entry per unknown transfer function, each ``(ref_edge, quantity)`` or
        ``(ref_edge, quantity, gain)`` -- the reference edge and flow quantity (``"u"``,
        ``"p"``, ``"rho"``, ``"mdot"``, or ``"Z:<name>"``) the response is written against.
    target : {"Qdot", "mdot"}, optional
        Modulated source quantity (default heat release).
    q_mean : float, optional
        Mean of the modulated quantity (auto-derived when ``None``).
    """
    ft: List[DynamicResponseTerm] = []
    for t in terms:
        ref_edge, quantity = t[0], t[1]
        gain = t[2] if len(t) > 2 else 1.0
        ft.append(DynamicResponseTerm(Constant(0.0), int(ref_edge), quantity, float(gain)))
    ds = DynamicSource(terms=ft, target=target, q_mean=q_mean)
    ds.is_unknown = True
    return ds
