"""Dynamic source descriptor ``S(omega)`` -- a forward-compatibility provision.

A mass source (and, later, a flame) may carry a **dynamic** part: a fluctuating
injection whose amplitude responds to the unsteady flow elsewhere in the network
-- e.g. a fuel mass-flow that fluctuates with the acoustic velocity ``u'`` at an
upstream reference edge (the classic ``n-tau`` flame-transfer-function coupling).

This module defines the *descriptor* only.  The mean-flow solve ignores it
entirely (a constant mean source is acoustically passive), and the perturbation
layer does not consume it yet -- the dynamic ``S(omega)`` stamping is the next
phase.  Carrying the descriptor end-to-end now means that phase needs no
architectural change: the reference edges, the modulated quantity, and the
transfer function are already attached to the element.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DynamicSource:
    """How a source term's fluctuation responds to the unsteady flow.

    Parameters
    ----------
    ref_edges : list of int
        Edge id(s) whose fluctuation drives the dynamic source (e.g. the edge
        just upstream of a flame for an ``n-tau`` model).
    quantity : str
        Which fluctuation at the reference edge drives it: ``"u"`` (velocity),
        ``"p"``, ``"rho"`` or ``"mdot"``.
    target : str
        Which source term is modulated: ``"mdot"`` (injected mass-flow),
        ``"h_t"`` (injected enthalpy) or ``"Qdot"`` (a heat-release flame).
    transfer : object, optional
        The transfer function ``S(omega)``: a callable ``omega -> complex`` or an
        ``(n, tau)`` pair for the ``n * exp(-i omega tau)`` model.  ``None`` is a
        placeholder (descriptor attached, response not yet specified).
    gain : float
        Optional scalar gain multiplying the response.
    """

    ref_edges: List[int]
    quantity: str = "u"
    target: str = "mdot"
    transfer: Optional[object] = None
    gain: float = 1.0
