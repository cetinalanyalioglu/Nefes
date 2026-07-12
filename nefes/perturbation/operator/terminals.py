"""Boundary terminals of the perturbation network.

A *terminal* is a single-port boundary element (an inlet, an outlet, or a wall) and
its one incident edge -- the place where an incoming characteristic enters the
domain and the reflected/transmitted one leaves.  Both the measurement driver
(``response.py``) and the physical boundary stamp (``stamps.py``) iterate terminals,
so the definition lives here to keep them decoupled.
"""

from dataclasses import dataclass
from typing import List

from ...assembly.recover import ES_MDOT
from ...elements.ids import BOUNDARY_RIDS
from ...solver.report import states_table
from .._meanstate import accepts_solution


@dataclass
class Terminal:
    """A 1-port boundary edge where an incoming wave can be injected/read."""

    node: int  # the boundary element
    rid: int  # its residual id (one of BOUNDARY_RIDS)
    edge: int  # the single incident edge
    at_tail: bool  # True if the boundary is the edge's tail (wave enters as f)
    row: int  # the boundary element's single equation row
    incoming: int  # acoustic wave index injected here: 0 (f) if at_tail else 1 (g)
    outgoing: int  # the reflected/transmitted acoustic wave index read here
    inflowing: bool  # True if the mean flow *enters* the domain here (carries entropy in)

    def __repr__(self) -> str:
        """Readable 1-port summary: node, edge/face, the wave injected vs read, and flow direction."""
        face = "tail" if self.at_tail else "head"
        inj = "f" if self.incoming == 0 else "g"
        out = "f" if self.outgoing == 0 else "g"
        flow = "inflow" if self.inflowing else "outflow"
        return f"Terminal: node {self.node} on edge {self.edge} ({face}), inject {inj}' / read {out}', mean {flow}"


@accepts_solution
def find_terminals(prob, x_bar=None) -> List[Terminal]:
    """All 1-port boundary terminals of the network (edges at a boundary node).

    When ``x_bar`` is given, ``inflowing`` is set from the mean flow direction so
    the incoming entropy excitation can be placed at genuine inlets.

    Parameters
    ----------
    prob : CompiledProblem or Solution
        The compiled network.  Pass a solved :class:`nefes.Solution` to have its problem and
        mean state supplied for you (then omit ``x_bar``).
    x_bar : ndarray, optional
        Converged mean-flow state, shape ``(n_solve, E)``.  When given, each
        terminal's ``inflowing`` flag is set from the local mean-flow direction.  Supplied
        automatically when ``prob`` is a ``Solution``.

    Returns
    -------
    list of Terminal
        One :class:`Terminal` per single-port boundary edge, in node order.
    """
    est = states_table(prob, x_bar) if x_bar is not None else None
    terms = []
    for n in range(prob.n_nodes):
        rid = int(prob.node_rid[n])
        if rid not in BOUNDARY_RIDS:
            continue
        base = int(prob.row_ptr[n])
        deg = int(prob.row_ptr[n + 1]) - base
        if deg != 1:
            raise ValueError(f"boundary node {n} has degree {deg}; a 1-port must have one edge")
        edge = int(prob.col_edge[base])
        at_tail = int(prob.tail_node[edge]) == n
        incoming = 0 if at_tail else 1
        inflowing = False
        if est is not None:
            mdot = float(est[ES_MDOT, edge])
            inflowing = (mdot > 0.0) if at_tail else (mdot < 0.0)
        terms.append(
            Terminal(
                node=n,
                rid=rid,
                edge=edge,
                at_tail=at_tail,
                row=int(prob.node_row_ptr[n]),
                incoming=incoming,
                outgoing=1 - incoming,
                inflowing=inflowing,
            )
        )
    return terms
