"""Network: the string-addressed builder that compiles to a CompiledProblem.

This is the user surface for the mean-flow solve.  Add elements (each returns a
node index), connect them with directed edges (each carries an area), then
``compile()`` to the immutable problem bundle or ``solve()`` straight to a
``Solution``.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..thermo.configure import ThermoConfig, perfect_gas
from ..connectivity import build_connectivity
from ..elements import catalog as cat
from ..elements.catalog import ElementSpec
from ..problem import CompiledProblem
from ..solver import solve as _solve
from ..solver.control import states_table, initial_guess
from ..derive import ES_MDOT, ES_P, ES_HT, ES_RHO, ES_U, ES_T, ES_C, ES_M, ES_PT, ES_AREA

_EDGE_FIELDS = {
    "mdot": ES_MDOT,
    "p": ES_P,
    "h_t": ES_HT,
    "rho": ES_RHO,
    "u": ES_U,
    "T": ES_T,
    "c": ES_C,
    "M": ES_M,
    "p_t": ES_PT,
    "area": ES_AREA,
}


class Network:
    """A compressible-flow network: elements (nodes) joined by directed edges."""

    def __init__(self, gas: Optional[ThermoConfig] = None, p_ref=101325.0, T_ref=300.0, mdot_ref=None):
        self.gas = gas if gas is not None else perfect_gas()
        self.p_ref = p_ref
        self.T_ref = T_ref
        self._mdot_ref = mdot_ref
        self._elements: List[ElementSpec] = []
        self._edges: List[Tuple[int, int, float]] = []
        self._ports: List[Tuple[Optional[int], Optional[int]]] = []
        self._edge_names: List[str] = []
        # UI-only metadata retained when the network was loaded from a UI save
        # file (None for networks built directly in Python); see fns.io.yaml_out.
        self.provenance = None

    # -- construction -------------------------------------------------------

    def add(self, spec: ElementSpec) -> int:
        self._elements.append(spec)
        return len(self._elements) - 1

    def connect(self, tail: int, head: int, area: float, name: str = "", tail_port=None, head_port=None) -> int:
        """Add a directed edge ``tail -> head``.

        ``tail_port``/``head_port`` pin the local port indices at each endpoint
        (e.g. from a UI export where ports carry meaning); leave them ``None`` to
        let the compiler auto-assign ports in attachment order.
        """
        idx = len(self._edges)
        self._edges.append((tail, head, float(area)))
        self._ports.append((tail_port, head_port))
        self._edge_names.append(name or f"e{idx}")
        return idx

    @property
    def h_ref(self) -> float:
        return self.gas.tf[0] * self.T_ref  # cp * T_ref

    @property
    def mdot_ref(self) -> float:
        if self._mdot_ref is not None:
            return self._mdot_ref
        specs = [el.fparams[0] for el in self._elements if el.residual_id == cat.MASS_FLOW_INLET]
        if specs and max(abs(s) for s in specs) > 0.0:
            return max(abs(s) for s in specs)
        cp, R = self.gas.tf[0], self.gas.tf[1]
        gamma = cp / (cp - R)
        rho = self.p_ref / (R * self.T_ref)
        c = np.sqrt(gamma * R * self.T_ref)
        a_med = float(np.median([a for (_t, _h, a) in self._edges]))
        return 0.3 * rho * c * a_med

    # -- compile / solve ----------------------------------------------------

    def compile(self) -> CompiledProblem:
        explicit = self._edges and all(tp is not None and hp is not None for (tp, hp) in self._ports)
        if explicit:
            endpoints = [(t, int(tp), h, int(hp)) for (t, h, _a), (tp, hp) in zip(self._edges, self._ports)]
            conn = build_connectivity(len(self._elements), endpoints)
            area = np.array([a for (_t, _h, a) in self._edges], dtype=np.float64)
            return cat.build_problem_from_connectivity(
                self.gas, self._elements, conn, area, self.mdot_ref, self.p_ref, self.h_ref
            )
        return cat.build_problem(self.gas, self._elements, self._edges, self.mdot_ref, self.p_ref, self.h_ref)

    def solve(self, x0=None, **kw) -> "Solution":
        prob = self.compile()
        res = _solve(prob, x0=x0, **kw)
        return Solution(self, prob, res)

    def initial_guess(self, **kw):
        return initial_guess(self.compile(), **kw)

    def save(self, path: str, **kwargs) -> None:
        """Write this network as a UI-readable YAML case (no result data).

        Thin wrapper over :func:`fns.io.save_case`; see it for the full set of
        keyword options (``solution``, ``fields``, ``forced``, ...).

        Parameters
        ----------
        path : str
            Destination ``.yaml`` path.
        **kwargs
            Forwarded to :func:`fns.io.save_case`.
        """
        from ..io import save_case

        save_case(self, path, **kwargs)


@dataclass
class Solution:
    """Converged mean-flow result with named edge-field access."""

    network: Network
    problem: CompiledProblem
    result: object  # SolveResult

    @property
    def converged(self) -> bool:
        return self.result.converged

    @property
    def iterations(self) -> int:
        return self.result.iterations

    @property
    def residual_norm(self) -> float:
        return self.result.residual_norm

    @property
    def x(self) -> np.ndarray:
        return self.result.x

    def table(self) -> np.ndarray:
        return states_table(self.problem, self.result.x)

    def edge(self, e: int) -> dict:
        col = self.table()[:, e]
        return {name: float(col[idx]) for name, idx in _EDGE_FIELDS.items()}

    def field(self, name: str) -> np.ndarray:
        return self.table()[_EDGE_FIELDS[name], :]

    def save(self, path: str, **kwargs) -> None:
        """Write the network and this solution's results as a UI-readable case.

        Convenience wrapper that calls :func:`fns.io.save_case` with this
        solution attached, so the mean-flow result fields are embedded as
        datasets the UI can load and visualize.

        Parameters
        ----------
        path : str
            Destination ``.yaml`` path.
        **kwargs
            Forwarded to :func:`fns.io.save_case` (e.g. ``fields``,
            ``node_data``, ``forced``, ``title``).
        """
        from ..io import save_case

        save_case(self.network, path, solution=self, **kwargs)
