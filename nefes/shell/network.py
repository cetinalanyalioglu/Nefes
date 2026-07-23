"""Network builder: add elements, connect them with directed edges, then compile or solve."""

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from nefes.thermo.constants import R_UNIVERSAL

from ..assembly.recover import ES_AREA, ES_C, ES_CP, ES_HT, ES_M, ES_MDOT, ES_P, ES_PT, ES_RHO, ES_T, ES_U, ES_W
from ..elements import catalog as cat
from ..elements.catalog import ElementSpec
from ..elements.composite import CompositeView, expand_composites, is_composite
from ..elements.ids import CHOKED_NOZZLE_OUTLET, ELEMENT_TYPE_NAMES, ISEN_AREA_CHANGE
from ..graph.connectivity import build_connectivity
from ..solver import solve as _solve
from ..solver.control import initial_guess
from ..solver.report import print_residuals, print_states, residual_breakdown, states_table
from ..thermo.api import EQ_KERNEL, PERFECT_GAS
from ..thermo.configure import ThermoConfig, perfect_gas
from . import checks
from .build import build_problem, build_problem_from_connectivity
from .diagnostics import diagnose_junctions
from .problem import CompiledProblem

# ES for "edge state"
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
    "W": ES_W,
    "cp": ES_CP,
}

# Cap on per-table rows in the Network repr (elements / edges); larger networks are truncated.
_REPR_MAX_ROWS = 20


class Network:
    """The main object for building and solving flow networks.

    A network can be built incrementally with :meth:`add` / :meth:`connect`, specified
    complete in one shot via the ``nodes`` / ``edges`` constructor arguments, or loaded from
    a saved case with :meth:`from_yaml` / :meth:`from_dict`.  Call :meth:`solve` for the
    steady mean flow (a :class:`Solution`) or :meth:`compile` / :attr:`problem` for the
    immutable compiled problem.  Parameter-swept stability follows from
    :meth:`eigenvalue_trajectory` / :meth:`nyquist_stability_map`.  Write it back out with
    :meth:`to_yaml`.
    """

    def __init__(
        self,
        gas: Optional[ThermoConfig] = None,
        nodes=None,
        edges=None,
        *,
        edge_models=None,
        require_connected=None,
        **refs,
    ):
        """Create a network, optionally fully specified in one shot.

        The three positional arguments -- ``gas``, ``nodes`` and ``edges`` -- are the whole
        interface for the common one-shot case; the one-shot form supersedes the lower-level
        :func:`nefes.shell.build.build_problem`.

        Parameters
        ----------
        gas : ThermoConfig, optional
            The thermodynamic model (default: dry-air perfect gas).
        nodes : sequence of ElementSpec, optional
            The elements, in node order -- attached via :meth:`add`.
        edges : sequence of tuple, optional
            Directed edges referencing node indices, attached via :meth:`connect`.  Each is
            ``(tail, head, area)`` or, to pin the local ports, ``(tail, head, area, tail_port,
            head_port)``; ports left unspecified are auto-assigned in attachment order.
        edge_models : sequence of int, optional
            Advanced per-edge thermo-model id override aligned with ``edges`` (a hard
            frozen/equilibrium closure); ``None`` entries use the gas default.  Normally left
            unset -- a reacting network with an equilibrium flame gates the closure
            automatically off the transported burnt marker.
        require_connected : bool, optional
            Reject a model that splits into disconnected sub-networks.  ``None`` (default)
            follows the process-wide :data:`nefes.shell.checks.CHECK_CONNECTED` toggle; pass
            ``True`` / ``False`` to force it for this network.

        Other Parameters
        ----------------
        p_ref : float, optional
            Absolute-pressure gauge reference [Pa] (default 101325).
        T_ref : float, optional
            Reference temperature [K] for the initial guess (default 300).
        mdot_ref, h_ref : float, optional
            Seed overrides for the residual scaling; normally auto-derived and re-measured
            during the solve, so rarely set.

        Notes
        -----
        The reference scales (``p_ref``, ``T_ref``, ``mdot_ref``, ``h_ref``) are keyword-only
        advanced overrides accepted through ``**refs``: the casual user leaves them alone and
        they are auto-derived, while an advanced user can still pin any of them by name.
        """
        self.gas = gas if gas is not None else perfect_gas()
        self.require_connected = require_connected
        self.p_ref = refs.pop("p_ref", 101325.0)
        self.T_ref = refs.pop("T_ref", 300.0)
        self._mdot_ref = refs.pop("mdot_ref", None)
        # Explicit absolute-enthalpy datum; if None, falls back to ``cp * T_ref`` (perfect-gas convention).
        # Reacting closures need the gas's absolute-enthalpy reference here instead.
        self._h_ref = refs.pop("h_ref", None)
        if refs:
            raise TypeError(f"unexpected keyword argument(s): {', '.join(sorted(refs))}")
        self._elements: List[ElementSpec] = []
        self._edges: List[Tuple[int, int, float]] = []
        self._ports: List[Tuple[Optional[int], Optional[int]]] = []
        self._edge_names: List[str] = []
        # Per-edge thermo-model override (None -> the gas config's model on that edge).
        self._edge_models: List[Optional[int]] = []
        # Provenance metadata for the network (e.g. from the UI).
        self.provenance = None
        # Lazily compiled problem, invalidated by any topology change (see ``_invalidate``).
        self._compiled: Optional[CompiledProblem] = None

        for spec in nodes or ():
            self.add(spec)
        if edge_models is not None and edges is None:
            raise ValueError("edge_models was given without edges")
        edges = list(edges or ())
        models = list(edge_models) if edge_models is not None else [None] * len(edges)
        if len(models) != len(edges):
            raise ValueError(f"edge_models has {len(models)} entries but there are {len(edges)} edges")
        for edge, model in zip(edges, models):
            # Accept a bare (tail, head, area) or a port-pinned (tail, head, area, tail_port, head_port).
            tail, head, area = edge[0], edge[1], edge[2]
            tail_port = edge[3] if len(edge) > 3 else None
            head_port = edge[4] if len(edge) > 4 else None
            self.connect(tail, head, area, tail_port=tail_port, head_port=head_port, edge_model=model)

    # -- construction -------------------------------------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> "Network":
        """Build a network from a saved UI/YAML case file.

        A convenience so callers need not reach into :mod:`nefes.io`; equivalent to
        :func:`nefes.io.load_case`.

        Parameters
        ----------
        path : str
            Path to a ``.yaml`` case file (as written by :meth:`to_yaml`).

        Returns
        -------
        Network
        """
        from ..io import load_case

        return load_case(path)

    @classmethod
    def from_dict(cls, data: dict) -> "Network":
        """Build a network from an in-memory case dictionary.

        The dictionary is the parsed form of the same schema :meth:`from_yaml` reads, so this
        is the file-less equivalent of :meth:`from_yaml`.

        Parameters
        ----------
        data : dict
            A case document (a ``model`` section with ``nodes`` / ``edges``, as in a YAML case).

        Returns
        -------
        Network
        """
        from ..io import case_from_dict

        return case_from_dict(data)

    def _invalidate(self) -> None:
        """Drop the cached compiled problem after a topology change."""
        self._compiled = None

    def add(self, spec: ElementSpec) -> int:
        """Add an element and return its node index.

        Element names must be unique.  A factory default (the caller did not pass ``name``) is
        always numbered -- a lone ``duct`` becomes ``duct-1`` -- while a name the caller chose is
        kept and only suffixed on an actual clash; the ``name_auto`` flag on the spec records which.
        """
        taken = {el.name for el in self._elements}
        base = spec.name or ""
        spec.name = cat.unique_name(base, taken, always_number=getattr(spec, "name_auto", False))
        spec.name_auto = False  # name is now concrete; a later re-add must not re-number it
        self._elements.append(spec)
        self._invalidate()
        return len(self._elements) - 1

    def connect(
        self, tail: int, head: int, area: float, name: str = "", *, tail_port=None, head_port=None, edge_model=None
    ) -> int:
        """Add a directed edge from element `tail` to element `head`, returning its edge id.

        The returned integer is the edge index in the compiled problem -- capture it to wire a
        dynamic source's ``ref_edge`` (e.g. the edge just upstream of a flame) without guessing.

        `tail_port`/`head_port` pin the local port indices at each endpoint; leave them `None` to let the
        compiler auto-assign ports in attachment order.

        `edge_model` is an advanced, keyword-only override of the per-edge thermo-model id (a hard
        frozen/equilibrium closure); leave it `None` to use the gas config's default model -- a reacting
        network gates the frozen/equilibrium split automatically off the transported burnt marker.
        """
        idx = len(self._edges)
        self._edges.append((tail, head, float(area)))
        self._ports.append((tail_port, head_port))
        self._edge_models.append(None if edge_model is None else int(edge_model))
        # Edge name is optional, defaulting to "e<index>".
        self._edge_names.append(name or f"e{idx}")
        self._invalidate()
        return idx

    def edge_between(self, a, b) -> int:
        """Return the id of the edge connecting elements `a` and `b`, in either orientation.

        A convenience for recovering an edge index after assembly (e.g. to set a dynamic source's
        ``ref_edge``) when the value returned by :meth:`connect` was not captured.  The lookup
        ignores wiring orientation: ``edge_between("cold", "flame")`` and
        ``edge_between("flame", "cold")`` return the same edge id, since an edge id names the edge
        regardless of which end you list first.  Read the stored orientation with :meth:`nodes_of`.

        Parameters
        ----------
        a, b : int or str
            The two endpoints, each given as a node index (as returned by :meth:`add`) or the
            element's display name.  Names are resolved through :meth:`element_index`, and the
            two forms may be mixed.

        Returns
        -------
        int

        Raises
        ------
        KeyError
            Unknown element name or node index out of range (via :meth:`element_index`).
        ValueError
            No edge joins the pair, or more than one does (parallel or antiparallel edges).

        See also
        --------
        edges_of : every edge incident to one element, when only one endpoint is known.
        nodes_of : the ``(tail, head)`` orientation of a known edge.
        element_index : the name-to-index resolution applied to each endpoint.
        """
        a, b = self.element_index(a), self.element_index(b)
        pair = {a, b}
        matches = [i for i, (t, h, _area) in enumerate(self._edges) if {t, h} == pair]
        if not matches:
            raise ValueError(f"no edge between elements {a} and {b}")
        if len(matches) > 1:
            raise ValueError(f"multiple edges between elements {a} and {b}: {matches}")
        return matches[0]

    def edges_of(self, element, direction: str = "both") -> List[int]:
        """Ids of the edges incident to an element, resolved by name or node index.

        The companion to :meth:`edge_between` for when only one endpoint is known: it returns
        every edge touching ``element`` rather than the single edge between an ordered pair.
        The ids index the same edge table as :meth:`connect`'s return and :meth:`edge_between`,
        so they read straight off a solution (``sol.edge(e)``, ``sol.field(name)[e]``).

        Parameters
        ----------
        element : int or str
            Node index (as returned by :meth:`add`) or the element's display name.
        direction : {"both", "in", "out"}, optional
            Which incident edges to return: ``"out"`` for edges *leaving* the element (it is
            the edge's tail), ``"in"`` for edges *entering* it (it is the edge's head), or
            ``"both"`` (default) for either.  This is the wiring orientation, not the solved
            flow direction (which may run either way along an edge).

        Returns
        -------
        list of int
            The incident edge ids, in ascending order (empty if the element has no edges).

        Raises
        ------
        KeyError
            Unknown element name or node index out of range (via :meth:`element_index`).
        ValueError
            ``direction`` is not one of ``"both"``, ``"in"``, ``"out"``.

        See also
        --------
        edge_between : the single edge between a known ordered pair of elements.
        nodes_of : the ``(tail, head)`` elements a known edge connects.

        Examples
        --------
        >>> net.edges_of("plenum", direction="out")  # doctest: +SKIP
        [2, 5, 8, 11]
        """
        if direction not in ("both", "in", "out"):
            raise ValueError(f"direction must be 'both', 'in', or 'out'; got {direction!r}")
        node = self.element_index(element)
        want_out = direction in ("both", "out")
        want_in = direction in ("both", "in")
        return [i for i, (t, h, _a) in enumerate(self._edges) if (want_out and t == node) or (want_in and h == node)]

    def nodes_of(self, edge: int) -> Tuple[int, int]:
        """The ``(tail, head)`` element indices an edge connects.

        The inverse of :meth:`edges_of`: given an edge id (from :meth:`connect`,
        :meth:`edge_between`, or :meth:`edges_of`) it returns the two elements the edge joins,
        tail first (the source endpoint in the wiring orientation), head second.  Resolve
        either index to its name or spec with :meth:`element`.

        Parameters
        ----------
        edge : int
            Edge id, in ``range(len(edges))``.

        Returns
        -------
        tuple of int
            The ``(tail, head)`` node indices.

        Raises
        ------
        IndexError
            ``edge`` is not a valid edge id for this network.

        See also
        --------
        edges_of : the edges incident to a known element.

        Examples
        --------
        >>> t, h = net.nodes_of(0)  # doctest: +SKIP
        >>> net.element(t).name, net.element(h).name  # doctest: +SKIP
        ('inlet', 'duct')
        """
        n = len(self._edges)
        if not 0 <= edge < n:
            raise IndexError(f"edge id {edge} out of range for a network with {n} edge(s)")
        t, h, _a = self._edges[edge]
        return int(t), int(h)

    def set_dynamic_source(self, node, source) -> int:
        """Attach (or replace) the dynamic-source descriptor on an *already-added* element.

        A named convenience over the generic :meth:`set` (``set(node, dynamic_source=...)``),
        so the write is validated against the element's parameter schema.

        Parameters
        ----------
        node : int or str
            Element index (as returned by :meth:`add`) or display name.
        source : DynamicSource or None
            The descriptor (e.g. from :func:`nefes.elements.dynamic_source.n_tau_flame`); ``None`` clears it.

        Returns
        -------
        int
            The element's node index, for chaining.
        """
        return self.set(node, dynamic_source=source)

    def set_perturbation_bc(self, node, bc) -> int:
        """Attach (or replace) the acoustic termination on an *already-added* boundary element.

        A named convenience over the generic :meth:`set` (``set(node, perturbation_bc=...)``),
        so the write is validated against the element's parameter schema.  Only the
        single-port boundary terminations (inlets, outlets, wall) carry a perturbation BC.

        Parameters
        ----------
        node : int or str
            Element index (as returned by :meth:`add`) or display name.
        bc : PerturbationBC or None
            The termination (e.g. ``PerturbationBC.open_end()``); ``None`` restores the
            inherited linearized boundary row.

        Returns
        -------
        int
            The element's node index, for chaining.
        """
        return self.set(node, perturbation_bc=bc)

    # -- parameter access ---------------------------------------------------------------------------------------------

    def element_index(self, key) -> int:
        """Resolve an element reference (node index or unique display name) to its node index.

        Parameters
        ----------
        key : int or str
            Node index (as returned by :meth:`add`), or the element's display name.

        Returns
        -------
        int

        Raises
        ------
        KeyError
            Unknown name (with near-match suggestions) or index out of range.
        """
        from .params import element_index

        return element_index(self, key)

    def element_name(self, key) -> str:
        """Display label of an element, the inverse of :meth:`element_index`.

        Turns a node index (as returned by :meth:`add`, :meth:`nodes_of`, or an ``edges_of``
        result) back into the element's name, the stable handle used across the API and in
        parameter addresses.  Passing a name returns it unchanged, so the two are interchangeable
        wherever a label is wanted.  Falls back to ``#<index>`` for the rare unnamed element.

        Parameters
        ----------
        key : int or str
            Node index or the element's display name.

        Returns
        -------
        str

        Raises
        ------
        KeyError
            Unknown name (with near-match suggestions) or index out of range (via
            :meth:`element_index`).

        Examples
        --------
        >>> t, h = net.nodes_of(net.edge_between("cold", "flame"))
        >>> net.element_name(t), net.element_name(h)
        ('cold', 'flame')

        See Also
        --------
        element_index : the inverse, resolving a label to its node index.
        element : the full element spec behind a label or index.
        """
        return self._node_label(self.element_index(key))

    def element(self, key) -> ElementSpec:
        """The element spec behind a node index or display name.

        Parameters
        ----------
        key : int or str
            Node index or the element's display name.

        Returns
        -------
        ElementSpec or CompositeElementSpec

        Examples
        --------
        >>> net.element("inlet").fparams
        [0.3, 700.0]

        See Also
        --------
        get : read a single named parameter instead of the raw spec.
        """
        return self._elements[self.element_index(key)]

    def parameters(self, advanced: bool = False, layer: Optional[str] = None):
        """The inventory of every addressable parameter: address, value, unit, bounds.

        The read-only companion of :meth:`get` / :meth:`set`: one row per named element
        parameter (in node order), per edge area, and per network-level reference, plus
        the scalar knobs of any attached object that exposes them (a dynamic source's
        gain and lag, a boundary condition's reflection magnitude and phase) under
        extended dotted addresses.  The returned
        :class:`~nefes.shell.params.ParameterInventory` is a list of
        :class:`~nefes.shell.params.ParameterInfo` rows with dict-style access by address
        and table reprs.

        Parameters
        ----------
        advanced : bool, optional
            Include the advanced knobs (smoothing ``eps``, ``ref_port``, the solver seed
            references) usually left alone (default ``False``).
        layer : str, optional
            Narrow to one solution layer: ``"mean"`` (parameters that reshape the mean
            flow) or ``"perturbation"`` (parameters entering only the acoustic operator:
            storage volumes, inertance lengths, source and boundary knobs).  Default:
            both.

        Returns
        -------
        ParameterInventory

        Examples
        --------
        >>> net.parameters()["inlet.mdot"].value
        0.3
        >>> net.parameters(layer="perturbation").addresses  # doctest: +SKIP
        ['plenum.volume', 'flame.dynamic_source.gain', ...]

        See Also
        --------
        get, set, update, with_params
        """
        from .params import inventory

        return inventory(self, advanced=advanced, layer=layer)

    def get(self, address: str):
        """Read one parameter by its dotted address.

        Addresses are ``"element.param"`` / ``"edge.area"`` strings (elements and edges by
        display name) plus the bare network references (``"p_ref"``, ``"T_ref"``).  An
        unknown address raises with near-match suggestions.  ``"element.area"`` reads the
        shared incident-edge area of a single-port or constant-area element.

        Parameters
        ----------
        address : str
            The dotted address, e.g. ``"inlet.mdot"``, ``"orifice.throat_area"``, ``"e3.area"``.

        Returns
        -------
        object
            The current value (a float for numeric parameters).

        Examples
        --------
        >>> net.get("inlet.mdot")
        0.3
        >>> net.get("e0.area")
        0.01

        See Also
        --------
        parameters : the full inventory of addressable parameters.
        """
        from .params import get_param

        return get_param(self, address)

    def set(self, element, **params) -> int:
        """Set named parameters on one element, in place, with fail-closed validation.

        Every value is validated against the element's declared schema (units, bounds,
        types) before anything is written -- an out-of-range value raises a named error,
        exactly as the element's factory would.  A composite element is rebuilt through
        its factory (its internals re-derived consistently), never patched.  ``area=`` on
        a single-port or constant-area element fans out to all its incident edges.  The
        compiled-problem cache is invalidated; topology is never touched, so a previous
        solution remains a valid warm start (``solve(x0=prev.x)``).

        Parameters
        ----------
        element : int or str
            Element node index or display name.
        **params
            ``name=value`` pairs from the element's parameter set (see :meth:`parameters`).

        Returns
        -------
        int
            The element's node index, for chaining.

        Examples
        --------
        >>> net.set("inlet", mdot=0.5, Tt=720.0)
        0
        >>> net.set("orifice", throat_area=1.2e-3)
        3

        See Also
        --------
        update : batch writes by dotted address.
        with_params : the functional (copying) variant recommended for parameter studies.
        """
        from .params import set_params

        return set_params(self, element, params)

    def update(self, mapping: dict) -> "Network":
        """Apply a batch of dotted-address parameter writes, in place.

        Every address is resolved before anything is written, so a mistyped address
        leaves the network untouched; values are then validated per element.  Element
        writes are grouped so a composite is rebuilt once with all its updates merged.

        Parameters
        ----------
        mapping : dict
            ``{address: value}``, e.g. ``{"orifice.throat_area": 1.2e-3, "e3.area": 0.01}``.

        Returns
        -------
        Network
            ``self``, for chaining (``net.update({...}).solve()``).

        See Also
        --------
        set : the single-element form.
        with_params : the functional (copying) variant recommended for parameter studies.
        """
        from .params import update_params

        update_params(self, mapping)
        return self

    def copy(self) -> "Network":
        """A deep copy of this network's specification (elements, edges, references).

        Edge order, port pins and names are preserved by construction, so the copy
        compiles to the same problem layout and a warm start from this network's solution
        stays valid.  The gas model is shared (an immutable configuration); the
        compiled-problem cache is not copied.

        Returns
        -------
        Network
        """
        from .params import copy_network

        return copy_network(self)

    def with_params(self, mapping: dict) -> "Network":
        """A modified deep copy: this network with the given parameter writes applied.

        The recommended idiom for parameter studies -- the loaded base stays pristine, no
        state accumulates across sweep points, and each point is safe to solve
        independently.  Addressing and validation are those of :meth:`update`.

        Parameters
        ----------
        mapping : dict
            ``{address: value}`` writes applied to the copy.

        Returns
        -------
        Network
            The modified copy; ``self`` is untouched.

        Examples
        --------
        >>> base = nefes.load_case("combustor.yaml")
        >>> prev = None
        >>> for mdot in np.linspace(0.3, 0.7, 20):
        ...     sol = base.with_params({"inlet.mdot": mdot}).solve(x0=prev.x if prev else None)
        ...     prev = sol

        See Also
        --------
        nefes.parameter_study : the warm-start-chained sweep driver built on this.
        builder : a one-parameter ``build(p)`` closure for the continuation drivers.
        """
        return self.copy().update(mapping)

    def builder(self, address: str, **fixed):
        """A one-parameter ``build(p)`` closure over :meth:`with_params`.

        The ``build`` contract the continuation drivers take
        (:func:`~nefes.perturbation.stability.trajectory.eigenvalue_trajectory`,
        :func:`~nefes.perturbation.response.nyquist.nyquist`): ``build(p)`` returns a
        fresh network with ``address`` set to ``p``, leaving this base pristine.

        Parameters
        ----------
        address : str
            The swept parameter's dotted address (e.g. ``"flame.Qdot"``).
        **fixed
            Additional ``{address: value}`` writes applied at every point (keyword form;
            dotted addresses with characters invalid in a keyword can be passed by
            building the closure manually with :meth:`with_params`).

        Returns
        -------
        callable
            ``build(p) -> Network``.

        Examples
        --------
        >>> traj = eigenvalue_trajectory(net.builder("flame.Qdot"), np.linspace(1e3, 5e3, 21),
        ...                              freq_band=(50.0, 400.0), param_name="Qdot")
        """
        fixed = dict(fixed)

        def build(p):
            return self.with_params({address: p, **fixed})

        return build

    def _seed_h(self) -> float:
        """Seed enthalpy scale threaded into the compiled ``var_scale`` (an explicit override or auto).

        Only the *seed* for the residual scaling -- the solve re-measures the enthalpy scale from the
        realized inflow, and the reacting initial guess seeds each edge from its feed enthalpy, so this
        need only be order-of-magnitude right.  Auto-derivation is the perfect-gas ``cp * T_ref``; an
        explicit ``h_ref=`` (as the reacting backend supplies) overrides it.
        """
        if self._h_ref is not None:
            return self._h_ref
        return self.gas.tf[0] * self.T_ref

    def _seed_mdot(self) -> float:
        """Seed mass-flow scale threaded into the compiled ``var_scale`` (an explicit override or auto).

        Only the *seed* for the residual scaling -- the solve re-measures it from the realized inflow at
        each continuation stage (``adaptive_scale``) -- so it need only be order-of-magnitude right.
        Auto-derivation: the **total** specified inflow when every inlet is a mass-flow inlet; otherwise
        a dP-based isentropic estimate ``A * sqrt(2 rho dP_max)`` from the boundary pressures; a quiescent
        / pressureless network falls back to an M=0.3 estimate.  An explicit ``mdot_ref=`` overrides it.
        """
        if self._mdot_ref is not None:
            return self._mdot_ref
        # ``getattr`` guards the composite specs (which carry no ``residual_id``); a composite
        # is never an inlet, so skipping it is correct.
        mass = [abs(el.fparams[0]) for el in self._elements if getattr(el, "residual_id", None) == cat.MASS_FLOW_INLET]
        has_pt = any(getattr(el, "residual_id", None) == cat.PT_INLET for el in self._elements)
        # every inlet a mass-flow inlet -> the total specified inflow is exactly known.
        if mass and not has_pt and sum(mass) > 0.0:
            return sum(mass)
        a_med = float(np.median([a for (_t, _h, a) in self._edges]))
        rho = self.p_ref / (self.gas.tf[1] * self.T_ref) if self.gas.model_id == PERFECT_GAS else 1.0
        # pressure-driven: an isentropic mass-flux estimate from the largest boundary dP.
        dp = self._boundary_dp()
        if dp > 0.0:
            return a_med * np.sqrt(2.0 * rho * dp)
        # quiescent / no pressure spread: the M=0.3 fallback (perfect gas) or a unit scale.
        if self.gas.model_id == PERFECT_GAS:
            cp, R = self.gas.tf[0], self.gas.tf[1]
            c = np.sqrt((cp / (cp - R)) * R * self.T_ref)
            return 0.3 * rho * c * a_med
        return max(sum(mass), 1.0)

    def _boundary_dp(self) -> float:
        """Largest a-priori pressure drop across the boundary pressure references (0 if < 2)."""
        refs = [
            el.fparams[0] for el in self._elements if getattr(el, "residual_id", None) in (cat.PT_INLET, cat.P_OUTLET)
        ]
        return (max(refs) - min(refs)) if len(refs) >= 2 else 0.0

    # -- compile / solve ----------------------------------------------------------------------------------------------

    def _resolve_edge_models(self):
        """Return the per-edge thermo-model array, or ``None`` if every edge uses the gas default."""
        if all(m is None for m in self._edge_models):
            return None
        default = int(self.gas.model_id)
        return np.array([default if m is None else m for m in self._edge_models], dtype=np.int64)

    @property
    def problem(self) -> CompiledProblem:
        """The compiled problem for the current topology, built on first access and cached.

        Most callers never need the compiled object directly -- :meth:`solve` and the
        :class:`Solution` it returns cover the common path -- but it is here for the lower-level
        routines that take a ``CompiledProblem``.  The cache is dropped whenever the network is
        mutated (:meth:`add` / :meth:`connect` / :meth:`set` / :meth:`update`), so it always
        reflects the live state.
        """
        if self._compiled is None:
            self._compiled = self._build()
        return self._compiled

    def compile(self) -> CompiledProblem:
        """Compile the elements and edges into an immutable ``CompiledProblem`` and cache it.

        Rebuilds unconditionally (refreshing the :attr:`problem` cache); prefer :attr:`problem`
        when a cached compile is enough.
        """
        self._compiled = self._build()
        return self._compiled

    def _build(self) -> CompiledProblem:
        """Assemble a fresh ``CompiledProblem`` from the current elements and edges."""
        prob = self._build_problem()
        # Surface the finalized config so ``net.gas.species_names`` / ``.species_set`` reflect the
        # resolved state after a build: a deferred automatic species set (equilibrium() with no
        # species_set) is inspectable once its species slate has been derived from the feeds.  The
        # config keeps its ``auto_species_set`` flag, so a later build re-derives it from the feeds.
        if getattr(self.gas, "auto_species_set", False) and prob.gas is not None:
            self.gas = prob.gas
        return prob

    def _build_problem(self) -> CompiledProblem:
        """Compile the current elements/edges into a ``CompiledProblem`` (no config surfacing)."""
        edge_models = self._resolve_edge_models()
        mdot_ref, h_ref = self._seed_mdot(), self._seed_h()
        # If the ports are explicitly set, use the connectivity builder.
        explicit = self._edges and all(tp is not None and hp is not None for (tp, hp) in self._ports)
        if explicit:
            # Composites expand here too: user pins survive at atomic endpoints, and the expansion
            # re-derives flow-aligned ports on the rewired sub-elements and the internal edges.
            elements, edges, cmap = expand_composites(self._elements, self._edges, ports=self._ports)
            if cmap is None:
                endpoints = [(t, int(tp), h, int(hp)) for (t, h, _a), (tp, hp) in zip(self._edges, self._ports)]
                area = np.array([a for (_t, _h, a) in self._edges], dtype=np.float64)
            else:
                endpoints = [(t, tp, h, hp) for (t, h, _a, tp, hp) in edges]
                area = np.array([a for (_t, _h, a, _tp, _hp) in edges], dtype=np.float64)
                if edge_models is not None:
                    # the appended internal edges follow the gas default closure
                    pad = np.full(len(edges) - len(edge_models), int(self.gas.model_id), dtype=np.int64)
                    edge_models = np.concatenate([edge_models, pad])
            conn = build_connectivity(len(elements), endpoints)
            return build_problem_from_connectivity(
                self.gas,
                elements,
                conn,
                area,
                mdot_ref,
                self.p_ref,
                h_ref,
                edge_models=edge_models,
                composite_map=cmap,
                require_connected=self.require_connected,
            )
        return build_problem(
            self.gas,
            self._elements,
            self._edges,
            mdot_ref,
            self.p_ref,
            h_ref,
            edge_models=edge_models,
            require_connected=self.require_connected,
        )

    def solve(self, x0=None, **kw) -> "Solution":
        """Compile and solve the steady mean flow, returning a ``Solution``.

        Keyword arguments
        ------------------
        x0 : ndarray, optional
            Initial state, shape ``(3, E)`` (default: a uniform co-directional guess).
        tol : float, optional
            Convergence tolerance on the scaled residual 2-norm (default ``1e-10``).
        max_iter : int, optional
            Maximum Newton iterations per continuation stage (default ``80``).
        kappa_stages : sequence of float, optional
            Artificial-resistance continuation schedule, warm-started in order (default ``(0.1, 0.01, 0.0)``).
        verbose : int or bool, optional
            Progress verbosity (default ``0``). ``0``/``False`` is silent; ``1``/``True`` prints a one-line
            gross-residual summary per continuation stage; ``2`` additionally prints the scaled residual broken down by
            equation kind (mass, pressure, energy, then each composition scalar) every ``progress_interval`` iterations.
        progress_interval : int, optional
            Iteration stride for the per-iteration prints at ``verbose >= 2`` (default ``1``).

        Returns
        -------
        Solution
            The converged mean-flow result with named edge-field access.

        Notes
        -----
        A solve that does not converge returns its (partial) ``Solution`` and emits a warning;
        reading a field off a state that cannot be recovered raises a clear error naming the
        non-convergence rather than an opaque linear-algebra failure from the closure.
        """
        prob = self.compile()
        for message in diagnose_junctions(self):
            warnings.warn(message, stacklevel=2)
        res = _solve(prob, x0=x0, **kw)
        sol = Solution(self, prob, res)
        if res.converged:
            for message in sol.verify():
                warnings.warn(message, stacklevel=2)
        else:
            # Surface the failure at the solve boundary rather than letting it resurface
            # opaquely when the caller first reads a field off a non-physical state.
            warnings.warn(
                f"Network.solve did not converge (residual_norm={res.residual_norm:.3e}, "
                f"iterations={res.iterations}); the returned state may be non-physical. "
                "Inspect it with sol.print_residuals().",
                stacklevel=2,
            )
        return sol

    def initial_guess(self, **kw):
        """Return the solver's initial state guess for the compiled problem."""
        return initial_guess(self.compile(), **kw)

    def plot(self, **kwargs):
        """Draw the network as a node/edge diagram (Plotly).

        A structural view by default: element indices/names and edge directions, with each edge's arrow
        **width scaled by its area** (``width_by="area"``), so the geometry reads at a glance. Pass
        ``width_by=None`` for uniform arrows, or another field (with a converged ``solution=``) to weight
        by it instead; ``color_by`` similarly tints the edges, and :meth:`Solution.plot` is the same diagram
        driven from a solution. Thin wrapper over :func:`nefes.plotting.plot_network_topology`; see it for
        the full keyword set.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        from ..plotting import plot_network_topology

        # Default the arrow width to edge area (geometry-weighted); the caller can override or disable it.
        kwargs.setdefault("width_by", "area")
        return plot_network_topology(self, **kwargs)

    # -- perturbation continuation (parameter-swept stability) ---------------------------------------------------------

    def eigenvalue_trajectory(self, address, params, **kwargs):
        """Track the eigenmode spectrum as one parameter is swept.

        A bound form of :func:`nefes.perturbation.eigenvalue_trajectory`: it builds each swept
        network with :meth:`builder` (this base stays pristine), seeds the spectrum once, then
        continues each mode along ``params``.

        Parameters
        ----------
        address : str
            The swept parameter's dotted address (e.g. ``"flame.Qdot"``).
        params : array_like
            The parameter values, in continuation order.
        **kwargs
            Forwarded to :func:`nefes.perturbation.eigenvalue_trajectory` (e.g. ``freq_band``,
            ``growth_band``, ``isentropic``); ``param_name`` defaults to ``address``.

        Returns
        -------
        TrajectoryResult

        See Also
        --------
        nefes.perturbation.eigenvalue_trajectory : the underlying routine.
        builder : the ``build(p)`` closure this passes through.
        """
        from ..perturbation import eigenvalue_trajectory

        kwargs.setdefault("param_name", address)
        return eigenvalue_trajectory(self.builder(address), params, **kwargs)

    def nyquist_stability_map(self, address, params, freqs, **kwargs):
        """Unstable-mode count across a parameter sweep, on the real-frequency axis.

        A bound form of :func:`nefes.perturbation.nyquist_stability_map`: it builds each swept
        network with :meth:`builder` (this base stays pristine).  The network must carry at least
        one dynamic source.

        Parameters
        ----------
        address : str
            The swept parameter's dotted address (e.g. ``"flame.n"``).
        params : array_like
            The parameter values to sweep.
        freqs : array_like
            Real frequencies (Hz) for the Nyquist locus at each point.
        **kwargs
            Forwarded to :func:`nefes.perturbation.nyquist_stability_map`; ``param_name`` defaults
            to ``address``.

        Returns
        -------
        NyquistStabilityMap

        See Also
        --------
        nefes.perturbation.nyquist_stability_map : the underlying routine.
        builder : the ``build(p)`` closure this passes through.
        """
        from ..perturbation import nyquist_stability_map

        kwargs.setdefault("param_name", address)
        return nyquist_stability_map(self.builder(address), params, freqs, **kwargs)

    def to_yaml(self, path: str, **kwargs) -> None:
        """Write this network as a UI-readable YAML case (no result data).

        The inverse of :meth:`from_yaml`.  Thin wrapper over :func:`nefes.io.save_case`; see it for
        the full set of keyword options.  To embed solved fields as well, use
        :meth:`Solution.to_yaml`.

        Parameters
        ----------
        path : str
            Destination ``.yaml`` path.
        **kwargs
            Forwarded to :func:`nefes.io.save_case`.
        """
        from ..io import save_case

        save_case(self, path, **kwargs)

    def save(self, path: str, **kwargs) -> None:
        """Alias for :meth:`to_yaml` (kept for intuitive usage)."""
        self.to_yaml(path, **kwargs)

    # -- display ------------------------------------------------------------------------------------------------------

    def _gas_summary(self) -> str:
        """One-line description of the thermo model (gas, scalars/streams).

        For an automatic product slate the species count is annotated with the reduction
        that selected it (candidate count and trace threshold), when a
        ``reduction_report`` is available on the species set.
        """
        g = self.gas
        if g.model_id == PERFECT_GAS:
            cp, R = float(g.tf[0]), float(g.tf[1])
            gamma = cp / (cp - R)
            text = f"perfect gas (R={R:.6g} J/kg/K, gamma={gamma:.4g})"
            if g.n_elem:
                text += f" + {g.n_elem} passive scalar(s): {', '.join(g.element_names)}"
            return text
        if g.model_id == EQ_KERNEL:
            if g.auto_species_set and g.species_set is None:
                text = "equilibrium (auto species, unresolved)"
            else:
                text = f"equilibrium ({g.n_species} species"
                report = getattr(g.species_set, "reduction_report", None) if g.species_set is not None else None
                if report:
                    n_cand = report.get("n_candidates")
                    thr = report.get("threshold")
                    reducer = report.get("reducer")
                    if reducer and reducer != "none" and n_cand is not None:
                        text += f", auto-reduced from {n_cand}"
                        cap = report.get("max_species")
                        if cap is not None:
                            text += f", max={cap}"
                        elif thr is not None:
                            text += f", threshold={thr:g}"
                    elif reducer == "none":
                        text += ", auto"
                text += ")"
            # Streams are discovered at build time, so the labels may not be populated yet.
            if g.element_names:
                text += f", streams: {', '.join(g.element_names)}"
            return text
        return f"model #{g.model_id}"

    def _refs(self):
        """``(p_ref, T_ref, mdot_seed_or_None, mdot_is_explicit)`` for the repr headers."""
        try:
            # The auto-derive medians the edge areas; an edge-less network yields a quiet NaN
            # (suppress numpy's "mean of empty slice" warning -- we report it as "n/a" below).
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                m = float(self._seed_mdot())
            if not np.isfinite(m):
                m = None
        except Exception:
            m = None
        return self.p_ref, self.T_ref, m, self._mdot_ref is not None

    def _node_label(self, i: int) -> str:
        """Compact identifier for an element endpoint, used in the edge listing."""
        if 0 <= i < len(self._elements):
            return self._elements[i].name or f"#{i}"
        return f"#{i}"

    @staticmethod
    def _type_name(el: ElementSpec) -> str:
        """Human-readable residual-type name for an element (or the kind for a composite)."""
        if is_composite(el):
            return el.kind or "composite"
        return ELEMENT_TYPE_NAMES.get(el.residual_id, f"residual#{el.residual_id}")

    def __repr__(self) -> str:
        """Compact text summary: size, thermo model, references, and the element / edge listings.

        Elements that carry a dynamic ``S(omega)`` source are flagged with a trailing ``*``; a
        footnote reports any per-edge thermo-model overrides.  Each listing is truncated past
        ``_REPR_MAX_ROWS`` rows.
        """
        n_el, n_ed = len(self._elements), len(self._edges)
        lines = [
            f"Network: {n_el} element{'' if n_el == 1 else 's'}, {n_ed} edge{'' if n_ed == 1 else 's'}",
            f"  gas: {self._gas_summary()}",
        ]
        p, T, m, explicit = self._refs()
        mdot = "n/a" if m is None else f"{m:.4g} kg/s ({'explicit' if explicit else 'auto'})"
        lines.append(f"  refs: p={p:.6g} Pa, T={T:.6g} K, mdot={mdot}")

        n_src = sum(1 for el in self._elements if getattr(el, "dynamic_source", None) is not None)
        n_ovr = sum(1 for mdl in self._edge_models if mdl is not None)

        if n_el:
            lines.append("")
            rows = [
                (
                    str(i),
                    self._node_label(i),
                    self._type_name(el) + (" *" if getattr(el, "dynamic_source", None) is not None else ""),
                )
                for i, el in enumerate(self._elements[:_REPR_MAX_ROWS])
            ]
            lines += _text_table(("#", "name", "type"), rows, ("r", "l", "l"), indent=2)
            if n_el > _REPR_MAX_ROWS:
                lines.append(f"    ... ({n_el - _REPR_MAX_ROWS} more)")

        if n_ed:
            lines.append("")
            rows = [
                (str(i), f"{self._node_label(t)} -> {self._node_label(h)}", f"{a:.4g}", self._edge_names[i])
                for i, (t, h, a) in enumerate(self._edges[:_REPR_MAX_ROWS])
            ]
            lines += _text_table(("#", "connection", "area [m^2]", "name"), rows, ("r", "l", "r", "l"), indent=2)
            if n_ed > _REPR_MAX_ROWS:
                lines.append(f"    ... ({n_ed - _REPR_MAX_ROWS} more)")

        notes = []
        if n_src:
            notes.append(f"* = carries a dynamic S(omega) source ({n_src})")
        if n_ovr:
            notes.append(f"{n_ovr} edge(s) carry a per-edge thermo-model override")
        if notes:
            lines.append("")
            lines += [f"  {note}" for note in notes]
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """Rich HTML summary for Jupyter: compact header plus element and edge listings."""
        n_el, n_ed = len(self._elements), len(self._edges)
        p, T, m, explicit = self._refs()
        mdot = "n/a" if m is None else f"{m:.4g} kg/s ({'explicit' if explicit else 'auto'})"
        parts = [
            f"{n_el} element{'' if n_el == 1 else 's'}, {n_ed} edge{'' if n_ed == 1 else 's'}",
            self._gas_summary(),
            f"p={p:.6g} Pa &nbsp;&middot;&nbsp; T={T:.6g} K &nbsp;&middot;&nbsp; mdot={mdot}",
        ]
        header = (
            "<div style='font-family:sans-serif;margin-bottom:6px'>"
            "<b>Network</b>"
            "<span style='color:#888'>&nbsp;&middot;&nbsp;</span>"
            + "<span style='color:#888'>&nbsp;|&nbsp;</span>".join(parts)
            + "</div>"
        )

        # Each listing becomes one flex column so the element and edge tables sit side by side.
        blocks = []
        if n_el:
            body = []
            for i, el in enumerate(self._elements[:_REPR_MAX_ROWS]):
                src = (
                    " <span style='color:#2a8a4a' title='carries a dynamic S(omega) source'>&#9733;</span>"
                    if getattr(el, "dynamic_source", None) is not None
                    else ""
                )
                body.append(
                    [str(i), self._node_label(i), self._type_name(el) + src],
                )
            block = [_caption("Elements"), _html_table(("#", "name", "type"), body, ("right", "left", "left"))]
            if n_el > _REPR_MAX_ROWS:
                block.append(f"<div style='color:#888;font-size:0.85em'>... ({n_el - _REPR_MAX_ROWS} more)</div>")
            blocks.append("".join(block))

        if n_ed:
            body = [
                [str(i), f"{self._node_label(t)} &rarr; {self._node_label(h)}", f"{a:.4g}", self._edge_names[i]]
                for i, (t, h, a) in enumerate(self._edges[:_REPR_MAX_ROWS])
            ]
            block = [
                _caption("Edges"),
                _html_table(("#", "connection", "area [m&sup2;]", "name"), body, ("right", "left", "right", "left")),
            ]
            if n_ed > _REPR_MAX_ROWS:
                block.append(f"<div style='color:#888;font-size:0.85em'>... ({n_ed - _REPR_MAX_ROWS} more)</div>")
            blocks.append("".join(block))

        columns = "".join(f"<div>{b}</div>" for b in blocks)
        flex = "display:flex;gap:24px;align-items:flex-start;flex-wrap:wrap"
        tables = f"<div style='{flex}'>{columns}</div>" if blocks else ""
        return header + tables


def _text_table(headers, rows, align, indent=0):
    """Render an aligned fixed-width text table as a list of lines (no trailing newline)."""
    cols = list(zip(*([headers] + rows))) if rows else [(h,) for h in headers]
    widths = [max(len(str(c)) for c in col) for col in cols]
    pad = " " * indent

    def fmt(cells):
        out = []
        for cell, w, a in zip(cells, widths, align):
            out.append(str(cell).rjust(w) if a == "r" else str(cell).ljust(w))
        return pad + "  " + "  ".join(out).rstrip()

    return [fmt(headers)] + [fmt(r) for r in rows]


def _caption(text):
    """Small bold caption above a repr table (labels the side-by-side element/edge columns)."""
    return f"<div style='font-family:sans-serif;font-size:0.85em;font-weight:bold;margin-bottom:2px'>{text}</div>"


def _html_table(headers, rows, align):
    """Render an HTML table (eigenmode-repr styling) from header and row cell lists."""
    th = "padding:2px 8px;border-bottom:1px solid #ccc"
    head = "<tr>" + "".join(f"<th style='text-align:{a};{th}'>{h}</th>" for h, a in zip(headers, align)) + "</tr>"
    body = [
        "<tr>" + "".join(f"<td style='text-align:{a};padding:2px 8px'>{c}</td>" for c, a in zip(r, align)) + "</tr>"
        for r in rows
    ]
    return (
        "<table style='border-collapse:collapse;font-family:monospace;font-size:0.9em;margin-bottom:6px'>"
        + head
        + "".join(body)
        + "</table>"
    )


@dataclass
class Solution:
    """Converged mean-flow result with named edge-field access.

    Key methods
    -----------
    field(name), edge(e)
        Read a per-edge field across the network, or every field on one edge.
    print_states(), table()
        Show / return the full per-edge mean-flow state table.
    residuals(), print_residuals()
        The converged residual broken down equation-by-equation.
    composite(key), composites
        Read a composite element's hidden interior (e.g. an orifice throat).
    species(e), mixture_fractions(e), marker(e)
        Per-edge chemistry: solved species, transported feed fractions, burnt marker.
    heat_release()
        Heating power [W] of every flame element (exact for the equilibrium flame).
    cuton_report()
        Per-duct plane-wave validity ceiling (higher-order-mode cut-on).
    eigenmodes(), forced_response(), perturbation_response(), nyquist_stability()
        Linear acoustic and stability analyses on this mean flow (see :mod:`nefes.perturbation`).
    to_yaml(path)
        Write the network and these results to a UI-readable YAML case.
    """

    network: Network
    problem: CompiledProblem
    result: object  # SolveResult

    @classmethod
    def from_yaml(cls, path: str, method: str = "warm", dataset: str = "Mean flow", **solve_kw) -> "Solution":
        """Restore a solution embedded in a saved UI/YAML case, skipping a cold re-solve.

        The inverse of :meth:`to_yaml`.  Reads the network and rebuilds the solver state from the
        saved datasets; ``method`` toggles how the stored state is trusted -- ``"warm"`` (default)
        verifies it with a single ``kappa = 0`` solve (returns at iteration ``0`` with no Jacobian
        when the state is faithful), ``"deserialize"`` returns it verbatim with no solve.  Thin
        wrapper over :func:`nefes.io.load_solution`; see it for the full parameter set.

        Parameters
        ----------
        path : str
            Path to a ``.yaml`` case carrying an embedded solution.
        method : {"warm", "deserialize"}, optional
            Restore strategy (default ``"warm"``).
        dataset : str, optional
            Name of the mean-flow dataset to restore (default ``"Mean flow"``).
        **solve_kw
            Forwarded to :meth:`Network.solve` when ``method="warm"``.

        Returns
        -------
        Solution
        """
        from ..io import load_solution

        return load_solution(path, method=method, dataset=dataset, **solve_kw)

    def __repr__(self) -> str:
        """Return a string representation of the solution."""
        return f"Solution(converged={self.converged}, iterations={self.iterations}, residual_norm={self.residual_norm})"

    def _repr_html_(self) -> str:
        """Return an HTML representation of the solution."""
        return (
            f"<div>Converged: {self.converged}</div><div>Iterations: {self.iterations}</div>"
            f"<div>Residual norm: {self.residual_norm}</div>"
        )

    @property
    def converged(self) -> bool:
        """Whether the solver reached the convergence tolerance."""
        return self.result.converged

    @property
    def iterations(self) -> int:
        """Number of Newton iterations taken."""
        return self.result.iterations

    @property
    def residual_norm(self) -> float:
        """Final scaled-residual 2-norm -- the quantity the solve drives below its ``tol``."""
        return self.result.residual_norm

    @property
    def elapsed(self) -> float:
        """Seconds the mean-flow solve took, on a monotonic clock.

        Covers the whole of :meth:`Network.solve` (the seed, every continuation stage, and any
        subsonic-scope re-solve).  The first solve of a session also carries the one-off
        compilation of the kernels, which typically dwarfs the solve itself, so a sweep's second
        and later solves are the ones that measure the solver.  A solution restored from a case
        file without re-solving reports ``0.0``.

        Being a reading of this machine and this run, it is the one field here that does not
        reproduce across runs, and it is not written to a case file.

        Examples
        --------
        >>> sol = net.solve()  # doctest: +SKIP
        >>> f"{sol.elapsed * 1e3:.1f} ms in {sol.iterations} iteration(s)"  # doctest: +SKIP
        '2.5 ms in 6 iteration(s)'
        """
        return self.result.elapsed

    @property
    def x(self) -> np.ndarray:
        """Raw converged state vector."""
        return self.result.x

    def _recovery_error(self) -> RuntimeError:
        """A clear diagnostic for a failed state recovery (replaces an opaque ``LinAlgError``)."""
        return RuntimeError(
            "could not recover the edge states from this solution: the equilibrium state "
            f"solve returned a non-finite result (converged={self.converged}, "
            f"residual_norm={self.residual_norm:.3e}). This points to an operating point "
            "outside the solver's envelope; inspect it with sol.print_residuals()."
        )

    def table(self, show_internal: bool = True) -> np.ndarray:
        """Return the per-edge state table (rows are fields, columns are edges).

        Parameters
        ----------
        show_internal : bool, optional
            When ``False`` and the network carries composite elements, drop the composite
            *internal* edge columns, leaving only the user-facing edges (which keep their
            ids; internals append at the tail).  Default ``True`` (every edge).
        """
        try:
            est = states_table(self.problem, self.result.x)
        except np.linalg.LinAlgError as exc:
            # A non-physical state can still drive the equilibrium recovery singular; surface it
            # as a clear diagnostic instead of the opaque LinAlgError from deep in the kernel.
            raise self._recovery_error() from exc
        cm = self.problem.composite_map
        if show_internal or cm is None or not cm.internal_edges:
            return est
        keep = [e for e in range(self.problem.n_edges) if e not in cm.internal_edges]
        return est[:, keep]

    def composite(self, key) -> "CompositeView":
        """Read the hidden interior of a solved composite element.

        A composite (e.g. an orifice or a tapered nozzle) is added as one element but expands into
        several sub-elements joined by internal edges that :meth:`table` hides by default.  This
        returns a :class:`~nefes.elements.composite.CompositeView` over that interior -- its internal
        edges, and, for a contracting composite, its throat.

        Parameters
        ----------
        key : str or int
            The composite's name or its user node id.

        Returns
        -------
        CompositeView
            A view exposing the composite's internal edges and (where it contracts) throat state.
        """
        cm = self.problem.composite_map
        if cm is None:
            raise ValueError("this network has no composite elements")
        if isinstance(key, str):
            nodes = [n for n, nm in cm.composite_name.items() if nm == key]
            if not nodes:
                raise ValueError(f"no composite named {key!r}; have {sorted(cm.composite_name.values())}")
            node = nodes[0]
        else:
            node = int(key)
            if node not in cm.composite_name:
                raise ValueError(f"user node {node} is not a composite element")
        expanded = set(cm.expanded_nodes(node))
        tail, head, area = self.problem.tail_node, self.problem.head_node, self.problem.area
        internal = tuple(sorted(e for e in cm.internal_edges if int(tail[e]) in expanded and int(head[e]) in expanded))
        throat = min(internal, key=lambda e: float(area[e])) if internal else None
        return CompositeView(
            name=cm.composite_name[node],
            kind=cm.composite_kind.get(node, ""),
            node=node,
            nodes=tuple(cm.expanded_nodes(node)),
            internal_edges=internal,
            throat=throat,
            _solution=self,
        )

    @property
    def composites(self) -> List["CompositeView"]:
        """Every composite element in the network, as :class:`CompositeView` projections."""
        cm = self.problem.composite_map
        if cm is None:
            return []
        return [self.composite(n) for n in sorted(cm.composite_name)]

    def cuton_report(self, section: str = "circular", aspect: float = 1.0):
        """Per-duct higher-order-mode cut-on frequencies and the plane-wave ceiling.

        The Nefes acoustic layer is plane-wave (1-D); it is valid only below the first
        duct cut-on frequency.  This reports the cut-on of every edge (from its area,
        sound speed and Mach) and the network-wide ceiling
        (:attr:`~nefes.perturbation.CutOnReport.f_cuton`) -- keep any perturbation
        analysis below it.

        Parameters
        ----------
        section : {"circular", "square", "rectangular"}, optional
            Assumed duct cross-section shape (Nefes ducts store only an area).
        aspect : float, optional
            Width-to-height ratio (``>= 1``) for ``section="rectangular"``, used to recover the
            larger transverse dimension (which sets the cut-on) from the area.  Ignored for the
            circular and square sections; default ``1.0`` (a square).

        Returns
        -------
        nefes.perturbation.CutOnReport
        """
        from ..perturbation.fields.cuton import duct_cuton_frequencies

        return duct_cuton_frequencies(
            self.problem, self.result.x, section=section, aspect=aspect, names=self.network._edge_names
        )

    def plot(self, color_by=None, width_by=None, **kwargs):
        """Draw this solved network as a node/edge diagram with the solved state on the edges (Plotly).

        The same diagram as :meth:`Network.plot`, with this solution attached: the edge hover carries
        the headline state, and ``color_by`` / ``width_by`` map any solved edge field onto edge color /
        arrow width.  Shares one backend (:func:`nefes.plotting.plot_network_topology`) with the
        structural view, so topology and results read the same way.

        Parameters
        ----------
        color_by : str, optional
            Solved edge field to color the edges by, e.g. ``"T"``, ``"M"``, ``"mdot"`` (keys of the
            per-edge state; see :meth:`field`).  Adds a colorbar and labels each edge with its value.
        width_by : str, optional
            Solved edge field whose magnitude scales each edge's arrow width (e.g. ``"mdot"`` for a
            flow-weighted diagram, ``"area"`` for a geometry-weighted one).
        **kwargs
            Forwarded to :func:`nefes.plotting.plot_network_topology` (e.g. ``colorscale``,
            ``show_edge_labels``, ``show_areas``, ``title``, ``height``, ``width``).

        Returns
        -------
        plotly.graph_objects.Figure
        """
        from ..plotting import plot_network_topology

        return plot_network_topology(self.network, solution=self, color_by=color_by, width_by=width_by, **kwargs)

    def print_states(self, edges=None, precision: int = 5, file=None) -> None:
        """Print the per-edge mean-flow state table to the screen.

        Thin wrapper over :func:`nefes.solver.report.print_states`; see it for the column layout.
        In a notebook (and when ``file`` is not given) the table renders as rich HTML;
        otherwise a fixed-width text table is printed.

        Parameters
        ----------
        edges : sequence of int, optional
            Edge indices to include, in the given order (default: every edge).
        precision : int, optional
            Number of significant digits printed per value (default 5).
        file : file-like, optional
            Destination stream forwarded to :func:`print` (default ``sys.stdout``).
        """
        print_states(self.problem, self.result.x, edges=edges, precision=precision, file=file)

    def residuals(self) -> dict:
        """Return the converged residual broken down equation-by-equation.

        Resolves the single :attr:`residual_norm` into its per-equation contributions,
        keyed by a human-readable label (element/equation for node rows, edge + scalar
        for transport rows).  Useful for spotting which equation a non-converged solve
        left large.

        Returns
        -------
        dict
            ``{label: scaled_residual}`` for every equation, in residual-row order.
        """
        labels, _R, R_hat = residual_breakdown(self.problem, self.result.x)
        return {label: float(v) for label, v in zip(labels, R_hat)}

    def print_residuals(self, sort: bool = True, top=None, precision: int = 4, file=None) -> None:
        """Print the residual broken down equation-by-equation.

        Thin wrapper over :func:`nefes.solver.report.print_residuals`; see it for the column layout.

        Parameters
        ----------
        sort : bool, optional
            Order rows by descending ``|scaled residual|`` (default ``True``).
        top : int, optional
            Show only the worst ``top`` equations (default: all).
        precision : int, optional
            Significant digits printed per residual value (default 4).
        file : file-like, optional
            Destination stream forwarded to :func:`print` (default ``sys.stdout``).
        """
        print_residuals(self.problem, self.result.x, sort=sort, top=top, precision=precision, file=file)

    def edge(self, e: int) -> dict:
        """Return a ``{field: value}`` dict of all derived quantities on edge ``e``."""
        col = self.table()[:, e]
        return {name: float(col[idx]) for name, idx in _EDGE_FIELDS.items()}

    def field(self, name: str) -> np.ndarray:
        """Return the named field across all edges.

        Names are the keys of the per-edge state: ``mdot, p, h_t, rho, u, T, c, M, p_t,
        area``, plus the mixture molar mass ``W`` [kg/mol] and specific heat ``cp``
        [J/(kg K)] (the latter consistent with the local sound speed -- exact for a
        perfect gas, the frozen value on an unburnt edge and the equilibrium value on a
        burnt one).
        """
        return self.table()[_EDGE_FIELDS[name], :]

    def unchoked_nozzles(self) -> list:
        """Choked-nozzle outlets whose set back pressure is too high for the throat to choke.

        A :func:`~nefes.elements.catalog.choked_nozzle_outlet` asserts a sonic throat, which
        holds only while the ambient back pressure sits below the throat's critical (sonic)
        pressure ``p* = p_t (2 / (g + 1))^(g / (g - 1))`` (``g`` the local ratio of specific
        heats, ``g = c^2 W / (R_u T)``).  For every such element given a ``back_pressure`` at
        construction, this compares it to ``p*`` at the converged state and returns the nozzles
        that would *not* actually choke -- the compact choked-nozzle model does not apply to
        them.  One of the checks :meth:`verify` runs after a solve; call it directly to inspect.

        Returns
        -------
        list of dict
            One entry per offending nozzle: ``{"node", "name", "edge", "back_pressure",
            "critical_pressure", "p_t"}``.  Empty when every nozzle with a set back pressure is
            genuinely choked (or none set one).
        """
        p_t, c, W, T = self.field("p_t"), self.field("c"), self.field("W"), self.field("T")
        cmap = getattr(self.problem, "composite_map", None)
        out = []
        for user_n, el in enumerate(self.network._elements):
            # composites expand to several atomic nodes and are never a choked nozzle; skip them
            if is_composite(el) or getattr(el, "residual_id", None) != CHOKED_NOZZLE_OUTLET:
                continue
            if el.back_pressure is None:
                continue
            # map the user node id to its compiled (expanded) node id, then to its single edge
            node = cmap.expanded_nodes(user_n)[0] if cmap is not None else user_n
            e = int(self.problem.col_edge[self.problem.row_ptr[node]])
            gamma = float(c[e]) ** 2 * float(W[e]) / (R_UNIVERSAL * float(T[e]))
            p_crit = float(p_t[e]) * (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))
            if float(el.back_pressure) > p_crit:
                out.append(
                    {
                        "node": node,
                        "name": el.name,
                        "edge": e,
                        "back_pressure": float(el.back_pressure),
                        "critical_pressure": p_crit,
                        "p_t": float(p_t[e]),
                    }
                )
        return out

    def unrealizable_lumped_shocks(self) -> list:
        """Choked area changes whose lumped total-pressure drop exceeds any realizable shock.

        When an :func:`~nefes.elements.catalog.isentropic_area_change` chokes -- its small port
        sonic with the flow entering there -- the total-pressure drop its complementarity row
        admits is the *lumped normal shock* standing somewhere in the diverging side (see the
        choking theory document).  That representation is faithful exactly while a normal shock
        inside the divergent can produce the converged loss: the strongest candidate stands at
        the exit plane, where the (hypothetical) supersonic branch reaches its largest Mach
        number.  Inverting the Rankine--Hugoniot total-pressure ratio on the converged loss
        yields the implied shock Mach; when it exceeds the exit-plane supersonic Mach, no shock
        position inside the element can host it -- the back pressure lies in the supersonic-exit
        regime, outside the subsonic scope, and the converged subsonic root is not physical.
        Evaluated on the compiled (composite-expanded) elements, so area changes inside
        composites are covered.  One of the checks :meth:`verify` runs after a solve; call it
        directly to inspect, including the implied shock Mach of the *realizable* cases.

        Returns
        -------
        list of dict
            One entry per choked area change taking a resolvable drop: ``{"node", "name",
            "edge_small", "edge_large", "pt_ratio", "implied_shock_mach", "max_shock_mach",
            "realizable"}``.  Filter on ``realizable`` for the offending ones.
        """
        prob = self.problem
        pt, c, W, T, M = (self.field(k) for k in ("p_t", "c", "W", "T", "M"))
        mdot, area = self.field("mdot"), self.field("area")

        def _bisect(f, lo, hi, n=80):
            flo = f(lo)
            for _ in range(n):
                mid = 0.5 * (lo + hi)
                if (f(mid) > 0.0) == (flo > 0.0):
                    lo = mid
                else:
                    hi = mid
            return 0.5 * (lo + hi)

        out = []
        for node in range(prob.n_nodes):
            if int(prob.node_rid[node]) != ISEN_AREA_CHANGE:
                continue
            ports = range(int(prob.row_ptr[node]), int(prob.row_ptr[node + 1]))
            edges = [int(prob.col_edge[p]) for p in ports]
            orients = [float(prob.orient[p]) for p in ports]
            if len(edges) != 2 or area[edges[0]] == area[edges[1]]:
                continue
            (e_s, o_s), (e_l, o_l) = sorted(zip(edges, orients), key=lambda eo: area[eo[0]])
            # choked-diverging operation: flow *enters* through the small port at M = 1
            if -o_s * mdot[e_s] <= 0.0 or M[e_s] < 0.995:
                continue
            ptr = float(pt[e_l] / pt[e_s])
            if ptr > 1.0 - 1e-3:  # within the complementarity smoothing bias: no lumped shock
                continue
            g = float(c[e_s]) ** 2 * float(W[e_s]) / (R_UNIVERSAL * float(T[e_s]))
            ar = float(area[e_l] / area[e_s])

            def _area_ratio(Ma):
                return (1.0 / Ma) * ((2.0 + (g - 1.0) * Ma * Ma) / (g + 1.0)) ** ((g + 1.0) / (2.0 * (g - 1.0)))

            def _pt_shock(Ma):
                a = ((g + 1.0) * Ma * Ma / (2.0 + (g - 1.0) * Ma * Ma)) ** (g / (g - 1.0))
                b = ((g + 1.0) / (2.0 * g * Ma * Ma - (g - 1.0))) ** (1.0 / (g - 1.0))
                return a * b

            m_exit = _bisect(lambda Ma: _area_ratio(Ma) - ar, 1.0 + 1e-12, 50.0)
            m_shock = _bisect(lambda Ma: _pt_shock(Ma) - ptr, 1.0 + 1e-12, 50.0)
            name = ""
            cmap = getattr(prob, "composite_map", None)
            if cmap is None:
                name = getattr(self.network._elements[node], "name", "")
            out.append(
                {
                    "node": node,
                    "name": name,
                    "edge_small": e_s,
                    "edge_large": e_l,
                    "pt_ratio": ptr,
                    "implied_shock_mach": m_shock,
                    "max_shock_mach": m_exit,
                    "realizable": m_shock <= m_exit * (1.0 + 1e-3),
                }
            )
        return out

    def verify(self) -> list:
        """Run the post-solve model-validity checks and return one message per issue found.

        The single home for checks that can only be evaluated once the mean flow is converged
        (as opposed to the structural checks :func:`nefes.shell.build.validate_network` runs at
        compile time).  Each check is gated by its ``CHECK_*`` toggle in
        :mod:`nefes.shell.checks`; currently these are the choked-nozzle back-pressure check
        (:meth:`unchoked_nozzles`, gated by ``CHECK_CHOKED_NOZZLE``) and the lumped-shock
        realizability check (:meth:`unrealizable_lumped_shocks`, gated by
        ``CHECK_LUMPED_SHOCK``).  :meth:`Network.solve` calls this on a converged solution and
        re-emits each message as a warning; call it directly to collect them without the
        warnings machinery.

        Returns
        -------
        list of str
            Human-readable messages, one per issue; empty when the solution passes every check.
        """
        messages = []
        if checks.CHECK_CHOKED_NOZZLE:
            for nz in self.unchoked_nozzles():
                messages.append(
                    f"choked_nozzle_outlet {nz['name']!r} (edge {nz['edge']}): the specified back pressure "
                    f"{nz['back_pressure']:.4g} Pa exceeds the throat's critical pressure "
                    f"{nz['critical_pressure']:.4g} Pa, so the nozzle would not choke -- the compact choked-nozzle "
                    f"model does not apply here; use a pressure_outlet, which handles the choked/unchoked "
                    f"transition against a back pressure."
                )
        if checks.CHECK_LUMPED_SHOCK:
            for sh in self.unrealizable_lumped_shocks():
                if sh["realizable"]:
                    continue
                label = f" {sh['name']!r}" if sh["name"] else ""
                messages.append(
                    f"isentropic_area_change{label} (node {sh['node']}, edges {sh['edge_small']}->"
                    f"{sh['edge_large']}): the choked total-pressure ratio {sh['pt_ratio']:.4f} implies a "
                    f"normal shock at Mach {sh['implied_shock_mach']:.3f}, stronger than any the diverging "
                    f"side can host (shock at the exit plane: Mach {sh['max_shock_mach']:.3f}). The back "
                    f"pressure lies in the supersonic-exit regime, outside the subsonic scope, so the "
                    f"converged subsonic root is not physical here."
                )
        return messages

    def mixture_fractions(self, e: int) -> dict:
        """Transported feed-stream mixture fractions ``{stream_label: xi}`` on edge ``e``.

        These are the conserved scalars the solver carries (one per distinct injected feed),
        not chemical species; for the actual species use :meth:`species`.  Empty for a
        perfect gas with no passive scalars.
        """
        names = self.problem.scalar_names
        xi = self.result.x[3 : 3 + self.problem.n_elem, e]
        return {name: float(v) for name, v in zip(names, xi)}

    def marker(self, e: int) -> float:
        """Transported burnt marker on edge ``e`` (``0`` fresh / unburnt, ``1`` burnt).

        The scalar that gates the reacting closure's frozen/equilibrium blend.  Bimodal at
        convergence (a flame is a discrete ``0 -> 1`` jump), so it reads ~0 on an unburnt edge
        and ~1 on a burnt one.  Returns ``0.0`` when the network carries no marker (a perfect
        gas, or a reacting network built with an explicit hard per-edge closure).
        """
        mr = int(getattr(self.problem, "marker_row", -1))
        return 0.0 if mr < 0 else float(self.result.x[mr, e])

    def _chemistry_caches(self):
        """Lazily build and cache the per-edge product moles and per-stream mass fractions."""
        if getattr(self, "_chem_cache", None) is None:
            from ..chem.chemistry import product_moles, stream_mass_fractions

            lib = self.network.gas.species_set
            moles = product_moles(self.problem, self.result.x)
            # declared streams (equilibrium(streams=...)) carry their fixed basis on the gas model;
            # otherwise rebuild the auto-discovered basis from the feed compositions.
            declared = getattr(self.network.gas, "stream_Y", None)
            if lib is None:
                stream_Y = None
            elif declared is not None:
                stream_Y = np.asarray(declared, dtype=float)
            else:
                stream_Y = stream_mass_fractions(self.network._elements, lib)
            self._chem_cache = (moles, stream_Y)
        return self._chem_cache

    def species(self, e: int, basis: str = "mole") -> dict:
        """Solved chemical species ``{name: fraction}`` on edge ``e``.

        A burnt (equilibrium) edge reports its HP-equilibrium products; an unburnt (frozen)
        edge reports the forward blend of its feed streams; a perfect-gas edge has no
        chemical species (use :meth:`mixture_fractions` for its passive scalars).

        Parameters
        ----------
        e : int
            Edge id.
        basis : {"mole", "mass"}, optional
            Mole or mass fractions (default ``"mole"``).

        Returns
        -------
        dict
            ``{species_name: fraction}`` for the species present on the edge.
        """
        from ..chem.chemistry import edge_species

        moles, stream_Y = self._chemistry_caches()
        lib = self.network.gas.species_set
        return edge_species(self.problem, self.result.x, e, lib, basis=basis, moles=moles, stream_Y=stream_Y)

    def heat_release(self) -> Dict[str, float]:
        """Heat release rate [W] of every flame element, ``{name: Q}``.

        Reads each flame's heating power off the converged mean flow as the sensible
        total-enthalpy rise of its through-flow.  For the perfect-gas
        :func:`~nefes.elements.catalog.heat_release_flame` this reproduces its ``Qdot``
        parameter; for the reacting :func:`~nefes.elements.catalog.equilibrium_flame` --
        whose power is an outcome of the equilibrium, not an input -- it is the exact
        formation-enthalpy drop from frozen reactants to equilibrium products at the
        converged composition, with no specific-heat approximation.  Positive heats the
        flow.  Empty when the network carries no flame.

        Returns
        -------
        dict
            ``{flame_name: Q_watts}``, in node order.

        Examples
        --------
        >>> sol.heat_release()  # doctest: +SKIP
        {'flame': 1483205.7}

        See Also
        --------
        nefes.chem.chemistry.node_heat_release : the per-element computation.
        """
        from ..chem.chemistry import node_heat_release, product_moles
        from ..elements.ids import FLAME_EQUILIBRIUM, FLAME_HEAT_RELEASE

        prob, x = self.problem, self.result.x
        flames = [
            n for n in range(int(prob.n_nodes)) if int(prob.node_rid[n]) in (FLAME_HEAT_RELEASE, FLAME_EQUILIBRIUM)
        ]
        if not flames:
            return {}
        est = self.table()
        moles = product_moles(prob, x)
        names = prob.node_names or tuple(f"node{n}" for n in range(int(prob.n_nodes)))
        return {names[n]: node_heat_release(prob, x, n, est=est, moles=moles) for n in flames}

    # -- perturbation / acoustics (linear analyses on this converged mean flow) ----------------------------------------

    def eigenmodes(self, freq_band=None, **kwargs):
        """Free-oscillation eigenmodes of the perturbation network on this mean flow.

        A bound form of :func:`nefes.perturbation.eigenmodes` that supplies this solution's
        compiled problem and mean state; see it for the full parameter set and the growth-rate
        sign convention.  Set the terminal :class:`~nefes.perturbation.PerturbationBC`\\ s on the
        network before solving.

        Parameters
        ----------
        freq_band : tuple of float
            ``(f_lo, f_hi)`` real-frequency search window, in Hz.
        **kwargs
            Forwarded to :func:`nefes.perturbation.eigenmodes` (e.g. ``growth_band``, ``isentropic``).

        Returns
        -------
        EigenmodeResult

        See Also
        --------
        nefes.perturbation.eigenmodes : the underlying routine.
        """
        from ..perturbation import eigenmodes

        res = eigenmodes(self.problem, self.x, freq_band, **kwargs)
        res._solution = self  # let EigenmodeResult.sensitivities() reach the network
        return res

    def forced_response(self, freqs, **kwargs):
        """Perturbation field under each terminal's declared boundary condition, on this mean flow.

        A bound form of :func:`nefes.perturbation.forced_response`: the forcing is whatever the
        terminals' ``driven`` :class:`~nefes.perturbation.PerturbationBC`\\ s inject.

        Parameters
        ----------
        freqs : array_like
            Frequencies (Hz) to solve at.
        **kwargs
            Forwarded to :func:`nefes.perturbation.forced_response` (e.g. ``isentropic``).

        Returns
        -------
        ForcedResponse

        See Also
        --------
        nefes.perturbation.forced_response : the underlying routine.
        """
        from ..perturbation import forced_response

        return forced_response(self.problem, self.x, freqs, **kwargs)

    def perturbation_response(self, freqs, forcing=None, **kwargs):
        """Transfer / scattering response by driving each terminal wave, on this mean flow.

        A bound form of :func:`nefes.perturbation.perturbation_response`; the matrices it yields
        are independent of the physical terminations.

        Parameters
        ----------
        freqs : array_like
            Frequencies (Hz) to solve at.
        forcing : tuple of int, optional
            The pair of terminal node ids to force (default: every open terminal).
        **kwargs
            Forwarded to :func:`nefes.perturbation.perturbation_response` (e.g. ``excite``).

        Returns
        -------
        PerturbationResponse

        See Also
        --------
        nefes.perturbation.perturbation_response : the underlying routine.
        """
        from ..perturbation import perturbation_response

        return perturbation_response(self.problem, self.x, freqs, forcing, **kwargs)

    def nyquist_stability(self, freqs, **kwargs):
        """Unstable-mode count from the real-frequency Nyquist sweep, on this mean flow.

        A bound form of :func:`nefes.perturbation.nyquist_stability`; the network must carry at
        least one dynamic source (a flame FTF or a fluctuating injector).

        Parameters
        ----------
        freqs : array_like
            Real frequencies (Hz), spanning ``~0`` to past the highest mode.
        **kwargs
            Forwarded to :func:`nefes.perturbation.nyquist_stability` (e.g. ``isentropic``).

        Returns
        -------
        NyquistResponse

        See Also
        --------
        nefes.perturbation.nyquist_stability : the underlying routine.
        """
        from ..perturbation import nyquist_stability

        return nyquist_stability(self.problem, self.x, freqs, **kwargs)

    def to_yaml(self, path: str, dataset: str = "Mean flow", **kwargs) -> None:
        """Write the network and this solution's results to a UI-readable YAML case.

        Embeds the mean-flow fields (and any transported chemistry) as a named dataset the UI can
        load.  If ``path`` does not yet exist, a fresh case is written.  If it exists -- and already
        holds this same network -- the results are *appended* as a new dataset, so several solutions
        (e.g. operating points) can be overlaid in one file from repeated calls with distinct
        ``dataset`` names.

        Parameters
        ----------
        path : str
            Destination ``.yaml`` path.  Appended to when it already exists.
        dataset : str, optional
            Name for this solution's mean-flow dataset (default ``"Mean flow"``).  Appending a
            dataset whose name is already present in the file raises ``ValueError``.
        **kwargs
            Forwarded to :func:`nefes.io.save_case` / :func:`nefes.io.dump_case` (e.g. ``fields``,
            ``node_data``, ``forced``, ``title``).
        """
        from ..io import save_solution

        save_solution(self.network, self, path, dataset=dataset, **kwargs)

    def save(self, path: str, dataset: str = "Mean flow", **kwargs) -> None:
        """Alias for :meth:`to_yaml` (kept for intuitive usage)."""
        self.to_yaml(path, dataset=dataset, **kwargs)
