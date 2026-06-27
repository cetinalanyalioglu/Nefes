"""Element catalog and CompiledProblem builder (Python, parse-time).

An ``ElementSpec`` names an element's residual id and its ordered float
parameters (the order the @njit kernels expect).  ``build_problem`` turns a list
of element specs plus directed edges into the immutable CompiledProblem.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ..composition import build_streams, enthalpy_mass, species_mass_fractions
from ..connectivity import connectivity_from_directed_edges, build_jacobian_pattern, Connectivity
from ..problem import CompiledProblem
from ..thermo.api import EQ_FROZEN, EQ_KERNEL, PERFECT_GAS
from ..thermo.configure import ThermoConfig
from .ids import (
    MASS_FLOW_INLET,
    PT_INLET,
    P_OUTLET,
    MASS_FLOW_OUTLET,
    CHOKED_NOZZLE_OUTLET,
    WALL,
    JUNCTION,
    SPLITTER,
    MASS_SOURCE,
    ACOUSTIC_DEFAULT,
    ACOUSTIC_DUCT,
    FIXED_NPORTS,
    ALLOWS_AREA_CHANGE,
    RESIDUAL_NAMES,
    KIND_MASS,
    KIND_PRESSURE,
    row_kind_tags,
)

# Relative tolerance for the equal-area check on constant-area elements.
_AREA_RTOL = 1e-9


@dataclass
class ElementSpec:
    """One network element: residual type + ordered float parameters.

    ``acoustic_id`` (implementation-plan.md s8.3) declares the optional acoustic
    face that overrides the default CSD linearization; ``ACOUSTIC_DEFAULT`` means
    the element contributes only through ``J_alg``.

    ``eps`` optionally overrides this element's smoothing width (the smooth-step /
    complementarity regularization, in mass-flow units, i.e. ~ a fraction of
    ``mdot_ref``).  ``None`` follows the global solve-time ``eps``.  Settable at
    creation or mutated later (``spec.eps = ...``) before ``build_problem``; a
    sharper value makes the frozen perturbation linearization track the exact
    (un-regularized) jump -- see the ``SUDDEN_AREA_CHANGE`` note in ``kernels.py``.
    """

    residual_id: int
    fparams: List[float] = field(default_factory=list)
    name: str = ""
    acoustic_id: int = ACOUSTIC_DEFAULT
    eps: Optional[float] = None
    perturbation_bc: Optional[object] = None  # PerturbationBC (None -> inherit)
    # Composition of the stream this element introduces (inlets, mass sources) or
    # would draw on backflow (outlets).  For the equilibrium model it is a named
    # species mixture -- ``{species: fraction}`` (or a full species array) in
    # ``basis`` units (``"mole"`` or ``"mass"``) -- resolved to the transported
    # elemental ``Z`` (and the ``Tt -> h_t`` datum) at build time.  For a perfect
    # gas with passive scalars it is the raw scalar values.  ``None`` -> zeros.
    composition_spec: object = None
    basis: str = "mole"
    # optional dynamic-source descriptor (fns.elements.dynamic_source.DynamicSource);
    # a forward-compatibility provision for the S(omega) perturbation phase -- the
    # mean flow ignores it.
    dynamic_source: object = None


def mass_flow_inlet(mdot, Tt, composition=None, basis="mole", name="inlet", perturbation_bc=None):
    """Prescribed mass-flow inlet feeding a stream of the given ``composition``.

    ``composition`` is a named species mixture (``{species: fraction}``) for the
    equilibrium model -- e.g. air as ``{"O2": 0.21, "N2": 0.79}`` with
    ``basis="mole"`` -- resolved to the transported elemental ``Z`` and the feed
    enthalpy at ``Tt`` during ``build_problem``.

    This is an **inflow boundary**: ``mdot`` must be non-negative (``>= 0``).  A
    positive value injects the feed stream; ``mdot = 0`` is a quiescent (closed) inlet.
    Reverse flow (a negative prescribed mass rate, i.e. suction out through the inlet)
    is not permitted -- use a :func:`pressure_outlet`, which models ingestion/backflow,
    for a boundary that may reverse.
    """
    if float(mdot) < 0.0:
        raise ValueError(
            f"mass_flow_inlet is an inflow boundary; mdot must be >= 0 (got {mdot}). Reverse flow is not "
            "permitted -- use a pressure_outlet (which models ingestion/backflow) for a reversing boundary."
        )
    return ElementSpec(
        MASS_FLOW_INLET,
        [float(mdot), float(Tt)],
        name,
        perturbation_bc=perturbation_bc,
        composition_spec=composition,
        basis=basis,
    )


def total_pressure_inlet(pt, Tt, composition=None, basis="mole", name="pt-inlet", perturbation_bc=None):
    """Prescribed total-pressure inlet feeding a stream of the given ``composition``."""
    return ElementSpec(
        PT_INLET,
        [float(pt), float(Tt)],
        name,
        perturbation_bc=perturbation_bc,
        composition_spec=composition,
        basis=basis,
    )


def pressure_outlet(p, Tt_backflow=300.0, composition=None, basis="mole", name="outlet", perturbation_bc=None):
    """Static-pressure outlet; ``composition`` is the backflow stream (on ingestion)."""
    return ElementSpec(
        P_OUTLET,
        [float(p), float(Tt_backflow)],
        name,
        perturbation_bc=perturbation_bc,
        composition_spec=composition,
        basis=basis,
    )


def mass_flow_outlet(mdot, name="outlet", perturbation_bc=None):
    """Prescribed-mass-flow outlet: the outflow rate is pinned to ``mdot``.

    The static pressure floats (it is whatever the interior produces); useful for a
    metered exhaust or as the mean-flow partner of a downstream choked throat whose
    critical mass flow is known.  Left at ``perturbation_bc=None`` the acoustic
    termination is the *inherited* linearization of this row, ``mdot' = 0`` -- the
    constant-mass-flow acoustic boundary condition (also available standalone as
    :meth:`~fns.perturbation.boundary_bc.PerturbationBC.constant_mass_flow`).

    This is an **outflow-only** boundary: ``mdot`` must be positive and the element does
    not model ingestion (use a :func:`pressure_outlet`, which models backflow, for a
    boundary that may reverse).

    Parameters
    ----------
    mdot : float
        Prescribed outflow mass rate [kg/s] (must be > 0, leaving the domain).
    name : str, optional
        Element label.
    perturbation_bc : PerturbationBC, optional
        Acoustic termination; ``None`` inherits ``mdot' = 0`` from this row.
    """
    if not float(mdot) > 0.0:
        raise ValueError(
            f"mass_flow_outlet is an outflow boundary; mdot must be > 0 (got {mdot}). "
            "Use a pressure_outlet (which models ingestion/backflow) for a reversing boundary."
        )
    return ElementSpec(MASS_FLOW_OUTLET, [float(mdot)], name, perturbation_bc=perturbation_bc)


def choked_nozzle_outlet(throat_area, name="outlet", perturbation_bc=None):
    """Compact choked-nozzle outlet of throat area ``throat_area`` lumped downstream.

    Models a sonic (M=1) throat just beyond the outlet plane: the outflow is the
    critical mass flux for the interior stagnation state and the throat area, so the
    static pressure floats (the choked mass flow is set by the upstream total state, not
    a back-pressure).  Because the nozzle is *compact* (lumped), the application plane
    stays **subsonic** -- the sonic point is in the throat, not the domain -- so the
    acoustic operator is non-degenerate.  Left at ``perturbation_bc=None`` the acoustic
    termination is the *inherited* linearization of the critical-mass-flux row, which is
    the compact choked-nozzle (Marble--Candel) reflection with its entropy -> acoustic
    coupling -- no separately specified boundary condition needed.

    Use it when the convergent section is acoustically compact; resolving the higher-Mach
    part of the contraction explicitly and lumping only the near-throat remainder makes
    the compact assumption progressively better.  A choked throat is one-way, so this is
    an **outflow-only** boundary: it does not model ingestion (the critical mass flux is
    always positive, so the converged flow cannot reverse here).

    This element **asserts** the nozzle is choked: it imposes the critical mass flux
    unconditionally (the mass flow scales with the interior total pressure and stays
    sonic-throat-consistent at any total pressure -- there is no back-pressure to detect
    unchoking).  ``throat_area`` must be smaller than the outlet edge area (a contraction;
    enforced at build time).  If the nozzle may be unchoked at low pressure ratio, use
    :func:`pressure_outlet` instead -- its emergent complementarity handles the
    choked/unchoked transition against a prescribed back-pressure.

    Parameters
    ----------
    throat_area : float
        Sonic-throat area ``A*`` [m^2] of the lumped nozzle (``A* < A_outlet`` so the
        approach stays subsonic).  Must be > 0.
    name : str, optional
        Element label.
    perturbation_bc : PerturbationBC, optional
        Acoustic termination; ``None`` inherits the linearized choked-nozzle reflection.
    """
    if not float(throat_area) > 0.0:
        raise ValueError(f"choked_nozzle_outlet throat area must be > 0 (got {throat_area})")
    return ElementSpec(CHOKED_NOZZLE_OUTLET, [float(throat_area)], name, perturbation_bc=perturbation_bc)


def wall(name="wall", perturbation_bc=None):
    """An impermeable single-port termination: ``mdot = 0`` on its incident edge.

    The wall blocks mean flow, so the leg behind it is stagnant (``M = 0``); its
    purpose is acoustic.  By default it closes the perturbation problem as a rigid
    hard wall (``u' = 0``, ``R = +1``) -- which at the wall's ``M = 0`` state is
    identical to the inherited ``mdot' = 0`` row.  Pass ``perturbation_bc`` to model
    a non-rigid termination (e.g. a liner impedance) instead.
    """
    from ..perturbation.boundary_bc import PerturbationBC

    bc = perturbation_bc if perturbation_bc is not None else PerturbationBC.hard_wall()
    return ElementSpec(WALL, [], name, perturbation_bc=bc)


def isentropic_area_change(name="iac"):
    from .ids import ISEN_AREA_CHANGE

    return ElementSpec(ISEN_AREA_CHANGE, [], name)


def sudden_area_change(name="sac", cc=1.0, eps=None):
    """Sudden area change: Borda-Carnot expansion, vena-contracta contraction.

    Forward flow (small -> large) follows the Borda-Carnot momentum balance
    (separation at the step, mixing loss).  Reverse flow (large -> small, a
    contraction) follows a vena-contracta total-pressure loss
    ``K_c * (1/2 rho u^2)_small`` with ``K_c = (1/cc - 1)^2``, referenced to the
    downstream (small-port) dynamic head.  The small/large sides are identified
    from the attached edge areas, so ``cc`` always acts on whichever direction is
    contracting.

    Parameters
    ----------
    name : str, optional
        Element label.
    cc : float, optional
        Vena-contracta contraction coefficient for the reverse (contracting)
        flow, in ``(0, 1]``.  ``cc = 1`` (default) is the loss-free contraction:
        the reverse branch reduces to exact total-pressure continuity (the
        historical behaviour).  Use a tabulated value for the geometry (e.g.
        ~0.62 for a sharp-edged contraction at a small area ratio; Weisbach /
        Idelchik).  Forward (expanding) flow is unaffected by ``cc``.

        The loss uses the incompressible ``1/2 rho u^2`` head, so it is accurate
        only to ``O(M^2)``; a dedicated contraction element resolving the vena-
        contracta state (exact at higher Mach) is planned.
    eps : float, optional
        Optionally sharpens this element's momentum<->contraction switch (see
        ``ElementSpec.eps``); use a small value (e.g. ``1e-6 * mdot_ref``) when
        the flow is firmly one-directional and an accurate perturbation jump is
        wanted.
    """
    from .ids import SUDDEN_AREA_CHANGE

    cc = float(cc)
    if not 0.0 < cc <= 1.0:
        raise ValueError(f"sudden_area_change: contraction coefficient cc must be in (0, 1]; got {cc}")
    return ElementSpec(SUDDEN_AREA_CHANGE, [cc], name, eps=eps)


def loss(K, name="loss", ref_port=0, eps=None):
    """A concentrated total-pressure loss ``Pt_in - Pt_out = K * (1/2 rho u^2)``.

    The element conserves mass and drops total pressure by ``K`` dynamic heads,
    with the head's sign tracking the flow direction so reverse flow reverses the
    drop (modeling-guide.md s4).  The static state on each port is reconstructed
    from that port's own area, so the loss may straddle an area change: the result
    is an isentropic area change (Pt-preserving static<->dynamic conversion) with
    the concentrated ``K``-loss superposed.

    Parameters
    ----------
    K : float
        Loss coefficient, referenced to the dynamic head at port ``ref_port``.
    name : str, optional
        Element name.
    ref_port : int, optional
        Which incident port's area and velocity define the reference dynamic head
        ``1/2 rho u^2`` that ``K`` multiplies -- ``0`` (default, the upstream edge
        in the canonical orientation) or ``1``.  Only matters when the ports carry
        different areas; tabulated ``K`` values always name their reference
        section, so set this to match the source.
    eps : float, optional
        Per-element smoothing-width override (see ``ElementSpec.eps``).
    """
    from .ids import LOSS

    rp = int(ref_port)
    if rp not in (0, 1):
        raise ValueError(f"loss: ref_port must be 0 or 1; got {ref_port}")
    return ElementSpec(LOSS, [float(K), float(rp)], name, eps=eps)


def heat_release_flame(Qdot, name="flame", dynamic_source=None):
    """A compact constant-area flame that adds heat power ``Qdot`` [W] to the flow.

    The perfect-gas counterpart of the reacting (equilibrium) flame: it conserves
    mass and total pressure (a low-Mach compact-flame idealization, neglecting the
    ``O(M^2)`` Rayleigh total-pressure loss) while raising the through-flow's total
    enthalpy by ``Q_dot / mdot`` -- so the downstream total temperature rises by
    ``Q_dot / (mdot * cp)``.

    With ``Q_dot`` fixed the linearized jump carries no fluctuating heat release, so
    the mean flame is acoustically passive.  Attach a ``dynamic_source`` (a
    :class:`~fns.elements.dynamic_source.DynamicSource`, e.g. an ``n-tau`` flame
    transfer function) to give it an active unsteady heat release ``S(omega)`` --
    the term that drives thermoacoustic instability (theory.md s12.4).

    Parameters
    ----------
    Qdot : float
        Heat-release rate [W] added across the flame (``> 0`` heats the flow).
    name : str, optional
        Element label.
    dynamic_source : DynamicSource, optional
        Unsteady heat-release response ``S(omega)`` for the perturbation analysis;
        ignored by the mean flow.  Build one with
        :func:`fns.elements.dynamic_source.n_tau_flame` or
        :func:`~fns.elements.dynamic_source.heat_release_response`.
    """
    from .ids import FLAME_HEAT_RELEASE

    return ElementSpec(FLAME_HEAT_RELEASE, [float(Qdot)], name, dynamic_source=dynamic_source)


def equilibrium_flame(name="flame", dynamic_source=None):
    """A compact reacting flame: frozen unburnt inflow -> equilibrium products.

    The reacting (headline) flame and counterpart of :func:`heat_release_flame`.
    It conserves mass, **static pressure** (a low-Mach compact-flame idealization)
    and total enthalpy (adiabatic), and conserves the elemental composition ``Z``.
    "Ignition" is the per-edge closure switch: the approach edge uses the frozen
    (``EQ_FROZEN``) closure and the product edge the equilibrium (``EQ_KERNEL``)
    closure (set via ``build_problem(..., edge_models=...)``), so the temperature
    rise emerges from the equilibrium solve at the shared ``(Z, h_t, p)``.

    The mean flame is acoustically passive (no explicit heat-release source -- the
    chemistry is quasi-steady).  Attach a ``dynamic_source`` to model the unsteady
    heat release ``S(omega)`` that lags the flow (the FTF), which makes the operator
    active and can drive instability.

    Parameters
    ----------
    name : str, optional
        Element label.
    dynamic_source : DynamicSource, optional
        Unsteady heat-release response ``S(omega)`` for the perturbation analysis;
        ignored by the mean flow.  Its mean heat release ``Q_bar`` auto-derives from
        the converged flame (``mdot * cp * dT``) unless given explicitly.
    """
    from .ids import FLAME_EQUILIBRIUM

    return ElementSpec(FLAME_EQUILIBRIUM, [], name, dynamic_source=dynamic_source)


def mass_source(mdot, T, composition, u_inj=0.0, basis="mole", name="source", dynamic_source=None):
    """A 2-port inline mass-injection element (e.g. a fuel injector).

    Injects a stream of mass-flow ``mdot`` [kg/s], total temperature ``T`` [K] and
    the given species ``composition`` into the through-flow, conserving mass,
    momentum and energy *with the appropriate source terms*:

    * **mass**: the outflow exceeds the inflow by ``mdot``;
    * **momentum**: the constant-area balance ``(rho u^2 + p)`` carries the
      injected axial momentum ``mdot * u_inj`` -- ``u_inj = 0`` (default) is normal
      (transverse) injection, which adds mass with no axial momentum;
    * **energy / composition**: the injected total enthalpy and elemental ``Z`` mix
      in mass-weighted with the through-flow (donor override).

    A fuel injector is just this element with a fuel ``composition``.  It performs
    no reaction -- *ignition is the flame element's job*; the source only sets the
    mixture the downstream flame then burns.

    Parameters
    ----------
    mdot : float
        Injected mass-flow [kg/s] (``> 0`` adds mass).
    T : float
        Injected stream total temperature [K] (sets its enthalpy datum).
    composition : dict or array_like
        Injected species mixture, e.g. ``{"CH4": 1.0}`` or ``{"O2": 0.21, "N2": 0.79}``.
    u_inj : float, optional
        Axial injection velocity [m/s] for the momentum source (default 0: normal
        injection).
    basis : {"mole", "mass"}, optional
        Units of ``composition``.
    name : str, optional
        Element label.
    dynamic_source : DynamicSource, optional
        Forward-compatibility provision for the dynamic ``S(omega)`` phase (e.g. a
        fuel-flow that fluctuates with an upstream ``u'``).  Ignored by the mean
        flow.
    """
    return ElementSpec(
        MASS_SOURCE,
        [float(mdot), float(u_inj), float(T)],
        name,
        composition_spec=composition,
        basis=basis,
        dynamic_source=dynamic_source,
    )


def junction(name="junction"):
    return ElementSpec(JUNCTION, [], name)


def splitter(name="splitter"):
    return ElementSpec(SPLITTER, [], name)


def duct(length=0.0, name="duct"):
    """A length-bearing, lossless, constant-area duct.

    The mean face is equal-area continuity (length-independent); ``length`` is
    inert in the steady residual and read only by the acoustic phase stamp
    (theory.md s12.3).  It rides ``fparams[0]`` as ordinary acoustic metadata.
    """
    from .ids import DUCT

    return ElementSpec(DUCT, [float(length)], name, acoustic_id=ACOUSTIC_DUCT)


def _node_label(n: int, el: ElementSpec) -> str:
    """Human-readable identifier for an element, for validation messages."""
    typ = RESIDUAL_NAMES.get(el.residual_id, f"residual {el.residual_id}")
    name = f" {el.name!r}" if el.name else ""
    return f"element {n}{name} ({typ})"


def ensure_unique_names(elements: List[ElementSpec]) -> None:
    """Make element display names unique, in place, by suffixing clashes.

    Names are display/reporting labels only (never touched by the kernels), but
    several surfaces -- the UI export, plot legends, the LaTeX reports -- assume
    they identify an element unambiguously.  Manually built networks routinely
    repeat the factory defaults (two ``duct`` elements both named ``"duct"``), so
    this normalizes them: the first occurrence of a name is kept, and each later
    duplicate is renamed ``<name>-1``, ``<name>-2``, ... (skipping any suffix
    already taken).  Idempotent -- a list whose names are already unique is left
    untouched.

    Parameters
    ----------
    elements : list of ElementSpec
        The network elements, in node order.  Mutated in place.
    """
    seen = set()
    for el in elements:
        name = el.name or ""
        if name not in seen:
            seen.add(name)
            continue
        k = 1
        candidate = f"{name}-{k}"
        while candidate in seen:
            k += 1
            candidate = f"{name}-{k}"
        el.name = candidate
        seen.add(candidate)


def validate_network(elements: List[ElementSpec], conn: Connectivity, area: np.ndarray) -> None:
    """Check structural and area-consistency invariants before compiling.

    Also normalizes element display names to be unique (see
    :func:`ensure_unique_names`) -- duplicates, common with the factory defaults,
    are suffixed in place rather than rejected.

    Raises ``ValueError`` (naming the offending element) on the first violation:

    * every edge area is finite and strictly positive;
    * each element's port count matches its arity -- exactly ``FIXED_NPORTS`` for
      fixed-arity elements, ``>= 2`` for the variable junction/splitter;
    * elements that do not permit an area change (``ALLOWS_AREA_CHANGE`` is
      ``False`` -- the constant-area duct) carry one shared area across all their
      incident edges.  An intended area change at an area-agnostic element (e.g. a
      sudden expansion) must use an ``isentropic_area_change`` or
      ``sudden_area_change`` element.

    Parameters
    ----------
    elements : list of ElementSpec
        The network elements, in node order.
    conn : Connectivity
        The compiled connectivity (per-node incident edges and degrees).
    area : ndarray
        Per-edge cross-sectional area, indexed by global edge id.
    """
    ensure_unique_names(elements)
    area = np.asarray(area, dtype=np.float64)
    if area.size != conn.n_edges:
        raise ValueError(f"area has {area.size} entries but the network has {conn.n_edges} edges")
    bad = np.nonzero(~(np.isfinite(area) & (area > 0.0)))[0]
    if bad.size:
        raise ValueError(f"edge areas must be finite and positive; offending edge id(s): {bad.tolist()}")
    if len(elements) != conn.n_nodes:
        raise ValueError(f"{len(elements)} elements but the connectivity has {conn.n_nodes} nodes")

    for n, el in enumerate(elements):
        rid = el.residual_id
        deg = conn.degree(n)
        label = _node_label(n, el)

        expected = FIXED_NPORTS.get(rid)
        if expected is not None:
            if deg != expected:
                raise ValueError(f"{label} expects {expected} port(s) but is connected to {deg} edge(s)")
        elif rid in (JUNCTION, SPLITTER):
            if deg < 2:
                raise ValueError(f"{label} is a manifold and needs >= 2 ports but is connected to {deg} edge(s)")

        if rid == CHOKED_NOZZLE_OUTLET:
            # the compact choked nozzle is a *contraction* to a sonic throat; the throat
            # area A* must be smaller than the outlet edge area so the approach plane stays
            # subsonic (A_out/A* > 1 has a unique subsonic area-Mach root).  A* >= A_out has
            # no subsonic choked solution (that is a converging-diverging / supersonic case).
            a_out = float(area[conn.incident_edges(n)[0]])
            a_star = float(el.fparams[0])
            if not a_star < a_out:
                raise ValueError(
                    f"{label}: choked-nozzle throat area A* = {a_star:g} m^2 must be smaller than "
                    f"the outlet area {a_out:g} m^2 (a contraction). A* >= A_out has no subsonic "
                    "choked approach; a converging-diverging (supersonic) nozzle is out of v1 scope."
                )

        if not ALLOWS_AREA_CHANGE.get(rid, True) and deg >= 2:
            inc = conn.incident_edges(n)
            a0 = float(area[inc[0]])
            for e in inc[1:]:
                ae = float(area[e])
                if abs(ae - a0) > _AREA_RTOL * max(abs(a0), abs(ae)):
                    raise ValueError(
                        f"{label} does not permit an area change but its ports carry different "
                        f"areas ({a0:g} vs {ae:g} m^2); model the area change with an "
                        f"isentropic_area_change or sudden_area_change element"
                    )

    _check_pressure_reference(elements)


# Boundaries that fix an *absolute pressure* (total_pressure_inlet, pressure_outlet) or tie
# the pressure level via a flow<->pressure relation (choked_nozzle_outlet).  At least one is
# needed or the mean-flow pressure level is a free gauge.
_PRESSURE_REFERENCE_RIDS = (PT_INLET, P_OUTLET, CHOKED_NOZZLE_OUTLET)


def _check_pressure_reference(elements: List[ElementSpec]) -> None:
    """Reject a boundary set with no absolute-pressure reference (a singular gauge).

    The steady residual fixes pressure only through *differences* (momentum, area-change
    and loss rows) plus the absolute pin a pressure boundary supplies.  If **every**
    boundary merely prescribes a mass flow (``mass_flow_inlet`` / ``mass_flow_outlet`` /
    ``wall``), the pressure level is undetermined: adding a constant to every pressure
    leaves the residual unchanged to leading order, so the Jacobian is singular and the
    solve cannot converge.  A ``total_pressure_inlet`` or ``pressure_outlet`` pins the
    level directly; a ``choked_nozzle_outlet`` pins it via its critical-mass-flux
    relation (the interior stagnation pressure is fixed once the flow is).
    """
    if any(el.residual_id in _PRESSURE_REFERENCE_RIDS for el in elements):
        return
    raise ValueError(
        "ill-posed boundary conditions: the network has no absolute-pressure reference. "
        "Every boundary fixes a mass flow (mass_flow_inlet / mass_flow_outlet / wall), so the "
        "pressure level is a free gauge and the steady solve is singular. Add a pressure_outlet "
        "or total_pressure_inlet (an absolute-pressure pin), or a choked_nozzle_outlet "
        "(a flow<->pressure relation), to set the level."
    )


def _row_kinds(rid: int, deg: int, mdot_ref, p_ref):
    """Residual-row scale magnitudes for one element (derived from its kind tags)."""
    scale = {KIND_MASS: mdot_ref, KIND_PRESSURE: p_ref}
    return [scale[tag] for tag in row_kind_tags(rid, deg)]


def _onehot(k: int, n: int):
    """Mixture-fraction unit vector: all mass from stream ``k`` (``[]`` if ``n==0``)."""
    xi = [0.0] * n
    if 0 <= k < n:
        xi[k] = 1.0
    return xi


def _boundary_scalars(thermo: ThermoConfig, el: ElementSpec, Tt: float, n_elem: int, label: str, stream: int):
    """Resolve a boundary's advected-scalar params ``[h_t, xi_0, ..., xi_{n_elem-1}]``.

    Converts the element's total temperature to the absolute total-enthalpy datum
    (D-1) and tags the stream it introduces with the mixture-fraction unit vector
    ``xi = e_stream``.  For a perfect gas ``h_t = cp*Tt`` and the composition (if
    any) is the raw passive-scalar values; for the equilibrium model the
    composition is a named species mixture whose own enthalpy at ``Tt`` is used.
    """
    if thermo.model_id == PERFECT_GAS:
        h_t = float(thermo.tf[0]) * Tt  # cp * Tt
        if n_elem == 0:
            return [h_t]
        comp = el.composition_spec
        if comp is None:
            return [h_t] + [0.0] * n_elem
        zvals = [float(c) for c in comp]
        if len(zvals) != n_elem:
            raise ValueError(f"{label} carries {len(zvals)} scalar(s) but the model has {n_elem}")
        return [h_t] + zvals

    # equilibrium / reacting backend: an explicit species composition is required
    # for any stream that introduces mass (inlets, mass sources); an outlet may
    # omit it (its backflow scalars are used only on ingestion).
    comp = el.composition_spec
    if comp is None:
        if el.residual_id in (MASS_FLOW_INLET, PT_INLET):
            raise ValueError(
                f"{label}: the equilibrium model requires an explicit species composition "
                f"(e.g. composition={{'O2': 0.21, 'N2': 0.79}})"
            )
        return [0.0] + [0.0] * n_elem  # inert backflow placeholder
    Y = species_mass_fractions(thermo.library, comp, el.basis)
    h_t = enthalpy_mass(thermo.library, Y, Tt)
    return [h_t] + _onehot(stream, n_elem)


def _mass_source_params(thermo: ThermoConfig, el: ElementSpec, n_elem: int, label: str, stream: int):
    """Resolve a mass source's params ``[mdot_src, u_inj, h_t_src, xi_src_0, ...]``.

    The injected total enthalpy carries the stream's enthalpy at ``T_src`` plus the
    injection kinetic energy ``0.5 u_inj^2`` (D-1 datum); the injected composition is
    the mixture-fraction unit vector of its feed stream (kernel donor index
    ``pb+2+s``).
    """
    mdot_src = float(el.fparams[0])
    u_inj = float(el.fparams[1])
    T_src = float(el.fparams[2])
    ke = 0.5 * u_inj * u_inj
    if thermo.model_id == PERFECT_GAS:
        h_t_src = float(thermo.tf[0]) * T_src + ke
        if n_elem == 0:
            return [mdot_src, u_inj, h_t_src]
        comp = el.composition_spec
        zvals = [float(c) for c in comp] if comp is not None else [0.0] * n_elem
        if len(zvals) != n_elem:
            raise ValueError(f"{label} carries {len(zvals)} scalar(s) but the model has {n_elem}")
        return [mdot_src, u_inj, h_t_src] + zvals

    comp = el.composition_spec
    if comp is None:
        raise ValueError(
            f"{label}: a mass source must specify its injected species composition "
            f"(e.g. composition={{'CH4': 1.0}})"
        )
    Y = species_mass_fractions(thermo.library, comp, el.basis)
    h_t_src = enthalpy_mass(thermo.library, Y, T_src) + ke
    return [mdot_src, u_inj, h_t_src] + _onehot(stream, n_elem)


# elements that introduce a feed stream (a distinct injected composition)
_STREAM_INTRODUCING = (MASS_FLOW_INLET, PT_INLET, P_OUTLET, MASS_SOURCE)


def finalize_thermo(thermo: ThermoConfig, elements: List[ElementSpec]):
    """Discover the network's feed streams and pack the equilibrium bundle.

    For the reacting (``EQ_KERNEL``) model the **streams are the distinct injected
    compositions** of the network's inlets, mass sources and (backflow-bearing)
    outlets.  This scans them in node order, auto-merges identical compositions, and
    packs the per-stream forward-blend maps -- so the user only ever names species at
    the elements that introduce them, and the transported scalar count equals the
    number of distinct feeds (never the chemical-element count, never the product
    species).  The deferred ``equilibrium(library)`` config is finalized here.

    Returns
    -------
    thermo : ThermoConfig
        The finalized config (unchanged for a perfect gas / passive-scalar model).
    node_stream : dict or None
        ``node -> stream index`` for every stream-introducing node (``-1`` if that
        element carries no composition, e.g. an inert-backflow outlet); ``None`` for
        a non-reacting model.
    """
    if thermo.model_id != EQ_KERNEL or thermo.library is None:
        return thermo, None

    from ..thermo.equilibrium import pack_equilibrium

    comps = []
    comp_nodes = []
    for n, el in enumerate(elements):
        if el.residual_id in _STREAM_INTRODUCING:
            comps.append((el.composition_spec, el.basis))
            comp_nodes.append(n)
    stream_Y, assignment = build_streams(thermo.library, comps)
    node_stream = {comp_nodes[i]: assignment[i] for i in range(len(comp_nodes))}

    # label each stream by the first element that introduces it (for reporting)
    K = stream_Y.shape[0]
    labels = [f"stream{k}" for k in range(K)]
    for i, k in enumerate(assignment):
        if k >= 0 and labels[k] == f"stream{k}":
            nm = elements[comp_nodes[i]].name
            if nm:
                labels[k] = nm

    tf, ti = pack_equilibrium(thermo.library, stream_Y, thermo.t_init, thermo.t_init_frozen)
    finalized = ThermoConfig(
        model_id=EQ_KERNEL,
        tf=tf,
        ti=ti,
        element_names=labels,
        species_names=thermo.species_names,
        library=thermo.library,
        t_init=thermo.t_init,
        t_init_frozen=thermo.t_init_frozen,
    )
    return finalized, node_stream


def build_problem(
    thermo: ThermoConfig,
    elements: List[ElementSpec],
    edges: List[Tuple[int, int, float]],
    mdot_ref: float,
    p_ref: float,
    h_ref: float,
    edge_models=None,
) -> CompiledProblem:
    """Assemble a CompiledProblem from elements and directed (tail, head, area) edges.

    Ports are auto-assigned in attachment order.  Use
    ``build_problem_from_connectivity`` to supply explicit ports (e.g. a UI
    export where the port ordinals carry meaning).  ``edge_models`` optionally
    overrides the per-edge thermo model id (default: the config's model on every
    edge) -- e.g. frozen upstream, equilibrium downstream of a flame.
    """
    n_nodes = len(elements)
    directed = [(t, h) for (t, h, _a) in edges]
    area = np.array([a for (_t, _h, a) in edges], dtype=np.float64)
    conn = connectivity_from_directed_edges(n_nodes, directed)
    return build_problem_from_connectivity(
        thermo, elements, conn, area, mdot_ref, p_ref, h_ref, edge_models=edge_models
    )


def build_problem_from_connectivity(
    thermo: ThermoConfig,
    elements: List[ElementSpec],
    conn: Connectivity,
    area: np.ndarray,
    mdot_ref: float,
    p_ref: float,
    h_ref: float,
    edge_models=None,
) -> CompiledProblem:
    """Assemble a CompiledProblem from elements and a prebuilt Connectivity.

    The connectivity carries explicit per-edge ports (``tail_port``/
    ``head_port``), so port-ordering conventions are preserved exactly.
    """
    n_nodes = len(elements)
    area = np.ascontiguousarray(area, dtype=np.float64)
    validate_network(elements, conn, area)

    # discover the feed streams from the network and finalize the (reacting) thermo
    # bundle: the transported mixture fractions are the distinct injected compositions.
    thermo, node_stream = finalize_thermo(thermo, elements)

    degrees = [conn.degree(n) for n in range(n_nodes)]
    node_rid = np.array([el.residual_id for el in elements], dtype=np.int64)
    node_acoustic_id = np.array([el.acoustic_id for el in elements], dtype=np.int64)

    # pack node float params in node order.  A boundary element that prescribes
    # advected scalars carries [base, h_t, Z_el...]: slot 0 is the prescribed
    # mdot/pt/p, slot 1 the absolute total enthalpy datum (converted from Tt), and
    # the remaining n_elem slots the feed/backflow elemental composition -- so the
    # donor kernel indexes npar_f[pb + 1 + s] for advected scalar s (s = 0 is h_t).
    n_elem = thermo.n_elem
    # boundaries that prescribe an advected-scalar (h_t + composition) donor on ingestion.
    # The mass-flow / choked-nozzle outlets are outflow-only (no backflow), so they carry no
    # such donor -- their edge inherits the interior scalars (scalar-transparent, see node_donor).
    boundary_rids = (MASS_FLOW_INLET, PT_INLET, P_OUTLET)
    npar_f = []
    npar_fptr = np.zeros(n_nodes + 1, dtype=np.int64)
    for n, el in enumerate(elements):
        fp = list(el.fparams)
        k = -1 if node_stream is None else node_stream.get(n, -1)
        if el.residual_id in boundary_rids and len(fp) >= 2:
            base, Tt = float(fp[0]), float(fp[1])
            fp = [base] + _boundary_scalars(thermo, el, Tt, n_elem, _node_label(n, el), k)
        elif el.residual_id == MASS_SOURCE:
            fp = _mass_source_params(thermo, el, n_elem, _node_label(n, el), k)
        npar_f.extend(fp)
        npar_fptr[n + 1] = npar_fptr[n] + len(fp)
    npar_f = np.array(npar_f, dtype=np.float64)

    # per-element smoothing-eps override (< 0 -> follow the global solve-time eps)
    node_eps = np.array([el.eps if el.eps is not None else -1.0 for el in elements], dtype=np.float64)

    # per-node perturbation BC (Python objects; read only by the perturbation layer)
    node_bc = tuple(getattr(el, "perturbation_bc", None) for el in elements)

    # per-node human-readable name (label); for plotting / reporting only
    node_names = tuple(getattr(el, "name", "") or "" for el in elements)

    # per-node dynamic-source descriptor (S(omega) provision; mean flow ignores it)
    node_dynamic_source = tuple(getattr(el, "dynamic_source", None) for el in elements)

    n_solve = 3 + thermo.n_elem
    pat = build_jacobian_pattern(conn, degrees, n_solve=n_solve)

    # residual scales: node rows, then the advected-scalar transport rows
    # (h_t for every edge, then each composition scalar for every edge).
    # Composition scalars are elemental mass fractions, O(1), so scale = 1.
    z_scale = 1.0
    res_scale = []
    for n, el in enumerate(elements):
        res_scale.extend(_row_kinds(el.residual_id, degrees[n], mdot_ref, p_ref))
    res_scale.extend([h_ref] * conn.n_edges)
    for _ in range(thermo.n_elem):
        res_scale.extend([z_scale] * conn.n_edges)
    res_scale = np.array(res_scale, dtype=np.float64)

    var_scale = np.array([mdot_ref, p_ref, h_ref] + [z_scale] * thermo.n_elem, dtype=np.float64)

    # per-edge thermo model (default: the config's model on every edge)
    if edge_models is None:
        edge_model = np.full(conn.n_edges, thermo.model_id, dtype=np.int64)
    else:
        edge_model = np.ascontiguousarray(edge_models, dtype=np.int64)
        if edge_model.shape[0] != conn.n_edges:
            raise ValueError(f"edge_models has {edge_model.shape[0]} entries but the network has {conn.n_edges} edges")

    # an unburnt (EQ_FROZEN) edge reconstructs its species from the feed streams; at
    # least one stream must exist (an inlet / mass source must inject a composition).
    if thermo.model_id != PERFECT_GAS and np.any(edge_model == EQ_FROZEN):
        n_streams = int(thermo.ti[5]) if thermo.ti.shape[0] > 5 else 0
        if n_streams == 0:
            raise ValueError(
                "the network has EQ_FROZEN (unburnt) edges but no feed streams were "
                "found; an inlet or mass source must inject an explicit species "
                "composition for the frozen closure to reconstruct from"
            )

    return CompiledProblem(
        model_id=thermo.model_id,
        tf=thermo.tf,
        ti=thermo.ti,
        n_elem=thermo.n_elem,
        n_solve=3 + thermo.n_elem,
        n_nodes=n_nodes,
        n_edges=conn.n_edges,
        n_eq=pat.n_eq,
        area=area,
        row_ptr=conn.row_ptr,
        col_edge=conn.col_edge,
        orient=conn.orient.astype(np.int64),
        tail_node=conn.tail_node,
        head_node=conn.head_node,
        node_rid=node_rid,
        node_acoustic_id=node_acoustic_id,
        npar_f=npar_f,
        npar_fptr=npar_fptr,
        node_row_ptr=pat.node_row_ptr,
        transport_row0=pat.transport_row0,
        indptr=pat.indptr,
        indices=pat.indices,
        var_scale=var_scale,
        res_scale=res_scale,
        edge_model=edge_model,
        node_eps=node_eps,
        node_bc=node_bc,
        node_names=node_names,
        node_dynamic_source=node_dynamic_source,
        scalar_names=tuple(thermo.element_names),
    )
