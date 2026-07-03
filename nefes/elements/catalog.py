"""Element catalog and CompiledProblem builder (Python, parse-time).

An ``ElementSpec`` names an element's residual id and its ordered float
parameters (the order the @njit kernels expect).  ``build_problem`` turns a list
of element specs plus directed edges into the immutable CompiledProblem.
"""
# CA: Why would we keep CompiledProblem builder here? What does it have to do with the elements catalog?
# We should probably move it to the shell module. Same goes for connectivity related routines, this is 
# part of the core functonality as well and probably belongs to the shell module.

import inspect
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np

from ..chem.composition import build_streams, enthalpy_mass, species_mass_fractions
from ..graph.connectivity import connectivity_from_directed_edges, build_jacobian_pattern, Connectivity
from ..graph.problem import CompiledProblem
from .composite import CompositeElementSpec, expand_composites
from ..thermo.api import EQ_FROZEN, EQ_KERNEL, EQ_MARKER, PERFECT_GAS
from ..thermo.configure import ThermoConfig
from .ids import (
    MASS_FLOW_INLET,
    PT_INLET,
    P_OUTLET,
    MASS_FLOW_OUTLET,
    CHOKED_NOZZLE_OUTLET,
    WALL,
    CAVITY,
    JUNCTION,
    SPLITTER,
    FORCED_SPLITTER,
    FLAME_EQUILIBRIUM,
    MASS_SOURCE,
    ACOUSTIC_DEFAULT,
    ACOUSTIC_DUCT,
    ACOUSTIC_VOLUME,
    FIXED_NPORTS,
    ALLOWS_AREA_CHANGE,
    RESIDUAL_NAMES,
    KIND_MASS,
    KIND_PRESSURE,
    row_kind_tags,
)

# Relative tolerance for the equal-area check on constant-area elements.
_AREA_RTOL = 1e-9


def _storage_block(name, l_up, l_down, end_correction):
    """Validate + pack the optional storage lengths ``[l_up, l_down, end_correction]``.

    The per-port half-lengths ``l_up``/``l_down`` (port 0 / port 1) set the element's
    compliance (each ``l_i`` times that port's edge area is a stored volume) and feed the
    series inertance ``L_eff = l_up + l_down + end_correction``; ``end_correction`` is the
    added-mass length the geometric extent omits (it contributes to the inertance only).
    All are lengths in metres, default zero (no storage -> the element is the lengthless
    jump it was before).  Read by :func:`nefes.perturbation.operator.stamps._inline_storage`.
    """
    lu, ld, ec = float(l_up), float(l_down), float(end_correction)
    for label, v in (("l_up", lu), ("l_down", ld), ("end_correction", ec)):
        if v < 0.0:
            raise ValueError(f"{name}: {label} must be non-negative (a length in metres); got {v}")
    return [lu, ld, ec]


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
    # optional dynamic-source descriptor (nefes.elements.dynamic_source.DynamicSource);
    # a forward-compatibility provision for the S(omega) perturbation phase -- the
    # mean flow ignores it.
    dynamic_source: object = None
    # optional transfer-matrix descriptor for a TRANSFER_MATRIX element: a
    # nefes.perturbation.matrix.TransferMatrix (or an UnknownTransferMatrix marker for
    # identification).  Read only by the perturbation layer; the mean flow (an
    # isentropic area change) ignores it.
    transfer_matrix: object = None
    # burnt-marker value injected at an inflow/source boundary (the last advected
    # scalar of the marker-gated reacting closure).  ``0.0`` is fresh reactant
    # (default); ``1.0`` is fully burnt gas (e.g. exhaust-gas recirculation as a feed),
    # which forces the equilibrium closure on the downstream edge.  Only meaningful on
    # a marker-gated network (an equilibrium-flame reacting net with no explicit
    # per-edge closure); a non-zero value elsewhere is rejected at build time.
    marker: float = 0.0


def _validate_marker(marker, name):
    """Validate a boundary burnt-marker value (in ``[0, 1]``); return it as a float.

    ``0.0`` is fresh reactant, ``1.0`` fully-burnt; the marker gate is calibrated on
    ``[0, 1]`` so values outside it have no calibrated meaning.
    """
    m = float(marker)
    if not 0.0 <= m <= 1.0:
        raise ValueError(f"{name}: marker must be in [0, 1] (0 fresh, 1 burnt); got {marker}")
    return m


def mass_flow_inlet(mdot, Tt, composition=None, basis="mole", name="inlet", perturbation_bc=None, marker=0.0):
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

    ``marker`` (default ``0.0``, fresh reactant) is the injected burnt-marker value of
    the marker-gated reacting closure; set ``1.0`` to feed already-burnt gas (e.g.
    exhaust-gas recirculation), forcing the equilibrium closure downstream.  It is only
    accepted on a marker-gated network (equilibrium-flame reacting, no explicit per-edge
    closure); a non-zero value elsewhere is rejected at build time.
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
        marker=_validate_marker(marker, name),
    )


def total_pressure_inlet(pt, Tt, composition=None, basis="mole", name="pt-inlet", perturbation_bc=None, marker=0.0):
    """Prescribed total-pressure inlet feeding a stream of the given ``composition``.

    ``marker`` injects the burnt-marker value (``0.0`` fresh, default; ``1.0`` burnt --
    e.g. recirculated exhaust gas as a feed); see :func:`mass_flow_inlet`.
    """
    return ElementSpec(
        PT_INLET,
        [float(pt), float(Tt)],
        name,
        perturbation_bc=perturbation_bc,
        composition_spec=composition,
        basis=basis,
        marker=_validate_marker(marker, name),
    )


def pressure_outlet(
    p, Tt_backflow=300.0, composition=None, basis="mole", name="outlet", perturbation_bc=None, marker=0.0
):
    """Static-pressure outlet; ``composition`` is the backflow stream (on ingestion).

    ``marker`` is the burnt-marker value of the backflow stream drawn in on ingestion
    (``0.0`` fresh, default; ``1.0`` burnt); see :func:`mass_flow_inlet`.
    """
    return ElementSpec(
        P_OUTLET,
        [float(p), float(Tt_backflow)],
        name,
        perturbation_bc=perturbation_bc,
        composition_spec=composition,
        basis=basis,
        marker=_validate_marker(marker, name),
    )


def mass_flow_outlet(mdot, name="outlet", perturbation_bc=None):
    """Prescribed-mass-flow outlet: the outflow rate is pinned to ``mdot``.

    The static pressure floats (it is whatever the interior produces); useful for a
    metered exhaust or as the mean-flow partner of a downstream choked throat whose
    critical mass flow is known.  Left at ``perturbation_bc=None`` the acoustic
    termination is the *inherited* linearization of this row, ``mdot' = 0`` -- the
    constant-mass-flow acoustic boundary condition (also available standalone as
    :meth:`~nefes.perturbation.operator.boundary_bc.PerturbationBC.constant_mass_flow`).

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
    from ..perturbation.operator.boundary_bc import PerturbationBC

    bc = perturbation_bc if perturbation_bc is not None else PerturbationBC.hard_wall()
    return ElementSpec(WALL, [], name, perturbation_bc=bc)


def cavity(volume, name="cavity"):
    """A lumped finite-volume cavity: a wall to the mean flow, a compliance to acoustics.

    Mean-flow-wise the cavity is impermeable -- its single port carries ``mdot = 0``,
    exactly like a :func:`wall`, so the leg behind it is stagnant and it needs no
    interior mean unknowns. Acoustically its enclosed gas stores energy: a finite
    volume ``V`` compresses isentropically, giving the lumped compliance ``C = V /
    (rho c^2)`` that populates the storage block ``M`` (the ``i*omega*M`` face of the
    operator ``A = J_alg + i*omega*M + P + S``). Paired with a neck inertance (a short
    :func:`duct`) off a :func:`junction`, it forms a Helmholtz resonator with
    ``omega_0 = c * sqrt(A_neck / (V * l_eff))`` (see :func:`helmholtz_resonator`).

    The cavity is *not* a boundary terminal: its acoustic response is the compliance
    itself (a reflection set by the storage), so the perturbation layer leaves its
    inherited ``mdot' = 0`` row in place and lets ``M`` add the storage onto it -- it is
    never neutralized or stamped with a reflection coefficient.

    The mean state of the cavity gas is slaved to its face (the local ``p``, ``T``,
    composition), so ``c`` is the local sound speed. An independently-stated cavity
    (a cold purge or a different gas) is a later provision.

    Parameters
    ----------
    volume : float
        Enclosed cavity volume ``V`` [m^3], strictly positive. Sets the compliance
        ``C = V / (rho c^2)`` and hence the resonance frequency.
    name : str, optional
        Element label.

    Returns
    -------
    ElementSpec
    """
    V = float(volume)
    if not V > 0.0:
        raise ValueError(f"cavity volume must be strictly positive; got {volume}")
    return ElementSpec(CAVITY, [V], name, acoustic_id=ACOUSTIC_VOLUME)


def isentropic_area_change(name="iac", l_up=0.0, l_down=0.0, end_correction=0.0):
    """A smooth (lossless) contraction or diffuser; optionally length-bearing.

    By default a lengthless jump.  A real diffuser/nozzle has axial extent, so the
    optional ``l_up``/``l_down`` (the passage half-lengths on the port-0 / port-1 sides)
    give it acoustic **compliance** (each side stores ``l_i * A_i`` of gas) and
    **inertance** (series effective length ``l_up + l_down + end_correction``,
    referenced to the throat).  These populate the storage block ``M`` and are inert in
    the mean flow.  See ``scratch/inertance-end-correction-theory.md``.

    Parameters
    ----------
    name : str, optional
        Element label.
    l_up, l_down : float, optional
        Passage half-length [m] on the port-0 (upstream) / port-1 (downstream) side.
    end_correction : float, optional
        Added-mass length [m] added to the inertance only (the entrained near-field the
        geometric length omits).
    """
    from .ids import ISEN_AREA_CHANGE

    return ElementSpec(ISEN_AREA_CHANGE, _storage_block("isentropic_area_change", l_up, l_down, end_correction), name)


def transfer_matrix_element(tm=None, name="tm"):
    """A 2-port whose acoustics are a **user-supplied transfer matrix**.

    In the mean flow this element is an :func:`isentropic_area_change` -- it conserves
    mass and energy, is isentropic, and permits an area change across it -- so it seats a
    well-defined mean state on both faces.  In the perturbation network it does **not**
    inherit the linearized area-change jump: its acoustic rows are overwritten with the
    relation ``w_down = TM(omega) . w_up`` carried by ``tm`` (theory.md s12.7), letting a
    measured / prescribed 2-port response stand in for an element that has no closed-form
    model.

    Parameters
    ----------
    tm : TransferMatrix or UnknownTransferMatrix, optional
        The frequency-domain 2-port descriptor
        (:class:`nefes.perturbation.matrix.TransferMatrix`) stamped in the perturbation
        layer.  Pass an :class:`~nefes.perturbation.identify.UnknownTransferMatrix` marker
        to leave it to be identified from a measured network response.  ``None`` (default)
        leaves the element acoustically an isentropic area change until a descriptor is
        attached (``spec.transfer_matrix = ...``).
    name : str, optional
        Element label.

    Returns
    -------
    ElementSpec
    """
    from .ids import TRANSFER_MATRIX

    # No mean-flow parameters and no lumped storage: the element's entire acoustic
    # identity is the transfer matrix stamped in the perturbation layer.
    return ElementSpec(TRANSFER_MATRIX, [], name, transfer_matrix=tm)


def sudden_area_change(name="sac", cc=1.0, eps=None, l_up=0.0, l_down=0.0, end_correction=0.0):
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
    l_up, l_down, end_correction : float, optional
        Optional storage lengths [m] (default 0).  A sudden change is geometrically thin
        (``l_up = l_down = 0``); supply ``end_correction`` for the entrained-mass inertance
        the step still carries.  See :func:`isentropic_area_change` and
        ``scratch/inertance-end-correction-theory.md``.
    """
    from .ids import SUDDEN_AREA_CHANGE

    cc = float(cc)
    if not 0.0 < cc <= 1.0:
        raise ValueError(f"sudden_area_change: contraction coefficient cc must be in (0, 1]; got {cc}")
    block = _storage_block("sudden_area_change", l_up, l_down, end_correction)
    return ElementSpec(SUDDEN_AREA_CHANGE, [cc] + block, name, eps=eps)


def loss(K, name="loss", ref_port=0, eps=None, l_up=0.0, l_down=0.0, end_correction=0.0):
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
    l_up, l_down, end_correction : float, optional
        Optional storage lengths [m] (default 0): an orifice's thickness / backing length
        (compliance + inertance) and its end correction (inertance only).  See
        :func:`isentropic_area_change` and ``scratch/inertance-end-correction-theory.md``.
        With these the loss becomes an orifice impedance ``Z = R(u) + i*omega*L_eff/A``
        (the steady resistance from ``J_alg``, the reactance from ``M``).
    """
    from .ids import LOSS

    rp = int(ref_port)
    if rp not in (0, 1):
        raise ValueError(f"loss: ref_port must be 0 or 1; got {ref_port}")
    block = _storage_block("loss", l_up, l_down, end_correction)
    return ElementSpec(LOSS, [float(K), float(rp)] + block, name, eps=eps)


def linear_resistance(R, name="resistance", l_up=0.0, l_down=0.0, end_correction=0.0):
    """A linear flow resistance ``Pt_in - Pt_out = R * mdot`` (a quiescent acoustic resistance).

    Drops total pressure in **linear** proportion to the through-flow mass rate, with the drop
    reversing with the flow direction.  Because it is linear in the flow (not the quadratic
    dynamic head that :func:`loss` uses), it survives the linearization with a non-zero
    coefficient even at **zero mean flow** -- the acoustic resistance of a screen, perforate or
    damper in an otherwise quiescent network, where the velocity-squared loss would vanish.  In a
    flowing network it adds an ordinary linear (Darcy-like) total-pressure drop on top of the mean
    state.  Mass is conserved and scalars (enthalpy, composition) pass through unchanged.

    Parameters
    ----------
    R : float
        Resistance coefficient ``>= 0`` in Pa per (kg/s): the total-pressure drop per unit mass
        flow.  As an acoustic resistance it sets the linear damping ``Pt' = R * mdot'``.
    name : str, optional
        Element label.
    l_up, l_down, end_correction : float, optional
        Optional storage lengths [m] (default 0): paired with ``R`` they make a screen /
        perforate / damper carrying both resistance and reactance -- the quiescent orifice
        impedance ``Z = R + i*omega*L_eff/A`` (the cleanest inertance test, since it is
        linear and survives at zero mean flow).  See :func:`isentropic_area_change`.

    Returns
    -------
    ElementSpec
    """
    from .ids import LINEAR_RESISTANCE

    R = float(R)
    if R < 0.0:
        raise ValueError(f"linear_resistance: R must be non-negative (a passive resistance); got {R}")
    block = _storage_block("linear_resistance", l_up, l_down, end_correction)
    return ElementSpec(LINEAR_RESISTANCE, [R] + block, name)


def heat_release_flame(Qdot, name="flame", dynamic_source=None):
    """A compact constant-area flame that adds heat power ``Qdot`` [W] to the flow.

    The perfect-gas counterpart of the reacting (equilibrium) flame: it conserves
    mass and total pressure (a low-Mach compact-flame idealization, neglecting the
    ``O(M^2)`` Rayleigh total-pressure loss) while raising the through-flow's total
    enthalpy by ``Q_dot / mdot`` -- so the downstream total temperature rises by
    ``Q_dot / (mdot * cp)``.

    With ``Q_dot`` fixed the linearized jump carries no fluctuating heat release, so
    the mean flame is acoustically passive.  Attach a ``dynamic_source`` (a
    :class:`~nefes.elements.dynamic_source.DynamicSource`, e.g. an ``n-tau`` flame
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
        :func:`nefes.elements.dynamic_source.n_tau_flame` or
        :func:`~nefes.elements.dynamic_source.heat_release_response`.
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


def mass_source(mdot, T, composition, u_inj=0.0, basis="mole", name="source", dynamic_source=None, marker=0.0):
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
    marker : float, optional
        Burnt-marker value of the injected stream (``0.0`` fresh, default; ``1.0``
        burnt); see :func:`mass_flow_inlet`.  An injector normally feeds fresh fuel,
        so the default is appropriate; set it only to inject already-burnt gas.
    """
    return ElementSpec(
        MASS_SOURCE,
        [float(mdot), float(u_inj), float(T)],
        name,
        composition_spec=composition,
        basis=basis,
        dynamic_source=dynamic_source,
        marker=_validate_marker(marker, name),
    )


def _manifold_block(name, volume):
    """Validate + pack a manifold's chamber ``volume`` into its float params.

    Packs ``fparams = [volume]``.  ``volume`` is the plenum compliance (default 0 -> no
    storage, a lengthless common-pressure node).  Read by
    :func:`nefes.perturbation.operator.stamps._manifold_storage`, which turns a non-zero
    volume into the chamber compliance ``C = V / (rho c^2)``.
    """
    V = float(volume)
    if V < 0.0:
        raise ValueError(f"{name}: volume must be non-negative (a chamber volume in m^3); got {volume}")
    return [V]


def junction(name="junction", volume=0.0):
    """A static-pressure manifold (header node) tying all ports to a common pressure.

    Optionally a **plenum**: a non-zero chamber ``volume`` [m^3] gives it the acoustic
    compliance ``C = V / (rho c^2)`` (populating the storage block ``M`` on the common
    pressure), so a header with a real internal volume resonates -- a junction with a
    volume is a cavity with through-flow.  The compliance is inert in the mean flow.

    A manifold branch's inertance (the "neck" of a Helmholtz element) is not a property of
    the junction itself -- it belongs to the passage attached to that branch.  Model it as
    an explicit neck :func:`duct` on the branch, which is exactly what
    :func:`helmholtz_resonator` assembles.

    Parameters
    ----------
    name : str, optional
        Display name.
    volume : float, optional
        Chamber volume [m^3] (default 0 -> no compliance).

    Returns
    -------
    ElementSpec
    """
    return ElementSpec(JUNCTION, _manifold_block("junction", volume), name)


def splitter(name="splitter", volume=0.0):
    """A lossless (total-pressure) manifold; optionally a finite-volume plenum.

    As :func:`junction`, a non-zero ``volume`` adds the chamber compliance to ``M`` (inert in
    the mean flow); a branch neck is modeled as an explicit neck :func:`duct`, not here.
    """
    return ElementSpec(SPLITTER, _manifold_block("splitter", volume), name)


def forced_splitter(fractions, name="splitter"):
    """A flow divider: one inflow split into N outflows at prescribed mass fractions.

    This is a :func:`splitter` whose branch flows are *controlled* rather than set
    by downstream resistance.  Exactly one edge is the inflow (port 0) and the rest
    are outflows; ``fractions[k]`` pins outflow port ``k + 1`` to that fraction of
    the port-0 inflow rate (``mdot_out = beta_k * mdot_in``).  With ``N`` outflows
    you give ``N - 1`` fractions: the last (highest-port) outflow carries the
    remainder ``1 - sum(fractions)`` and keeps total-pressure continuity with the
    inflow (the one branch whose pressure does not float).

    Reverse flow is not modelled -- the inflow direction is taken as fixed, so the
    constraint is linear in the flow state (hence complex-step-exact and inherited
    unchanged by the perturbation network).  Replacing the splitter's pressure
    couplings on the controlled branches means those branch total pressures float;
    the downstream elements must absorb the resulting pressure jump (a
    control-valve / ideal flow-divider idealization).

    Parameters
    ----------
    fractions : sequence of float
        The ``N - 1`` controlled outflow fractions ``beta_k``, each in ``(0, 1)``,
        with ``sum(fractions) < 1`` so the remainder branch carries positive flow.
        ``fractions[k]`` applies to outflow port ``k + 1`` (attachment order).
    name : str, optional
        Display name.

    Returns
    -------
    ElementSpec

    Notes
    -----
    Port order follows attachment order, so wire the **inflow edge first** (port 0)
    and the **remainder outflow last** (highest port).  The build-time check
    requires the wired port count to be ``len(fractions) + 2`` (1 inflow + the
    controlled outflows + the remainder).  Not representable in the UI export format.

    Because the controlled branches float in pressure, the manifold has weaker
    pressure coupling than a plain splitter and is harder to converge as the inflow
    nears choke; with the default continuation it is robust to roughly inflow ``M ~ 0.6``.
    """
    betas = [float(b) for b in fractions]
    if len(betas) < 1:
        raise ValueError(
            "forced_splitter needs at least one split fraction (>= 2 outflow ports); got none -- "
            "use splitter() for an uncontrolled manifold"
        )
    for b in betas:
        if not (0.0 < b < 1.0):
            raise ValueError(f"forced_splitter fractions must each lie in (0, 1); got {betas}")
    total = math.fsum(betas)
    if not total < 1.0:
        raise ValueError(
            f"forced_splitter fractions must sum to < 1 so the remainder branch carries positive "
            f"flow (the remainder = 1 - sum is the last outflow's share); got sum = {total:g}"
        )
    return ElementSpec(FORCED_SPLITTER, betas, name)


def duct(length=0.0, name="duct"):
    """A length-bearing, lossless, constant-area duct.

    The mean face is equal-area continuity (length-independent); ``length`` is
    inert in the steady residual and read only by the acoustic phase stamp
    (theory.md s12.3).  It rides ``fparams[0]`` as ordinary acoustic metadata.
    """
    from .ids import DUCT

    return ElementSpec(DUCT, [float(length)], name, acoustic_id=ACOUSTIC_DUCT)


def pipe(length, diameter, friction_factor, name="pipe") -> ElementSpec:
    """A length-bearing pipe: Darcy-Weisbach wall friction + the duct acoustic phase.

    The ``DUCT (+) LOSS`` unification (Greyvenstein-Laurie): one element that drops total
    pressure ``pt0 - pt1 = K * (1/2 rho u^2)`` with the Darcy-Weisbach loss coefficient
    ``K = friction_factor * length / diameter`` on the mean flow, **and** carries its
    ``length`` for the acoustic phase stamp ``P(omega)`` (so it propagates waves like a
    duct).  Constant area -- its two ports share one flow area (set on the wired edges; for
    a circular pipe that is ``pi * diameter^2 / 4``).  ``diameter`` is the hydraulic
    diameter used only in the friction term, so a non-circular passage is supported by
    passing its hydraulic diameter and matching flow area.

    The lumped pipe is exact in the low-Mach limit (the paper's Example 3, M ~ 0.01); a
    long or fast pipe develops Fanno gradients -- chain several with :func:`fanno_pipe`
    (which uses this as its atom) to resolve the Mach rise toward choke.

    Parameters
    ----------
    length : float
        Pipe length ``L`` [m] (the acoustic propagation length and the friction length).
    diameter : float
        Hydraulic diameter ``D`` [m] for the friction term ``K = f L / D``.
    friction_factor : float
        Darcy friction factor ``f`` (e.g. ``64/Re`` laminar, Haaland/Colebrook turbulent).
    name : str, optional
        Element label.

    Returns
    -------
    ElementSpec
    """
    from .ids import PIPE

    L, D, f = float(length), float(diameter), float(friction_factor)
    if not L > 0.0 or not D > 0.0 or not f >= 0.0:
        raise ValueError(f"pipe {name!r}: length and diameter must be positive and friction_factor >= 0")
    return ElementSpec(PIPE, [L, D, f], name, acoustic_id=ACOUSTIC_DUCT)


# ---------------------------------------------------------------------------
# Composite elements (Class-1 macros): a single user element that expands to a
# fixed recipe of atomic elements + internal edges at build time.  The expander
# (nefes.elements.composite.expand_composites) and round-trip-stability guarantees
# are element-agnostic; these factories just name the recipe.
# ---------------------------------------------------------------------------


def orifice(throat_area, name="orifice", eps=None) -> CompositeElementSpec:
    """Orifice plate: an isentropic contraction to the throat, then a Borda-Carnot loss.

    The De Domenico (2019) assembly at maximum loss -- an
    :func:`isentropic_area_change` to the throat area ``throat_area`` followed by a
    :func:`sudden_area_change` (Borda re-expansion) back to the downstream edge area.
    The two external edges carry the upstream (``A1``) and downstream (``A2``) areas; the
    internal throat edge carries ``throat_area``.

    Parameters
    ----------
    throat_area : float
        Throat (vena-contracta plane) area [m^2]; must be smaller than both external areas
        for a genuine orifice.
    name : str, optional
        Display name; sub-elements are namespaced (``orifice.iac`` / ``orifice.sac``).
    eps : float, optional
        Sharpens the embedded sudden-area-change switch (see :func:`sudden_area_change`);
        use a small value when the flow is one-directional and an accurate perturbation
        jump is wanted.

    Returns
    -------
    CompositeElementSpec
    """
    AT = float(throat_area)
    if not AT > 0.0:
        raise ValueError(f"orifice {name!r}: throat_area must be positive; got {throat_area}")
    return CompositeElementSpec(
        name=name,
        sub_elements=[
            isentropic_area_change(name=f"{name}.iac"),
            sudden_area_change(name=f"{name}.sac", eps=eps),
        ],
        internal_edges=[(0, 1, AT)],  # iac -> sac at the throat area
        kind="orifice",
    )


def lossy_nozzle(throat_area, beta, downstream_area, name="nozzle", eps=None) -> CompositeElementSpec:
    """General lossy nozzle (De Domenico): ``A1 ->isen-> AT ->isen-> Aj ->Borda-> A2``.

    A converging nozzle to the throat, a second isentropic change to the jet plane
    ``Aj = beta * A2``, then a Borda re-expansion to the downstream area.  ``beta`` knobs
    the loss between the two physical limits:

    * ``beta = throat_area / downstream_area`` -> the orifice (maximum loss): the second
      isentropic change is ``AT -> AT`` (trivial) and the Borda is the full ``AT -> A2``;
    * ``beta = 1`` -> the lossless nozzle: the Borda is ``A2 -> A2`` (equal areas, the
      momentum/loss terms vanish), recovering the isentropic limit.

    Parameters
    ----------
    throat_area : float
        Throat area ``AT`` [m^2].
    beta : float
        Jet-to-downstream area ratio ``Aj / A2`` in ``[AT/A2, 1]``.
    downstream_area : float
        Downstream edge area ``A2`` [m^2] (must match the external outflow edge).
    name : str, optional
        Display name.
    eps : float, optional
        Sharpens the embedded sudden-area-change switch (see :func:`sudden_area_change`).

    Returns
    -------
    CompositeElementSpec
    """
    AT, A2, b = float(throat_area), float(downstream_area), float(beta)
    if not AT > 0.0 or not A2 > 0.0:
        raise ValueError(f"lossy_nozzle {name!r}: throat_area and downstream_area must be positive")
    lo = AT / A2
    if not lo - 1e-12 <= b <= 1.0 + 1e-12:
        raise ValueError(
            f"lossy_nozzle {name!r}: beta must lie in [AT/A2, 1] = [{lo:.4g}, 1] "
            f"(AT/A2 -> orifice, 1 -> lossless); got {beta}"
        )
    Aj = b * A2
    return CompositeElementSpec(
        name=name,
        sub_elements=[
            isentropic_area_change(name=f"{name}.iac0"),
            isentropic_area_change(name=f"{name}.iac1"),
            sudden_area_change(name=f"{name}.sac", eps=eps),
        ],
        internal_edges=[(0, 1, AT), (1, 2, Aj)],  # iac0 -> iac1 at AT, iac1 -> sac at Aj
        kind="lossy_nozzle",
    )


def sudden_contraction(downstream_area, cc=0.62, name="contraction", eps=None) -> CompositeElementSpec:
    """Sudden contraction that resolves the vena-contracta state (composite).

    A flow contracting into a smaller pipe necks to a **vena contracta** of area
    ``cc * downstream_area`` just past the contraction plane, where the static pressure
    is at its minimum, then re-expands (with mixing loss) to fill the downstream pipe.
    This composite resolves that explicitly -- an :func:`isentropic_area_change` from the
    upstream area to the vena contracta, then a :func:`sudden_area_change` (Borda-Carnot)
    re-expansion to the downstream area -- so the total-pressure loss and the **minimum
    static pressure** are exact at higher Mach.

    This is the compressible upgrade to :func:`sudden_area_change`'s ``cc``-loss, whose
    incompressible ``1/2 rho u^2`` head is accurate only to ``O(M^2)``.  Read the
    vena-contracta state off the composite's throat edge
    (``solution.composite(name).throat_state``).

    Parameters
    ----------
    downstream_area : float
        The downstream pipe area ``A2`` [m^2] (must match the wired outflow edge).
    cc : float, optional
        Vena-contracta contraction coefficient in ``(0, 1]`` (default 0.62, a sharp-edged
        contraction; ``cc = 1`` is the loss-free limit).
    name : str, optional
        Display name.
    eps : float, optional
        Sharpens the embedded Borda re-expansion switch (see :func:`sudden_area_change`).

    Returns
    -------
    CompositeElementSpec
    """
    A2, c = float(downstream_area), float(cc)
    if not A2 > 0.0:
        raise ValueError(f"sudden_contraction {name!r}: downstream_area must be positive; got {downstream_area}")
    if not 0.0 < c <= 1.0:
        raise ValueError(f"sudden_contraction {name!r}: cc must be in (0, 1]; got {cc}")
    A_vc = c * A2  # the vena-contracta (minimum) area
    return CompositeElementSpec(
        name=name,
        sub_elements=[
            isentropic_area_change(name=f"{name}.contract"),
            sudden_area_change(name=f"{name}.borda", eps=eps),
        ],
        internal_edges=[(0, 1, A_vc)],  # isentropic acceleration to the vena contracta, then Borda re-expansion
        kind="sudden_contraction",
    )


def helmholtz_resonator(volume, neck_length, neck_area, name="hr") -> CompositeElementSpec:
    """Side-branch Helmholtz resonator: a tee, a neck duct, and a backing cavity.

    The "first user" of the composite mechanism (TODO item 3): a single element wrapping
    the ``junction`` + ``duct`` (neck) + ``cavity`` build that is otherwise placed by hand.
    The main line passes straight through the tee (the two external edges both attach to
    it); the neck duct and cavity hang off as an internal side branch.  Resonates at the
    lumped Helmholtz frequency ``f0 = c * sqrt(neck_area / (volume * neck_length)) / 2pi``,
    the neck's inertance (its duct phase) against the cavity's compliance (storage ``M``).

    Parameters
    ----------
    volume : float
        Backing cavity volume [m^3].
    neck_length : float
        Neck length [m] (the acoustic inertance; lengthen by an end correction to model
        the entrained near-field mass).
    neck_area : float
        Neck cross-sectional area [m^2]; the two internal edges (tee->neck, neck->cavity)
        carry it.
    name : str, optional
        Display name.

    Returns
    -------
    CompositeElementSpec
    """
    V, ln, an = float(volume), float(neck_length), float(neck_area)
    if not V > 0.0 or not ln > 0.0 or not an > 0.0:
        raise ValueError(f"helmholtz_resonator {name!r}: volume, neck_length and neck_area must all be positive")
    return CompositeElementSpec(
        name=name,
        sub_elements=[
            junction(name=f"{name}.tee"),
            duct(length=ln, name=f"{name}.neck"),
            cavity(volume=V, name=f"{name}.cavity"),
        ],
        internal_edges=[(0, 1, an), (1, 2, an)],  # tee -> neck -> cavity (the side branch)
        upstream_sub=0,  # the main line enters and leaves through the tee
        downstream_sub=0,
        kind="helmholtz_resonator",
    )


# ---------------------------------------------------------------------------
# Composite elements (Class-2 discretization): a continuous 1-D element resolved
# as an N-segment chain of compact atoms.  N is a fidelity knob -- the chain
# converges to the true distributed solution as N grows, and grid-refinement (solve
# at N and 2N) *is* the verification.  Same element-agnostic expander as Class 1.
# ---------------------------------------------------------------------------


def segments_for_frequency(length, sound_speed, f_max, points_per_wavelength=12):
    """Smallest segment count ``N`` that keeps each segment acoustically compact at ``f_max``.

    A discretization composite resolves waves only where each segment is short against the
    wavelength (``k dL << 1``).  With ``P`` points per wavelength, ``N >= P * f_max * L / c``.
    Use it to auto-size :func:`fanno_pipe` / :func:`tapered_duct` from the highest analysis
    frequency rather than guessing.

    Parameters
    ----------
    length : float
        Element length ``L`` [m].
    sound_speed : float
        A representative mean sound speed ``c`` [m/s] (e.g. ``solution.field("c").mean()``).
    f_max : float
        Highest analysis frequency [Hz] the chain must resolve.
    points_per_wavelength : int, optional
        Target points per wavelength ``P`` (default 12; 10-20 is typical).

    Returns
    -------
    int
        The recommended segment count ``N`` (>= 1).
    """
    L, c, fmax = float(length), float(sound_speed), float(f_max)
    if not (L > 0.0 and c > 0.0 and fmax > 0.0):
        raise ValueError("segments_for_frequency needs positive length, sound_speed and f_max")
    return max(1, int(math.ceil(points_per_wavelength * fmax * L / c)))


def fanno_pipe(length, diameter, friction_factor, n_segments, name="pipe") -> CompositeElementSpec:
    """Distributed (Fanno) pipe: an ``n_segments`` chain of :func:`pipe` atoms.

    A long or fast pipe is **Fanno flow** -- wall friction drives the subsonic flow toward
    ``M = 1``, so density, velocity and Mach vary continuously along the length and a single
    lumped ``K`` misses it.  This chains ``n_segments`` pipe atoms, each of length ``L/N``
    and the same ``diameter``/``friction_factor``, joined by single internal edges (each
    internal edge *is* the intermediate flow state -- no junctions).  As ``N`` grows the
    chain converges to the true Fanno solution and can approach choke at the pipe exit; the
    locally-uniform per-segment mean state also propagates acoustics through the mean
    gradient far better than one lumped duct stamp.

    Parameters
    ----------
    length, diameter, friction_factor : float
        Total length ``L`` [m], hydraulic diameter ``D`` [m] and Darcy friction factor ``f``
        (see :func:`pipe`).
    n_segments : int
        Number of pipe-atom segments ``N`` (>= 1; a fidelity knob, see
        :func:`segments_for_frequency` and :func:`grid_refine`).
    name : str, optional
        Display name.

    Returns
    -------
    CompositeElementSpec
    """
    L, D, f, N = float(length), float(diameter), float(friction_factor), int(n_segments)
    if not L > 0.0 or not D > 0.0 or not f >= 0.0:
        raise ValueError(f"fanno_pipe {name!r}: length and diameter must be positive, friction_factor >= 0")
    if N < 1:
        raise ValueError(f"fanno_pipe {name!r}: n_segments must be >= 1; got {n_segments}")
    if N == 1:
        return pipe(L, D, f, name=name)  # the lumped (N=1) limit is a single pipe atom, no composite
    area = math.pi * D * D / 4.0  # the constant flow area shared by every segment
    subs = [pipe(L / N, D, f, name=f"{name}.seg{i}") for i in range(N)]
    internal = [(i, i + 1, area) for i in range(N - 1)]  # one edge between consecutive segments
    return CompositeElementSpec(name=name, sub_elements=subs, internal_edges=internal, kind="fanno_pipe")


def _taper_stations(area, length, n_segments):
    """Resolve a taper's station areas ``[A0 .. AN]`` and axial coordinates ``[x0 .. xN]``.

    Two input forms:

    * an ``(x, A)`` table -- a sequence of ``(position, area)`` pairs, the axial positions
      ``x`` in metres and strictly increasing.  The stations (hence the per-segment lengths)
      may be **non-uniform**, and the total length is *inferred* as ``xN - x0``; the segment
      count is ``len(table) - 1``.  A ``length`` may be passed only as a consistency check
      (it must match the inferred span); ``n_segments`` likewise.
    * a callable ``A(x)`` -- sampled at ``n_segments + 1`` equispaced stations over
      ``[0, length]`` (both ``length`` and ``n_segments`` required).
    """
    if callable(area):
        if length is None:
            raise ValueError("tapered_duct: pass length when area is a callable A(x)")
        L = float(length)
        if not L > 0.0:
            raise ValueError(f"tapered_duct: length must be positive; got {length}")
        if n_segments is None or int(n_segments) < 1:
            raise ValueError("tapered_duct: pass n_segments >= 1 when area is a callable A(x)")
        N = int(n_segments)
        xs = [L * k / N for k in range(N + 1)]
        areas = [float(area(x)) for x in xs]
    else:
        try:
            pairs = [(float(x), float(a)) for (x, a) in area]
        except (TypeError, ValueError):
            raise ValueError(
                "tapered_duct: the area table must be a sequence of (x, area) pairs -- axial position "
                "x [m] and area A [m^2], e.g. [(0.0, 3e-3), (0.15, 1.5e-3), (0.3, 3e-3)]"
            )
        if len(pairs) < 2:
            raise ValueError("tapered_duct: the (x, area) table needs >= 2 stations")
        xs = [p[0] for p in pairs]
        areas = [p[1] for p in pairs]
        if any(xs[i + 1] <= xs[i] for i in range(len(xs) - 1)):
            raise ValueError(f"tapered_duct: station positions x must be strictly increasing; got {xs}")
        span = xs[-1] - xs[0]
        if length is not None and abs(float(length) - span) > 1e-9 * max(span, 1.0):
            raise ValueError(
                f"tapered_duct: passed length {length} does not match the (x, area) table span {span:g} "
                "(length is inferred from x -- omit it, or make it match the span)"
            )
        if n_segments is not None and int(n_segments) != len(pairs) - 1:
            raise ValueError(
                f"tapered_duct: n_segments ({n_segments}) must equal len(table) - 1 ({len(pairs) - 1}) "
                "for an (x, area) table (the stations set the segment count)"
            )
    if any(a <= 0.0 for a in areas):
        raise ValueError("tapered_duct: every station area must be positive")
    return areas, xs


def tapered_duct(area, length=None, n_segments=None, name="taper") -> CompositeElementSpec:
    """Tapered duct / horn / con-di nozzle resolved from an ``(x, A)`` profile.

    Discretizes a continuously area-varying acoustic passage into ``N`` segments, each a
    compact area change ``A_i -> A_{i+1}`` followed by a length-``(x_{i+1} - x_i)`` duct at
    the segment's downstream area (the catalog has no length-bearing area-change atom, so a
    segment is two atoms).  As ``N`` grows the chain converges to the true horn, and a con-di
    profile **chokes at its true throat** -- the min-area edge -- with the
    isentropic-area-change complementarity engaging on exactly that segment.

    The standard input is a table of ``(x, A)`` pairs: the axial positions ``x`` [m] set the
    station spacing (which **may be non-uniform** -- cluster stations where the area varies
    fastest, e.g. near a throat) and the total length is *inferred* from them.  A callable
    ``A(x)`` is also accepted; it is sampled at ``n_segments + 1`` equispaced stations.

    Because each segment carries a real :func:`duct`, the taper **propagates acoustic waves**
    through its interior (each duct spans its own station interval); the area-change atoms are
    compact.

    Parameters
    ----------
    area : sequence of (float, float), or callable
        A table of ``(x, A)`` station pairs -- axial position ``x`` [m] (strictly increasing)
        and area ``A`` [m^2] -- or a callable ``A(x)`` [m^2] sampled at ``n_segments + 1``
        equispaced stations over ``[0, length]``.  The two external edges must carry ``A0``
        (upstream) and ``AN`` (downstream).
    length : float, optional
        Total axial length ``L`` [m].  **Required** for a callable ``area``; for an
        ``(x, A)`` table it is inferred as ``xN - x0`` and, if given, checked against that span.
    n_segments : int, optional
        Segment count ``N`` -- **required** when ``area`` is a callable; for an ``(x, A)``
        table it is ``len(table) - 1`` (and, if given, checked against it).
    name : str, optional
        Display name.

    Returns
    -------
    CompositeElementSpec
    """
    areas, xs = _taper_stations(area, length, n_segments)
    N = len(areas) - 1
    # segment i: iac (A_i -> A_{i+1}) at sub-index 2i, then a duct spanning this station's own
    # interval (x_{i+1} - x_i) at A_{i+1} at 2i+1 -- non-uniform stations give non-uniform ducts
    subs = []
    for i in range(N):
        subs.append(isentropic_area_change(name=f"{name}.iac{i}"))
        subs.append(duct(length=xs[i + 1] - xs[i], name=f"{name}.duct{i}"))
    internal = []
    for i in range(N):
        internal.append((2 * i, 2 * i + 1, areas[i + 1]))  # iac_i -> duct_i at the segment's downstream area
        if i < N - 1:
            internal.append((2 * i + 1, 2 * i + 2, areas[i + 1]))  # duct_i -> iac_{i+1}
    return CompositeElementSpec(
        name=name,
        sub_elements=subs,
        internal_edges=internal,
        upstream_sub=0,
        downstream_sub=2 * N - 1,
        kind="tapered_duct",
    )


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
    bases = default_name_bases()
    for el in elements:
        base = el.name or ""
        # Factory defaults always start numbered ("inlet" -> "inlet-1"); user-chosen names keep
        # their bare form and are suffixed only on an actual clash.
        el.name = unique_name(base, seen, always_number=base in bases)
        seen.add(el.name)


def unique_name(name: str, taken, always_number: bool = False) -> str:
    """Return a copy of ``name`` not already in ``taken``.

    If ``name`` is free it is returned unchanged, unless ``always_number`` is set, in which case the
    first free ``<name>-1``, ``<name>-2``, ... is returned instead (so a lone factory default still
    reads ``inlet-1``).  ``taken`` is any container of used names; the result is *not* added to it --
    the caller records it, which keeps this usable both for a running build and for a one-shot pass
    over a finished list.
    """
    if not always_number and name not in taken:
        return name
    k = 1
    while f"{name}-{k}" in taken:
        k += 1
    return f"{name}-{k}"


@lru_cache(maxsize=1)
def default_name_bases() -> frozenset:
    """The set of factory-default element names, read from the catalog factory signatures.

    Every public factory here declares its default label as a ``name=`` keyword (``duct`` for
    :func:`duct`, ``inlet`` for :func:`mass_flow_inlet`, ...).  Collecting those defaults lets the
    dedup pass tell a factory default apart from a name the user chose, so only the defaults are
    force-numbered.  Cached: the signatures are fixed at import.
    """
    bases = set()
    module = sys.modules[__name__]
    for obj in vars(module).values():
        if not (inspect.isfunction(obj) and obj.__module__ == __name__):
            continue
        param = inspect.signature(obj).parameters.get("name")
        if param is not None and isinstance(param.default, str) and param.default:
            bases.add(param.default)
    return frozenset(bases)


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
        elif rid == FORCED_SPLITTER:
            if deg < 3:
                raise ValueError(
                    f"{label} is a forced splitter and needs >= 3 ports (1 inflow + >= 2 outflows) "
                    f"but is connected to {deg} edge(s)"
                )
            n_frac = len(el.fparams)
            if n_frac != deg - 2:
                raise ValueError(
                    f"{label}: a forced splitter with {deg} ports (1 inflow + {deg - 1} outflows) needs "
                    f"{deg - 2} split fraction(s) -- one per controlled outflow, the last outflow being the "
                    f"remainder -- but {n_frac} were given"
                )

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


def _burnt_seed(conn: Connectivity, flame_nodes) -> np.ndarray:
    """Per-edge burnt-marker initial guess ``(0 fresh / 1 burnt)`` by a topology flood-fill.

    Seeds ``b = 1`` on every edge leaving an equilibrium flame (along the *declared* tail->head
    arrows) and floods it downstream.  This is only the marker transport's **initial guess** --
    the signed-mass-flow transport self-corrects a backward-drawn flame at convergence -- so its
    job is purely to warm the start (a correct drawing converges in one shot).
    """
    n_edges = int(conn.n_edges)
    tail = np.asarray(conn.tail_node)
    head = np.asarray(conn.head_node)
    out_edges = defaultdict(list)  # node -> outgoing edges (declared tail -> head)
    for e in range(n_edges):
        out_edges[int(tail[e])].append(e)
    burnt = np.zeros(n_edges, dtype=np.float64)
    stack = []
    for e in range(n_edges):  # seed: every edge leaving a flame is burnt
        if int(tail[e]) in flame_nodes:
            burnt[e] = 1.0
            stack.append(int(head[e]))
    while stack:  # flood downstream; each edge is marked at most once -> terminates on cycles
        for e in out_edges[stack.pop()]:
            if burnt[e] == 0.0:
                burnt[e] = 1.0
                stack.append(int(head[e]))
    return burnt


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

    This is the lower-level functional builder; the user-facing path is
    :class:`nefes.shell.network.Network`, whose constructor accepts the same ``nodes`` /
    ``edges`` lists (and auto-derives the reference scales), then ``.solve()``.

    Ports are auto-assigned in attachment order.  Use
    ``build_problem_from_connectivity`` to supply explicit ports (e.g. a UI
    export where the port ordinals carry meaning).  ``edge_models`` optionally
    overrides the per-edge thermo model id (default: the config's model on every
    edge) -- e.g. frozen upstream, equilibrium downstream of a flame.
    """
    # expand any composite elements (build-time graph transform) into atomic elements +
    # internal edges; a composite-free network passes through unchanged (composite_map None).
    elements, edges, composite_map = expand_composites(elements, edges)
    n_nodes = len(elements)
    directed = [(t, h) for (t, h, _a) in edges]
    area = np.array([a for (_t, _h, a) in edges], dtype=np.float64)
    conn = connectivity_from_directed_edges(n_nodes, directed)
    return build_problem_from_connectivity(
        thermo, elements, conn, area, mdot_ref, p_ref, h_ref, edge_models=edge_models, composite_map=composite_map
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
    composite_map=None,
) -> CompiledProblem:
    """Assemble a CompiledProblem from elements and a prebuilt Connectivity.

    The connectivity carries explicit per-edge ports (``tail_port``/
    ``head_port``), so port-ordering conventions are preserved exactly.
    ``composite_map`` (set by :func:`build_problem` when the network carried composite
    elements) bridges the user-facing ids to the expanded ones; ``None`` otherwise.
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

    # Marker-gated reacting closure (the default/auto reacting path): a reacting network with at
    # least one equilibrium flame and no explicit per-edge override runs EQ_MARKER on every edge
    # and transports one extra "burnt" marker scalar (the last advected scalar) that gates the
    # frozen/equilibrium blend.  The marker rides the *signed* mass flow, so it labels "downstream
    # of a flame" robustly regardless of how the edges were drawn -- demoting the old topology
    # flood-fill to the marker's initial guess.  An explicit ``edge_models`` keeps the hard
    # per-edge closure (EQ_FROZEN/EQ_KERNEL, no marker), the power-user escape hatch.
    flame_nodes = {n for n in range(n_nodes) if int(node_rid[n]) == FLAME_EQUILIBRIUM}
    marker_gated = thermo.model_id == EQ_KERNEL and bool(flame_nodes) and edge_models is None
    n_marker = 1 if marker_gated else 0

    # A user-set inflow marker only has a transport scalar to ride when the network is
    # marker-gated; reject a non-zero marker elsewhere rather than silently dropping it.
    if not marker_gated:
        stray = [el.name or f"node {n}" for n, el in enumerate(elements) if float(getattr(el, "marker", 0.0)) != 0.0]
        if stray:
            raise ValueError(
                "a non-zero burnt marker requires a marker-gated reacting network (an equilibrium-flame "
                "reacting model with no explicit per-edge closure); marker was set on: " + ", ".join(stray)
            )

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
    # The burnt marker is the *last* advected scalar, so its donor param sits after the
    # composition.  A fresh feed/backflow enters with marker = 0 (the default); a boundary
    # may set ``marker = 1`` to inject already-burnt gas (e.g. exhaust-gas recirculation).
    # Appended only when the network is marker-gated.
    npar_f = []
    npar_fptr = np.zeros(n_nodes + 1, dtype=np.int64)
    for n, el in enumerate(elements):
        fp = list(el.fparams)
        k = -1 if node_stream is None else node_stream.get(n, -1)
        marker_param = [float(el.marker)] if n_marker else []
        if el.residual_id in boundary_rids and len(fp) >= 2:
            base, Tt = float(fp[0]), float(fp[1])
            fp = [base] + _boundary_scalars(thermo, el, Tt, n_elem, _node_label(n, el), k) + marker_param
        elif el.residual_id == MASS_SOURCE:
            fp = _mass_source_params(thermo, el, n_elem, _node_label(n, el), k) + marker_param
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

    # per-node transfer-matrix descriptor (TRANSFER_MATRIX element; mean flow ignores it)
    node_transfer_matrix = tuple(getattr(el, "transfer_matrix", None) for el in elements)

    n_scalars = thermo.n_elem + n_marker  # composition mixture fractions + the optional burnt marker
    n_solve = 3 + n_scalars
    marker_row = (3 + thermo.n_elem) if marker_gated else -1  # the marker is the last band-1 row
    pat = build_jacobian_pattern(conn, degrees, n_solve=n_solve)

    # residual scales: node rows, then the advected-scalar transport rows (h_t for every edge,
    # then each composition scalar for every edge, then the marker for every edge).  Composition
    # mixture fractions and the marker are O(1) (in [0, 1]), so their scale = 1.
    z_scale = 1.0
    res_scale = []
    for n, el in enumerate(elements):
        res_scale.extend(_row_kinds(el.residual_id, degrees[n], mdot_ref, p_ref))
    res_scale.extend([h_ref] * conn.n_edges)
    for _ in range(n_scalars):
        res_scale.extend([z_scale] * conn.n_edges)
    res_scale = np.array(res_scale, dtype=np.float64)

    var_scale = np.array([mdot_ref, p_ref, h_ref] + [z_scale] * n_scalars, dtype=np.float64)

    # per-edge thermo model: marker-gated -> EQ_MARKER everywhere; explicit override -> verbatim;
    # otherwise the config's model on every edge.
    if edge_models is None:
        edge_model = np.full(conn.n_edges, EQ_MARKER if marker_gated else thermo.model_id, dtype=np.int64)
    else:
        edge_model = np.ascontiguousarray(edge_models, dtype=np.int64)
        if edge_model.shape[0] != conn.n_edges:
            raise ValueError(f"edge_models has {edge_model.shape[0]} entries but the network has {conn.n_edges} edges")

    # an unburnt (EQ_FROZEN) or marker-gated (EQ_MARKER, which runs the frozen leg) edge
    # reconstructs species from the feed streams; at least one stream must exist (an inlet /
    # mass source must inject a composition).
    if thermo.model_id != PERFECT_GAS and np.any((edge_model == EQ_FROZEN) | (edge_model == EQ_MARKER)):
        n_streams = int(thermo.ti[5]) if thermo.ti.shape[0] > 5 else 0
        if n_streams == 0:
            raise ValueError(
                "the network has frozen / marker-gated (unburnt-capable) edges but no feed streams "
                "were found; an inlet or mass source must inject an explicit species composition "
                "for the frozen closure to reconstruct from"
            )

    # burnt-marker initial guess: the old topology flood-fill, demoted to the marker's seed
    # (b = 1 downstream of a flame along the declared arrows, b = 0 elsewhere).  Correctness no
    # longer depends on it -- the transport self-corrects a backward-drawn flame -- it only warms
    # the start.  None when the network carries no marker.
    marker_seed = _burnt_seed(conn, flame_nodes) if marker_gated else None

    return CompiledProblem(
        model_id=thermo.model_id,
        tf=thermo.tf,
        ti=thermo.ti,
        n_elem=thermo.n_elem,
        n_solve=n_solve,
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
        node_transfer_matrix=node_transfer_matrix,
        scalar_names=tuple(thermo.element_names),
        marker_row=marker_row,
        marker_seed=marker_seed,
        composite_map=composite_map,
    )
