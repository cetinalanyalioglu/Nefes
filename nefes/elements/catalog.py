"""Element catalog (Python, parse-time): the factory functions for network elements.

Each factory returns an ``ElementSpec`` naming an element's residual id and its ordered
float parameters (the order the ``@njit`` kernels expect).  The CompiledProblem builder
that assembles these specs into a solvable problem lives in :mod:`nefes.shell.build`.
"""

import inspect
import math
import sys
from dataclasses import dataclass, field
from functools import lru_cache, wraps
from typing import List, Optional

from .composite import CompositeElementSpec
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
    MASS_SOURCE,
    STAMP_DEFAULT,
    STAMP_DUCT,
    STAMP_VOLUME,
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
    """One network element: its residual type and the ordered parameters its kernel reads.

    The immutable parse-time building block that ``build_problem`` assembles (with the
    directed edges) into a :class:`~nefes.shell.problem.CompiledProblem`.  The mean-flow
    solve needs only ``residual_id`` and ``fparams``; the remaining fields are optional
    acoustic / reacting / perturbation extras, one line each below.
    """

    residual_id: int  # element type (an ``ids`` constant); the @njit kernel dispatch key
    fparams: List[float] = field(default_factory=list)  # ordered float params, in kernel order
    name: str = ""  # human-readable label (reporting / plotting)
    acoustic_stamp: int = STAMP_DEFAULT  # perturbation stamp; STAMP_DEFAULT -> only J_alg
    # smoothing-width override in mass-flow units; None follows the global solve-time eps
    eps: Optional[float] = None
    perturbation_bc: Optional[object] = None  # PerturbationBC (None -> inherit)
    # stream this element introduces / draws on backflow: a named species mixture
    # {species: fraction} in ``basis`` units, or raw passive scalars for a perfect gas
    composition_spec: object = None
    basis: str = "mole"  # units of composition_spec: "mole" or "mass"
    dynamic_source: object = None  # DynamicSource for the S(omega) block; mean flow ignores it
    transfer_matrix: object = None  # TransferMatrix for a TRANSFER_MATRIX element; perturbation-only
    # injected burnt marker at an inflow/source (0 fresh, 1 burnt); marker-gated networks only
    marker: float = 0.0
    # ambient back pressure [Pa] a choked_nozzle_outlet discharges into; diagnostic only
    # (never read by a kernel), used post-solve to warn if the nozzle would not actually choke
    back_pressure: Optional[float] = None
    # True when ``name`` was left at the factory default (not chosen by the caller); the dedup pass
    # numbers a lone default ("duct" -> "duct-1") but keeps an explicitly chosen name bare.  Set by
    # the factory wrapper, not the user; excluded from equality / repr.
    name_auto: bool = field(default=False, compare=False, repr=False)


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

    ``Tt`` is the **total** (stagnation) temperature.  The transported total enthalpy
    is ``h_t = h(Tt)`` -- the mixture's enthalpy evaluated at ``Tt`` -- so the static
    temperature ``T < Tt`` is recovered on the edge from ``h = h_t - u^2/2``; no kinetic
    energy is double-counted.  ``composition`` is a named species mixture
    (``{species: fraction}``) for the equilibrium model -- e.g. dry air as
    ``{"O2": 0.21, "N2": 0.79}`` with ``basis="mole"`` -- resolved to the transported
    feed streams and this ``h_t`` at ``build_problem``.

    This is an **inflow boundary**: ``mdot`` must be non-negative (``>= 0``).  A
    positive value injects the feed stream; ``mdot = 0`` is a quiescent (closed) inlet.
    Reverse flow (a negative prescribed mass rate, i.e. suction out through the inlet)
    is not permitted -- use a :func:`mass_flow_outlet` for an outflow with a prescribed
    mass flow rate.

    ``marker`` (default ``0.0``, fresh reactant) is the injected burnt-marker value of
    the marker-gated reacting closure; set ``1.0`` to feed already-burnt gas (e.g.
    exhaust-gas recirculation), forcing the equilibrium closure downstream.  It is only
    accepted on a marker-gated network (equilibrium-flame reacting, no explicit per-edge
    closure); a non-zero value elsewhere is rejected at build time.

    Parameters
    ----------
    mdot : float
        Prescribed inflow mass rate [kg/s] (``>= 0``).
    Tt : float
        Total (stagnation) temperature [K] of the feed.
    composition : dict or array_like, optional
        Feed composition -- a named species mixture ``{species: fraction}`` for the
        equilibrium model, or raw passive-scalar values for a perfect gas.  ``None`` -> zeros.
    basis : {"mole", "mass"}, optional
        Units of ``composition`` (default ``"mole"``).
    name : str, optional
        Element label.
    perturbation_bc : PerturbationBC, optional
        Acoustic termination; ``None`` inherits the linearized inlet reflection.
    marker : float, optional
        Injected burnt marker (``0.0`` fresh, default; ``1.0`` burnt); marker-gated networks only.

    Returns
    -------
    ElementSpec
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

    Parameters
    ----------
    pt : float
        Prescribed total (stagnation) pressure [Pa].
    Tt : float
        Total (stagnation) temperature [K] of the feed.
    composition : dict or array_like, optional
        Feed composition; see :func:`mass_flow_inlet`.
    basis : {"mole", "mass"}, optional
        Units of ``composition`` (default ``"mole"``).
    name : str, optional
        Element label.
    perturbation_bc : PerturbationBC, optional
        Acoustic termination; ``None`` inherits the linearized inlet reflection.
    marker : float, optional
        Injected burnt marker (``0.0`` fresh, default; ``1.0`` burnt); marker-gated networks only.

    Returns
    -------
    ElementSpec
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
    """Static-pressure outlet; becomes a total-pressure inlet on backflow, feeding ``composition`` at ``Tt_backflow``
    and a total pressure of ``pt_backflow = p``.

    ``marker`` is the burnt-marker value of the backflow stream drawn in on ingestion
    (``0.0`` fresh, default; ``1.0`` burnt); see :func:`mass_flow_inlet`.

    Parameters
    ----------
    p : float
        Prescribed static (and backflow total) pressure [Pa].
    Tt_backflow : float, optional
        Total temperature [K] of gas drawn in on ingestion (default ``300.0``).
    composition : dict or array_like, optional
        Backflow composition; see :func:`mass_flow_inlet`.
    basis : {"mole", "mass"}, optional
        Units of ``composition`` (default ``"mole"``).
    name : str, optional
        Element label.
    perturbation_bc : PerturbationBC, optional
        Acoustic termination; ``None`` inherits the linearized outlet reflection.
    marker : float, optional
        Backflow burnt marker (``0.0`` fresh, default; ``1.0`` burnt); marker-gated networks only.

    Returns
    -------
    ElementSpec
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
    critical mass flow is known.

    This is an **outflow-only** boundary: ``mdot`` must be positive (positive means
    leaving the domain per element definition).

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


def choked_nozzle_outlet(throat_area, back_pressure=None, name="outlet", perturbation_bc=None):
    """Compact choked-nozzle outlet of throat area ``throat_area`` lumped downstream.

    Models a sonic (M=1) throat just beyond the outlet plane: the outflow is the
    critical mass flux for the interior stagnation state and the throat area, so the
    static pressure floats (the choked mass flow is set by the upstream total state, not
    a back-pressure).  Because the nozzle is *compact* (lumped), the application plane
    stays **subsonic** (the sonic point is in the throat, not explicitly in the domain).
    Left at ``perturbation_bc=None`` the acoustic termination is the *inherited* linearization
    of the critical-mass-flux row, which is the compact choked-nozzle (Marble-Candel) reflection
    with its entropy -> acoustic coupling -- no separately specified boundary condition needed.

    Use it when the convergent section is acoustically compact; resolving the higher-Mach
    part of the contraction explicitly and lumping only the near-throat remainder makes
    the compact assumption progressively better.  A choked throat is one-way, so this is
    an **outflow-only** boundary: it does not model ingestion (the critical mass flux is
    always positive, so the converged flow cannot reverse here).

    This element **asserts** the nozzle is choked: it imposes the critical mass flux
    unconditionally (the mass flow scales with the interior total pressure and stays
    sonic-throat-consistent at any total pressure, i.e. there is no back-pressure to detect
    unchoking.  ``throat_area`` must be smaller than the outlet edge area (a contraction;
    enforced at build time).  If the nozzle may be unchoked at low pressure ratio, use
    :func:`pressure_outlet` instead -- its emergent complementarity handles the
    choked/unchoked transition against a prescribed back-pressure.

    Parameters
    ----------
    throat_area : float
        Sonic-throat area ``A*`` [m^2] of the lumped nozzle (``A* < A_outlet`` so the
        approach stays subsonic).  Must be > 0.
    back_pressure : float, optional
        Ambient (discharge) pressure [Pa] the nozzle exhausts into.  Purely a diagnostic:
        no kernel reads it, but when given, the solve checks it against the throat's
        critical (sonic) pressure at the converged state and warns if it is too high for the
        nozzle to choke (i.e. the compact choked-nozzle model would not apply).  ``None``
        (default) skips the check.  See :meth:`~nefes.shell.network.Solution.unchoked_nozzles`.
    name : str, optional
        Element label.
    perturbation_bc : PerturbationBC, optional
        Acoustic termination; ``None`` inherits the linearized choked-nozzle reflection.
    """
    if not float(throat_area) > 0.0:
        raise ValueError(f"choked_nozzle_outlet throat area must be > 0 (got {throat_area})")
    if back_pressure is not None and not float(back_pressure) > 0.0:
        raise ValueError(f"choked_nozzle_outlet back_pressure must be > 0 (got {back_pressure})")
    bp = None if back_pressure is None else float(back_pressure)
    return ElementSpec(
        CHOKED_NOZZLE_OUTLET, [float(throat_area)], name, perturbation_bc=perturbation_bc, back_pressure=bp
    )


def wall(name="wall", perturbation_bc=None):
    """An impermeable single-port termination: ``mdot = 0`` on its incident edge.

    The wall blocks mean flow, so the leg behind it is stagnant (``M = 0``); its
    purpose is acoustic.  By default it closes the perturbation problem as a rigid
    hard wall (``u' = 0``, ``R = +1``) -- which at the wall's ``M = 0`` state is
    identical to the inherited ``mdot' = 0`` row.  Pass ``perturbation_bc`` to model
    a non-rigid termination (e.g. a liner impedance) instead.

    Parameters
    ----------
    name : str, optional
        Element label.
    perturbation_bc : PerturbationBC, optional
        Acoustic termination; ``None`` (default) is a rigid hard wall.

    Returns
    -------
    ElementSpec
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
    (rho c^2)`` that populates the storage block ``M`` (the ``i*omega*M`` term of the
    operator ``A = J_alg + i*omega*M + P + S``). Paired with a neck inertance (a short
    :func:`duct`) off a :func:`junction`, it forms a Helmholtz resonator with
    ``omega_0 = c * sqrt(A_neck / (V * l_eff))`` (see :func:`helmholtz_resonator`).

    Although it is a terminal element (single-port), the cavity is *not* a boundary terminal:
    its acoustic response is the compliance itself (a reflection set by the storage),
    so the perturbation layer leaves its inherited ``mdot' = 0`` row in place and lets ``M``
    add the storage onto it. It is never neutralized or stamped with a reflection coefficient.

    The mean state of the cavity gas is tied to its face (the local ``p``, ``T``,
    composition), so ``c`` is the local sound speed. An independently-stated cavity
    is currently not supported.

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
    return ElementSpec(CAVITY, [V], name, acoustic_stamp=STAMP_VOLUME)


def isentropic_area_change(name="iac", l_up=0.0, l_down=0.0, end_correction=0.0):
    """A smooth (lossless) contraction or diffuser; optionally length-bearing to model
    acoustic inertance and compliance (**not** propagation).

    By default a lengthless jump.  A real diffuser/nozzle has axial extent, so the
    optional ``l_up``/``l_down`` (the passage half-lengths on the port-0 / port-1 sides)
    give it acoustic **compliance** (each side stores ``l_i * A_i`` of gas) and
    **inertance** (series effective length ``l_up + l_down + end_correction``,
    referenced to the throat).  These populate the storage block ``M`` and are inert in
    the mean flow.

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
    """A 2-port whose acoustic response is a **user-supplied transfer matrix**.

    In the mean flow this element is an :func:`isentropic_area_change` -- it conserves
    mass and energy, is isentropic, and permits an area change across it -- so it seats a
    well-defined mean state on both faces.  In the perturbation network it does **not**
    inherit the linearized area-change jump: only its acoustic rows are overwritten, with
    the relation ``w_down = TM(omega) . w_up`` carried by ``tm``, letting a measured /
    prescribed 2-port response stand in for an element that has no closed-form model.  The
    mean state is unaffected by ``tm``.

    In the UI this is the ``TransferMatrix`` node.  The descriptor itself is a Python
    object with no YAML form, so it does not round-trip: attach it after loading
    (``net.set(node, transfer_matrix=...)``).

    Parameters
    ----------
    tm : TransferMatrix or UnknownTransferMatrix, optional
        The frequency-domain 2-port descriptor
        (:class:`nefes.perturbation.matrix.TransferMatrix`) stamped in the perturbation
        layer.  Pass an :class:`~nefes.perturbation.identify.UnknownTransferMatrix` marker
        to leave it to be identified from a measured network response.  ``None`` (default)
        leaves the element acoustically an isentropic area change until a descriptor is
        attached (``net.set(node, transfer_matrix=...)``).
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
        the reverse branch reduces to exact total-pressure continuity.  Use a
        tabulated value for the geometry (e.g.
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
        the step still carries.  See :func:`isentropic_area_change`.
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
    drop.  The static state on each port is reconstructed
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
        :func:`isentropic_area_change`.
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
    the term that drives thermoacoustic instability.

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

    Parameters
    ----------
    name : str, optional
        Element label.
    volume : float, optional
        Chamber volume [m^3] for the acoustic compliance (default ``0.0``, a lengthless split).

    Returns
    -------
    ElementSpec
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
    controlled outflows + the remainder).  In the UI this is the ``ForcedSplitter``
    node, whose ``fractions`` string carries the betas in port order.

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

    The mean residual is equal-area continuity (length-independent); ``length`` is inert in the
    steady residual and read only by the acoustic phase stamp.  It rides ``fparams[0]`` as
    ordinary acoustic metadata.

    Parameters
    ----------
    length : float, optional
        Duct length [m] for the acoustic phase (default ``0.0``, a lengthless jump).
    name : str, optional
        Element label.

    Returns
    -------
    ElementSpec
    """
    from .ids import DUCT

    return ElementSpec(DUCT, [float(length)], name, acoustic_stamp=STAMP_DUCT)


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
    return ElementSpec(PIPE, [L, D, f], name, acoustic_stamp=STAMP_DUCT)


# ---------------------------------------------------------------------------
# Composite elements (Class-1 macros): a single user element that expands to a
# fixed recipe of atomic elements + internal edges at build time.  The expander
# (nefes.elements.composite.expand_composites) and round-trip-stability guarantees
# do not depend on the element; these factories just name the recipe.
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
        params={"throat_area": AT},
    )


def lossy_nozzle(throat_area, beta, name="nozzle", eps=None) -> CompositeElementSpec:
    """General lossy nozzle (De Domenico): ``A1 ->isen-> AT ->isen-> Aj ->Borda-> A2``.

    A converging nozzle to the throat, a second isentropic change to the jet plane
    ``Aj = beta * A2``, then a Borda re-expansion to the downstream area.  The downstream
    area ``A2`` is read off the attached outflow edge at build time (areas live on edges,
    never on elements).  ``beta`` knobs the loss between the two physical limits:

    * ``beta = AT / A2`` -> the orifice (maximum loss): the second isentropic change is
      ``AT -> AT`` (trivial) and the Borda is the full ``AT -> A2``;
    * ``beta = 1`` -> the lossless nozzle: the Borda is ``A2 -> A2`` (equal areas, the
      momentum/loss terms vanish), recovering the isentropic limit.

    Parameters
    ----------
    throat_area : float
        Throat area ``AT`` [m^2].
    beta : float
        Jet-to-downstream area ratio ``Aj / A2`` in ``[AT/A2, 1]``; the lower bound needs
        the outflow edge area, so it is checked at build time.
    name : str, optional
        Display name.
    eps : float, optional
        Sharpens the embedded sudden-area-change switch (see :func:`sudden_area_change`).

    Returns
    -------
    CompositeElementSpec
    """
    AT, b = float(throat_area), float(beta)
    if not AT > 0.0:
        raise ValueError(f"lossy_nozzle {name!r}: throat_area must be positive; got {throat_area}")
    if not 0.0 < b <= 1.0 + 1e-12:
        raise ValueError(f"lossy_nozzle {name!r}: beta must lie in [AT/A2, 1] (1 -> lossless); got {beta}")

    def _jet_area(a_up, a_down):
        lo = AT / a_down
        if not lo - 1e-12 <= b:
            raise ValueError(
                f"lossy_nozzle {name!r}: beta must lie in [AT/A2, 1] = [{lo:.4g}, 1] "
                f"(AT/A2 -> orifice, 1 -> lossless); got {b}"
            )
        return b * a_down

    return CompositeElementSpec(
        name=name,
        sub_elements=[
            isentropic_area_change(name=f"{name}.iac0"),
            isentropic_area_change(name=f"{name}.iac1"),
            sudden_area_change(name=f"{name}.sac", eps=eps),
        ],
        internal_edges=[(0, 1, AT), (1, 2, _jet_area)],  # iac0 -> iac1 at AT, iac1 -> sac at Aj = beta*A2
        kind="lossy_nozzle",
        params={"throat_area": AT, "beta": b},
    )


def sudden_contraction(*, cc=0.62, name="contraction", eps=None) -> CompositeElementSpec:
    """Sudden contraction that resolves the vena-contracta state (composite).

    A flow contracting into a smaller pipe necks to a **vena contracta** of area
    ``cc * A2`` just past the contraction plane, where the static pressure is at its
    minimum, then re-expands (with mixing loss) to fill the downstream pipe.  The
    downstream area ``A2`` is read off the attached outflow edge at build time (areas
    live on edges, never on elements).  This composite resolves the neck explicitly --
    an :func:`isentropic_area_change` from the upstream area to the vena contracta, then
    a :func:`sudden_area_change` (Borda-Carnot) re-expansion to the downstream area --
    so the total-pressure loss and the **minimum static pressure** are exact at higher
    Mach.

    This is the compressible upgrade to :func:`sudden_area_change`'s ``cc``-loss, whose
    incompressible ``1/2 rho u^2`` head is accurate only to ``O(M^2)``.  Read the
    vena-contracta state off the composite's throat edge
    (``solution.composite(name).throat_state``).

    Parameters
    ----------
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
    c = float(cc)
    if not 0.0 < c <= 1.0:
        raise ValueError(f"sudden_contraction {name!r}: cc must be in (0, 1]; got {cc}")
    return CompositeElementSpec(
        name=name,
        sub_elements=[
            isentropic_area_change(name=f"{name}.contract"),
            sudden_area_change(name=f"{name}.borda", eps=eps),
        ],
        # isentropic acceleration to the vena contracta cc*A2, then Borda re-expansion
        internal_edges=[(0, 1, lambda a_up, a_down: c * a_down)],
        kind="sudden_contraction",
        params={"cc": c},
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
        params={"volume": V, "neck_length": ln, "neck_area": an},
    )


# ---------------------------------------------------------------------------
# Composite elements (Class-2 discretization): a continuous 1-D element resolved
# as an N-segment chain of compact atoms.  N is a fidelity knob -- the chain
# converges to the true distributed solution as N grows, and grid-refinement (solve
# at N and 2N) *is* the verification.  Same element-independent expander as Class 1.
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
    return CompositeElementSpec(
        name=name,
        sub_elements=subs,
        internal_edges=internal,
        kind="fanno_pipe",
        params={"length": L, "diameter": D, "friction_factor": f, "n_segments": N},
    )


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
        params={"stations": [(float(x), float(a)) for x, a in zip(xs, areas)]},
    )


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
        base = el.name or ""
        # A factory default always starts numbered ("inlet" -> "inlet-1"); a name the caller chose
        # keeps its bare form and is suffixed only on an actual clash.  ``name_auto`` records which
        # (set by the factory wrapper); a spec built directly, without it, counts as chosen.
        el.name = unique_name(base, seen, always_number=getattr(el, "name_auto", False))
        # The name is now concrete: clear the flag so a second pass over the same list is idempotent.
        el.name_auto = False
        seen.add(el.name)


def unique_name(name: str, taken, always_number: bool = False) -> str:
    """Return a copy of ``name`` not already in ``taken``.

    If ``name`` is free it is returned unchanged, unless ``always_number`` is set, in which case the
    first free ``<name>-1``, ``<name>-2``, ... is returned instead (so a lone factory default still
    reads ``inlet-1``).

    Parameters
    ----------
    name : str
        The desired base name.
    taken : container of str
        The names already in use.  The result is *not* added to it -- the caller records it,
        which keeps this usable both for a running build and for a one-shot pass over a list.
    always_number : bool, optional
        Force a ``-k`` suffix even when ``name`` is free (default ``False``).

    Returns
    -------
    str
        A name not present in ``taken``.
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
    :func:`duct`, ``inlet`` for :func:`mass_flow_inlet`, ...); this collects those defaults.
    Informational -- the dedup pass numbers factory defaults off each spec's ``name_auto`` flag
    (set by the factory wrapper), not off this set.  Cached: the signatures are fixed at import.
    """
    bases = set()
    module = sys.modules[__name__]
    for obj in vars(module).values():
        if not (inspect.isfunction(obj) and getattr(obj, "__module__", None) == __name__):
            continue
        param = inspect.signature(obj).parameters.get("name")
        if param is not None and isinstance(param.default, str) and param.default:
            bases.add(param.default)
    return frozenset(bases)


def _track_default_name(factory):
    """Wrap a catalog factory so its returned spec records whether ``name`` was defaulted.

    When the caller does not pass ``name``, the returned spec's ``name_auto`` is set ``True``, so the
    dedup pass (:func:`ensure_unique_names`, :meth:`nefes.shell.Network.add`) numbers a lone factory
    default (``duct`` -> ``duct-1``) while leaving an explicitly chosen name bare.  A user-chosen name
    equal to the default (``name="duct"``) is kept as-is when free.
    """
    sig = inspect.signature(factory)

    @wraps(factory)
    def wrapper(*args, **kwargs):
        spec = factory(*args, **kwargs)
        try:
            explicit = "name" in sig.bind(*args, **kwargs).arguments
        except TypeError:
            explicit = "name" in kwargs
        if hasattr(spec, "name_auto"):
            spec.name_auto = not explicit
        return spec

    return wrapper


def _install_default_name_tracking() -> None:
    """Wrap every catalog factory (a module function with a defaulted string ``name=``) so a lone
    factory default is numbered while an explicitly chosen name is kept.  Runs once at import."""
    module = sys.modules[__name__]
    for attr, obj in list(vars(module).items()):
        if not (inspect.isfunction(obj) and obj.__module__ == __name__):
            continue
        param = inspect.signature(obj).parameters.get("name")
        if param is not None and isinstance(param.default, str) and param.default:
            setattr(module, attr, _track_default_name(obj))


_install_default_name_tracking()
