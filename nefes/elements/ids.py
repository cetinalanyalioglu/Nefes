"""Element residual-type ids and per-type metadata (Python-side constants).

The integer ``residual_id`` is the dispatch key the compiled (``@njit``) kernels
branch on to select an element's residual and donor.
"""

MASS_FLOW_INLET = 0
PT_INLET = 1
P_OUTLET = 2
ISEN_AREA_CHANGE = 3
SUDDEN_AREA_CHANGE = 4
LOSS = 5
JUNCTION = 6
SPLITTER = 7
DUCT = 8
SUPERSONIC_INLET = 9  # reserved (deferred)
SUPERSONIC_OUTLET = 10  # reserved (deferred)
WALL = 11  # impermeable single-port termination (mdot = 0); acoustic hard wall
FLAME_HEAT_RELEASE = 12  # 2-port heat-addition flame (perfect-gas): h_t jump from a power source
FLAME_EQUILIBRIUM = 13  # 2-port reacting flame: frozen inflow -> equilibrium products (closure switch)
MASS_SOURCE = 14  # 2-port mass-injection element: adds mass/momentum/energy + composition source
MASS_FLOW_OUTLET = 15  # 1-port outlet: prescribed outflow mass rate (acoustic mdot' = 0 by inheritance)
CHOKED_NOZZLE_OUTLET = 16  # 1-port outlet: compact choked nozzle of throat area A* (critical mass flux)
LINEAR_RESISTANCE = 17  # 2-port linear flow resistance: Pt drop proportional to mass flow (Pt0 - Pt1 = R*mdot)
CAVITY = 18  # 1-port finite-volume cavity: mean flow is a wall (mdot = 0); acoustically a compliance (storage M)
FORCED_SPLITTER = 19  # flow-divider manifold: 1 inflow (port 0) + N outflows, (N-1) outflow rates forced to fractions
PIPE = 20  # 2-port length-bearing pipe: Darcy-Weisbach friction (Pt drop K = f*L/D) + the duct acoustic phase
TRANSFER_MATRIX = 21  # 2-port element: mean flow == isentropic area change; perturbation uses a user transfer matrix

# Acoustic-face ids: the acoustic stamp an element uses in place of its default
# CSD face.  The dynamic-source S is carried on the element's DynamicSource
# descriptor, not this tag.
ACOUSTIC_DEFAULT = 0  # contributes only through J_alg (the CSD linearization)
ACOUSTIC_DUCT = 1  # phase-propagation stamp P(omega)
ACOUSTIC_VOLUME = 2  # finite-volume storage stamp M (the cavity compliance)
ACOUSTIC_FLAME = 3  # heat-release source stamp S(omega) (reserved)

# Equation-kind tags (for residual-row scaling); mirror prototype KIND_*.
KIND_MASS = 0
KIND_PRESSURE = 1
KIND_ENTHALPY = 2

# Human-readable kind names, for per-equation residual reporting.
KIND_NAMES = {KIND_MASS: "mass", KIND_PRESSURE: "pressure", KIND_ENTHALPY: "enthalpy"}


def row_kind_tags(rid, deg):
    """Equation-kind tag (``KIND_*``) for each balance row an element emits.

    An element emits one balance row per incident edge (its *band-1* rows -- the mass
    / pressure / enthalpy balances the solver carries, as distinct from the *band-2*
    thermodynamic fields the recovery reconstructs).  This is the single source of
    truth for their kinds; both the residual-row scaling
    (:func:`nefes.shell.build._row_kinds`) and the per-equation residual report
    (:func:`nefes.solver.report.residual_labels`) derive from it, so they cannot drift.

    Parameters
    ----------
    rid : int
        The element's ``residual_id``.
    deg : int
        The element's degree (its number of balance rows = its port count).

    Returns
    -------
    list of int
        One ``KIND_*`` tag per residual row, in row order.
    """
    if rid in (MASS_FLOW_INLET, WALL, CAVITY, MASS_FLOW_OUTLET, CHOKED_NOZZLE_OUTLET):
        return [KIND_MASS]  # single mass-flux row (WALL/CAVITY pin mdot = 0; outlets pin a mass-flux residual)
    if rid in (PT_INLET, P_OUTLET):
        return [KIND_PRESSURE]  # single absolute-pressure row
    if rid == FORCED_SPLITTER:
        # net mass balance + (deg - 2) forced-fraction (mass-flux) rows + 1 remainder
        # pressure-coupling row; the fraction rows are mass-flow dimensioned.
        return [KIND_MASS] * (deg - 1) + [KIND_PRESSURE]
    # interior element: a mass balance plus (deg - 1) pressure-coupling rows
    return [KIND_MASS] + [KIND_PRESSURE] * (deg - 1)


# Port count of each element with a fixed number of ports (absent from this map ->
# variable port count: the junction / splitter manifolds).
FIXED_NPORTS = {
    MASS_FLOW_INLET: 1,
    PT_INLET: 1,
    P_OUTLET: 1,
    MASS_FLOW_OUTLET: 1,
    CHOKED_NOZZLE_OUTLET: 1,
    WALL: 1,
    CAVITY: 1,
    ISEN_AREA_CHANGE: 2,
    SUDDEN_AREA_CHANGE: 2,
    LOSS: 2,
    DUCT: 2,
    FLAME_HEAT_RELEASE: 2,
    FLAME_EQUILIBRIUM: 2,
    MASS_SOURCE: 2,
    LINEAR_RESISTANCE: 2,
    PIPE: 2,
    TRANSFER_MATRIX: 2,
}

# Whether an element permits an area change across it (its incident edges may carry
# different areas).  Most elements are area-agnostic; the exceptions are the ones
# whose mean face assumes equal-area continuity -- the constant-area duct / pipe and
# the compact flames / mass source.  Elements absent from this map default to ``True``
# (unconstrained); add an entry when a new element type needs the equal-area rule.
ALLOWS_AREA_CHANGE = {
    MASS_FLOW_INLET: True,
    PT_INLET: True,
    P_OUTLET: True,
    MASS_FLOW_OUTLET: True,
    CHOKED_NOZZLE_OUTLET: True,
    WALL: True,
    CAVITY: True,  # single port: nothing to compare (its volume tag carries no area constraint)
    ISEN_AREA_CHANGE: True,
    SUDDEN_AREA_CHANGE: True,
    JUNCTION: True,
    SPLITTER: True,
    FORCED_SPLITTER: True,  # manifold (flow divider); imposes no area-equality constraint
    DUCT: False,
    PIPE: False,  # constant-area length-bearing pipe (like the duct, both ports share one area)
    LOSS: True,
    FLAME_HEAT_RELEASE: False,  # constant-area compact flame (Pt-continuity pressure row)
    FLAME_EQUILIBRIUM: False,  # constant-area compact reacting flame (static-p continuity)
    MASS_SOURCE: False,  # constant-area inline injection (momentum balance with a source)
    LINEAR_RESISTANCE: True,  # area-agnostic: the Pt drop is set by mass flow, not area
    TRANSFER_MATRIX: True,  # mean flow is an isentropic area change (contraction/diffuser allowed)
}

# Human-readable element-type names, for validation / reporting messages.
ELEMENT_TYPE_NAMES = {
    MASS_FLOW_INLET: "MassFlowInlet",
    PT_INLET: "TotalPressureInlet",
    P_OUTLET: "PressureOutlet",
    ISEN_AREA_CHANGE: "IsentropicAreaChange",
    SUDDEN_AREA_CHANGE: "SuddenAreaChange",
    LOSS: "LossElement",
    JUNCTION: "JunctionStaticP",
    SPLITTER: "LosslessSplitter",
    FORCED_SPLITTER: "ForcedSplitter",
    DUCT: "Duct",
    SUPERSONIC_INLET: "SupersonicInlet",
    SUPERSONIC_OUTLET: "SupersonicOutlet",
    WALL: "Wall",
    CAVITY: "Cavity",
    FLAME_HEAT_RELEASE: "HeatReleaseFlame",
    FLAME_EQUILIBRIUM: "EquilibriumFlame",
    MASS_SOURCE: "MassSource",
    MASS_FLOW_OUTLET: "MassFlowOutlet",
    CHOKED_NOZZLE_OUTLET: "ChokedNozzleOutlet",
    LINEAR_RESISTANCE: "LinearResistance",
    PIPE: "Pipe",
    TRANSFER_MATRIX: "TransferMatrixElement",
}

# Elements that introduce a feed stream (a distinct injected composition): the boundaries
# and the mass source.  Shared by the builder (stream discovery) and the post-solve
# per-edge chemistry recovery.
STREAM_INTRODUCING = (MASS_FLOW_INLET, PT_INLET, P_OUTLET, MASS_SOURCE)

# Single-port boundary terminations (one equation row, one incident edge): the inlets,
# outlets and wall.  Shared by the perturbation terminals and the YAML writer.
BOUNDARY_RIDS = (MASS_FLOW_INLET, PT_INLET, P_OUTLET, MASS_FLOW_OUTLET, CHOKED_NOZZLE_OUTLET, WALL)
