"""Element residual-type ids and per-type metadata (Python-side constants).

The integer ``residual_id`` is the @njit dispatch key (a big switch that
constant-folds).  Supersonic boundaries are reserved but not implemented in v1.
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

# Acoustic-face ids (implementation-plan.md s8.3): which acoustic stamp an
# element overrides its default CSD face with.  Only DUCT is active in v1;
# VOLUME (storage M) and FLAME (sources S) are reserved provisions.
ACOUSTIC_DEFAULT = 0  # contributes only through J_alg (the CSD linearization)
ACOUSTIC_DUCT = 1  # phase-propagation stamp P(omega)
ACOUSTIC_VOLUME = 2  # finite-volume storage stamp M (reserved)
ACOUSTIC_FLAME = 3  # heat-release source stamp S(omega) (reserved)

# Equation-kind tags (for residual-row scaling); mirror prototype KIND_*.
KIND_MASS = 0
KIND_PRESSURE = 1
KIND_ENTHALPY = 2

# Human-readable kind names, for per-equation residual reporting.
KIND_NAMES = {KIND_MASS: "mass", KIND_PRESSURE: "pressure", KIND_ENTHALPY: "enthalpy"}


def row_kind_tags(rid, deg):
    """Equation-kind tag (``KIND_*``) for each band-1 residual row of an element.

    The single source of truth for the per-row kinds; both the residual-row scaling
    (:func:`fns.elements.catalog._row_kinds`) and the per-equation residual report
    (:func:`fns.solver.control.residual_labels`) derive from it, so they cannot drift.

    Parameters
    ----------
    rid : int
        The element's ``residual_id``.
    deg : int
        The element's degree (its number of band-1 residual rows = its port count).

    Returns
    -------
    list of int
        One ``KIND_*`` tag per residual row, in row order.
    """
    if rid in (MASS_FLOW_INLET, WALL, MASS_FLOW_OUTLET, CHOKED_NOZZLE_OUTLET):
        return [KIND_MASS]  # single mass-flux row (WALL pins mdot = 0; outlets pin a mass-flux residual)
    if rid in (PT_INLET, P_OUTLET):
        return [KIND_PRESSURE]  # single absolute-pressure row
    # interior element: a mass balance plus (deg - 1) pressure-coupling rows
    return [KIND_MASS] + [KIND_PRESSURE] * (deg - 1)


# Fixed n_ports for fixed-arity elements (None -> variable: junction/splitter).
FIXED_NPORTS = {
    MASS_FLOW_INLET: 1,
    PT_INLET: 1,
    P_OUTLET: 1,
    MASS_FLOW_OUTLET: 1,
    CHOKED_NOZZLE_OUTLET: 1,
    WALL: 1,
    ISEN_AREA_CHANGE: 2,
    SUDDEN_AREA_CHANGE: 2,
    LOSS: 2,
    DUCT: 2,
    FLAME_HEAT_RELEASE: 2,
    FLAME_EQUILIBRIUM: 2,
    MASS_SOURCE: 2,
}

# Whether an element permits an area change across it (its incident edges may
# carry different areas).  Most elements are area-agnostic: the dedicated
# area-change elements carry the static<->dynamic conversion, and the concentrated
# loss reconstructs each port's static state from that port's own area, so it may
# straddle an area change (the loss rides on top of an isentropic area change; its
# K is pinned to a definite port via catalog.loss's ref_port).  The duct alone is
# truly constant-area (theory.md s12.3): its mean face is equal-area continuity,
# so its two ports must share one area.  Junctions/splitters are manifolds and
# impose no area-equality constraint.  Single-port boundaries are exempt (one
# edge, nothing to compare).  Elements absent from this map default to True
# (unconstrained); add an entry here when a new element type needs the equal-area
# rule enforced.
ALLOWS_AREA_CHANGE = {
    MASS_FLOW_INLET: True,
    PT_INLET: True,
    P_OUTLET: True,
    MASS_FLOW_OUTLET: True,
    CHOKED_NOZZLE_OUTLET: True,
    WALL: True,
    ISEN_AREA_CHANGE: True,
    SUDDEN_AREA_CHANGE: True,
    JUNCTION: True,
    SPLITTER: True,
    DUCT: False,
    LOSS: True,
    FLAME_HEAT_RELEASE: False,  # constant-area compact flame (Pt-continuity pressure row)
    FLAME_EQUILIBRIUM: False,  # constant-area compact reacting flame (static-p continuity)
    MASS_SOURCE: False,  # constant-area inline injection (momentum balance with a source)
}

# Human-readable element-type names, for validation / reporting messages.
RESIDUAL_NAMES = {
    MASS_FLOW_INLET: "MassFlowInlet",
    PT_INLET: "TotalPressureInlet",
    P_OUTLET: "PressureOutlet",
    ISEN_AREA_CHANGE: "IsentropicAreaChange",
    SUDDEN_AREA_CHANGE: "SuddenAreaChange",
    LOSS: "LossElement",
    JUNCTION: "JunctionStaticP",
    SPLITTER: "LosslessSplitter",
    DUCT: "Duct",
    SUPERSONIC_INLET: "SupersonicInlet",
    SUPERSONIC_OUTLET: "SupersonicOutlet",
    WALL: "Wall",
    FLAME_HEAT_RELEASE: "HeatReleaseFlame",
    FLAME_EQUILIBRIUM: "EquilibriumFlame",
    MASS_SOURCE: "MassSource",
    MASS_FLOW_OUTLET: "MassFlowOutlet",
    CHOKED_NOZZLE_OUTLET: "ChokedNozzleOutlet",
}
