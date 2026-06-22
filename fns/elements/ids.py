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

# Fixed n_ports for fixed-arity elements (None -> variable: junction/splitter).
FIXED_NPORTS = {
    MASS_FLOW_INLET: 1,
    PT_INLET: 1,
    P_OUTLET: 1,
    WALL: 1,
    ISEN_AREA_CHANGE: 2,
    SUDDEN_AREA_CHANGE: 2,
    LOSS: 2,
    DUCT: 2,
}

# Whether an element permits an area change across it (its incident edges may
# carry different areas).  Area changes are physically carried by the dedicated
# area-change elements; the constant-area elements -- the duct (theory.md s12.3)
# and the concentrated loss (whose K-referenced loss assumes equal up/downstream
# area, modeling-guide.md s4) -- require one shared area across both ports.
# Junctions/splitters are manifolds and impose no area-equality constraint.
# Single-port boundaries are exempt (one edge, nothing to compare).  Elements
# absent from this map default to True (unconstrained); add an entry here when a
# new element type needs the equal-area rule enforced.
ALLOWS_AREA_CHANGE = {
    MASS_FLOW_INLET: True,
    PT_INLET: True,
    P_OUTLET: True,
    WALL: True,
    ISEN_AREA_CHANGE: True,
    SUDDEN_AREA_CHANGE: True,
    JUNCTION: True,
    SPLITTER: True,
    DUCT: False,
    LOSS: False,
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
}
