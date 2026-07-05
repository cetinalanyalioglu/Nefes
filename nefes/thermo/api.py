"""Thermo boundary: the single gateway through which all gas models are reached.

The gas model is selected by an integer ``model_id``, so the registry, assembly
kernel, and solver stay model-agnostic.  Two entry points share that dispatch:

* :func:`thermo_update` fills the band-2 fields ``(T, rho, c, W)`` in place from a
  thermodynamic point ``(Z_el, h, p)`` -- ``Z_el`` being the transported
  composition vector (the feed-stream mixture fractions ``xi`` for a reacting gas).
* :func:`thermo_state` returns the same ``(T, rho, c, W)`` as a scalar tuple on the
  hot recover path, and :func:`thermo_total_pressure` maps a Mach number to the
  isentropic total pressure.

Derived quantities beyond band-2 (mixture molar mass, specific heat, caloric
derivatives) are formed downstream from the recovered state in
:mod:`nefes.assembly.recover`, not here.
"""

from numba import njit

from .equilibrium import eq_frozen_state, eq_kernel_state, eq_total_pressure
from .perfect_gas import pg_update, pg_state, pg_total_pressure

# --- model ids -------------------------------------------------------------
PERFECT_GAS = 0  # The standard perfect gas model
EQ_KERNEL = 1  # thermolib element-potential HP equilibrium (burnt side)
EQ_TABLE = 2  # reserved: precomputed equilibrium table
EQ_FROZEN = 3  # thermolib frozen real-gas of the reactant composition (unburnt side)
EQ_MARKER = 4  # burnt-marker-gated blend of EQ_FROZEN (b=0) and EQ_KERNEL (b=1)

# --- evaluation modes (how much of `out` to fill) --------------------------
MODE_STATE = 0  # T, rho, c, W
MODE_SPECIES = 1  # + species mass fractions (reactive, later)
MODE_RATES = 2  # + net production rates (reactive, later)

# --- band-2 output slot layout (what thermo_update writes) -----------------
T_OUT = 0
RHO_OUT = 1
C_OUT = 2
W_OUT = 3
N_THERMO_OUT = 4


@njit(cache=True)
def thermo_update(model_id, tf, ti, Z_el, h, p, mode, out):
    """Fill ``out`` with band-2 thermo fields for the selected model."""
    if model_id == PERFECT_GAS:
        pg_update(tf, ti, Z_el, h, p, mode, out)
    else:
        raise ValueError("unknown thermo model_id")


@njit(cache=True)
def thermo_state(model_id, tf, ti, Z_el, h, p):
    """Return scalar ``(T, rho, c, W)`` from a thermodynamic point (hot path)."""
    if model_id == PERFECT_GAS:
        return pg_state(tf, h, p)
    if model_id == EQ_KERNEL:
        return eq_kernel_state(tf, ti, Z_el, h, p)
    if model_id == EQ_FROZEN:
        return eq_frozen_state(tf, ti, Z_el, h, p)
    raise ValueError("unknown thermo model_id")


@njit(cache=True)
def thermo_total_pressure(model_id, tf, M, p, T, c, W):
    """Return isentropic total pressure from static pressure and Mach number.

    Parameters
    ----------
    model_id : int
        Gas-model selector.
    tf : numpy.ndarray
        Packed model data (the perfect gas reads ``gamma`` from it).
    M : float or complex
        Local Mach number.
    p : float or complex
        Static pressure [Pa].
    T, c, W : float or complex
        Recovered band-2 temperature [K], sound speed [m/s] and molar mass
        [kg/mol]; a variable-gamma gas (equilibrium/frozen) forms ``gamma`` from
        them.  The perfect gas ignores them and uses its constant-gamma closed
        form, so its result is unchanged.

    Returns
    -------
    float or complex
        Total pressure [Pa].
    """
    if model_id == PERFECT_GAS:
        return pg_total_pressure(tf, M, p)
    if model_id == EQ_KERNEL or model_id == EQ_FROZEN or model_id == EQ_MARKER:
        # variable-gamma isentropic relation from the (blended, for EQ_MARKER) band-2 fields
        return eq_total_pressure(M, p, T, c, W)
    raise ValueError("unknown thermo model_id")
