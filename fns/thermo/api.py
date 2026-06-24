"""Thermo boundary: the single gateway through which all gas models are reached.

``thermo_update`` dispatches on an integer ``model_id`` to a model kernel that
fills band-2 fields ``(T, rho, c, W)`` from a thermodynamic point
``(Z_el, h, p)`` -- ``Z_el`` being the transported composition vector (the
feed-stream mixture fractions ``xi`` for the reacting model, which the equilibrium
kernels map to elements / feed-species moles).  Only ``PerfectGas`` is implemented in v1, but the signature
and the integer dispatch are fixed so an equilibrium ``thermolib`` backend drops
in later as an extra branch -- without touching the registry, the assembly
kernel, or the solver (reactive-flow AD-3).
"""

from numba import njit

from .equilibrium import eq_frozen_state, eq_kernel_state, eq_total_pressure
from .perfect_gas import pg_update, pg_state, pg_total_pressure

# --- model ids -------------------------------------------------------------
PERFECT_GAS = 0
EQ_KERNEL = 1  # thermolib element-potential HP equilibrium (burnt side)
EQ_TABLE = 2  # reserved: precomputed equilibrium table
EQ_FROZEN = 3  # thermolib frozen real-gas of the reactant composition (unburnt side)

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
def thermo_total_pressure(model_id, tf, ti, Z_el, M, p, T, c, W):
    """Return total pressure from static pressure and Mach (isentropic).

    ``(T, c, W)`` are the already-recovered band-2 fields; a variable-gamma gas
    (equilibrium/frozen) needs them to form ``gamma`` -- the perfect gas ignores
    them and uses its constant-gamma closed form, so its result is unchanged.
    """
    if model_id == PERFECT_GAS:
        return pg_total_pressure(tf, M, p)
    if model_id == EQ_KERNEL or model_id == EQ_FROZEN:
        return eq_total_pressure(M, p, T, c, W)
    raise ValueError("unknown thermo model_id")
