"""Residual / variable nondimensionalization scales.

The Newton solve nondimensionalizes the system by a characteristic magnitude per
equation row (``res_scale``) and per variable column (``var_scale``).  Historically
these were fixed boundary references frozen into the :class:`~fns.problem.CompiledProblem`
at compile time.  :func:`compose_scales` builds them from four kind-scales (mass,
pressure, enthalpy, composition); :func:`measure_inflow_scales` reads the mass and
enthalpy scales off a state vector so the solve can *adapt* them to the realized flow
(see :func:`fns.solver.control.solve`).  The pressure scale stays the user gauge anchor
``p_ref`` and the composition scale stays ``1`` (mixture fractions are O(1)).
"""

import numpy as np

from ..elements.ids import KIND_MASS, row_kind_tags
from ..elements.ids import MASS_FLOW_INLET, PT_INLET

# row layout: the node band-1 rows (mass / pressure), then the h_t transport rows
# (one per edge), then the composition transport rows (n_elem per edge).


def compose_scales(node_rid, degrees, n_edges, n_scalars, mass, p, h, z=1.0):
    """Assemble ``(res_scale, var_scale)`` from the four kind-scales.

    Parameters
    ----------
    node_rid : sequence of int
        Per-node residual id (``CompiledProblem.node_rid``).
    degrees : sequence of int
        Per-node degree (port count).
    n_edges, n_scalars : int
        Edge count and the number of transported scalars *beyond* ``h_t`` -- the
        composition mixture fractions plus the optional burnt marker (``n_solve - 3``).
    mass, p, h, z : float
        Characteristic scales for the mass-flux, pressure, total-enthalpy and
        composition/marker rows/variables.

    Returns
    -------
    res_scale : ndarray
        One scale per residual row, in row order.
    var_scale : ndarray
        One scale per band-1 variable ``(mdot, p, h_t, Z..., marker)``.
    """
    res = []
    for n in range(len(node_rid)):
        for tag in row_kind_tags(int(node_rid[n]), int(degrees[n])):
            res.append(mass if tag == KIND_MASS else p)
    res.extend([h] * n_edges)
    res.extend([z] * n_edges * n_scalars)
    var = np.array([mass, p, h] + [z] * n_scalars, dtype=np.float64)
    return np.array(res, dtype=np.float64), var


def _degrees(prob):
    return np.diff(prob.row_ptr)


def inlet_edges(prob):
    """Edge ids incident to a mass-flow / total-pressure inlet (the domain's feeds)."""
    edges = []
    for n in range(prob.n_nodes):
        if int(prob.node_rid[n]) in (MASS_FLOW_INLET, PT_INLET):
            edges.extend(int(e) for e in prob.col_edge[prob.row_ptr[n] : prob.row_ptr[n + 1]])
    return edges


def measure_inflow_scales(prob, x2d, seed_mass, seed_h):
    """Mass and enthalpy scales measured from the realized inflow in ``x2d``.

    The mass scale is the total inlet mass flow ``sum |mdot|`` and the enthalpy scale is
    the mass-weighted mean inlet ``|h_t|``.  When the network carries no inlet, or the
    inflow is (near) zero -- the quiescent ``mdot = 0`` case, which must keep working --
    the seed scales are returned unchanged, so the norm never collapses onto a vanishing
    measured scale.

    Parameters
    ----------
    prob : CompiledProblem
        The compiled network.
    x2d : ndarray
        Current state, shape ``(n_solve, n_edges)``.
    seed_mass, seed_h : float
        Fallback scales (the build-time references) used when no inflow is realized.

    Returns
    -------
    (float, float)
        ``(mass_scale, h_scale)``.
    """
    edges = inlet_edges(prob)
    if not edges:
        return seed_mass, seed_h
    mdot = np.abs(np.real(x2d[0, edges]))
    mass = float(mdot.sum())
    if mass <= 1e-9 * seed_mass:  # quiescent / explicit mdot = 0 -> keep the seed scales
        return seed_mass, seed_h
    h_flux = float(np.abs(np.real(x2d[0, edges]) * np.real(x2d[2, edges])).sum())
    h = h_flux / mass
    return mass, max(h, 1e-6 * seed_h)
