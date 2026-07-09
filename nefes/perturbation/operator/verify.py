"""Pre-assembly checks for the acoustic layer.

The acoustic operator is assembled on top of a converged mean state; a handful of
preconditions must hold before any duct phase stamp is built.  ``verify_acoustic``
raises a clear ``ValueError`` if one is violated, so failures surface at the
boundary rather than as a malformed operator.

For every duct node the acoustic assembly requires:

* degree 2 with the **pinned orientation** ``orient == (-1, +1)`` -- port 0 points
  into the duct (tail station), port 1 out of it (head station).  Fixing the axis
  this way lets the characteristic decomposition run without sign/label-swap
  algebra; it is a graph-construction convention and does **not** constrain the mean
  flow direction (the mean solve does not depend on direction, which is why it carries no
  such check -- see the boundary-reversal note below);
* a positive ``length``;
* a strictly **subsonic** mean state on both incident edges -- the acoustic layer
  models subsonic propagation, so a sonic or supersonic edge is rejected.

Boundary flow reversal (the mean flow entering at a head terminal or leaving at a
tail) is supported: the genuine inlet/outlet are read from the mean flow direction,
so the entropy seat and the duct entropy-phase row follow the flow rather than the
element geometry (see ``response._seats_entropy`` and ``stamps.build_duct_stamps``).
"""

from ...solver.report import states_table
from ...assembly.recover import ES_M
from ...elements.ids import ACOUSTIC_DUCT


def duct_nodes(prob):
    """Indices of nodes carrying the duct acoustic face."""
    return [n for n in range(prob.n_nodes) if int(prob.node_acoustic_id[n]) == ACOUSTIC_DUCT]


def verify_acoustic(prob, x_bar):
    """Raise ``ValueError`` unless ``prob`` admits a subsonic acoustic assembly at ``x_bar``.

    Parameters
    ----------
    prob : CompiledProblem
        The compiled network.
    x_bar : ndarray
        Converged mean-flow state, shape ``(n_solve, E)``.

    Raises
    ------
    ValueError
        If any duct node is not a 2-port, or the mean flow is not everywhere
        subsonic and flow-aligned, so the linear acoustic assembly is ineligible.
    """
    est = states_table(prob, x_bar)
    for n in duct_nodes(prob):
        base = int(prob.row_ptr[n])
        deg = int(prob.row_ptr[n + 1]) - base
        if deg != 2:
            raise ValueError(f"duct node {n} has degree {deg}; a duct must be a 2-port")
        e0 = int(prob.col_edge[base])
        s0 = int(prob.orient[base])
        e1 = int(prob.col_edge[base + 1])
        s1 = int(prob.orient[base + 1])
        if not (s0 == -1 and s1 == +1):
            raise ValueError(
                f"duct node {n} is not flow-aligned: port orientations are ({s0:+d}, {s1:+d}), "
                "expected (-1, +1) -- port 0 must point into the duct, port 1 out of it"
            )
        length = float(prob.npar_f[int(prob.npar_fptr[n])])
        if length <= 0.0:
            raise ValueError(f"duct node {n} has non-positive length {length}")
        for e in (e0, e1):
            mach = abs(float(est[ES_M, e]))
            if mach >= 1.0:
                raise ValueError(
                    f"duct node {n} edge {e} has mean Mach {mach:.3f} >= 1; "
                    "the acoustic layer supports subsonic propagation only"
                )
