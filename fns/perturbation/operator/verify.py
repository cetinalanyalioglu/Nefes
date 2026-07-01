"""Pre-assembly checks for the acoustic layer.

The acoustic operator is assembled on top of a converged mean state; a handful of
preconditions must hold before any duct phase stamp is built.  ``verify_acoustic``
raises a clear ``ValueError`` if one is violated, so failures surface at the
boundary rather than as a malformed operator.

v1 (subsonic, flow-aligned ducts) requires, for every DUCT node:

* degree 2 with the **pinned orientation** ``orient == (-1, +1)`` -- port 0 points
  into the duct (tail station), port 1 out of it (head station).  This makes the
  duct axis the flow axis, so the characteristic decomposition needs no
  sign/label-swap algebra (the UI constrains port 0 to be the inlet side);
* a positive ``length``;
* a strictly **subsonic** mean state on both incident edges (supersonic acoustic
  propagation is deferred with the supersonic mean flow).

Boundary flow reversal (the mean flow entering at a head terminal or leaving at a
tail) is supported: genuine inlet/outlet are read from the mean flow direction, so
the entropy seat and the duct entropy-phase row follow the flow rather than the
element geometry (see ``response._seats_entropy`` and ``stamps.build_duct_stamps``).
"""

from ...solver.control import states_table
from ...assembly.derive import ES_M
from ...elements.ids import ACOUSTIC_DUCT


def duct_nodes(prob):
    """Indices of nodes carrying the duct acoustic face."""
    return [n for n in range(prob.n_nodes) if int(prob.node_acoustic_id[n]) == ACOUSTIC_DUCT]


def verify_acoustic(prob, x_bar):
    """Raise ``ValueError`` unless ``prob`` admits a v1 acoustic assembly at ``x_bar``."""
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
                    "supersonic acoustic propagation is deferred in v1"
                )
