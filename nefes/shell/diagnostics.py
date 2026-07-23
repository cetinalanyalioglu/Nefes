"""Static well-posedness diagnostics run on a network's wiring before it is solved.

These checks read the topology only (never a solved state) to catch setups that compile but
have no unique steady flow, so the cause is reported up front rather than surfacing as an
opaque non-convergence.  The current check is the under-pinned junction: near its
``recovery = 1`` limit a junction on the geometry-free recovery closure adds no flow resistance
of its own, so each inflow must be pinned by the network; two total-pressure sources reaching it
through no resistance leave the division of flow among the inflows undetermined.

Exports: ``diagnose_junctions``.
"""

from typing import List

from ..elements.ids import (
    CHOKED_NOZZLE_OUTLET,
    DUCT,
    FLAME_EQUILIBRIUM,
    FLAME_HEAT_RELEASE,
    ISEN_AREA_CHANGE,
    JUNCTION,
    LINEAR_RESISTANCE,
    LOSS,
    MASS_FLOW_INLET,
    MASS_FLOW_OUTLET,
    MASS_SOURCE,
    PIPE,
    PT_INLET,
    SUDDEN_AREA_CHANGE,
    TRANSFER_MATRIX,
)

# At or above this recovery the junction's self-supplied dump resistance is too weak to pin
# the flow split on its own, so an inflow reaching a fixed total pressure through no resistance
# leaves the merge ill-conditioned (and, at recovery = 1 exactly, with no unique steady flow).
# Below it the dump term conditions the split on any wiring, so the warning would be spurious.
SIGMA_PIN_WARN = 0.99

# A branch reaching one of these through no resistance is a fixed-pressure inflow the manifold
# cannot pin at high recovery.
UNPINNED_SOURCE = (PT_INLET,)

# Elements that fix the mass flow or drop total pressure with it: an inflow through one of these is
# pinned by the network, so the branch is well posed at any recovery.
PINNING = (
    MASS_FLOW_INLET,
    MASS_FLOW_OUTLET,
    LOSS,
    LINEAR_RESISTANCE,
    PIPE,
    SUDDEN_AREA_CHANGE,
    CHOKED_NOZZLE_OUTLET,
)

# Lossless two-port elements that pass the flow-pressure relation straight through: the walk
# continues past them to whatever feeds the branch.
PASS_THROUGH = (
    ISEN_AREA_CHANGE,
    DUCT,
    TRANSFER_MATRIX,
    FLAME_HEAT_RELEASE,
    FLAME_EQUILIBRIUM,
    MASS_SOURCE,
)


def _rid(net, node: int):
    """Residual id of an element, or ``None`` for a composite (opaque to this walk)."""
    return getattr(net._elements[node], "residual_id", None)


def _branch_reaches_unpinned_source(net, manifold: int, edge: int) -> bool:
    """Walk a branch outward from a manifold and report whether it is an unpinned inflow.

    Starting on ``edge`` leaving ``manifold``, follow lossless pass-through elements until a
    terminal element is reached.  The branch is an unpinned inflow only if that terminal is a
    fixed total-pressure source with no resistance anywhere along the way.
    """
    prev = manifold
    here = edge
    seen = set()
    while True:
        t, h = net.nodes_of(here)
        far = h if t == prev else t
        rid = _rid(net, far)
        if rid in UNPINNED_SOURCE:
            return True
        if rid in PINNING:
            return False
        if rid in PASS_THROUGH:
            others = [e for e in net.edges_of(far) if e != here]
            if len(others) != 1 or far in seen:
                # not a simple two-port pass-through (or a loop); stop on the safe side
                return False
            seen.add(far)
            prev = far
            here = others[0]
            continue
        # a pressure outlet, another manifold, a wall/cavity, or an unknown element: not an
        # unpinned inflow the manifold has to pin
        return False


def diagnose_junctions(net) -> List[str]:
    """Warning messages for junctions whose flow split is left under-determined.

    A junction on the geometry-free recovery closure at ``recovery`` above :data:`SIGMA_PIN_WARN`
    imposes only total-pressure equalities, so the network must pin every inflow's rate.  When two
    or more of its branches reach a total-pressure inlet through no resistance, the split among
    those inflows is undetermined and the solve is unlikely to converge.  A junction on the
    per-branch loss-coefficient closure carries its own branch resistance and is exempt.

    Parameters
    ----------
    net : Network
        The network to inspect (read only).

    Returns
    -------
    list of str
        One message per under-pinned junction (empty when none is found).
    """
    messages: List[str] = []
    for node, spec in enumerate(net._elements):
        if _rid(net, node) != JUNCTION:
            continue
        sel = float(spec.fparams[1])
        edges = net.edges_of(node)
        if sel < -2.5:
            # per-branch recovery: only the branches whose own factor is near 1 go unpinned
            sigmas = [float(s) for s in spec.fparams[2:]]
            candidates = [e for e, s in zip(edges, sigmas) if s >= SIGMA_PIN_WARN]
            shown = "recovery=[" + ", ".join(f"{s:g}" for s in sigmas) + "]"
        elif sel < -0.5 or len(spec.fparams) > 2:
            # the common-static-pressure header, or the loss coefficients that pin the split
            continue
        else:
            if sel < SIGMA_PIN_WARN:
                continue
            candidates = edges
            shown = f"recovery={sel:g}"
        unpinned = [e for e in candidates if _branch_reaches_unpinned_source(net, node, e)]
        if len(unpinned) < 2:
            continue
        feeds = []
        for e in unpinned:
            t, h = net.nodes_of(e)
            other = h if t == node else t
            feeds.append(net._elements[other].name)
        messages.append(
            f"junction '{spec.name}' has {shown}, but {len(unpinned)} of its "
            f"inflow branches reach a fixed total-pressure inlet ({', '.join(sorted(feeds))}) with "
            f"no flow resistance in between. Near recovery = 1 the junction adds no flow "
            f"resistance of its own, so the division of flow among these inflows is not pinned: the "
            f"solve is ill-conditioned approaching recovery = 1 and has no unique steady flow at "
            f"recovery = 1. Prescribe each inflow's rate with a mass-flow inlet, add a resistance in "
            f"each branch (a loss, orifice, or pipe), or lower the recovery toward 0 (the robust "
            f"full dump)."
        )
    return messages
