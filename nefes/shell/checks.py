"""Toggleable network validation checks and their global on/off constants.

Every discretionary check the builder runs before compiling (and the post-solve report runs
after a solve) is gated by a module-level boolean here, so any of them can be disabled
process-wide by flipping the constant::

    import nefes.shell.checks as checks
    checks.CHECK_CONNECTED = False        # allow disconnected sub-networks everywhere

These are guardrails against likely modeling slips, not the hard structural invariants
(finite/positive areas, admissible port counts, an absolute-pressure reference), which are
always enforced because the mean-flow solve is ill-posed without them.

Public: the ``CHECK_*`` toggles, :func:`assert_single_component` and
:func:`assert_allowed_connections`.
"""

from collections import defaultdict

from ..graph.connectivity import Connectivity
from ..elements.ids import DISALLOWED_NEIGHBORS, ELEMENT_TYPE_NAMES

# --- global check toggles (flip to disable a check process-wide) ---
# Reject a model that splits into more than one disconnected sub-network.
CHECK_CONNECTED = True
# Reject an edge joining two element types flagged as incompatible neighbors.
CHECK_CONNECTIONS = True
# Post-solve: warn when a choked-nozzle outlet's back pressure is too high for it to choke.
CHECK_CHOKED_NOZZLE = True


def _label(elements, i: int) -> str:
    """Human-readable identifier for an element, for validation messages."""
    el = elements[i]
    name = f" {el.name!r}" if getattr(el, "name", "") else ""
    typ = ELEMENT_TYPE_NAMES.get(el.residual_id, f"residual {el.residual_id}")
    return f"element {i}{name} ({typ})"


def assert_single_component(conn: Connectivity) -> None:
    """Reject a model whose elements split into more than one disconnected sub-network.

    A single :class:`~nefes.shell.network.Network` must describe one connected flow circuit.
    Two independent sub-networks that share no edge are almost always a wiring slip -- a
    forgotten connection or a mistyped node index -- and would hand the steady solve an
    independent, separately-gauged block per component (each needing its own pressure
    reference), so the failure is easier to read here than as a singular Jacobian later.
    Edges are treated as undirected; raises ``ValueError`` naming the lowest node of each
    component when more than one is found.

    Parameters
    ----------
    conn : Connectivity
        The compiled connectivity (edge endpoints and node count).
    """
    n = conn.n_nodes
    if n <= 1:
        return
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]  # path halving
            a = parent[a]
        return a

    for e in range(conn.n_edges):
        ra, rb = find(int(conn.tail_node[e])), find(int(conn.head_node[e]))
        if ra != rb:
            parent[ra] = rb

    components = defaultdict(list)
    for node in range(n):
        components[find(node)].append(node)
    if len(components) > 1:
        reps = sorted(min(members) for members in components.values())
        raise ValueError(
            f"the model has {len(components)} disconnected sub-networks (representative nodes {reps}); "
            "a single network must be fully connected. Wire them together, split them into separate "
            "Network objects, or pass require_connected=False to allow it."
        )


def assert_allowed_connections(elements, conn: Connectivity) -> None:
    """Reject an edge joining two element types flagged as incompatible neighbors.

    Consults the symmetric :data:`~nefes.elements.ids.DISALLOWED_NEIGHBORS` table: an edge is
    rejected when either endpoint's element type is in the other's disallow set (so the result
    does not depend on which way the edge was drawn).  A guardrail against likely modeling slips
    -- e.g. two prescribed-inflow boundaries back to back, which would doubly fix the same
    edge's flow -- not a physical law; disable it with ``CHECK_CONNECTIONS = False``.

    Parameters
    ----------
    elements : list of ElementSpec
        The network elements, in node order.
    conn : Connectivity
        The compiled connectivity (edge endpoints).
    """
    for e in range(conn.n_edges):
        t, h = int(conn.tail_node[e]), int(conn.head_node[e])
        a, b = elements[t].residual_id, elements[h].residual_id
        if b in DISALLOWED_NEIGHBORS.get(a, ()) or a in DISALLOWED_NEIGHBORS.get(b, ()):
            raise ValueError(
                f"edge {e} connects {_label(elements, t)} and {_label(elements, h)}, an incompatible "
                "pairing (e.g. two prescribed-inflow boundaries back to back). Separate them with an "
                "interior element, or set nefes.shell.checks.CHECK_CONNECTIONS = False to allow it."
            )
