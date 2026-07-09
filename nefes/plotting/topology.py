"""Network topology diagram for visual diagnostics in a notebook.

Draws the element graph -- one node per element (labelled by index and name), one
arrow per directed edge -- using a simple longest-path **layered** layout so the
flow reads left to right.  It is a structural view (no solve needed): a quick way
to confirm the connectivity, element indices and edge directions before solving.

    from nefes.plotting import plot_network_topology
    plot_network_topology(net).show()      # or net.plot() / sol.plot(color_by="T")
"""

import numpy as np
from ._deps import go, sample_colorscale

from .theme import NEFES_TEMPLATE_NAME
from ..shell.network import _EDGE_FIELDS

# Per-edge solved field -> (display label, unit, value format) for the overlay
# colorbar / labels / hover.  Keyed to match :data:`nefes.shell.network._EDGE_FIELDS`
# one-for-one; the guard below fails at import if the two ever drift, so a field added
# to the network cannot silently lose (or invent) an overlay entry.
_FIELD_INFO = {
    "mdot": ("mass flow", "kg/s", ".3g"),
    "p": ("static pressure", "Pa", ".4g"),
    "h_t": ("total enthalpy", "J/kg", ".3g"),
    "rho": ("density", "kg/m³", ".3g"),
    "u": ("velocity", "m/s", ".4g"),
    "T": ("temperature", "K", ".0f"),
    "c": ("sound speed", "m/s", ".4g"),
    "M": ("Mach", "", ".3f"),
    "p_t": ("total pressure", "Pa", ".4g"),
    "area": ("area", "m²", ".3g"),
    "W": ("molar mass", "kg/mol", ".4g"),
    "cp": ("cp", "J/kgK", ".4g"),
}

if set(_FIELD_INFO) != set(_EDGE_FIELDS):
    _missing = sorted(set(_EDGE_FIELDS) - set(_FIELD_INFO))
    _extra = sorted(set(_FIELD_INFO) - set(_EDGE_FIELDS))
    raise RuntimeError(
        f"_FIELD_INFO is out of sync with network._EDGE_FIELDS (missing {_missing}, unexpected {_extra})"
    )

# Semantic node colours by element role (matched on the residual-type name).  A
# role keyword -> fill colour; anything unmatched is an interior element.
_ROLE_COLORS = (
    ("Inlet", "#10b981"),  # sources of flow: green
    ("Source", "#8b5cf6"),  # mass source: violet
    ("Outlet", "#ef4444"),  # sinks: red
    ("Flame", "#f97316"),  # heat release: orange
    ("Wall", "#6b7280"),  # dead end: grey
)
_INTERIOR_COLOR = "#2563eb"  # ducts, junctions, area changes, ...: blue


def _role_color(type_name):
    for key, color in _ROLE_COLORS:
        if key in type_name:
            return color, key.lower()
    return _INTERIOR_COLOR, "interior"


def _layers(n, edges):
    """Longest-path layer (x rank) of each node from the sources.

    Sources (no incoming edge) sit at layer 0; every other node is one past the
    deepest predecessor.  Relaxed up to ``n`` passes so a cyclic graph (e.g. a
    recirculation) still terminates, capping the layer at ``n - 1``.
    """
    incoming = [[] for _ in range(n)]
    for t, h in edges:
        if 0 <= t < n and 0 <= h < n:
            incoming[h].append(t)
    layer = [0] * n
    for _ in range(n):
        changed = False
        for v in range(n):
            best = 0
            for u in incoming[v]:
                best = max(best, min(layer[u] + 1, n - 1))
            if best != layer[v]:
                layer[v] = best
                changed = True
        if not changed:
            break
    return layer


def _positions(n, edges):
    """``(x, y)`` per node: x is the layer; y stacks the layer's nodes, centered."""
    layer = _layers(n, edges)
    by_layer = {}
    for v in range(n):
        by_layer.setdefault(layer[v], []).append(v)
    x = np.zeros(n)
    y = np.zeros(n)
    for lx, nodes in by_layer.items():
        nodes = sorted(nodes)  # stable within-layer order by index
        offset = (len(nodes) - 1) / 2.0
        for k, v in enumerate(nodes):
            x[v] = float(lx)
            y[v] = offset - k  # top-down, centered on 0
    return x, y


def _edge_field(network, solution, name, n_edges, kind):
    """Per-edge values of a field for an overlay (clear errors on misuse).

    ``area`` lives on the network edges, so it is available without a solve (the geometry-weighted
    diagram of an unsolved network); every other field is read from the converged ``solution``.
    """
    if name not in _FIELD_INFO:
        raise ValueError(f"{kind}={name!r} is not a known edge field; choose from {sorted(_FIELD_INFO)}")
    if name == "area":
        return np.asarray([a for (_t, _h, a) in network._edges[:n_edges]], dtype=float)
    vals = np.asarray(solution.field(name), dtype=float)
    # A composite network solves on an expanded graph (internal edges appended at the tail),
    # so the solution may carry MORE edges than the drawn (user) topology -- the user edges keep
    # ids 0..n_edges-1, so take that leading slice.  Fewer edges is a genuine mismatch.
    if vals.shape[0] < n_edges:
        raise ValueError(
            f"the solution has {vals.shape[0]} edges but the network has {n_edges}; "
            "pass the Solution from solving *this* network"
        )
    return vals[:n_edges]


def plot_network_topology(
    network,
    *,
    solution=None,
    color_by=None,
    width_by=None,
    colorscale="Viridis",
    show_edge_labels=True,
    show_areas=False,
    title=None,
    height=None,
    width=None,
):
    """Plot the element/edge topology of a :class:`~nefes.shell.network.Network`.

    With no ``solution`` this is a structural view (no solve needed): the element
    graph with one arrow per directed edge.  Pass a converged ``solution`` to overlay
    the solved field on the edges -- the edges carry the state in Nefes, so a solved
    quantity (temperature, Mach, mass flow, ...) is drawn *on the edges*: ``color_by``
    tints each edge (with a colorbar) and ``width_by`` scales its arrow width.

    Parameters
    ----------
    network : nefes.shell.network.Network
        The network to draw (elements and directed edges).
    solution : nefes.shell.network.Solution, optional
        A converged mean-flow solution of *this* network.  Required for ``color_by``
        / ``width_by``; when given it also enriches the edge hover with ``mdot``,
        ``T`` and ``M``.
    color_by : str, optional
        Edge field to color the edges by (e.g. ``"T"``, ``"M"``, ``"mdot"``; keys of
        :data:`nefes.shell.network._EDGE_FIELDS`).  Adds a colorbar and labels each edge
        with its value.  ``"area"`` needs no solve; every other field requires ``solution``.
    width_by : str, optional
        Edge field whose magnitude scales the arrow width (e.g. ``"mdot"`` for a
        flow-weighted diagram, ``"area"`` for a geometry-weighted one).  ``"area"`` needs no
        solve -- it is the width :meth:`Network.plot` uses by default; every other field
        requires ``solution``.
    colorscale : str, optional
        Plotly colorscale name for ``color_by`` (default ``"Viridis"``).
    show_edge_labels : bool, optional
        Label each edge at its midpoint (the edge index, or the ``color_by`` value
        when coloring; default ``True``).
    show_areas : bool, optional
        Append the edge area to the edge label (default ``False``; always in hover).
    title : str, optional
        Figure title.  Defaults to ``"Network topology"``, or names the overlaid
        field when ``color_by`` is set.
    height, width : int, optional
        Figure size in pixels.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    elements = network._elements
    edges = network._edges  # list of (tail, head, area)
    edge_names = network._edge_names
    n = len(elements)
    x, y = _positions(n, [(t, h) for (t, h, _a) in edges])

    # Every overlay field but ``area`` (which lives on the network) needs a converged solution.
    if any(f and f != "area" for f in (color_by, width_by)) and solution is None:
        raise ValueError("color_by / width_by need a converged `solution` of this network")

    # -- field overlays (edge-centric: Nefes state lives on edges; area lives on the network)
    cvals = _edge_field(network, solution, color_by, len(edges), "color_by") if color_by else None
    wvals = _edge_field(network, solution, width_by, len(edges), "width_by") if width_by else None
    edge_colors = None
    if cvals is not None and len(cvals):
        cmin, cmax = float(np.min(cvals)), float(np.max(cvals))
        span = (cmax - cmin) or 1.0
        edge_colors = sample_colorscale(colorscale, [(v - cmin) / span for v in cvals])
    edge_widths = None
    if wvals is not None and len(wvals):
        mag = np.abs(wvals)
        wmax = float(np.max(mag)) or 1.0
        edge_widths = [1.0 + 6.0 * (m / wmax) for m in mag]  # 1..7 px, scaled by magnitude
    sol_field = solution.field if solution is not None else None

    fig = go.Figure()

    # -- edges: an arrow per directed edge, plus a midpoint marker for hover/label
    annotations = []
    mids_x, mids_y, mid_text, mid_hover = [], [], [], []
    for i, (t, h, a) in enumerate(edges):
        annotations.append(
            dict(
                x=x[h],
                y=y[h],
                ax=x[t],
                ay=y[t],
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.2,
                arrowwidth=edge_widths[i] if edge_widths is not None else 1.4,
                arrowcolor=edge_colors[i] if edge_colors is not None else "#9aa5b1",
                standoff=14,  # stop short of the head marker
                startstandoff=14,  # start past the tail marker
                opacity=0.9,
            )
        )
        mids_x.append((x[t] + x[h]) / 2.0)
        mids_y.append((y[t] + y[h]) / 2.0)
        if cvals is not None:  # label the edge with its colored field value
            _lab, _unit, _fmt = _FIELD_INFO[color_by]
            label = format(cvals[i], _fmt)
        else:
            label = str(i)
        if show_areas:
            label += f"\nA={a:.3g}"
        mid_text.append(label)
        nm = edge_names[i] if i < len(edge_names) else f"e{i}"
        hov = f"edge {i} ({nm})<br>{network._node_label(t)} → {network._node_label(h)}<br>A = {a:.4g} m²"
        if sol_field is not None:  # enrich with the headline solved state
            hov += (
                f"<br>mdot = {sol_field('mdot')[i]:.3g} kg/s"
                f"<br>T = {sol_field('T')[i]:.0f} K"
                f"<br>M = {sol_field('M')[i]:.3f}"
            )
        mid_hover.append(hov)

    if edges:
        if cvals is not None:
            clabel, cunit, _ = _FIELD_INFO[color_by]
            marker = dict(
                size=16,
                color=cvals,
                colorscale=colorscale,
                cmin=float(np.min(cvals)),
                cmax=float(np.max(cvals)),
                showscale=True,
                colorbar=dict(title=clabel + (f" [{cunit}]" if cunit else ""), thickness=14),
                line=dict(width=0),
            )
        else:
            marker = dict(size=14, color="rgba(255,255,255,0.85)", line=dict(width=0))
        fig.add_trace(
            go.Scatter(
                x=mids_x,
                y=mids_y,
                mode="markers+text" if show_edge_labels else "markers",
                text=mid_text if show_edge_labels else None,
                textposition="middle center" if cvals is None else "top center",
                textfont=dict(size=10, color="#52606d"),
                marker=marker,
                hovertext=mid_hover,
                hoverinfo="text",
                showlegend=False,
                name="edges",
            )
        )

    # -- nodes: one scatter per role so the legend reads as a key
    roles = {}
    for i, el in enumerate(elements):
        color, role = _role_color(network._type_name(el))
        roles.setdefault((role, color), []).append(i)

    for (role, color), idxs in roles.items():
        fig.add_trace(
            go.Scatter(
                x=[x[i] for i in idxs],
                y=[y[i] for i in idxs],
                mode="markers+text",
                text=[f"{i}: {network._node_label(i)}" for i in idxs],
                textposition="bottom center",
                textfont=dict(size=11, color="#1f2933"),
                marker=dict(size=22, color=color, line=dict(width=1.5, color="#ffffff")),
                hovertext=[f"{i}: {network._node_label(i)} ({network._type_name(elements[i])})" for i in idxs],
                hoverinfo="text",
                name=role,
            )
        )

    if title is None:
        if color_by:
            clabel, cunit, _ = _FIELD_INFO[color_by]
            title = f"Network solution: {clabel}" + (f" [{cunit}]" if cunit else "")
        else:
            title = "Network topology"

    pad = 0.6
    fig.update_layout(
        template=NEFES_TEMPLATE_NAME,
        title=title,
        annotations=annotations,
        height=height or (260 + 60 * int(max(y) - min(y) if n else 0)),
        width=width,
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    axis = dict(showgrid=False, zeroline=False, showticklabels=False, showline=False, ticks="")
    fig.update_xaxes(**axis, range=[(min(x) - pad) if n else 0, (max(x) + pad) if n else 1])
    fig.update_yaxes(**axis, range=[(min(y) - pad - 0.4) if n else 0, (max(y) + pad) if n else 1])
    return fig
