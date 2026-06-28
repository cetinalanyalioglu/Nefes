"""Network topology diagram for visual diagnostics in a notebook.

Draws the element graph -- one node per element (labelled by index and name), one
arrow per directed edge -- using a simple longest-path **layered** layout so the
flow reads left to right.  It is a structural view (no solve needed): a quick way
to confirm the connectivity, element indices and edge directions before solving.

    from fns.plotting import plot_network_topology
    plot_network_topology(net).show()      # or net.plot_topology()
"""

import numpy as np
import plotly.graph_objects as go

from .theme import FNS_TEMPLATE_NAME

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


def plot_network_topology(
    network,
    *,
    show_edge_labels=True,
    show_areas=False,
    title="Network topology",
    height=None,
    width=None,
):
    """Plot the element/edge topology of a :class:`~fns.shell.network.Network`.

    Parameters
    ----------
    network : fns.shell.network.Network
        The network to draw (elements and directed edges; no solve required).
    show_edge_labels : bool, optional
        Label each edge with its index at the edge midpoint (default ``True``).
    show_areas : bool, optional
        Append the edge area to the edge label (default ``False``; the area is
        always in the hover text).
    title : str, optional
        Figure title.
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

    fig = go.Figure()

    # -- edges: a grey arrow per directed edge, plus a midpoint marker for hover/label
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
                arrowwidth=1.4,
                arrowcolor="#9aa5b1",
                standoff=14,  # stop short of the head marker
                startstandoff=14,  # start past the tail marker
                opacity=0.9,
            )
        )
        mids_x.append((x[t] + x[h]) / 2.0)
        mids_y.append((y[t] + y[h]) / 2.0)
        label = str(i)
        if show_areas:
            label += f"\nA={a:.3g}"
        mid_text.append(label)
        nm = edge_names[i] if i < len(edge_names) else f"e{i}"
        mid_hover.append(f"edge {i} ({nm})<br>{network._node_label(t)} → {network._node_label(h)}<br>A = {a:.4g} m²")

    if edges:
        fig.add_trace(
            go.Scatter(
                x=mids_x,
                y=mids_y,
                mode="markers+text" if show_edge_labels else "markers",
                text=mid_text if show_edge_labels else None,
                textposition="middle center",
                textfont=dict(size=10, color="#52606d"),
                marker=dict(size=14, color="rgba(255,255,255,0.85)", line=dict(width=0)),
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

    pad = 0.6
    fig.update_layout(
        template=FNS_TEMPLATE_NAME,
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
