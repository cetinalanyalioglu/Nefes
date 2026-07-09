"""Post-solve per-edge chemistry: the solved species composition of each edge.

The mean-flow solve transports the **feed-stream mixture fractions** ``xi`` (one per
distinct injected composition), not chemical species.  This module recovers the actual
species per edge from a converged state, for diagnostics / output:

* a **burnt** (``EQ_KERNEL``) edge sits at HP equilibrium, its product-species moles
  are captured by passing a sized warm-start cache to :func:`~nefes.derive.recover_all`
  (the equilibrium kernel writes its converged composition there);
* an **unburnt** (``EQ_FROZEN``) edge is the forward blend of the feed streams -- its
  species mass fractions are ``xi @ stream_Y`` over the network's distinct streams.

A perfect-gas edge carries no chemical species (its scalars, if any, are passive).
"""

import numpy as np

from ..assembly.recover import recover_all, NS_EST
from ..thermo.api import PERFECT_GAS, EQ_KERNEL, EQ_MARKER
from ..elements.ids import STREAM_INTRODUCING
from ..elements.composite import is_composite
from .composition import build_streams


def product_moles(prob, x):
    """Per-edge converged product-species moles ``[mol/kg]`` for ``EQ_KERNEL`` edges.

    Captures each burnt edge's converged composition in one recovery pass (no extra solve).
    Rows for perfect-gas / frozen edges stay zero, those kernels do not populate the cache.

    Parameters
    ----------
    prob : CompiledProblem
        The compiled network.
    x : ndarray
        Converged state, shape ``(n_solve, n_edges)``.

    Returns
    -------
    ndarray
        Shape ``(n_edges, Np)`` product-species moles per kg; ``Np == 0`` for a
        non-reacting model.
    """
    if prob.ti.shape[0] <= 6:  # not an equilibrium bundle (perfect gas / passive scalars)
        return np.zeros((prob.n_edges, 0))
    Np = int(prob.ti[6])
    cache = np.zeros((prob.n_edges, Np + 1))  # [moles_0..moles_{Np-1}, T]
    est = np.zeros((NS_EST, prob.n_edges))
    mr = int(getattr(prob, "marker_row", -1))
    recover_all(prob.edge_model, prob.tf, prob.ti, np.ascontiguousarray(x), prob.area, prob.n_elem, mr, est, cache)
    return cache[:, :Np]


def stream_mass_fractions(elements, library):
    """The ``(K, Ns)`` full-library mass fractions of the network's distinct feed streams.

    Re-runs the build-time stream discovery (deterministic auto-merge in node order), so
    stream ``k`` aligns with the transported mixture fraction ``xi[k]`` and the scalar
    label ``scalar_names[k]``.

    Parameters
    ----------
    elements : list of ElementSpec
        The network elements (in node order).
    library : nefes.thermo.SpeciesLibrary
        The species data.

    Returns
    -------
    ndarray
        Shape ``(K, Ns)`` -- each distinct stream's species mass fractions.
    """
    # Composites carry atomic sub-elements; flatten so ``residual_id`` is always defined.
    # Stream-introducing elements (inlets/sources/outlets) are top-level atomic and keep their
    # node order under the build's composite expansion, so the flattened stream sequence -- and
    # thus stream ``k`` <-> ``xi[k]`` -- stays aligned with the compiled problem.
    atomic = []
    for el in elements:
        atomic.extend(el.sub_elements if is_composite(el) else (el,))
    comps = [(el.composition_spec, el.basis) for el in atomic if el.residual_id in STREAM_INTRODUCING]
    stream_Y, _assignment = build_streams(library, comps)
    return stream_Y


def _fractions(names, moles, W, basis, threshold):
    """``{name: fraction}`` from per-kg moles, dropping entries below ``threshold``."""
    moles = np.asarray(moles, dtype=float)
    W = np.asarray(W, dtype=float)
    if basis == "mole":
        total = moles.sum()
        frac = moles / total if total > 0.0 else moles
    elif basis == "mass":
        frac = moles * W  # moles per kg times molar mass = mass fraction
    else:
        raise ValueError(f"basis must be 'mole' or 'mass'; got {basis!r}")
    return {name: float(f) for name, f in zip(names, frac) if abs(float(f)) > threshold}


def edge_species(prob, x, e, library, *, basis="mole", moles=None, stream_Y=None, threshold=1e-12):
    """Solved chemical species ``{name: fraction}`` on edge ``e``.

    Selects the reconstruction that matches the edge's thermo model: the converged
    equilibrium products for a burnt edge, or the unburnt forward blend of the feed
    streams (``xi @ stream_Y``) for a frozen / fresh one.  A perfect-gas edge (or a
    ``None`` library) carries no chemical species and returns an empty dict.

    Parameters
    ----------
    prob : CompiledProblem
        The compiled network.
    x : ndarray
        Converged state, shape ``(n_solve, n_edges)``.
    e : int
        Edge id.
    library : nefes.thermo.SpeciesLibrary or None
        Species data; ``None`` (perfect gas) yields an empty result.
    basis : {"mole", "mass"}, optional
        Whether the fractions are mole or mass fractions (default ``"mole"``).
    moles, stream_Y : ndarray, optional
        Precomputed :func:`product_moles` / :func:`stream_mass_fractions` (pass them when
        querying many edges to avoid recomputing per call).
    threshold : float, optional
        Drop species whose fraction magnitude is below this (default ``1e-12``).

    Returns
    -------
    dict
        ``{species_name: fraction}`` in library order, restricted to the present species.
    """
    model = int(prob.edge_model[e])
    if library is None or model == PERFECT_GAS:
        return {}
    # A marker-gated edge is bimodal at convergence: report the equilibrium products on a burnt
    # edge (marker >= 1/2) and the frozen feed-stream blend on a fresh one, matching the state's
    # blend gate.  The equilibrium product moles are captured for every reacting edge by
    # product_moles (the EQ_MARKER recovery runs the equilibrium leg regardless of the marker).
    burnt = model == EQ_KERNEL
    if model == EQ_MARKER:
        marker_row = int(getattr(prob, "marker_row", -1))
        # The marker is a sticky reachability label (a noisy-OR transport), so it stays bimodal
        # even where burnt gas is diluted by a fresh stream -- ~1 on any edge downstream of a
        # flame, ~0 upstream.  ``0.5`` is the crossover of ``marker_gate`` (equal frozen/equilibrium
        # weight), so reporting the equilibrium products above it and the frozen blend below it
        # matches the converged state.  The equilibrium solve runs on every marker-gated edge
        # regardless (product_moles), so a burnt edge is never denied its products.
        burnt = marker_row >= 0 and float(x[marker_row, e]) >= 0.5
    if burnt:
        nj = (product_moles(prob, x) if moles is None else moles)[e]
        idx = np.nonzero(np.asarray(library.product_mask))[0]
        names = [library.species[i].name for i in idx]
        W = np.asarray(library.molar_masses)[idx]
        return _fractions(names, nj, W, basis, threshold)
    # frozen / fresh marker-gated edge: the unburnt forward blend of the feed streams
    sY = stream_mass_fractions_for(prob, x, library) if stream_Y is None else stream_Y
    n_elem = prob.n_elem
    xi = np.asarray(x[3 : 3 + n_elem, e], dtype=float)
    Y = xi @ sY  # full-library mass fractions
    W = np.asarray(library.molar_masses)
    nj = Y / W  # mass fraction -> moles per kg
    names = [s.name for s in library.species]
    return _fractions(names, nj, W, basis, threshold)


def stream_mass_fractions_for(prob, x, library):  # pragma: no cover - convenience shim
    """Internal fallback used by :func:`edge_species` when ``stream_Y`` is not supplied.

    Requires the network elements, which a bare ``CompiledProblem`` does not carry, so the
    higher-level :class:`~nefes.shell.network.Solution` always passes ``stream_Y`` explicitly.

    Parameters
    ----------
    prob : CompiledProblem
        The compiled network (does not carry the element specs this needs).
    x : ndarray
        Converged state.
    library : nefes.thermo.SpeciesLibrary
        The species data.

    Raises
    ------
    ValueError
        Always -- the caller must supply ``stream_Y`` via ``Solution.species``.
    """
    raise ValueError("frozen-edge species need the network's stream compositions; query via Solution.species")
