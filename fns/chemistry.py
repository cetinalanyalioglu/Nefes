"""Post-solve per-edge chemistry: the solved species composition of each edge.

The mean-flow solve transports the **feed-stream mixture fractions** ``xi`` (one per
distinct injected composition), not chemical species.  This module recovers the actual
species per edge from a converged state, for diagnostics / output:

* a **burnt** (``EQ_KERNEL``) edge sits at HP equilibrium -- its product-species moles
  are captured for free by passing a sized warm-start cache to :func:`~fns.derive.recover_all`
  (the equilibrium kernel writes its converged composition there);
* an **unburnt** (``EQ_FROZEN``) edge is the forward blend of the feed streams -- its
  species mass fractions are ``xi @ stream_Y`` over the network's distinct streams.

A perfect-gas edge carries no chemical species (its scalars, if any, are passive).
"""

import numpy as np

from .derive import recover_all, NS_EST
from .thermo.api import PERFECT_GAS, EQ_KERNEL
from .elements.ids import MASS_FLOW_INLET, PT_INLET, P_OUTLET, MASS_SOURCE
from .composition import build_streams

# elements that introduce a feed stream (mirrors catalog._STREAM_INTRODUCING)
_STREAM_INTRODUCING = (MASS_FLOW_INLET, PT_INLET, P_OUTLET, MASS_SOURCE)


def product_moles(prob, x):
    """Per-edge converged product-species moles ``[mol/kg]`` for ``EQ_KERNEL`` edges.

    Reuses the equilibrium warm-start write-back: a per-edge cache passed to
    :func:`~fns.derive.recover_all` captures each burnt edge's converged composition in
    one recovery pass (no extra solve).  Rows for perfect-gas / frozen edges stay zero --
    those kernels do not populate the cache.

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
    recover_all(prob.edge_model, prob.tf, prob.ti, np.ascontiguousarray(x), prob.area, prob.n_elem, est, cache)
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
    library : thermolib.SpeciesLibrary
        The species data.

    Returns
    -------
    ndarray
        Shape ``(K, Ns)`` -- each distinct stream's species mass fractions.
    """
    comps = [(el.composition_spec, el.basis) for el in elements if el.residual_id in _STREAM_INTRODUCING]
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

    Parameters
    ----------
    prob : CompiledProblem
        The compiled network.
    x : ndarray
        Converged state, shape ``(n_solve, n_edges)``.
    e : int
        Edge id.
    library : thermolib.SpeciesLibrary or None
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
    if model == EQ_KERNEL:
        nj = (product_moles(prob, x) if moles is None else moles)[e]
        idx = np.nonzero(np.asarray(library.product_mask))[0]
        names = [library.species[i].name for i in idx]
        W = np.asarray(library.molar_masses)[idx]
        return _fractions(names, nj, W, basis, threshold)
    # EQ_FROZEN: the unburnt forward blend of the feed streams
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
    higher-level :class:`~fns.shell.network.Solution` always passes ``stream_Y`` explicitly.
    """
    raise ValueError("frozen-edge species need the network's stream compositions; query via Solution.species")
