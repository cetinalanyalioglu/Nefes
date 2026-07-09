"""Warm-start-chained parameter studies over named parameter addresses.

The sweep driver on top of the generic parameter API: :func:`parameter_study` walks an
N-dimensional grid (or a zipped list) of dotted-address values, solves each point on a
fresh :meth:`~nefes.shell.network.Network.with_params` copy of the pristine base, chains
warm starts through ``solve(x0=prev.x)``, and collects probed scalar outputs into
grid-shaped arrays (:class:`StudyResult`).

For eigenvalue continuation over a parameter, use
:func:`~nefes.perturbation.stability.trajectory.eigenvalue_trajectory` with
:meth:`Network.builder` -- same ``build(p)`` contract, no parallel sweep concept.

Main exports: :func:`parameter_study`, :class:`StudyResult`.
"""

import itertools
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class StudyResult:
    """The outcome of a :func:`parameter_study`: the grid, convergence and probed outputs.

    Attributes
    ----------
    addresses : tuple of str
        The swept parameter addresses, in the order given.
    shape : tuple of int
        The grid shape (one axis per address for ``mode="grid"``; a single axis for
        ``mode="zip"``).
    grid : dict of str -> ndarray
        Each address's value at every point, shaped ``shape``.
    converged : ndarray of bool
        Whether the mean flow converged at each point, shaped ``shape``.
    probes : dict of str -> ndarray
        The probed scalar outputs, shaped ``shape`` (``NaN`` at non-converged points);
        empty when no ``probe`` was given.
    solutions : list of Solution or None
        Every point's solution in iteration order (row-major over ``shape``);
        ``None`` when ``keep_solutions=False``.

    Examples
    --------
    >>> res = parameter_study(base, {"inlet.mdot": np.linspace(0.3, 0.7, 20)},
    ...                       probe=lambda sol: {"M_max": sol.field("M").max()})
    >>> res.probes["M_max"].shape
    (20,)
    """

    addresses: Tuple[str, ...]
    shape: Tuple[int, ...]
    grid: Dict[str, np.ndarray]
    converged: np.ndarray
    probes: Dict[str, np.ndarray] = field(default_factory=dict)
    solutions: Optional[List] = None

    @property
    def n_points(self) -> int:
        """Total number of sweep points."""
        return int(np.prod(self.shape))

    def __repr__(self) -> str:
        ok = int(np.count_nonzero(self.converged))
        probes = ", ".join(self.probes) or "none"
        return (
            f"StudyResult({' x '.join(str(s) for s in self.shape)} points over {list(self.addresses)}, "
            f"{ok}/{self.n_points} converged, probes: {probes})"
        )


def _points(params: Dict[str, Sequence], mode: str):
    """The sweep points as ``(shape, [ {address: value} ... ])`` in iteration order."""
    addresses = list(params)
    values = [np.asarray(v).ravel() for v in params.values()]
    if not addresses:
        raise ValueError("parameter_study needs at least one swept address")
    for addr, v in zip(addresses, values):
        if v.size == 0:
            raise ValueError(f"parameter_study: no values given for {addr!r}")
    if mode == "zip":
        sizes = {v.size for v in values}
        if len(sizes) != 1:
            raise ValueError(
                f"mode='zip' needs equal-length value lists; got {dict(zip(addresses, (v.size for v in values)))}"
            )
        shape = (values[0].size,)
        pts = [{a: v[k] for a, v in zip(addresses, values)} for k in range(values[0].size)]
        return tuple(addresses), shape, pts
    if mode == "grid":
        shape = tuple(v.size for v in values)
        pts = [dict(zip(addresses, combo)) for combo in itertools.product(*values)]
        return tuple(addresses), shape, pts
    raise ValueError(f"unknown mode {mode!r}; choose 'grid' (outer product) or 'zip' (aligned lists)")


def parameter_study(
    base,
    params: Dict[str, Sequence],
    probe: Optional[Callable] = None,
    *,
    mode: str = "grid",
    warm_start: bool = True,
    keep_solutions: bool = True,
    on_fail: str = "raise",
    **solve_kw,
):
    """Solve the mean flow over a grid of parameter values, warm-started point to point.

    Each point solves ``base.with_params({address: value, ...})`` -- a fresh copy, so the
    base stays pristine and no state accumulates across points.  Because parameter writes
    never touch topology, the previous point's converged state is a valid warm start and
    is chained through ``solve(x0=prev.x)`` (points march last-address-fastest, so
    neighbouring solves differ in one value).

    Parameters
    ----------
    base : Network
        The pristine base network (never mutated).
    params : dict
        ``{address: values}`` -- each key a dotted parameter address (see
        :meth:`Network.parameters`), each value a 1-D sequence to sweep.
    probe : callable, optional
        ``probe(solution) -> {name: scalar}`` evaluated at every converged point; the
        outputs are collected into grid-shaped arrays (:attr:`StudyResult.probes`).
    mode : {"grid", "zip"}, optional
        ``"grid"`` (default) sweeps the outer product of the value lists (N-D grid);
        ``"zip"`` aligns equal-length lists into a single 1-D path.
    warm_start : bool, optional
        Chain each solve from the previous converged state (default ``True``).
    keep_solutions : bool, optional
        Retain every point's :class:`~nefes.shell.network.Solution` (default ``True``);
        set ``False`` on large sweeps to save memory (probes are still collected).
    on_fail : {"raise", "continue"}, optional
        What to do when a point fails to converge: ``"raise"`` (default) stops with a
        pointed error; ``"continue"`` records ``converged=False`` (probes ``NaN``) and
        marches on, warm-starting from the last converged state.
    **solve_kw
        Forwarded to :meth:`Network.solve` at every point (e.g. ``tol``, ``verbose``).

    Returns
    -------
    StudyResult

    Examples
    --------
    A 1-D operating-line sweep with a scalar probe:

    >>> res = parameter_study(base, {"inlet.mdot": np.linspace(0.3, 0.7, 20)},
    ...                       probe=lambda sol: {"p_drop": sol.field("p")[0] - sol.field("p")[-1]})

    A 2-D grid over a composite knob and a boundary value:

    >>> res = parameter_study(base, {"orifice.throat_area": areas, "outlet.p": pressures})
    >>> res.probes["p_drop"].shape == (len(areas), len(pressures))

    See Also
    --------
    Network.with_params : the functional single-point idiom this driver chains.
    Network.builder : the ``build(p)`` closure for eigenvalue/Nyquist continuation.
    nefes.perturbation.stability.trajectory.eigenvalue_trajectory : modal continuation.
    """
    if on_fail not in ("raise", "continue"):
        raise ValueError(f"unknown on_fail {on_fail!r}; choose 'raise' or 'continue'")
    addresses, shape, pts = _points(params, mode)
    # resolve every address once up front (fail-closed before any solve)
    for addr in addresses:
        base.get(addr)

    n = len(pts)
    converged = np.zeros(n, dtype=bool)
    probe_rows: List[Optional[dict]] = [None] * n
    solutions: Optional[List] = [] if keep_solutions else None
    x_prev = None
    for k, point in enumerate(pts):
        net = base.with_params(point)
        sol = net.solve(x0=x_prev if warm_start else None, **solve_kw)
        if solutions is not None:
            solutions.append(sol)
        converged[k] = bool(sol.converged)
        if sol.converged:
            x_prev = sol.x
            if probe is not None:
                probe_rows[k] = dict(probe(sol))
        elif on_fail == "raise":
            at = ", ".join(f"{a} = {point[a]:g}" for a in addresses)
            raise RuntimeError(
                f"parameter_study: the mean flow failed to converge at point {k} ({at}); "
                "narrow the range, refine the step, or pass on_fail='continue' to record and march on"
            )

    grid = {a: np.array([p[a] for p in pts]).reshape(shape) for a in addresses}
    probes: Dict[str, np.ndarray] = {}
    names: List[str] = []
    for row in probe_rows:
        if row is not None:
            names = list(row)
            break
    for name in names:
        vals = np.full(n, np.nan)
        for k, row in enumerate(probe_rows):
            if row is not None:
                vals[k] = float(row[name])
        probes[name] = vals.reshape(shape)
    return StudyResult(
        addresses=addresses,
        shape=shape,
        grid=grid,
        converged=converged.reshape(shape),
        probes=probes,
        solutions=solutions,
    )
