"""Eigenvalue sensitivities: how each setup parameter moves each mode.

For an eigenpair ``A(omega; p) x = 0`` of the perturbation network, the derivative of the
eigenvalue with respect to any setup parameter ``p`` follows from one left eigenvector
``y`` (``y^H A = 0``) per mode:

    d omega / d p = - (y^H (dA/dp) x) / (y^H (dA/d omega) x).

The parameter enters the operator twice: directly (a duct length in the phase term, a
volume in the storage block) and through the mean flow it reshapes.  Both routes are
captured by evaluating ``dA/dp`` along the *solution path*: the network is rebuilt with
the parameter stepped, the mean-state shift is obtained from the already-factorized mean
Jacobian (``d x_bar / d p = - J^-1 dR/dp``, one back-substitution), and the operator is
re-assembled at the shifted state.  No mean flow is re-solved and no eigenvalue is
re-searched, so probing every parameter of a network costs a fraction of a single
eigenmode search.

Main exports: :func:`eigenvalue_sensitivities`, :class:`EigenmodeSensitivityResult`,
:class:`SensitivityWarning`.

See also
--------
eigenmodes : produces the spectrum this module differentiates.
trajectory.eigenvalue_trajectory : finite-parameter continuation (the brute-force check).
"""

import fnmatch
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ...assembly.assemble import jacobian, residual
from ...solver.linear import unflatten
from ..operator.operator import assemble_acoustic, build_acoustic_blocks

# Relative finite step in omega for the eigenvalue-derivative denominator (matches the
# Newton polish in eigenmodes.py).
_OMEGA_FD_REL = 1e-6

# Default parameter steps: relative for a nonzero value, absolute (in the parameter's own
# unit) when the base value is zero.  The operator entries are smooth in every parameter,
# so a small step keeps the truncation error near the difference roundoff floor.
_REL_STEP = 1e-6
_ABS_STEP = 1e-6

# A mode pair closer than this (relative, in the normalization term) is treated as
# effectively repeated: its individual sensitivities are ill-conditioned.
_DEGENERATE_TOL = 1e-8

# Inverse-iteration sweeps for the left eigenvector (each is one triangular solve).
_LEFT_ITERS = 3

# Consistency guard: the supplied modes must satisfy the freshly assembled operator to
# within this factor of their recorded residuals (catches a mismatched solution/result pair).
_CONSISTENCY_FACTOR = 1e3

_REPR_MAX_ROWS = 20
_REPR_MAX_MODES = 6


class SensitivityWarning(UserWarning):
    """Diagnostic from the sensitivity evaluation: a skipped parameter or a near-repeated mode."""


def _match(address: str, patterns) -> bool:
    """Whether ``address`` matches any of the glob ``patterns`` (a string or a list of strings)."""
    if patterns is None:
        return False
    if isinstance(patterns, str):
        patterns = [patterns]
    return any(fnmatch.fnmatch(address, pat) for pat in patterns)


def _select_addresses(inventory, params, include, exclude) -> List:
    """The inventory rows to probe, honoring ``params`` / ``include`` / ``exclude``.

    ``params`` (explicit address list) wins outright; otherwise every scalar numeric row is
    taken, optionally narrowed by ``include`` glob patterns, minus ``exclude`` matches.
    """
    if params is not None:
        rows = []
        for addr in params:
            rows.append(inventory[addr])  # KeyError with the unknown address, early and loud
        return rows
    rows = [
        r
        for r in inventory
        if r.kind == "float" and isinstance(r.value, (int, float)) and not isinstance(r.value, bool)
    ]
    if include is not None:
        rows = [r for r in rows if _match(r.address, include)]
    if exclude is not None:
        rows = [r for r in rows if not _match(r.address, exclude)]
    return rows


def _left_eigenvector(A, rng):
    """Left eigenvector of the (near-singular) assembled operator ``A`` by inverse iteration.

    Solves ``A^H y = b`` repeatedly with the LU factors of ``A`` (transpose-conjugate
    solves), which converges onto the null direction of ``A^H`` in a step or two because
    ``A`` is evaluated at the eigenvalue.  Returns ``(y, quality)`` with ``quality`` the
    scaled left residual ``||A^H y|| / max|A|``.
    """
    A = A.tocsc()
    scale = float(np.max(np.abs(A.data))) if A.nnz else 1.0
    try:
        lu = spla.splu(A)
    except RuntimeError:
        # exactly singular in floating point: nudge the diagonal by a few roundoff quanta
        A = (A + (1e-13 * scale) * sp.identity(A.shape[0], dtype=A.dtype, format="csc")).tocsc()
        lu = spla.splu(A)
    y = rng.standard_normal(A.shape[0]) + 1j * rng.standard_normal(A.shape[0])
    y /= np.linalg.norm(y)
    for _ in range(_LEFT_ITERS):
        y = lu.solve(y, trans="H")
        nrm = np.linalg.norm(y)
        if not np.isfinite(nrm) or nrm == 0.0:
            break
        y /= nrm
    quality = float(np.linalg.norm(A.conj().T @ y) / scale)
    return y, quality


def _step_for(row, rel_step, abs_step, steps) -> float:
    """The (positive) probe step for one parameter row."""
    if steps and row.address in steps:
        return float(steps[row.address])
    p0 = float(row.value)
    return rel_step * abs(p0) if p0 != 0.0 else abs_step


def eigenvalue_sensitivities(
    solution,
    eigs,
    *,
    params=None,
    include=None,
    exclude=None,
    rel_step=_REL_STEP,
    abs_step=_ABS_STEP,
    steps=None,
    scheme="forward",
    chain=True,
    eps=None,
    eps_fb=1e-6,
    u_floor=1e-8,
    isentropic=False,
    rng=None,
):
    """Differentiate every eigenvalue of ``eigs`` with respect to the network's parameters.

    One left eigenvector per mode turns the eigenvalue derivative into a pair of inner
    products, so each parameter costs a single operator re-assembly (plus one residual
    evaluation and one back-substitution for its mean-flow shift) -- no re-solve, no
    re-search.  All modes are differentiated against all selected parameters in one pass;
    the result is a single object holding the full mode-by-parameter table.

    Parameters
    ----------
    solution : Solution
        The solved network that produced ``eigs`` (supplies the network for parameter
        writes, the compiled problem, and the converged mean state).
    eigs : EigenmodeResult
        The spectrum to differentiate.  Its assembly settings (``isentropic`` etc.) must be
        repeated here -- :meth:`EigenmodeResult.sensitivities` does that automatically.
    params : list of str, optional
        Explicit parameter addresses to probe (as in :meth:`nefes.Network.parameters`).
        Overrides ``include``/``exclude``.
    include, exclude : str or list of str, optional
        Glob patterns over addresses (e.g. ``include="*.length"``,
        ``exclude=["*.mdot", "*.Tt"]``).  Default: every scalar numeric parameter.
    rel_step : float, optional
        Relative probe step for a nonzero parameter value (default ``1e-6``).
    abs_step : float, optional
        Absolute probe step for a parameter whose current value is zero (default ``1e-6``
        in the parameter's own unit).  A zero-valued parameter is differentiated at the
        edge of its admissible range, so its derivative is one-sided by construction.
    steps : dict, optional
        Per-address step overrides ``{address: step}`` (positive; the sign is flipped
        automatically when the stepped value would leave the admissible range).
    scheme : {"forward", "central"}, optional
        Difference scheme for ``dA/dp`` (default ``"forward"``: one re-assembly per
        parameter; ``"central"`` doubles the cost for a second-order truncation error).
    chain : bool, optional
        Include the mean-flow route (default True): the parameter shifts the mean state,
        and the operator moves with it.  ``False`` freezes the mean state -- useful to
        isolate how much of a sensitivity acts through the mean flow.
    eps, eps_fb, u_floor : float, optional
        Operator-assembly regularizers, matching the ``eigenmodes`` call.
    isentropic : bool, optional
        Assemble with the entropy wave pinned, matching the ``eigenmodes`` call.
    rng : numpy.random.Generator, optional
        Random source for the left-eigenvector start (default: a fixed seed).

    Returns
    -------
    EigenmodeSensitivityResult
        The mode-by-parameter derivative table, with skipped parameters and their reasons.

    Warns
    -----
    SensitivityWarning
        When a parameter cannot be probed (it is skipped and recorded), when a mode pair is
        near-repeated (its individual sensitivities are ill-conditioned), or when the
        supplied modes do not satisfy the freshly assembled operator (a mismatched
        ``solution``/``eigs`` pair).

    Examples
    --------
    >>> eigs = sol.eigenmodes(freq_band=(100, 600), isentropic=True)
    >>> sens = eigs.sensitivities(include="*.length")
    >>> sens  # doctest: +SKIP
    EigenmodeSensitivityResult: 3 mode(s) x 12 parameter(s)
    ...

    See also
    --------
    EigenmodeResult.sensitivities : the bound form that re-supplies the assembly settings.
    trajectory.eigenvalue_trajectory : finite-parameter continuation of the same spectrum.
    """
    if scheme not in ("forward", "central"):
        raise ValueError(f"scheme must be 'forward' or 'central'; got {scheme!r}")
    network, prob0 = solution.network, solution.problem
    x0 = np.ascontiguousarray(solution.x)
    rng = np.random.default_rng(0) if rng is None else rng
    if eps is None:
        eps = 1e-4 * prob0.var_scale[0]

    inventory = network.parameters()
    rows = _select_addresses(inventory, params, include, exclude)

    n_modes = eigs.n_modes
    omega = np.asarray(eigs.omega, dtype=np.complex128)

    blocks0 = build_acoustic_blocks(prob0, x0, eps=eps, eps_fb=eps_fb, u_floor=u_floor, isentropic=isentropic)

    # Per-mode ingredients: the right eigenvector (stored), a left eigenvector from the
    # factorized operator at the eigenvalue, and the omega-derivative normalization.
    lefts, denoms, left_res = [], np.zeros(n_modes, np.complex128), np.zeros(n_modes)
    a0 = np.zeros(n_modes, np.complex128)  # y^H A0(omega_i) x_i, the (tiny) base bilinear form
    for i in range(n_modes):
        w = complex(omega[i])
        x_i = np.asarray(eigs.modes[i], dtype=np.complex128)
        A_i = assemble_acoustic(w, blocks0)
        scale = float(np.max(np.abs(A_i.data))) if A_i.nnz else 1.0
        r_here = float(np.linalg.norm(A_i @ x_i) / scale)
        r_ref = max(float(eigs.residuals[i]), 1e-14)
        if r_here > _CONSISTENCY_FACTOR * r_ref:
            warnings.warn(
                f"mode {i} ({w.real / 2 / np.pi:.1f} Hz) does not satisfy the operator assembled from "
                f"this solution (residual {r_here:.1e} vs {r_ref:.1e} at the search): the solution and "
                "the eigenmode result likely belong to different setups, or the assembly settings "
                "(isentropic, eps) differ from the eigenmode call.",
                SensitivityWarning,
                stacklevel=2,
            )
        y_i, q_i = _left_eigenvector(A_i, rng)
        h_w = _OMEGA_FD_REL * (abs(w) + 1.0)
        Ap_x = (assemble_acoustic(w + h_w, blocks0) @ x_i - assemble_acoustic(w - h_w, blocks0) @ x_i) / (2.0 * h_w)
        denom = np.vdot(y_i, Ap_x)
        if abs(denom) < _DEGENERATE_TOL * np.linalg.norm(Ap_x):
            warnings.warn(
                f"mode {i} ({w.real / 2 / np.pi:.1f} Hz) is near-repeated: the normalization "
                "y^H (dA/domega) x is tiny, so its individual sensitivity is ill-conditioned "
                "(a nearly coincident pair moves by splitting, not independently).",
                SensitivityWarning,
                stacklevel=2,
            )
        lefts.append(y_i)
        denoms[i] = denom
        left_res[i] = q_i
        a0[i] = np.vdot(y_i, A_i @ x_i)

    # Mean-flow chain: one factorization of the mean Jacobian, shared by every parameter.
    luJ, R0 = None, None
    if chain and rows:
        J = jacobian(prob0, x0, eps, eps_fb, 0.0).tocsc()
        luJ = spla.splu(J)
        R0 = residual(prob0, x0, eps, eps_fb)

    def _perturbed(addr, p0, h):
        """Blocks of the operator with ``addr`` stepped by ``h`` (mean state carried along)."""
        net_h = network.with_params({addr: p0 + h})
        prob_h = net_h.compile()
        if prob_h.n_eq != prob0.n_eq:
            raise ValueError("the stepped network compiles to a different system size")
        x_h = x0
        if luJ is not None:
            dR = residual(prob_h, x0, eps, eps_fb) - R0
            x_h = x0 - unflatten(luJ.solve(dR), prob0.n_edges, prob0.n_solve)
        return build_acoustic_blocks(prob_h, x_h, eps=eps, eps_fb=eps_fb, u_floor=u_floor, isentropic=isentropic)

    addresses, values, units, used_steps = [], [], [], []
    columns = []  # per parameter: (n_modes,) d omega / d p
    failed: Dict[str, str] = {}
    for row in rows:
        addr, p0 = row.address, float(row.value)
        h = _step_for(row, rel_step, abs_step, steps)
        try:
            try:
                blocks_p = _perturbed(addr, p0, h)
            except Exception as exc_first:
                try:
                    h = -h  # the stepped value left the admissible range: probe the other side
                    blocks_p = _perturbed(addr, p0, h)
                except Exception:
                    raise exc_first  # both sides failed: the original error names the real obstacle
            if scheme == "central":
                blocks_m = _perturbed(addr, p0, -h)
        except Exception as exc:
            failed[addr] = f"{type(exc).__name__}: {exc}"
            continue
        dw = np.zeros(n_modes, np.complex128)
        for i in range(n_modes):
            w, x_i, y_i = complex(omega[i]), eigs.modes[i], lefts[i]
            a_p = np.vdot(y_i, assemble_acoustic(w, blocks_p) @ x_i)
            if scheme == "central":
                a_m = np.vdot(y_i, assemble_acoustic(w, blocks_m) @ x_i)
                dA = (a_p - a_m) / (2.0 * h)
            else:
                dA = (a_p - a0[i]) / h
            dw[i] = -dA / denoms[i]
        addresses.append(addr)
        values.append(p0)
        units.append(row.unit)
        used_steps.append(h)
        columns.append(dw)

    dw_dp = np.array(columns, dtype=np.complex128).T if columns else np.empty((n_modes, 0), np.complex128)
    for addr, reason in failed.items():
        warnings.warn(f"parameter {addr!r} skipped: {reason}", SensitivityWarning, stacklevel=2)

    return EigenmodeSensitivityResult(
        omega=omega,
        addresses=addresses,
        values=np.array(values, dtype=float),
        units=units,
        dw_dp=dw_dp,
        steps=np.array(used_steps, dtype=float),
        failed=failed,
        denominators=denoms,
        left_residuals=left_res,
        chain=bool(chain),
        isentropic=bool(isentropic),
        scheme=scheme,
    )


@dataclass
class EigenmodeSensitivityResult:
    """Mode-by-parameter eigenvalue derivatives of a perturbation network.

    Rows are modes (in the order of the originating :class:`EigenmodeResult`), columns are
    parameters.  Derivatives are reported both per parameter unit and per +1% of the
    parameter's current value; a positive growth-rate entry means increasing the parameter
    pushes that mode toward instability.

    Attributes
    ----------
    omega : ndarray
        Complex modal angular frequencies (rad/s), shape ``(n_modes,)``.
    addresses : list of str
        Probed parameter addresses, shape ``(n_params,)``.
    values : ndarray
        Base parameter values.
    units : list of str
        Parameter units (display only).
    dw_dp : ndarray
        ``d omega / d p``, complex, shape ``(n_modes, n_params)`` (rad/s per parameter unit).
    steps : ndarray
        Signed probe step actually taken per parameter.
    failed : dict
        ``{address: reason}`` for parameters that could not be probed.
    denominators : ndarray
        Per-mode normalization ``y^H (dA/domega) x`` (diagnostic; tiny for a near-repeated pair).
    left_residuals : ndarray
        Scaled left-eigenvector residuals ``||A^H y|| / max|A|`` (diagnostic).
    chain : bool
        Whether the mean-flow route was included.
    isentropic : bool
        Whether the operator was assembled with the entropy wave pinned.
    scheme : str
        The difference scheme used for ``dA/dp``.
    """

    omega: np.ndarray
    addresses: List[str]
    values: np.ndarray
    units: List[str]
    dw_dp: np.ndarray
    steps: np.ndarray
    failed: Dict[str, str] = field(default_factory=dict)
    denominators: Optional[np.ndarray] = None
    left_residuals: Optional[np.ndarray] = None
    chain: bool = True
    isentropic: bool = False
    scheme: str = "forward"

    @property
    def n_modes(self) -> int:
        """Number of modes differentiated."""
        return int(self.omega.size)

    @property
    def n_params(self) -> int:
        """Number of parameters probed (successfully)."""
        return len(self.addresses)

    @property
    def freqs(self) -> np.ndarray:
        """Modal frequencies ``Re(omega)/(2*pi)`` in Hz."""
        return self.omega.real / (2.0 * np.pi)

    @property
    def growth_rates(self) -> np.ndarray:
        """Modal growth rates ``-Im(omega)`` in 1/s (positive = unstable)."""
        return -self.omega.imag

    @property
    def dgrowth_dp(self) -> np.ndarray:
        """Growth-rate derivative ``-Im(d omega/d p)``, 1/s per parameter unit, shape ``(n_modes, n_params)``."""
        return -self.dw_dp.imag

    @property
    def dfreq_dp(self) -> np.ndarray:
        """Frequency derivative ``Re(d omega/d p)/(2*pi)``, Hz per parameter unit, shape ``(n_modes, n_params)``."""
        return self.dw_dp.real / (2.0 * np.pi)

    @property
    def dgrowth_pct(self) -> np.ndarray:
        """Growth-rate change for a +1% parameter change, 1/s (zero for a zero-valued parameter)."""
        return self.dgrowth_dp * (0.01 * self.values)[None, :]

    @property
    def dfreq_pct(self) -> np.ndarray:
        """Frequency change for a +1% parameter change, Hz (zero for a zero-valued parameter)."""
        return self.dfreq_dp * (0.01 * self.values)[None, :]

    def _influence_scale(self) -> np.ndarray:
        """Per-parameter ranking scale: +1% of the value, or the probe step for a zero value."""
        return np.where(self.values != 0.0, 0.01 * np.abs(self.values), np.abs(self.steps))

    def influence(self, mode: Optional[int] = None) -> np.ndarray:
        """Growth-rate influence used for ranking, shape ``(n_params,)``.

        The magnitude of the growth-rate change over a +1% parameter change -- or over the
        probe step when the base value is zero, so zero-valued parameters (an unset volume,
        an unset end correction) still rank by their leverage.

        Parameters
        ----------
        mode : int, optional
            Rank against one mode; default: the maximum over all modes.
        """
        scaled = np.abs(self.dgrowth_dp) * self._influence_scale()[None, :]
        return scaled[mode] if mode is not None else scaled.max(axis=0) if self.n_modes else np.zeros(self.n_params)

    def ranking(self, mode: Optional[int] = None) -> np.ndarray:
        """Parameter indices sorted by decreasing :meth:`influence`."""
        return np.argsort(-self.influence(mode), kind="stable")

    def top(self, n: int = 10, mode: Optional[int] = None) -> List[str]:
        """The ``n`` most influential parameter addresses (see :meth:`influence`)."""
        return [self.addresses[k] for k in self.ranking(mode)[:n]]

    def __getitem__(self, address: str) -> np.ndarray:
        """Column ``d omega / d p`` for one parameter address, shape ``(n_modes,)``."""
        try:
            k = self.addresses.index(address)
        except ValueError:
            raise KeyError(address) from None
        return self.dw_dp[:, k]

    def summary(self) -> dict:
        """One-line-per-field dict: sizes, per-mode frequencies, the top parameters."""
        return {
            "n_modes": self.n_modes,
            "n_params": self.n_params,
            "freqs": self.freqs.tolist(),
            "growth_rates": self.growth_rates.tolist(),
            "top": self.top(5),
            "n_failed": len(self.failed),
            "chain": self.chain,
            "isentropic": self.isentropic,
        }

    def _mode_headers(self, mode_ids) -> List[str]:
        return [f"#{i} {self.freqs[i]:.4g} Hz" for i in mode_ids]

    def __repr__(self) -> str:
        """Ranked table: growth-rate change per +1% (per probe step for zero-valued parameters)."""
        head = f"EigenmodeSensitivityResult: {self.n_modes} mode(s) x {self.n_params} parameter(s)"
        if self.failed:
            head += f", {len(self.failed)} skipped"
        lines = [head, "  d(growth)/d(parameter) scaled to +1% of value (probe step if zero); positive = destabilizing"]
        if not self.n_params or not self.n_modes:
            return "\n".join(lines)
        mode_ids = list(range(min(self.n_modes, _REPR_MAX_MODES)))
        scaled = self.dgrowth_dp * self._influence_scale()[None, :]
        order = self.ranking()[:_REPR_MAX_ROWS]
        headers = self._mode_headers(mode_ids)
        addr_w = max(len("parameter"), max(len(self.addresses[k]) for k in order))
        lines.append("")
        lines.append(f"  {'parameter':<{addr_w}}  {'value':>11}  " + "  ".join(f"{h:>14}" for h in headers))
        for k in order:
            cells = "  ".join(f"{scaled[i, k]:>+14.4g}" for i in mode_ids)
            lines.append(f"  {self.addresses[k]:<{addr_w}}  {self.values[k]:>11.5g}  {cells}")
        if self.n_params > _REPR_MAX_ROWS:
            lines.append(f"  ... ({self.n_params - _REPR_MAX_ROWS} more)")
        if self.n_modes > _REPR_MAX_MODES:
            lines.append(f"  (showing {_REPR_MAX_MODES} of {self.n_modes} modes)")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """Rich HTML table for Jupyter: ranked parameters, one growth/frequency pair per mode."""
        n, m = self.n_modes, self.n_params
        parts = [f"{n} mode{'' if n == 1 else 's'}", f"{m} parameter{'' if m == 1 else 's'}"]
        if self.failed:
            parts.append(f"<span style='color:#c0392b'>{len(self.failed)} skipped</span>")
        parts.append("entropy wave pinned" if self.isentropic else "entropy/composition included")
        header = (
            "<div style='font-family:sans-serif;margin-bottom:4px'><b>EigenmodeSensitivityResult</b>"
            " &nbsp;&middot;&nbsp; " + " &nbsp;|&nbsp; ".join(parts) + "</div>"
        )
        sub = (
            "<div style='font-family:sans-serif;font-size:0.85em;color:#888;margin-bottom:4px'>"
            "growth-rate / frequency change per +1% of the parameter value (per probe step for a "
            "zero value); <span style='color:#c0392b'>positive growth = destabilizing</span></div>"
        )
        if not m or not n:
            return header + sub
        mode_ids = list(range(min(n, _REPR_MAX_MODES)))
        scale = self._influence_scale()
        g = self.dgrowth_dp * scale[None, :]
        f = self.dfreq_dp * scale[None, :]
        order = self.ranking()[:_REPR_MAX_ROWS]
        th = "style='text-align:right;padding:2px 8px;border-bottom:1px solid #ccc'"
        thl = "style='text-align:left;padding:2px 8px;border-bottom:1px solid #ccc'"
        cols = "".join(
            f"<th {th}>&Delta;g [1/s]<br><span style='font-weight:normal'>{h}</span></th>"
            f"<th {th}>&Delta;f [Hz]</th>"
            for h in self._mode_headers(mode_ids)
        )
        rows_html = []
        for k in order:
            cells = []
            for i in mode_ids:
                color = "#c0392b" if g[i, k] > 0 else "#2a8a4a"
                cells.append(f"<td style='text-align:right;padding:2px 8px;color:{color}'>{g[i, k]:+.4g}</td>")
                cells.append(f"<td style='text-align:right;padding:2px 8px;color:#888'>{f[i, k]:+.4g}</td>")
            rows_html.append(
                f"<tr><td style='text-align:left;padding:2px 8px'>{self.addresses[k]}</td>"
                f"<td style='text-align:right;padding:2px 8px'>{self.values[k]:.5g}"
                f"<span style='color:#888'> {self.units[k]}</span></td>" + "".join(cells) + "</tr>"
            )
        more = (
            f"<div style='font-family:sans-serif;font-size:0.85em;color:#888'>... " f"({m - _REPR_MAX_ROWS} more)</div>"
            if m > _REPR_MAX_ROWS
            else ""
        )
        table = (
            "<table style='border-collapse:collapse;font-family:monospace;font-size:0.9em'>"
            f"<tr><th {thl}>parameter</th><th {th}>value</th>{cols}</tr>" + "".join(rows_html) + "</table>"
        )
        return header + sub + table + more

    def plot(self, **kwargs):
        """Bar chart of the ranked growth-rate sensitivities (see :func:`nefes.plotting.plot_sensitivities`)."""
        from ...plotting.sensitivity import plot_sensitivities

        return plot_sensitivities(self, **kwargs)
