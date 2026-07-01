"""Forced perturbation response under physical boundary conditions (theory.md s12.7).

Where ``response.py`` *measures* a network -- driving every terminal with independent
unit waves to read out a transfer/scattering matrix, regardless of how the boundaries
are actually closed -- this module *solves* the network as it is physically
terminated.  Each single-port element carries a :class:`PerturbationBC`; the operator
``A(omega)`` is assembled with the terminal reflection face stamped
(``with_boundaries=True``) and the forcing right-hand side ``b(omega)`` built from the
terminals that drive an incoming wave (``driven``).  One sparse solve per frequency
gives the nodal perturbation field.

A purely-reflective, undamped, unforced network is singular at its resonances -- that
is the (deferred) stability eigenvalue problem ``det A(omega) = 0``.  With no driven
terminal the forcing vanishes and the forced response is the trivial zero field (and is
singular exactly at those resonances); a single driven terminal (or some loss) makes it
well posed off resonance.
"""

import warnings
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import scipy.sparse.linalg as spla

from ..operator.operator import build_acoustic_blocks, assemble_acoustic
from ..operator.stamps import boundary_forcing
from ..operator.characteristics import edge_transforms, basis_block_from_state
from ...solver.control import states_table


class CompositionalNoiseWarning(UserWarning):
    """A **hand-written compact-nozzle closure** drops composition -> acoustic (compositional /
    indirect) noise.

    Composition -> acoustic coupling is captured *everywhere the linearization is inherited*: the
    full algebraic Jacobian carries it (a flame, an area change, a resolved nozzle, the compact
    ``choked_nozzle_outlet`` *element* -- whose critical-mass-flux row is complex-stepped through
    its composition dependence).  It is dropped only by the explicit analytic terminal closures
    :meth:`~fns.perturbation.operator.boundary_bc.PerturbationBC.choked_nozzle` /
    :meth:`~fns.perturbation.operator.boundary_bc.PerturbationBC.constant_mass_flow`, which overwrite that
    row with a 3-wave ``(f, g, h)`` relation: they keep the entropy off-diagonal ``R_s`` but have
    no composition column ``R_xi``.  Use the inherited element (or resolve the nozzle) to capture
    it."""


# The explicit analytic terminal closures whose hand-written 3-wave (f, g, h) form drops the
# composition -> acoustic off-diagonal R_xi (everywhere else the inherited J_alg retains it).
_COMPOSITIONAL_NOISE_DROPPING_KINDS = ("choked_nozzle", "constant_mass_flow")


def _compositional_noise_gap(prob):
    """Compact analytic closures that drop composition -> acoustic noise, *if* scalars are present.

    Returns the sorted, de-duplicated closure kinds (e.g. ``["choked_nozzle"]``) only when the
    network actually transports reacting scalars (``prob.scalar_names`` non-empty) and at least one
    terminal is closed by a hand-written compact-nozzle BC; empty otherwise.  This is the *only*
    configuration where a composition fluctuation reaches a section that physically radiates sound
    yet has its composition -> acoustic coupling discarded -- see :class:`CompositionalNoiseWarning`.
    """
    if not getattr(prob, "scalar_names", ()):
        return []
    kinds = set()
    for bc in getattr(prob, "node_bc", None) or []:
        if bc is not None and getattr(bc, "kind", None) in _COMPOSITIONAL_NOISE_DROPPING_KINDS:
            kinds.add(bc.kind)
    return sorted(kinds)


def forced_response(prob, x_bar, freqs, *, eps=None, eps_fb=1e-6, u_floor=1e-8, isentropic=False):
    """Solve the perturbation field under each terminal's declared boundary condition.

    The forcing is whatever the terminals' :class:`PerturbationBC`\\ s drive (their
    ``driven`` families); with no driven terminal the response is the trivial zero field.

    Parameters
    ----------
    prob : CompiledProblem
        Compiled flow network whose single-port elements carry ``PerturbationBC``s
        (``prob.node_bc``).  Terminals left at ``inherit`` keep their linearized mean
        boundary row.
    x_bar : ndarray
        Converged mean-flow state, shape ``(n_solve, E)``.
    freqs : array_like
        Frequencies (Hz) to solve at.
    eps, eps_fb, u_floor : float, optional
        Operator-assembly regularizers forwarded to :func:`build_acoustic_blocks`.
    isentropic : bool, optional
        Force isentropic perturbations (``rho' = p'/c^2``): the entropy wave is pinned to
        zero on every edge, leaving the two acoustic waves (default False).  Standard
        acoustic analysis; uses the same operator and solve path.

    Returns
    -------
    ForcedResponse
        The nodal perturbation field at every frequency.
    """
    freqs = np.asarray(freqs, dtype=float)
    gap = _compositional_noise_gap(prob)
    if gap:  # one reminder per call: the compact closure radiates entropy noise but not compositional
        warnings.warn(
            f"compact nozzle closure(s) {gap} terminate a reacting (multi-stream) flow: they carry the "
            "entropy -> acoustic (entropy/indirect) noise R_s but drop the composition -> acoustic "
            "(compositional/indirect) noise R_xi. Use the inherited choked_nozzle_outlet element, or "
            "resolve the nozzle, to capture it (the inherited linearization keeps the composition column).",
            CompositionalNoiseWarning,
            stacklevel=2,
        )
    omegas = 2.0 * np.pi * freqs  # operator assembly works in angular frequency (rad/s)
    blocks = build_acoustic_blocks(prob, x_bar, eps=eps, eps_fb=eps_fb, u_floor=u_floor, isentropic=isentropic)
    K = float(prob.tf[0]) / float(prob.tf[1])
    est = states_table(prob, x_bar)
    cals = blocks.cals
    _, L = edge_transforms(est, K, cals)

    X = np.zeros((omegas.size, int(prob.n_col)), dtype=np.complex128)
    for i, omega in enumerate(omegas):
        A = assemble_acoustic(omega, blocks, with_boundaries=True)
        b = boundary_forcing(prob, x_bar, omega, cals)
        X[i] = spla.spsolve(A.tocsc(), b)
    # The length-bearing ducts are carried along so the stored-energy diagnostics
    # (ForcedResponse.stored_energy / .plot_response) need no second pass over prob.
    from ..fields.modeshape import build_geometry

    return ForcedResponse(
        freqs=freqs,
        X=X,
        L=L,
        est=est,
        K=K,
        n_solve=int(prob.n_solve),
        cals=cals,
        scalar_names=tuple(getattr(prob, "scalar_names", ())),
        ducts=build_geometry(prob).ducts,
    )


@dataclass
class ForcedResponse:
    """Nodal perturbation field of a physically-terminated network over a sweep."""

    freqs: np.ndarray  # (n_freq,) in Hz
    X: np.ndarray  # (n_freq, n_col) nodal perturbation vectors
    L: List[np.ndarray]  # per-edge dx_to_char (3x3) at the mean state
    est: np.ndarray  # frozen mean edge-state table
    K: float  # cp / R # CA: Why do we have cp/R here? There is no point.
    n_solve: int
    cals: Optional[list] = None  # per-edge caloric rows (reacting "network" flavor)
    scalar_names: tuple = ()  # transported reacting scalars (feed-stream labels), in band-1 order
    ducts: Optional[list] = None  # length-bearing DuctSegments, for the stored-energy diagnostics

    def __repr__(self) -> str:
        """One-line summary: sweep extent, edge count, and the transported wave labels."""
        f = np.asarray(self.freqs, dtype=float)
        n = f.size
        span = "empty" if n == 0 else (f"f = {f[0]:.1f} Hz" if n == 1 else f"f in [{f.min():.1f}, {f.max():.1f}] Hz")
        return (
            f"ForcedResponse: {n} frequenc{'y' if n == 1 else 'ies'} ({span}), "
            f"{len(self.L)} edges, waves {self.wave_labels}"
        )

    @property
    def _n_acoustic(self) -> int:
        """The acoustic+entropy characteristic count handled by the ``L`` transform (3)."""
        return self.L[0].shape[0]

    @property
    def n_char(self) -> int:
        """Wave count per edge: the acoustic+entropy chars plus one per convected scalar (== ``n_solve``)."""
        return self.n_solve

    @property
    def wave_labels(self) -> tuple:
        """Per-wave symbols: ``("f", "g", "h")`` then the reacting-scalar names (:attr:`scalar_names`)."""
        return ("f", "g", "h") + tuple(self.scalar_names)

    def waves(self, edge):
        """Wave amplitudes at ``edge``: ``(f, g, h)`` then one convected amplitude per reacting scalar.

        Shape ``(n_omega, n_char)``.  The acoustic/entropy block is the characteristic
        transform ``L_e`` of the network unknowns; each transported scalar's perturbation is
        already its own convected wave (the operator propagates it at the mean speed ``u``), so
        it is surfaced as-is (identity) under the name in :attr:`wave_labels`.  Inert flow
        (no scalars) returns just ``(f, g, h)`` exactly as before.
        """
        ns = self.n_solve
        na = self._n_acoustic
        Xe = self.X[:, ns * edge : ns * (edge + 1)]  # (n_omega, ns): every unknown on this edge
        w = np.empty((Xe.shape[0], ns), dtype=Xe.dtype)
        w[:, :na] = np.einsum("ij,oj->oi", self.L[edge], Xe[:, :na])  # (f, g, h)
        w[:, na:] = Xe[:, na:]  # convected scalar waves (identity)
        return w

    def field(self, edge, basis="network"):
        """Perturbation at ``edge`` in a variable flavor; shape ``(n_omega, n_char)``.

        ``basis`` is any of ``characteristics.BASIS_LABELS`` (default ``"network"`` -- the
        solver's own ``(mdot', p', h_t')``) and re-expresses the acoustic+entropy block; the
        reacting-scalar waves pass through unchanged (they are already in network units).
        """
        w = self.waves(edge)  # (n_omega, n_char)
        na = self._n_acoustic
        cal = None if self.cals is None else self.cals[edge]
        B = basis_block_from_state(basis, self.est[:, edge], self.K, cal)
        out = np.empty_like(w)
        out[:, :na] = np.einsum("ij,oj->oi", B, w[:, :na])
        out[:, na:] = w[:, na:]
        return out

    def reflection_at(self, edge):
        """Local acoustic reflection ``g/f`` at ``edge``; shape ``(n_omega,)``.

        The ratio of the upstream- to downstream-running acoustic amplitude -- the
        reflection seen looking downstream at this station.
        """
        w = self.waves(edge)
        return w[:, 1] / w[:, 0]

    def stored_energy(self, *, n_x: int = 120) -> np.ndarray:
        """Total acoustic energy stored in the domain at each swept frequency.

        The Myers acoustic energy density integrated over every length-bearing duct
        (the field reconstructed from each duct's face waves) and summed across the
        network.  Boundary flux is deliberately excluded: this is the energy *held*
        in the domain, a probe-independent resonance indicator that peaks at every
        lightly-damped mode -- unlike a single-point transfer function, it cannot be
        blinded by a probe sitting on a pressure node.

        Parameters
        ----------
        n_x : int, optional
            Interior samples per duct for the spatial integral (default 120).

        Returns
        -------
        ndarray
            Stored acoustic energy at each frequency, shape ``(n_freq,)``
            (arbitrary drive-scale units).
        """
        from ..fields.power import duct_energy_spectrum

        if not self.ducts:
            return np.zeros(np.asarray(self.freqs, dtype=float).size)
        return duct_energy_spectrum(self, self.ducts, n_x=n_x)

    def plot_response(self, *, n_x: int = 120, log: bool = True, title: Optional[str] = None, **layout):
        """Plot the total stored acoustic energy across the domain versus frequency.

        A first-look resonance map for a forced sweep: peaks mark the lightly-damped
        modes and their height/width the damping.  It plots :meth:`stored_energy` --
        the whole-domain energy, with no probe and no boundary flux -- so it cannot be
        fooled by a sensor that happens to sit on a pressure node.

        Parameters
        ----------
        n_x : int, optional
            Interior samples per duct for the energy integral (default 120).
        log : bool, optional
            Logarithmic energy axis (default True), so weak and strong resonances are
            both legible.
        title : str, optional
            Figure title; a sensible default is used when omitted.
        **layout
            Extra Plotly ``update_layout`` keyword arguments.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        import plotly.graph_objects as go

        from ...plotting.theme import COLORWAY, FNS_TEMPLATE_NAME

        f = np.asarray(self.freqs, dtype=float)
        energy = self.stored_energy(n_x=n_x)
        fig = go.Figure(
            go.Scatter(
                x=f,
                y=energy,
                mode="lines",
                line=dict(color=COLORWAY[0], width=2),
                name="stored acoustic energy",
                hovertemplate="f = %{x:.1f} Hz<br>E = %{y:.3g}<extra></extra>",
            )
        )
        fig.update_xaxes(title_text="frequency [Hz]")
        fig.update_yaxes(title_text="stored acoustic energy  [drive-scale]", type="log" if log else "linear")
        fig.update_layout(template=FNS_TEMPLATE_NAME, title=title or "Forced response: stored acoustic energy")
        fig.update_layout(**layout)
        return fig
