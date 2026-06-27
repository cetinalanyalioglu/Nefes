"""Acoustic-power diagnostics for the perturbation network (theory.md s12).

Linear disturbances on a *moving* mean flow do not carry energy in proportion to
``|p'|^2``: the mean convection biases the two acoustic characteristics, so a
downstream wave ``f`` and an upstream wave ``g`` of equal amplitude carry
different energy.  The time-averaged acoustic energy flux (Myers 1991) and energy
density across a uniform section ``(rho, c, M = u / c)`` are

.. math::

    I &= \\tfrac12 \\rho c \\,[\\,(1+M)^2 |f|^2 - (1-M)^2 |g|^2\\,]
        \\quad (\\text{flux / area, downstream } +) \\\\
    e &= \\tfrac12 \\rho   \\,[\\,(1+M)   |f|^2 + (1-M)   |g|^2\\,]
        \\quad (\\text{energy / volume})

so a wave's energy travels at its group speed ``u +/- c`` (``I / e = u + c`` for a
pure ``f`` wave).  Two consequences drive these diagnostics:

* The energy-neutral reflection magnitude of a through-flow boundary is **not** 1.
  An outlet returns all the acoustic power it receives at ``|R| = (1+M)/(1-M)``
  (the constant-mass-flow limit); an inlet at ``|R| = (1-M)/(1+M)``.  A larger
  ``|R|`` *adds* acoustic energy -- such a boundary is a source even though it was
  specified as merely "partially reflecting".  See :func:`passive_reflection_bound`.
* For a self-sustained eigenmode (complex ``omega``, growth ``sigma = -Im omega``)
  with no volume source, the global balance is
  ``dE/dt = 2 sigma E = sum of boundary power into the domain``.  Since ``E > 0``
  the **net boundary power and the growth rate share a sign**: :func:`boundary_power`
  attributes any instability to the boundaries that feed it.

These are post-processing diagnostics on an already-converged complex mode shape,
not residual math -- they use ``|.|^2`` freely and carry no complex-step
constraint.
"""

from dataclasses import dataclass
from typing import List

import numpy as np

from ..derive import ES_RHO, ES_C, ES_U, ES_M, ES_AREA
from ..elements.ids import MASS_FLOW_INLET, PT_INLET, WALL

_INLET_RIDS = (MASS_FLOW_INLET, PT_INLET)

# np.trapezoid is the NumPy >= 2.0 spelling; fall back to the older np.trapz.
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def acoustic_intensity(rho, c, mach, f, g):
    """Net time-averaged acoustic energy flux per unit area (downstream positive).

    Parameters
    ----------
    rho, c : float
        Mean density and sound speed of the section.
    mach : float
        Mean Mach number ``u / c`` in the downstream (``f``-wave) direction; may be
        signed (negative for reverse mean flow).
    f, g : complex or ndarray
        Downstream- and upstream-running characteristic wave amplitudes (arrays are
        taken element-wise, e.g. one entry per swept frequency).

    Returns
    -------
    float or ndarray
        ``1/2 rho c [(1+M)^2 |f|^2 - (1-M)^2 |g|^2]`` -- positive when net acoustic
        power flows downstream.
    """
    f2 = np.real(f * np.conj(f))
    g2 = np.real(g * np.conj(g))
    return 0.5 * rho * c * ((1.0 + mach) ** 2 * f2 - (1.0 - mach) ** 2 * g2)


def acoustic_energy_density(rho, mach, f, g):
    """Time-averaged acoustic energy per unit volume of a section.

    Parameters
    ----------
    rho : float
        Mean density of the section.
    mach : float
        Mean Mach number ``u / c`` (downstream positive).
    f, g : complex or ndarray
        Downstream- and upstream-running characteristic wave amplitudes (arrays are
        taken element-wise, e.g. one entry per swept frequency / axial station).

    Returns
    -------
    float or ndarray
        ``1/2 rho [(1+M) |f|^2 + (1-M) |g|^2]`` -- always non-negative for subsonic
        flow.
    """
    f2 = np.real(f * np.conj(f))
    g2 = np.real(g * np.conj(g))
    return 0.5 * rho * ((1.0 + mach) * f2 + (1.0 - mach) * g2)


def passive_reflection_bound(mach, side="outlet"):
    """Largest ``|R|`` a passive (non-energy-adding) through-flow boundary can have.

    With mean flow the energy-neutral reflection magnitude is shifted away from 1 by
    the ``(1 +/- M)^2`` energy bias.  A boundary with ``|R|`` above this bound feeds
    acoustic energy into the domain (an acoustic source); below it, it absorbs.

    Parameters
    ----------
    mach : float
        Approach Mach magnitude at the boundary.
    side : {'outlet', 'inlet'}, optional
        ``'outlet'`` (flow leaving, incident wave is downstream) gives
        ``(1+M)/(1-M)``; ``'inlet'`` (flow entering) gives ``(1-M)/(1+M)``.

    Returns
    -------
    float
        The energy-neutral reflection-coefficient magnitude.
    """
    m = abs(mach)
    if side == "outlet":
        return (1.0 + m) / (1.0 - m)
    if side == "inlet":
        return (1.0 - m) / (1.0 + m)
    raise ValueError("side must be 'inlet' or 'outlet'")


@dataclass
class BoundaryPower:
    """Per-terminal acoustic-power budget of one eigenmode.

    The eigenvector carries an arbitrary complex scale, so absolute powers are
    meaningless; use :attr:`fraction` (signed share of the gross throughput) and the
    *sign* of :attr:`net`.  By the global energy law ``2 sigma E = net``, a positive
    :attr:`net` must accompany a positive :attr:`growth_rate` -- see
    :attr:`sign_consistent`.

    Attributes
    ----------
    entries : list of dict
        One per terminal, with keys ``name``, ``edge``, ``kind``
        (``inlet``/``outlet``/``wall``), ``mach``, ``reflection`` (``|R|``),
        ``passive_bound`` (energy-neutral ``|R|``), ``power_in`` (acoustic power into
        the domain, mode-scale units) and ``fraction`` (``power_in`` / gross).
    growth_rate, freq_hz : float
        The mode's growth rate (1/s) and frequency (Hz).
    omega : complex
        The modal angular frequency (rad/s).
    """

    entries: List[dict]
    growth_rate: float
    freq_hz: float
    omega: complex = 0.0 + 0.0j

    @property
    def net(self) -> float:
        """Net acoustic power delivered to the domain (``= dE/dt``, mode-scale units)."""
        return float(sum(e["power_in"] for e in self.entries))

    @property
    def gross(self) -> float:
        """Sum of ``|power_in|`` over terminals (the normalizer for :attr:`fraction`)."""
        return float(sum(abs(e["power_in"]) for e in self.entries)) or 1.0

    @property
    def sources(self) -> List[dict]:
        """Terminals feeding acoustic energy into the domain (``power_in > 0``)."""
        return [e for e in self.entries if e["power_in"] > 0.0]

    @property
    def sinks(self) -> List[dict]:
        """Terminals draining acoustic energy from the domain (``power_in < 0``)."""
        return [e for e in self.entries if e["power_in"] < 0.0]

    @property
    def sign_consistent(self) -> bool:
        """Whether the net boundary power and the growth rate agree in sign.

        The energy-budget law ``2 sigma E = net`` (``E > 0``) requires this for any
        resolved eigenmode of a source-free network; a near-marginal mode
        (``net ~ 0``) is reported consistent.
        """
        return (self.net * self.growth_rate >= 0.0) or abs(self.net) < 1e-9 * self.gross

    def table(self) -> str:
        """A human-readable per-terminal power table (signed share of throughput)."""
        lines = [
            f"acoustic-power budget  f = {self.freq_hz:.2f} Hz  growth = {self.growth_rate:+.3f} 1/s",
            f"{'boundary':<16}{'kind':<8}{'M':>7}{'|R|':>9}{'neutral':>9}{'share':>9}",
        ]
        for e in sorted(self.entries, key=lambda d: d["power_in"]):
            lines.append(
                f"{e['name']:<16}{e['kind']:<8}{e['mach']:>7.3f}{e['reflection']:>9.3f}"
                f"{e['passive_bound']:>9.3f}{100.0 * e['fraction']:>8.1f}%"
            )
        tag = "drives growth" if self.net > 0 else "net dissipative"
        lines.append(f"{'NET':<16}{'':8}{'':7}{'':9}{'':9}{100.0 * self.net / self.gross:>8.1f}%   ({tag})")
        return "\n".join(lines)

    def plot(self, **kwargs):
        """Horizontal bar chart of each terminal's signed power share.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        import plotly.graph_objects as go

        ents = sorted(self.entries, key=lambda d: d["power_in"])
        names = [f"{e['name']} ({e['kind']})" for e in ents]
        shares = [100.0 * e["fraction"] for e in ents]
        colors = ["#d62728" if s > 0 else "#1f77b4" for s in shares]
        fig = go.Figure(go.Bar(x=shares, y=names, orientation="h", marker_color=colors))
        fig.add_vline(x=0.0, line_width=1, line_color="#888")
        title = kwargs.pop(
            "title", f"Boundary acoustic-power budget  ({self.freq_hz:.1f} Hz, growth {self.growth_rate:+.2f} 1/s)"
        )
        fig.update_layout(
            title=title,
            xaxis_title="share of acoustic-power throughput  [%]   (red = source, blue = sink)",
            template=kwargs.pop("template", "fns"),
            **kwargs,
        )
        return fig


def _kind(rid: int) -> str:
    if rid in _INLET_RIDS:
        return "inlet"
    if rid == WALL:
        return "wall"
    return "outlet"


def boundary_power(result, mode: int = 0, terminals=None) -> BoundaryPower:
    """Acoustic-power budget across the boundaries for one eigenmode.

    For each terminal, the net acoustic power *into the domain* is the Myers energy
    flux through the boundary face, signed by which side the domain is on.  Summed,
    it is the mode's energy growth rate ``dE/dt`` (up to the arbitrary mode scale);
    its sign must match the growth rate (:attr:`BoundaryPower.sign_consistent`).

    Parameters
    ----------
    result : EigenmodeResult
        A resolved eigenmode set (carries the frozen mean state and, when available,
        the boundary terminals).
    mode : int, optional
        Mode index (default 0).
    terminals : list of Terminal, optional
        Boundary terminals; taken from ``result.terminals`` when omitted.

    Returns
    -------
    BoundaryPower
        The per-terminal budget; see that class for the scale caveat.
    """
    terms = terminals if terminals is not None else getattr(result, "terminals", None)
    if not terms:
        raise ValueError("boundary_power needs the network terminals; pass terminals=find_terminals(prob, x_bar)")
    est = result.est
    names = result.node_names
    entries = []
    for t in terms:
        e = t.edge
        rho = float(est[ES_RHO, e])
        c = float(est[ES_C, e])
        mach = float(est[ES_M, e])
        area = float(est[ES_AREA, e])
        w = result.mode_waves(mode, e)
        f, g = complex(w[0]), complex(w[1])
        flux = acoustic_intensity(rho, c, mach, f, g) * area
        power_in = flux if t.at_tail else -flux
        arriving = abs(complex(w[t.outgoing]))  # wave leaving the domain (incident on boundary)
        returning = abs(complex(w[t.incoming]))  # wave the boundary sends into the domain
        refl = returning / arriving if arriving > 0.0 else float("inf")
        kind = _kind(t.rid)
        bound = passive_reflection_bound(mach, "inlet" if kind == "inlet" else "outlet")
        name = names[t.node] if t.node < len(names) else f"node{t.node}"
        entries.append(
            {
                "name": str(name),
                "edge": e,
                "kind": kind,
                "mach": mach,
                "reflection": refl,
                "passive_bound": bound,
                "power_in": power_in,
            }
        )
    gross = sum(abs(en["power_in"]) for en in entries) or 1.0
    for en in entries:
        en["fraction"] = en["power_in"] / gross
    return BoundaryPower(
        entries=entries,
        growth_rate=float(result.growth_rates[mode]),
        freq_hz=float(result.freqs[mode]),
        omega=complex(result.omega[mode]),
    )


# ---------------------------------------------------------------------------
# Forced-sweep power balance (a real-frequency drive, not a single eigenmode)
# ---------------------------------------------------------------------------


def duct_energy_spectrum(fr, ducts, *, n_x: int = 120):
    """Acoustic energy stored in the duct volumes at each forced frequency.

    A :class:`~fns.perturbation.ForcedResponse` carries the wave amplitudes only at
    the edge stations.  Inside each uniform duct the field is reconstructed in closed
    form -- the duct's own phase relation, ``f`` riding downstream at ``u + c`` and
    ``g`` upstream at ``c - u`` (theory.md s12.3) -- and the Myers energy density
    (:func:`acoustic_energy_density`) is integrated along the length and over the
    cross-section.

    Parameters
    ----------
    fr : ForcedResponse
        The solved forced field over a frequency sweep.
    ducts : iterable of DuctSegment
        The length-bearing duct segments (e.g. ``build_geometry(prob).ducts``); each
        carries its two face edges and its length.
    n_x : int, optional
        Interior samples per duct for the spatial integral (default 120).

    Returns
    -------
    ndarray
        Stored acoustic energy at each frequency, shape ``(n_freq,)``, in mode-scale
        units (the forcing amplitude sets the overall scale).
    """
    est = fr.est
    omega = 2.0 * np.pi * np.asarray(fr.freqs, dtype=float)
    energy = np.zeros(omega.size)
    for d in ducts:
        e_t, e_h, length = d.e_tail, d.e_head, d.length
        # A duct is uniform, so its tail-face mean state holds along the whole length.
        rho = float(est[ES_RHO, e_t])
        c = float(est[ES_C, e_t])
        u = float(est[ES_U, e_t])
        mach = float(est[ES_M, e_t])
        area = float(est[ES_AREA, e_t])
        s = np.linspace(0.0, length, n_x)
        # f propagates from the tail face, g from the head face; broadcast over (freq, station).
        f = fr.waves(e_t)[:, 0][:, None] * np.exp(-1j * omega[:, None] * s[None, :] / (u + c))
        g = fr.waves(e_h)[:, 1][:, None] * np.exp(-1j * omega[:, None] * (length - s)[None, :] / (c - u))
        e_dens = acoustic_energy_density(rho, mach, f, g)
        energy += _trapz(e_dens, s, axis=1) * area
    return energy


def boundary_power_spectrum(fr, terminals):
    """Net acoustic power into the domain across all terminals, per forced frequency.

    The forced-sweep analogue of :func:`boundary_power`'s single-mode net: the Myers
    acoustic intensity (:func:`acoustic_intensity`) through each terminal face, signed
    positive *into* the domain and summed.  A positive value means the boundaries feed
    net acoustic power to the drive; negative means they absorb it.

    Parameters
    ----------
    fr : ForcedResponse
        The solved forced field over a frequency sweep.
    terminals : iterable of Terminal
        The boundary terminals (e.g. ``find_terminals(prob)``).

    Returns
    -------
    ndarray
        Net into-domain acoustic power at each frequency, shape ``(n_freq,)``.
    """
    est = fr.est
    power = np.zeros(np.asarray(fr.freqs).size)
    for t in terminals:
        e = t.edge
        rho = float(est[ES_RHO, e])
        c = float(est[ES_C, e])
        mach = float(est[ES_M, e])
        area = float(est[ES_AREA, e])
        w = fr.waves(e)
        flux = acoustic_intensity(rho, c, mach, w[:, 0], w[:, 1]) * area
        # At a tail terminal the incident f-wave enters here, so downstream-positive flux is
        # power into the domain; at a head terminal the sign flips.
        power = power + (flux if t.at_tail else -flux)
    return power


@dataclass
class ForcedPowerBalance:
    """Energy stored in the ducts and net boundary power over a forced sweep.

    Both are per-frequency traces in arbitrary (mode-scale) units fixed by the drive
    amplitude, so read them by *shape*: the energy peaks locate the resonances, and
    the sign of :attr:`net_power` says whether the domain absorbs the drive
    (``> 0``, power into the domain) or radiates net acoustic power back out
    (``< 0`` -- the fingerprint of an active, self-amplifying element).

    Attributes
    ----------
    freqs : ndarray
        Sweep frequencies [Hz], shape ``(n_freq,)``.
    energy : ndarray
        Acoustic energy stored in the duct volumes at each frequency.
    net_power : ndarray
        Net acoustic power crossing all boundaries into the domain at each frequency.
    """

    freqs: np.ndarray
    energy: np.ndarray
    net_power: np.ndarray


def forced_power_balance(fr, prob, *, n_x: int = 120) -> ForcedPowerBalance:
    """Energy stored in the ducts and net boundary power, frequency by frequency.

    A one-call diagnostic for a forced frequency sweep: it reconstructs the intra-duct
    field to integrate the stored acoustic energy (:func:`duct_energy_spectrum`) and
    sums the Myers flux through the terminals (:func:`boundary_power_spectrum`).  The
    energy trace locates the resonances; the boundary-power trace shows whether the
    domain is absorbing the drive or feeding energy back out.

    Parameters
    ----------
    fr : ForcedResponse
        The solved forced field over a frequency sweep.
    prob : CompiledProblem
        The compiled network ``fr`` was solved on; supplies the duct geometry and the
        boundary terminals.
    n_x : int, optional
        Interior samples per duct for the energy integral (default 120).

    Returns
    -------
    ForcedPowerBalance
        The per-frequency stored-energy and net boundary-power traces.
    """
    from .modeshape import build_geometry
    from .terminals import find_terminals

    geo = build_geometry(prob)
    terms = find_terminals(prob)
    energy = duct_energy_spectrum(fr, geo.ducts, n_x=n_x)
    net_power = boundary_power_spectrum(fr, terms)
    return ForcedPowerBalance(freqs=np.asarray(fr.freqs, dtype=float), energy=energy, net_power=net_power)
