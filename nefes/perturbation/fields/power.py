"""Acoustic-power diagnostics for the perturbation network.

On a moving mean flow, acoustic energy is not proportional to ``|p'|^2``: mean
convection biases the two characteristics, so downstream ``f`` and upstream ``g``
waves of equal amplitude carry different energy.  The time-averaged Myers (1991)
energy flux and density across a uniform section ``(rho, c, M = u / c)`` are

.. math::

    I &= \\tfrac12 \\rho c \\,[\\,(1+M)^2 |f|^2 - (1-M)^2 |g|^2\\,]
        \\quad (\\text{flux / area, downstream } +) \\\\
    e &= \\tfrac12 \\rho   \\,[\\,(1+M)   |f|^2 + (1-M)   |g|^2\\,]
        \\quad (\\text{energy / volume})

These drive three diagnostics:

* The energy-neutral reflection magnitude of a through-flow boundary is not 1 (an
  outlet at ``|R| = (1+M)/(1-M)``, an inlet at ``(1-M)/(1+M)``); a larger ``|R|``
  adds energy.  See :func:`passive_reflection_bound`.
* For an eigenmode (growth ``sigma = -Im omega``), net boundary power and growth
  rate share a sign, so :func:`boundary_power` attributes an instability to the
  boundaries feeding it.
* With interior sources this becomes a node-wise ledger
  ``2 sigma E = sum_interior Phi_n + boundary flux``, powering the forced-sweep
  budget (:func:`forced_power_balance`) and an energy-derived growth rate
  (:func:`modal_energy_balance`).

These are post-processing diagnostics on a converged complex mode shape, not
residual math -- they use ``|.|^2`` freely and carry no complex-step constraint.
"""

from dataclasses import dataclass
from typing import List

import numpy as np

from ...assembly.recover import ES_RHO, ES_C, ES_U, ES_M, ES_P, ES_AREA
from ...elements.ids import MASS_FLOW_INLET, PT_INLET, WALL
from ..operator.stamps import storage_stamps_from_est

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

    def __repr__(self) -> str:
        """Per-terminal acoustic-power budget table (see :meth:`table`)."""
        flag = "" if self.sign_consistent else "   [sign-inconsistent: net vs growth disagree]"
        return f"BoundaryPower ({len(self.entries)} terminals){flag}\n" + self.table()

    def _repr_html_(self) -> str:
        """Rich HTML budget table for notebooks: sources (red) and sinks (blue) by share."""
        tag = "drives growth" if self.net > 0 else "net dissipative"
        warn = "" if self.sign_consistent else " &nbsp;<b style='color:#c0392b'>sign-inconsistent</b>"
        header = (
            "<div style='font-family:sans-serif;margin-bottom:4px'>"
            "<b>BoundaryPower</b> &nbsp;&middot;&nbsp; "
            f"f = {self.freq_hz:.2f} Hz &nbsp;|&nbsp; growth = {self.growth_rate:+.3f} s<sup>-1</sup> "
            f"&nbsp;|&nbsp; net {100.0 * self.net / self.gross:+.1f}% ({tag}){warn}</div>"
        )
        th = "style='text-align:right;padding:2px 8px;border-bottom:1px solid #ccc'"
        thl = "style='text-align:left;padding:2px 8px;border-bottom:1px solid #ccc'"
        head_row = (
            f"<tr><th {thl}>boundary</th><th {thl}>kind</th><th {th}>M</th>"
            f"<th {th}>|R|</th><th {th}>neutral |R|</th><th {th}>share</th></tr>"
        )
        td = "style='text-align:right;padding:2px 8px'"
        tdl = "style='text-align:left;padding:2px 8px'"
        body = []
        for e in sorted(self.entries, key=lambda d: d["power_in"]):
            color = "#d62728" if e["power_in"] > 0 else "#1f77b4"
            body.append(
                f"<tr><td {tdl}>{e['name']}</td><td {tdl}>{e['kind']}</td>"
                f"<td {td}>{e['mach']:.3f}</td><td {td}>{e['reflection']:.3f}</td>"
                f"<td {td}>{e['passive_bound']:.3f}</td>"
                f"<td style='text-align:right;padding:2px 8px;color:{color}'>"
                f"<b>{100.0 * e['fraction']:+.1f}%</b></td></tr>"
            )
        table = (
            "<table style='border-collapse:collapse;font-family:monospace;font-size:0.9em'>"
            + head_row
            + "".join(body)
            + "</table>"
        )
        return header + table

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
            template=kwargs.pop("template", "nefes"),
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
# Acoustic-energy budget (node-wise): one ledger for forced sweeps and eigenmodes
# ---------------------------------------------------------------------------
#
# The time-averaged acoustic energy flux F_e = I_e * A_e lives on each edge.  At every
# node the net flux it sends into the network is its *produced* power,
#
#     Phi_n = sum_{e at n} sigma_{n,e} F_e ,   sigma = +1 at the tail, -1 at the head,
#
# (a flame is a source, Phi > 0; a passive scatterer / loss a sink, Phi < 0; a lossless
# duct conserves flux, Phi = 0).  Summing over all nodes telescopes to zero, which *is*
# the energy balance once the storage carried by the (length-bearing) ducts is split out:
#
#     d E / d t  =  sum_interior Phi_n  +  boundary flux ,
#
# zero on average for a steady forced state, and 2*sigma*E for a free eigenmode -- so the
# same ledger yields the growth rate, sigma = (generation + boundary flux) / (2 E).  Every
# term is a real time-averaged power.  Forced sweeps (per real frequency) and eigenmodes
# (per complex mode) share the helpers below, differing only in the field they read
# (`fr.waves` vs `result.mode_waves`) and whether omega is real.


def _edge_flux(waves, est, e):
    """Downstream-positive acoustic energy flux ``F_e`` at one edge, shape ``(K,)``.

    ``waves`` maps an edge to its ``(K, n_char)`` wave amplitudes (``K`` swept frequencies,
    or 1 for a single mode); the rest is the Myers intensity times the edge area.
    """
    e = int(e)
    w = waves(e)
    intensity = acoustic_intensity(float(est[ES_RHO, e]), float(est[ES_C, e]), float(est[ES_M, e]), w[:, 0], w[:, 1])
    return intensity * float(est[ES_AREA, e])


def _node_power(waves, est, tail, head, n_edges, node):
    """Net acoustic power produced at ``node``: ``sum_e sigma_{n,e} F_e``, shape ``(K,)``."""
    node = int(node)
    power = 0.0
    for e in range(int(n_edges)):
        if int(tail[e]) == node:
            power = power + _edge_flux(waves, est, e)  # downstream flux leaves the node here
        elif int(head[e]) == node:
            power = power - _edge_flux(waves, est, e)  # downstream flux enters the node here
    return power


def _stored_energy(waves, est, ducts, omega, n_x):
    """Acoustic energy stored in the duct volumes, shape ``(K,)``; ``omega`` shape ``(K,)``.

    The field is reconstructed inside each uniform duct from its end amplitudes -- ``f``
    riding downstream at ``u + c``, ``g`` upstream at ``c - u`` (a complex ``omega``
    capturing a mode's interior growth) -- and the Myers energy density
    (:func:`acoustic_energy_density`) integrated along the length and cross-section.
    """
    omega = np.asarray(omega, dtype=np.complex128)
    energy = np.zeros(omega.shape, dtype=float)
    for d in ducts:
        e_t, e_h, length = int(d.e_tail), int(d.e_head), float(d.length)
        # A duct is uniform, so its tail-face mean state holds along the whole length.
        rho = float(est[ES_RHO, e_t])
        c = float(est[ES_C, e_t])
        u = float(est[ES_U, e_t])
        mach = float(est[ES_M, e_t])
        area = float(est[ES_AREA, e_t])
        s = np.linspace(0.0, length, n_x)
        f = waves(e_t)[:, 0][:, None] * np.exp(-1j * omega[:, None] * s[None, :] / (u + c))
        g = waves(e_h)[:, 1][:, None] * np.exp(-1j * omega[:, None] * (length - s)[None, :] / (c - u))
        energy = energy + _trapz(acoustic_energy_density(rho, mach, f, g), s, axis=1) * area
    return energy


def _edge_network_var(w, est, e, v):
    """Network-variable perturbation amplitude on edge ``e`` from its wave amplitudes ``w``.

    ``w`` is ``(K, n_char)`` (``K`` swept frequencies or 1 mode) with columns ``(f, g, h)``;
    the maps are ``p' = rho c (f + g)`` and ``mdot' = A (u rho' + rho u')`` with
    ``rho' = h + p'/c^2`` and ``u' = f - g``.  Only ``v = 0`` (mass flow)
    and ``v = 1`` (pressure) are reconstructed -- the only solve variables a storage stamp
    couples -- and neither needs the caloric (``h_t``) row.
    """
    f = w[:, 0]
    g = w[:, 1]
    rho = float(est[ES_RHO, e])
    c = float(est[ES_C, e])
    p_prime = rho * c * (f + g)
    if v == 1:  # static pressure
        return p_prime
    if v == 0:  # mass flow
        h = w[:, 2] if w.shape[1] > 2 else 0.0  # entropy/convected wave (absent in the isentropic operator)
        rho_prime = h + p_prime / (c * c)
        u_prime = f - g
        return float(est[ES_AREA, e]) * (float(est[ES_U, e]) * rho_prime + rho * u_prime)
    raise ValueError(
        f"lumped-storage energy for solve-variable {v} is unsupported (only mass flow v=0 and "
        "pressure v=1 carry a storage stamp; the total-enthalpy store is the deferred heated-volume element)"
    )


def _lumped_storage_energy(stamps, est, waves, n_solve):
    """Acoustic energy stored in the lumped storage block ``M``, shape ``(K,)``.

    The element-independent complement of :func:`_stored_energy` (which integrates the
    *distributed* duct field): every storage stamp ``(row, col = n_solve*e + v, val)`` adds a
    term ``i*omega*val*x'_{e,v}`` onto a conservation row, and stores the time-averaged
    acoustic energy ``0.25 * |val|/rho_e * |x'_{e,v}|^2`` -- a compliance entry (``v = 1``) its
    potential energy ``0.25 (V/c^2)/rho |p'|^2``, an inertance entry (``v = 0``) its kinetic
    energy ``0.25 (L_eff/A)/rho |mdot'|^2``.  Walking the stamp triplets makes the ledger pick
    up any storage element -- cavity, inline area-change/loss, manifold plenum + neck -- with no
    per-element bookkeeping; the duct stores live only in ``_stored_energy``, so the two never
    double-count.

    Returns scalar ``0.0`` when the network carries no storage stamp (so it adds onto the duct
    energy spectrum harmlessly).
    """
    ns = int(n_solve)
    energy = 0.0
    for st in stamps:
        for col, val in zip(st.cols, st.vals):
            e = int(col) // ns
            v = int(col) % ns
            amp = _edge_network_var(waves(e), est, e, v)
            energy = energy + 0.25 * abs(complex(val)) / float(est[ES_RHO, e]) * np.real(amp * np.conj(amp))
    return energy


def _interior_nodes(geo, terminals):
    """Node ids that are neither length-bearing ducts nor 1-port terminals (the compact sources/sinks)."""
    duct_nodes = {int(d.node) for d in geo.ducts}
    term_nodes = {int(t.node) for t in terminals}
    return [n for n in range(int(geo.n_nodes)) if n not in duct_nodes and n not in term_nodes]


def _boundary_split(waves, est, terminals, node_bc, freqs):
    """Split the net boundary flux into a passive reflection part and an excitation source.

    The wave returning into the domain at a terminal is ``g = R f_out + d`` -- the reflection of
    the outgoing wave plus any imposed drive.  Scoring the boundary on the reflected wave alone
    (``R f_out``) gives the *reflection* flux (``~0`` for an energy-neutral reflector); the rest of
    the terminal's net flux is the *excitation source* ``d``.  An undriven terminal contributes its
    whole flux to the reflection part (no source).

    Returns ``(reflection, source)``, each shape ``(K,)`` and signed *into* the domain.
    """
    freqs = np.asarray(freqs, dtype=float)
    reflection = np.zeros(freqs.size)
    source = np.zeros(freqs.size)
    for t in terminals:
        e = int(t.edge)
        rho, c = float(est[ES_RHO, e]), float(est[ES_C, e])
        mach, area = float(est[ES_M, e]), float(est[ES_AREA, e])
        p = float(est[ES_P, e])
        sign = 1.0 if t.at_tail else -1.0
        w = waves(e)
        face_total = sign * acoustic_intensity(rho, c, mach, w[:, 0], w[:, 1]) * area
        bc = node_bc[t.node] if node_bc is not None and t.node < len(node_bc) else None
        if bc is not None and "acoustic" in getattr(bc, "driven", ()):
            # pass the mean pressure so the choked-nozzle gamma matches the operator assembly
            # (state form gamma = rho c^2 / p, backend-consistent) rather than the perfect-gas K
            r = np.array([complex(bc.reflection_coefficient(f, rho, c, mach, p=p)) for f in freqs])
            w_refl = w.copy()
            w_refl[:, t.incoming] = r * w[:, t.outgoing]  # the return wave without the drive
            face_refl = sign * acoustic_intensity(rho, c, mach, w_refl[:, 0], w_refl[:, 1]) * area
            reflection = reflection + face_refl
            source = source + (face_total - face_refl)
        else:
            reflection = reflection + face_total
    return reflection, source


# -- public per-quantity spectra (thin wrappers over the shared ledger) --------------------------------------------


def acoustic_flux_spectrum(fr, edge):
    """Downstream-positive time-averaged acoustic energy flux through ``edge``, per frequency.

    The Myers intensity (:func:`acoustic_intensity`) across the edge face times the area.  At a
    boundary edge it is the flux the terminal exchanges with the domain; the *jump* across a
    compact element is that element's acoustic power production (:func:`compact_power_spectrum`).

    Returns
    -------
    ndarray
        Shape ``(n_freq,)``.
    """
    return _edge_flux(fr.waves, fr.est, edge)


def compact_power_spectrum(fr, prob, node):
    """Net acoustic power produced by a compact element, ``Phi_n``, per frequency.

    The flux is conserved along a lossless duct, so the net flux a compact element injects is its
    own acoustic power production -- the signed sum of its face fluxes (out on the tail side, in on
    the head side).  An active flame gives ``> 0`` (it pumps the field); a passive scatterer ``< 0``.

    Returns
    -------
    ndarray
        Shape ``(n_freq,)``.
    """
    return _node_power(fr.waves, fr.est, prob.tail_node, prob.head_node, prob.n_edges, node)


def intensity_along_network(geometry, chars_of_edge, est, omega, *, energy_density=False, root=None, n_x=160):
    """Acoustic **intensity** (or energy density) along the developed length of the network.

    Reconstructs the interior field inside every duct (``f`` riding downstream at
    ``u + c``, ``g`` upstream at ``c - u``) and evaluates the Myers
    energy flux per unit area :func:`acoustic_intensity` (downstream positive) at each
    station, returning one developed-length :class:`~nefes.perturbation.fields.modeshape.PathField`
    per root->leaf path.  The companion to the mode-shape field reconstruction, but the
    value is a **real** time-averaged power density (its ``values`` carry no imaginary
    part), so it reads where acoustic power flows and where it is produced or absorbed.

    Parameters
    ----------
    geometry : NetworkGeometry
        Topology and duct lengths (from :func:`nefes.perturbation.build_geometry`).
    chars_of_edge : callable
        ``edge -> (f, g, h)`` complex wave amplitudes at that edge's face (the same
        accessor the mode-shape reconstruction uses).
    est : ndarray
        Frozen mean edge-state table.
    omega : complex
        Angular frequency [rad/s]; complex for an eigenmode, real for a forced field.
    energy_density : bool, optional
        Return the Myers energy **density** [J/m^3] (:func:`acoustic_energy_density`)
        instead of the intensity [W/m^2] (default ``False``).
    root : int, optional
        Developed-length origin element (default: a mean-flow inlet).
    n_x : int, optional
        Interior samples per duct (default 160).

    Returns
    -------
    list of nefes.perturbation.fields.modeshape.PathField
        One per root->leaf path; ``values`` is the real intensity (or energy density).
    """
    from .modeshape import walk_paths, _duct_chars, PathField

    def _quantity(rho, c, mach, f, g):
        if energy_density:
            return acoustic_energy_density(rho, mach, f, g)
        return acoustic_intensity(rho, c, mach, f, g)

    def duct_fn(seg):
        e_t, e_h, length = seg.e_tail, seg.e_head, seg.length
        rho, c = float(est[ES_RHO, e_t]), float(est[ES_C, e_t])
        u, mach = float(est[ES_U, e_t]), float(est[ES_M, e_t])
        s, chars = _duct_chars(chars_of_edge(e_t), chars_of_edge(e_h), c, u, omega, length, n_x)
        return s, _quantity(rho, c, mach, chars[:, 0], chars[:, 1])

    def point_fn(rep):
        rho, c, mach = float(est[ES_RHO, rep]), float(est[ES_C, rep]), float(est[ES_M, rep])
        w = chars_of_edge(rep)
        return _quantity(rho, c, mach, complex(w[0]), complex(w[1]))

    fields = walk_paths(geometry, duct_fn, point_fn, root=root, n_x=n_x)
    # The diagnostic is real; drop the (numerically zero) imaginary part the walker carries.
    return [PathField(name=f.name, x=f.x, values=np.real(f.values), markers=f.markers) for f in fields]


def duct_energy_spectrum(fr, ducts, *, n_x: int = 120):
    """Acoustic energy stored in the duct volumes at each forced frequency.

    Parameters
    ----------
    fr : ForcedResponse
        The solved forced field over a frequency sweep.
    ducts : iterable of DuctSegment
        The length-bearing duct segments (e.g. ``build_geometry(prob).ducts``).
    n_x : int, optional
        Interior samples per duct for the spatial integral (default 120).

    Returns
    -------
    ndarray
        Stored acoustic energy at each frequency, shape ``(n_freq,)`` (drive-scale units).
    """
    return _stored_energy(fr.waves, fr.est, ducts, 2.0 * np.pi * np.asarray(fr.freqs, dtype=float), n_x)


# -- forced sweep: the node-wise energy budget ---------------------------------------------------------------------


@dataclass
class ForcedPowerBalance:
    """Node-wise acoustic-energy budget of a forced sweep, frequency by frequency.

    In a steady forced state the stored energy is constant on average, so the budget closes::

        generation + boundary_reflection + boundary_source  ~  0   ( = -dissipation ).

    The three buckets are the net power produced by the interior elements (:attr:`generation`,
    the flame's Rayleigh source ``> 0``), the flux through the boundary *reflectors*
    (:attr:`boundary_reflection`, ``~0`` -- a rigid wall ``u'=0`` or open end ``p'=0`` carries no
    net flux of its own), and the power injected by the boundary *excitation*
    (:attr:`boundary_source`, the drive).  Keeping the excitation out of the reflector flux is the
    point of the split: the reflectors stay near zero and the drive mirrors the generation.  All
    terms are real time-averaged powers in arbitrary (drive-scale) units; read them by shape and
    sign.

    Attributes
    ----------
    freqs : ndarray
        Sweep frequencies [Hz].
    energy : ndarray
        Acoustic energy stored in the duct volumes.
    generation : ndarray
        Net acoustic power produced by the interior (compact) elements.
    boundary_reflection : ndarray
        Net flux through the boundary reflectors (excitation excluded).
    boundary_source : ndarray
        Net acoustic power injected by the boundary excitation.
    """

    freqs: np.ndarray
    energy: np.ndarray
    generation: np.ndarray
    boundary_reflection: np.ndarray
    boundary_source: np.ndarray

    @property
    def net_boundary_flux(self) -> np.ndarray:
        """Total net flux into the domain across the boundaries, ``boundary_reflection + boundary_source``."""
        return self.boundary_reflection + self.boundary_source

    @property
    def residual(self) -> np.ndarray:
        """Budget closure ``generation + net_boundary_flux`` -- ``~0``; its size is the numerical dissipation."""
        return self.generation + self.net_boundary_flux

    def __repr__(self) -> str:
        """One-line sweep summary: frequency span and the worst budget-closure residual."""
        f = np.asarray(self.freqs, dtype=float)
        n = f.size
        if n == 0:
            return "ForcedPowerBalance (empty sweep)"
        gross = float(np.max(np.abs(self.generation))) or 1.0
        worst = float(np.max(np.abs(self.residual)))
        span = f"f in [{f.min():.1f}, {f.max():.1f}] Hz" if n > 1 else f"f = {f[0]:.1f} Hz"
        return (
            f"ForcedPowerBalance: {n} frequenc{'y' if n == 1 else 'ies'}, {span}\n"
            f"  max |generation| = {gross:.3g}, worst closure |residual| = {worst:.2g} "
            f"({100.0 * worst / gross:.2g}% of generation)"
        )


def forced_power_balance(fr, prob, *, n_x: int = 120) -> ForcedPowerBalance:
    """Node-wise acoustic-energy budget of a forced frequency sweep.

    Assembles the ledger: the interior generation/sink (summed over the compact nodes), the
    boundary flux split into its reflector and excitation parts, and the stored duct energy.  In a
    steady forced state ``generation + boundary_reflection + boundary_source ~ 0``.

    Parameters
    ----------
    fr : ForcedResponse
        The solved forced field over a frequency sweep.
    prob : CompiledProblem
        The compiled network ``fr`` was solved on (geometry, terminals, and terminal BCs).
    n_x : int, optional
        Interior samples per duct for the energy integral (default 120).

    Returns
    -------
    ForcedPowerBalance
    """
    from .modeshape import build_geometry
    from ..operator.terminals import find_terminals

    geo = build_geometry(prob)
    terms = find_terminals(prob)
    waves, est = fr.waves, fr.est
    generation = np.zeros(np.asarray(fr.freqs).size)
    for node in _interior_nodes(geo, terms):
        generation = generation + _node_power(waves, est, geo.tail_node, geo.head_node, geo.n_edges, node)
    reflection, source = _boundary_split(waves, est, terms, getattr(prob, "node_bc", None), fr.freqs)
    energy = _stored_energy(waves, est, geo.ducts, 2.0 * np.pi * np.asarray(fr.freqs, dtype=float), n_x)
    # add the lumped storage block's energy (cavity/manifold compliance + neck inertance)
    energy = energy + _lumped_storage_energy(storage_stamps_from_est(prob, est), est, waves, prob.n_solve)
    return ForcedPowerBalance(
        freqs=np.asarray(fr.freqs, dtype=float),
        energy=energy,
        generation=generation,
        boundary_reflection=reflection,
        boundary_source=source,
    )


# -- eigenmode: the same ledger, yielding the growth rate ----------------------------------------------------------


@dataclass
class ModalEnergyBalance:
    """Acoustic-energy budget of one eigenmode, and the growth rate it implies.

    A free mode carries no excitation, so the budget is ``2 sigma E = generation + boundary_flux``
    (energy-neutral ends make ``boundary_flux ~ 0``, so the growth is the trapped generation).
    This gives an **energy-derived growth rate** independent of the contour eigenvalue,

        growth_rate_energy = (generation + boundary_flux) / (2 * stored_energy),

    which must match :attr:`growth_rate` -- a cross-check on the eigensolver, not a restatement.

    Attributes
    ----------
    freq_hz : float
        Modal frequency (Hz).
    growth_rate : float
        The contour eigenvalue's growth rate ``-Im(omega)`` (1/s).
    growth_rate_energy : float
        The growth rate from the energy budget, ``(generation + boundary_flux) / (2 E)`` (1/s).
    generation : float
        Net acoustic power produced by the interior elements (mode-scale units).
    boundary_flux : float
        Net acoustic energy flux into the domain across the boundaries (mode-scale units).
    stored_energy : float
        Total acoustic energy stored in the ducts (mode-scale units).
    """

    freq_hz: float
    growth_rate: float
    growth_rate_energy: float
    generation: float
    boundary_flux: float
    stored_energy: float

    @property
    def consistent(self) -> bool:
        """Whether the energy-derived and contour growth rates agree (to 1% of the modal scale)."""
        scale = max(abs(self.growth_rate), 2.0 * np.pi * abs(self.freq_hz), 1.0)
        return abs(self.growth_rate_energy - self.growth_rate) < 1e-2 * scale

    def __repr__(self) -> str:
        """Energy budget of the mode plus the two growth rates it must reconcile."""
        ok = "agree" if self.consistent else "DISAGREE"
        return (
            f"ModalEnergyBalance: f = {self.freq_hz:.2f} Hz\n"
            f"  generation     = {self.generation:+.4g}\n"
            f"  boundary_flux  = {self.boundary_flux:+.4g}\n"
            f"  stored_energy  = {self.stored_energy:.4g}\n"
            f"  growth (contour) = {self.growth_rate:+.4g} 1/s\n"
            f"  growth (energy)  = {self.growth_rate_energy:+.4g} 1/s   [{ok}]"
        )

    def _repr_html_(self) -> str:
        """Rich HTML energy budget and growth-rate cross-check for notebooks."""
        ok = self.consistent
        verdict = (
            "<span style='color:#2a8a4a'>agree</span>"
            if ok
            else "<span style='color:#c0392b;font-weight:bold'>disagree</span>"
        )
        td = "style='text-align:right;padding:2px 8px'"
        tdl = "style='text-align:left;padding:2px 8px'"
        rows = [
            ("generation", f"{self.generation:+.4g}"),
            ("boundary flux", f"{self.boundary_flux:+.4g}"),
            ("stored energy", f"{self.stored_energy:.4g}"),
            ("growth rate (contour)", f"{self.growth_rate:+.4g} s<sup>-1</sup>"),
            ("growth rate (energy)", f"{self.growth_rate_energy:+.4g} s<sup>-1</sup>"),
        ]
        body = "".join(f"<tr><td {tdl}>{k}</td><td {td}>{v}</td></tr>" for k, v in rows)
        return (
            "<div style='font-family:sans-serif;margin-bottom:4px'>"
            f"<b>ModalEnergyBalance</b> &nbsp;&middot;&nbsp; f = {self.freq_hz:.2f} Hz "
            f"&nbsp;|&nbsp; growth rates {verdict}</div>"
            "<table style='border-collapse:collapse;font-family:monospace;font-size:0.9em'>" + body + "</table>"
        )


def modal_energy_balance(result, mode: int = 0, *, n_x: int = 160) -> ModalEnergyBalance:
    """Acoustic-energy budget and energy-derived growth rate of one eigenmode.

    Reads the mode shape off ``result`` and forms the same ledger the forced balance uses --
    interior generation, boundary flux, stored duct energy -- then returns the growth rate the
    budget implies, ``(generation + boundary_flux) / (2 E)``, beside the contour eigenvalue's.

    Parameters
    ----------
    result : EigenmodeResult
        A resolved eigenmode set (must carry its geometry and terminals).
    mode : int, optional
        Mode index (default 0).
    n_x : int, optional
        Interior samples per duct for the stored-energy integral (default 160).

    Returns
    -------
    ModalEnergyBalance
    """
    geo = getattr(result, "geometry", None)
    if geo is None:
        raise ValueError("modal_energy_balance needs the network geometry; rebuild via eigenmodes()")
    est = result.est
    terms = result.terminals or []

    def waves(e):
        return result.mode_waves(mode, int(e))[None, :]  # (1, n_char) for the shared ledger

    generation = 0.0
    for node in _interior_nodes(geo, terms):
        generation += float(_node_power(waves, est, geo.tail_node, geo.head_node, geo.n_edges, node)[0])
    boundary_flux = float(boundary_power(result, mode, terminals=terms).net) if terms else 0.0
    duct_energy = _stored_energy(waves, est, geo.ducts, np.array([complex(result.omega[mode])]), n_x)
    # add the lumped storage block's energy (cavity/manifold compliance + neck inertance)
    lumped = _lumped_storage_energy(getattr(result, "storage", None) or [], est, waves, result.n_solve)
    energy = float((duct_energy + lumped)[0])
    sigma = (generation + boundary_flux) / (2.0 * energy) if energy != 0.0 else float("nan")
    return ModalEnergyBalance(
        freq_hz=float(result.freqs[mode]),
        growth_rate=float(result.growth_rates[mode]),
        growth_rate_energy=sigma,
        generation=generation,
        boundary_flux=boundary_flux,
        stored_energy=energy,
    )
