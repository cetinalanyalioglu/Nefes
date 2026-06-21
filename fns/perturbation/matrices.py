"""Transfer- and scattering-matrix algebra for the N-variable perturbation
network (theory.md s12.2, s12.7).

A **transfer matrix** (TM) relates the perturbation variables of two stations
(edges) along their own arrow: ``v_b = T_ba @ v_a``.  A **scattering matrix**
(SM) relates the *incoming* waves of the two stations to the *outgoing* ones,
the split coming from characteristic analysis -- the sign of each wave speed
``(u+c, u-c, u)`` at the mean state.  The two encode identical information and
convert into each other; a TM additionally re-expresses in any flavor
(``characteristics.basis_matrix``) by a per-station similarity.

Everything here is plain NumPy and dimension-generic (``n = 3`` for inert flow,
larger with reacting scalars): inputs are single ``(n, n)`` matrices or batched
``(n_omega, n, n)`` stacks.
"""

import numpy as np


def _as_batch(M):
    """Return (M3, was_2d) with M3 of shape (n_omega, n, n)."""
    M = np.asarray(M)
    if M.ndim == 2:
        return M[None, ...], True
    if M.ndim == 3:
        return M, False
    raise ValueError(f"expected (n, n) or (n_omega, n, n), got shape {M.shape}")


def wave_speeds(u, c, n=3):
    """Characteristic speeds ``(u+c, u-c, u)`` (padded with ``u`` for scalars)."""
    return np.array([u + c, u - c] + [u] * (n - 2), dtype=float)


def wave_signs(u, c, n=3, u_floor=1e-8):
    """Propagation sign of each characteristic: +1 downstream, -1 upstream.

    The two acoustic waves are unambiguous when subsonic; the convected waves
    (entropy and any scalar) carry the flow, so at a quiescent station
    (``|u| < u_floor``) they are pinned downstream (+1) -- their ``u -> 0+`` limit.
    """
    s = np.sign(wave_speeds(u, c, n))
    s[1] = -1.0 if (c - u) > 0 else 1.0  # upstream acoustic: -1 whenever subsonic
    for k in [0] + list(range(2, n)):
        if abs(wave_speeds(u, c, n)[k]) < u_floor:
            s[k] = 1.0
    return s


def partition(u, c, side, n=3, u_floor=1e-8):
    """Incoming/outgoing characteristic indices at one face of a cut.

    ``side='a'`` is the upstream face of an a->b segment: waves with speed > 0
    travel into the segment (incoming).  ``side='b'`` is the downstream face:
    waves with speed < 0 are incoming.  Returns ``(incoming_idx, outgoing_idx)``.
    """
    s = wave_signs(u, c, n, u_floor)
    into = 1.0 if side == "a" else -1.0
    incoming = tuple(int(i) for i in range(n) if s[i] == into)
    outgoing = tuple(int(i) for i in range(n) if s[i] != into)
    return incoming, outgoing


# --------------------------------------------------------------------------
# Basis (flavor) change of a transfer matrix
# --------------------------------------------------------------------------


def tm_in_basis(T_char, Ba, Bb):
    """Re-express a characteristic-basis TM in another flavor.

    ``T_char`` maps ``w_a -> w_b``; with ``v = B w`` at each station the same map
    reads ``v_b = (Bb T_char Ba^-1) v_a``.  ``Ba, Bb`` are the per-station
    ``basis_matrix`` blocks.  Accepts a single matrix or an ``(n_omega, n, n)``
    stack.
    """
    Tb, was2d = _as_batch(T_char)
    Bb = np.asarray(Bb, dtype=complex)
    Ba_inv = np.linalg.inv(np.asarray(Ba, dtype=complex))
    out = Bb[None, ...] @ Tb @ Ba_inv[None, ...]
    return out[0] if was2d else out


# --------------------------------------------------------------------------
# Transfer <-> scattering (characteristic amplitudes, any n)
# --------------------------------------------------------------------------


def scattering_labels(ua, ca, ub, cb, n=3, u_floor=1e-8):
    """Ordered (station, char-index) tags of the incoming and outgoing waves.

    Incoming = a's downstream-running waves then b's upstream-running waves;
    outgoing = a's upstream-running waves then b's downstream-running ones.
    """
    Ia, Oa = partition(ua, ca, "a", n, u_floor)
    Ib, Ob = partition(ub, cb, "b", n, u_floor)
    incoming = [("a", i) for i in Ia] + [("b", i) for i in Ib]
    outgoing = [("a", i) for i in Oa] + [("b", i) for i in Ob]
    return incoming, outgoing


def tm_to_sm(T_char, ua, ca, ub, cb, u_floor=1e-8):
    """Scattering matrix from a characteristic-basis transfer matrix.

    Given ``w_b = T w_a`` and the per-station wave split, returns ``S`` mapping the
    incoming amplitudes to the outgoing ones, ordered by ``scattering_labels``.
    Accepts a single matrix or an ``(n_omega, n, n)`` stack.
    """
    Tb, was2d = _as_batch(T_char)
    n = Tb.shape[-1]
    incoming, outgoing = scattering_labels(ua, ca, ub, cb, n, u_floor)
    n_in = len(incoming)
    if n_in != n:
        raise ValueError(
            f"non-square wave split: {n_in} incoming vs {n} characteristics "
            "(supersonic or degenerate station -- deferred in v1)"
        )

    def row_of(tag):  # selector row in the [w_a; w_b] (2n) layout
        st, i = tag
        r = np.zeros(2 * n, dtype=complex)
        r[i if st == "a" else n + i] = 1.0
        return r

    S = np.empty_like(Tb)
    sel_out = np.array([row_of(t) for t in outgoing])
    sel_in = np.array([row_of(t) for t in incoming])
    for k in range(Tb.shape[0]):
        # constraints: [T | -I] z = 0  (n rows);  selection of incoming = identity
        M = np.empty((2 * n, 2 * n), dtype=complex)
        M[:n, :n] = Tb[k]
        M[:n, n:] = -np.eye(n)
        M[n:, :] = sel_in
        rhs = np.vstack([np.zeros((n, n)), np.eye(n)])
        z = np.linalg.solve(M, rhs)  # (2n, n_in): full state per unit incoming
        S[k] = sel_out @ z
    return (S[0] if was2d else S), incoming, outgoing


def sm_to_tm(S, ua, ca, ub, cb, u_floor=1e-8):
    """Inverse of :func:`tm_to_sm`: characteristic transfer matrix from an SM."""
    Sb, was2d = _as_batch(S)
    n = Sb.shape[-1]
    incoming, outgoing = scattering_labels(ua, ca, ub, cb, n, u_floor)

    def emb(tags):  # 2n x n embedding of an ordered wave list into [w_a; w_b]
        E = np.zeros((2 * n, n), dtype=complex)
        for col, (st, i) in enumerate(tags):
            E[i if st == "a" else n + i, col] = 1.0
        return E

    Ein, Eout = emb(incoming), emb(outgoing)
    T = np.empty_like(Sb)
    for k in range(Sb.shape[0]):
        # full state z = Ein @ in + Eout @ out, with out = S @ in  -> z = (Ein+Eout S) in
        Z = Ein + Eout @ Sb[k]  # (2n, n): [w_a; w_b] per unit incoming
        Wa, Wb = Z[:n], Z[n:]
        T[k] = Wb @ np.linalg.inv(Wa)
    return T[0] if was2d else T


# --------------------------------------------------------------------------
# 2x2 acoustic helpers (entropy dropped) -- the classic conventions, kept for
# the "acoustics-only" preset and round-trips with external 2x2 data.
# --------------------------------------------------------------------------

_OMG = np.array([[0.5, 0.5], [0.5, -0.5]])  # (f, g) = OMG @ (p'/(rho c), u')


def tm_pu_to_fg(tm_pu):
    """2x2 acoustic TM from (p'/(rho c), u') to (f, g) coordinates."""
    M, was2d = _as_batch(tm_pu)
    out = _OMG[None] @ M @ np.linalg.inv(_OMG)[None]
    return out[0] if was2d else out


def tm_fg_to_pu(tm_fg):
    """2x2 acoustic TM from (f, g) to (p'/(rho c), u') coordinates."""
    M, was2d = _as_batch(tm_fg)
    out = np.linalg.inv(_OMG)[None] @ M @ _OMG[None]
    return out[0] if was2d else out


def tm_fg_to_sm2(tm_fg):
    """Classic 2x2 acoustic scattering matrix from a 2x2 (f, g) transfer matrix."""
    M, was2d = _as_batch(tm_fg)
    S = np.zeros_like(M)
    t11, t12, t21, t22 = M[:, 0, 0], M[:, 0, 1], M[:, 1, 0], M[:, 1, 1]
    S[:, 0, 0] = t11 - t12 * t21 / t22
    S[:, 0, 1] = t12 / t22
    S[:, 1, 0] = -t21 / t22
    S[:, 1, 1] = 1.0 / t22
    return S[0] if was2d else S
