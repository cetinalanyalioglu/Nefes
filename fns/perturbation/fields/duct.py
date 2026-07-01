"""The length-bearing acoustic duct: phase relations and modal analysis.

A uniform duct of length L carrying mean state ``(c, u)`` delays its three
characteristic waves by ``tau_+ = L/(u+c)``, ``tau_- = L/(c-u)`` and
``tau_0 = L/u`` (theory.md s12.3).  With reflection coefficients at the two ends
the acoustic eigenmodes solve ``det A(omega) = 0``; for a closed-closed
(rigid-rigid, R = +1) quiescent duct these are ``omega_n = n*pi*c/L``.
"""

# CA: Do we need this file at all?

from dataclasses import dataclass

import numpy as np

from .drivers import modes_from_det


@dataclass
class DuctAcoustics:
    """Acoustic model of a single uniform duct with end reflections."""

    c: float
    length: float
    u: float = 0.0

    @property
    def tau_plus(self):
        return self.length / (self.u + self.c)

    @property
    def tau_minus(self):
        return self.length / (self.c - self.u)

    def system(self, omega, R0, R1):
        """4x4 acoustic system in wave amplitudes ``(f0, g0, f1, g1)``.

        Rows: downstream phase ``f1 = Pp f0``, upstream phase ``g0 = Pm g1``, and the
        two end reflections expressed as *reflected = R x incident* at each
        termination -- the standard convention, matching the network operator: at
        the tail the upstream ``g0`` is incident and ``f0`` reflected (``f0 = R0
        g0``); at the head the downstream ``f1`` is incident and ``g1`` reflected
        (``g1 = R1 f1``).  The free modes then satisfy ``Pp Pm = 1/(R0 R1)``, so a
        passive end (``|R| < 1``) decays (``Im(omega) > 0`` under ``e^{+i*omega*t}``).
        """
        Pp = np.exp(-1j * omega * self.tau_plus)
        Pm = np.exp(-1j * omega * self.tau_minus)
        A = np.zeros((4, 4), dtype=np.complex128)
        # f1 - Pp*f0 = 0
        A[0, 2] = 1.0
        A[0, 0] = -Pp
        # g0 - Pm*g1 = 0
        A[1, 1] = 1.0
        A[1, 3] = -Pm
        # tail reflection: f0 - R0*g0 = 0   (incident g0 -> reflected f0)
        A[2, 0] = 1.0
        A[2, 1] = -R0
        # head reflection: g1 - R1*f1 = 0   (incident f1 -> reflected g1)
        A[3, 3] = 1.0
        A[3, 2] = -R1
        return A

    def det(self, omega, R0, R1):
        # LAPACK's complex LU trips the divide-by-zero/invalid FP flags during
        # intermediate steps even when the determinant is finite; the modal search
        # also probes omega at the genuine det -> 0 roots.  Both are expected here.
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.linalg.det(self.system(omega, R0, R1))


def duct_modes(c, length, n_modes=4, R0=1.0, R1=1.0, u=0.0, n_grid=4000):
    """First ``n_modes`` acoustic eigenfrequencies of a uniform duct.

    Closed-closed quiescent default (R0 = R1 = 1, u = 0) gives ``n*pi*c/L``.
    """
    da = DuctAcoustics(c, length, u)
    w_max = 1.15 * (n_modes + 1) * np.pi * c / length
    grid = np.linspace(0.05 * np.pi * c / length, w_max, n_grid)
    roots = modes_from_det(lambda w: da.det(w, R0, R1), grid)
    return roots[:n_modes]
