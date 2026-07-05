"""Analytic acoustics of a single uniform duct (verification oracle).

A uniform duct of length L carrying mean state ``(c, u)`` delays its three
characteristic waves by ``tau_+ = L/(u+c)``, ``tau_- = L/(c-u)`` and
``tau_0 = L/u``.  With reflection coefficients at the two ends the acoustic
eigenmodes solve ``det A(omega) = 0``; for a closed-closed (rigid-rigid, R = +1)
quiescent duct these are ``omega_n = n*pi*c/L``.

``DuctAcoustics`` (the 4x4 system ``A(omega)``) is the known-answer oracle the
network eigensolver is checked against; ``scattering_2port`` is the analytic
plane-wave transfer the operator's duct transfer matrix is checked against.
"""

from dataclasses import dataclass

import numpy as np


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


def scattering_2port(c, length, omega, u=0.0):
    """Acoustic transfer of a uniform duct: phase delays of the two acoustic waves.

    Returns ``diag(exp(-i*omega*tau_+), exp(-i*omega*tau_-))`` mapping the
    incoming wave amplitudes (downstream f at the tail, upstream g at the head)
    to the outgoing ones.  Lossless: both entries have unit modulus.
    """
    tau_p = length / (u + c)
    tau_m = length / (c - u)
    return np.array([[np.exp(-1j * omega * tau_p), 0.0], [0.0, np.exp(-1j * omega * tau_m)]])
