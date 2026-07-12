"""Frequency-domain complex-matrix descriptors: TransferMatrix / ScatteringMatrix.

Round-trips of the transfer<->scattering and flavor conversions (machine precision),
plus real-grid evaluation and the rational continuation that makes a table analytic.
"""

import numpy as np
import pytest

from nefes.perturbation import PortState, ScatteringMatrix, TransferMatrix

FREQS = np.linspace(50.0, 1500.0, 25)
PA = PortState(rho=1.1, c=340.0, u=30.0, p=101325.0, area=0.05)
PB = PortState(rho=0.9, c=360.0, u=55.0, p=99000.0, area=0.03)


def _rand_tm(N, seed=0):
    rng = np.random.default_rng(seed)
    d = rng.standard_normal((FREQS.size, N, N)) + 1j * rng.standard_normal((FREQS.size, N, N))
    return d + 3.0 * np.eye(N)[None]  # diagonally dominant -> invertible per frequency


def _rel(a, b):
    return np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-12)


@pytest.mark.parametrize("N", [2, 3])
def test_transfer_scattering_round_trip(N):
    T = TransferMatrix(FREQS, _rand_tm(N), basis="char", ports=(PA, PB))
    S = T.to_scattering()
    assert isinstance(S, ScatteringMatrix) and S.n == N
    assert _rel(S.to_transfer().data, T.data) < 1e-9


@pytest.mark.parametrize("N", [2, 3])
def test_flavor_round_trip(N):
    T = TransferMatrix(FREQS, _rand_tm(N), basis="char", ports=(PA, PB))
    assert _rel(T.to_basis("primitive").to_basis("char").data, T.data) < 1e-9


@pytest.mark.parametrize("N", [2, 3])
def test_eval_and_resample(N):
    T = TransferMatrix(FREQS, _rand_tm(N), basis="char", ports=(PA, PB))
    assert _rel(T(FREQS), T.data) < 1e-9  # evaluating on the grid reproduces the data
    coarse = T.resample(FREQS[::2])
    assert coarse.n_freqs == FREQS[::2].size


@pytest.mark.parametrize("N", [2, 3])
def test_continuation_is_analytic(N):
    T = TransferMatrix(FREQS, _rand_tm(N), basis="char", ports=(PA, PB))
    with pytest.raises(ValueError):
        T(np.array([500.0 + 5.0j]))  # a raw table is real-axis only
    Tc = T.continue_(rtol=1e-10)
    assert Tc.analytic and _rel(Tc(FREQS), T.data) < 1e-8
    assert Tc(np.array([500.0 + 5.0j])).shape == (1, N, N)  # off the real axis: no raise


def test_impulse_continuation_of_a_finite_memory_scattering_matrix_is_exact():
    # entries built from finite-memory responses continue off the real axis exactly
    rng = np.random.default_rng(4)
    dt = 0.5 / FREQS.max()
    h = rng.standard_normal((2, 2, 7)) * 0.3 + 0.5 * np.eye(2)[..., None]
    lags = np.arange(7) * dt

    def exact(f):
        return (h * np.exp(-2j * np.pi * np.asarray(f, complex)[..., None, None, None] * lags)).sum(-1)

    S = ScatteringMatrix(FREQS, exact(FREQS), basis="char", ports=(PA, PB))
    Sc = S.continue_(method="impulse", duration=6.0 * dt, smoothing=0.0)
    assert Sc.analytic and Sc.max_fit_error() < 1e-10
    z = np.array([400.0 - 20.0j, 900.0 + 10.0j])
    assert _rel(Sc(z), exact(z)) < 1e-9
    with pytest.raises(ValueError):
        S.continue_(method="nope")


def test_constant_matrix_broadcasts():
    M = np.eye(3, dtype=complex)
    T = TransferMatrix(FREQS, M, basis="char", ports=(PA, PB))
    assert T.data.shape == (FREQS.size, 3, 3)
    assert _rel(T(np.array([123.0])), M[None]) < 1e-12


def test_missing_ports_rejects_conversion():
    T = TransferMatrix(FREQS, _rand_tm(3))  # no ports
    with pytest.raises(ValueError):
        T.to_scattering()
    with pytest.raises(ValueError):
        T.to_basis("primitive")


def test_scattering_basis_must_be_diagonal():
    with pytest.raises(ValueError):
        ScatteringMatrix(FREQS, _rand_tm(3), basis="primitive", ports=(PA, PB))
