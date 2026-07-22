"""Analytical validation of the distributed momentum-pipe formulation.

The oracle is the classical perfect-gas Fanno relation, inverted independently
with SciPy.  Nefes supplies only the numerical network solution being tested.
"""

import math

import numpy as np
import pytest
from scipy.optimize import brentq

import nefes
from nefes.elements import catalog as cat

R_AIR, GAMMA = 287.0, 1.4
CFG = nefes.perfect_gas(R_AIR, GAMMA)
D, F_DARCY = 0.05, 0.02
AREA = math.pi * D**2 / 4.0
P_IN, T_IN = 2.0e5, 300.0
M_IN = 0.3


def _fanno_parameter(M):
    return (1.0 - M**2) / (GAMMA * M**2) + (GAMMA + 1.0) / (2.0 * GAMMA) * np.log(
        (GAMMA + 1.0) * M**2 / (2.0 + (GAMMA - 1.0) * M**2)
    )


def _p_over_pstar(M):
    return np.sqrt((GAMMA + 1.0) / (2.0 + (GAMMA - 1.0) * M**2)) / M


def _T_over_Tstar(M):
    return (GAMMA + 1.0) / (2.0 + (GAMMA - 1.0) * M**2)


def _case(n_segments, M_out=0.7, reverse=False):
    length = D / F_DARCY * (_fanno_parameter(M_IN) - _fanno_parameter(M_out))
    Tt_in = T_IN * (1.0 + 0.5 * (GAMMA - 1.0) * M_IN**2)
    pt_in = P_IN * (1.0 + 0.5 * (GAMMA - 1.0) * M_IN**2) ** (GAMMA / (GAMMA - 1.0))
    p_out = P_IN * _p_over_pstar(M_out) / _p_over_pstar(M_IN)
    pipe = cat.fanno_pipe(length, D, F_DARCY, n_segments, name="fanno")
    if reverse:
        nodes = [cat.pressure_outlet(p_out, Tt_in), pipe, cat.total_pressure_inlet(pt_in, Tt_in)]
    else:
        nodes = [cat.total_pressure_inlet(pt_in, Tt_in), pipe, cat.pressure_outlet(p_out, Tt_in)]
    net = nefes.Network(CFG, nodes=nodes, edges=[(0, 1, AREA), (1, 2, AREA)])
    sol = net.solve()
    assert sol.converged, f"residual norm {sol.residual_norm}"
    sol.verify()

    cv = sol.composite("fanno")
    edge_ids = [0] + list(cv.internal_edges) + [1]
    x = np.linspace(0.0, length, n_segments + 1)
    if reverse:
        x = length - x
    M_exact = np.array(
        [
            brentq(
                lambda M: _fanno_parameter(M) - (_fanno_parameter(M_IN) - F_DARCY * xi / D),
                M_IN,
                M_out,
            )
            for xi in x
        ]
    )
    p_exact = P_IN * _p_over_pstar(M_exact) / _p_over_pstar(M_IN)
    T_exact = T_IN * _T_over_Tstar(M_exact) / _T_over_Tstar(M_IN)
    profile = {field: np.array([sol.edge(e)[field] for e in edge_ids]) for field in ("M", "p", "T")}
    profile["M"] = np.abs(profile["M"])
    errors = {
        "M": np.max(np.abs(profile["M"] - M_exact) / M_exact),
        "p": np.max(np.abs(profile["p"] - p_exact) / p_exact),
        "T": np.max(np.abs(profile["T"] - T_exact) / T_exact),
    }
    return sol, errors


def test_momentum_fanno_converges_to_analytic_profile():
    errors = [max(_case(n)[1].values()) for n in (8, 16, 32, 64)]
    assert all(fine < coarse for coarse, fine in zip(errors, errors[1:]))
    assert errors[-1] < 1.0e-3
    observed_order = np.log2(errors[-2] / errors[-1])
    assert observed_order > 1.8


def test_momentum_fanno_near_choke_converges_to_analytic_profile():
    errors = [max(_case(n, M_out=0.95)[1].values()) for n in (16, 32, 64)]
    assert all(fine < coarse for coarse, fine in zip(errors, errors[1:]))
    assert errors[-1] < 3.0e-3


def test_momentum_fanno_is_orientation_safe_in_reverse_flow():
    sol, errors = _case(64, reverse=True)
    assert np.all(sol.field("mdot") < 0.0)
    assert max(errors.values()) < 1.0e-3


@pytest.mark.parametrize("formulation", ["darcy-weisbach", "momentum"])
def test_zero_friction_pipe_is_lossless(formulation):
    net = nefes.Network(
        CFG,
        nodes=[
            cat.total_pressure_inlet(1.2e5, 300.0),
            cat.pipe(1.0, D, 0.0, formulation=formulation),
            cat.pressure_outlet(1.1e5, 300.0),
        ],
        edges=[(0, 1, AREA), (1, 2, AREA)],
    )
    sol = net.solve()
    assert sol.converged
    assert sol.edge(0)["p_t"] == pytest.approx(sol.edge(1)["p_t"], rel=1e-9)
