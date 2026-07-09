"""Toggleable network validation checks (`nefes.shell.checks`).

Each guardrail check is gated by a global ``CHECK_*`` boolean; flipping it disables the check
process-wide, and the per-network ``require_connected`` override wins over the global.
"""

import pytest

from nefes.elements import catalog as cat
from nefes.shell import checks
from nefes.shell.network import Network
from nefes.thermo.configure import perfect_gas

CFG = perfect_gas(287.0, 1.4)


def _two_disconnected_circuits():
    """Two independent inlet -> outlet circuits sharing no edge (one model, two components)."""
    nodes = [
        cat.mass_flow_inlet(0.1, 300.0),
        cat.pressure_outlet(1e5, 300.0),
        cat.mass_flow_inlet(0.1, 300.0),
        cat.pressure_outlet(1e5, 300.0),
    ]
    edges = [(0, 1, 0.01), (2, 3, 0.01)]
    return nodes, edges


@pytest.fixture(autouse=True)
def _restore_toggles():
    """Every test starts and ends with the default toggle values."""
    saved = (checks.CHECK_CONNECTED, checks.CHECK_CONNECTIONS, checks.CHECK_CHOKED_NOZZLE)
    checks.CHECK_CONNECTED, checks.CHECK_CONNECTIONS, checks.CHECK_CHOKED_NOZZLE = True, True, True
    yield
    checks.CHECK_CONNECTED, checks.CHECK_CONNECTIONS, checks.CHECK_CHOKED_NOZZLE = saved


def test_disconnected_network_rejected_by_default():
    nodes, edges = _two_disconnected_circuits()
    with pytest.raises(ValueError, match="disconnected sub-networks"):
        Network(gas=CFG, nodes=nodes, edges=edges).compile()


def test_global_toggle_disables_connectivity_check():
    nodes, edges = _two_disconnected_circuits()
    checks.CHECK_CONNECTED = False
    prob = Network(gas=CFG, nodes=nodes, edges=edges).compile()  # no raise
    assert prob.n_nodes == 4


def test_per_network_override_beats_global():
    nodes, edges = _two_disconnected_circuits()
    # global on, but this network opts out
    prob = Network(gas=CFG, nodes=nodes, edges=edges, require_connected=False).compile()
    assert prob.n_nodes == 4
    # global off, but this network forces the check on
    checks.CHECK_CONNECTED = False
    with pytest.raises(ValueError, match="disconnected sub-networks"):
        Network(gas=CFG, nodes=nodes, edges=edges, require_connected=True).compile()


def test_connected_network_passes():
    nodes = [cat.mass_flow_inlet(0.1, 300.0), cat.duct(), cat.pressure_outlet(1e5, 300.0)]
    edges = [(0, 1, 0.01), (1, 2, 0.01)]
    Network(gas=CFG, nodes=nodes, edges=edges).compile()  # no raise


def test_adjacent_inlets_rejected():
    # two prescribed-inflow boundaries back to back: an incompatible pairing.  Both are total
    # pressure inlets so a pressure reference exists and the connection check is what fires.
    nodes = [cat.total_pressure_inlet(1.2e5, 300.0), cat.total_pressure_inlet(1.1e5, 300.0)]
    edges = [(0, 1, 0.01)]
    with pytest.raises(ValueError, match="incompatible pairing"):
        Network(gas=CFG, nodes=nodes, edges=edges).compile()


def test_connection_check_is_direction_independent():
    # the disallow rule is symmetric: the edge is rejected whichever inlet is the tail
    nodes = [cat.mass_flow_inlet(0.1, 300.0), cat.total_pressure_inlet(1.2e5, 300.0)]
    for edges in ([(0, 1, 0.01)], [(1, 0, 0.01)]):
        with pytest.raises(ValueError, match="incompatible pairing"):
            Network(gas=CFG, nodes=nodes, edges=edges).compile()


def test_global_toggle_disables_connection_check():
    nodes = [cat.total_pressure_inlet(1.2e5, 300.0), cat.total_pressure_inlet(1.1e5, 300.0)]
    edges = [(0, 1, 0.01)]
    checks.CHECK_CONNECTIONS = False
    Network(gas=CFG, nodes=nodes, edges=edges).compile()  # no raise
