"""Per-port nominal flow-direction metadata (`ids.port_kinds`).

Phase-1 metadata only: the table declares each element's per-local-port role
(``PORT_TARGET`` / ``PORT_SOURCE`` / ``PORT_ANY``).  These tests pin the table against the
port-count map and the known conventions so it cannot drift; enforcement (the source->target
edge rule) is a later phase.
"""

import pytest

from nefes.elements import ids


ALL_KINDS = {ids.PORT_TARGET, ids.PORT_SOURCE, ids.PORT_ANY}

# Every residual id an element can carry (the ones with a human-readable type name).
ALL_RIDS = sorted(ids.ELEMENT_TYPE_NAMES)


def test_every_element_has_a_port_kind_rule():
    """port_kinds covers every residual id, for a representative degree."""
    for rid in ALL_RIDS:
        deg = ids.FIXED_NPORTS.get(rid, 3)  # manifolds: probe a valid variable degree
        kinds = ids.port_kinds(rid, deg)
        assert kinds, f"empty port-kind list for {ids.ELEMENT_TYPE_NAMES[rid]}"
        assert set(kinds) <= ALL_KINDS


def test_fixed_element_kind_count_matches_port_count():
    """A fixed-port element declares exactly one kind per port."""
    for rid, nports in ids.FIXED_NPORTS.items():
        assert len(ids.port_kinds(rid, nports)) == nports


@pytest.mark.parametrize("rid", [ids.MASS_FLOW_INLET, ids.PT_INLET, ids.SUPERSONIC_INLET])
def test_inlets_are_source_ports(rid):
    assert ids.port_kinds(rid, 1) == [ids.PORT_SOURCE]


@pytest.mark.parametrize(
    "rid",
    [ids.P_OUTLET, ids.MASS_FLOW_OUTLET, ids.CHOKED_NOZZLE_OUTLET, ids.SUPERSONIC_OUTLET, ids.WALL, ids.CAVITY],
)
def test_outlets_and_terminations_are_target_ports(rid):
    assert ids.port_kinds(rid, 1) == [ids.PORT_TARGET]


@pytest.mark.parametrize(
    "rid",
    [
        ids.DUCT,
        ids.PIPE,
        ids.LOSS,
        ids.LINEAR_RESISTANCE,
        ids.ISEN_AREA_CHANGE,
        ids.SUDDEN_AREA_CHANGE,
        ids.FLAME_HEAT_RELEASE,
        ids.FLAME_EQUILIBRIUM,
        ids.MASS_SOURCE,
        ids.TRANSFER_MATRIX,
    ],
)
def test_two_port_through_elements_are_target_then_source(rid):
    """A two-port through element takes flow in at port 0 and out at port 1."""
    assert ids.port_kinds(rid, 2) == [ids.PORT_TARGET, ids.PORT_SOURCE]


@pytest.mark.parametrize("rid", [ids.JUNCTION, ids.SPLITTER])
@pytest.mark.parametrize("deg", [2, 3, 5])
def test_symmetric_manifolds_are_all_any(rid, deg):
    assert ids.port_kinds(rid, deg) == [ids.PORT_ANY] * deg


@pytest.mark.parametrize("deg", [3, 4, 6])
def test_forced_splitter_is_inflow_target_then_outflow_sources(deg):
    """Port 0 is the single inflow (target); the remaining ports are forced outflows (sources)."""
    assert ids.port_kinds(ids.FORCED_SPLITTER, deg) == [ids.PORT_TARGET] + [ids.PORT_SOURCE] * (deg - 1)


def test_unknown_residual_id_raises():
    with pytest.raises(KeyError):
        ids.port_kinds(9999, 2)


def test_port_kind_names_cover_all_kinds():
    assert set(ids.PORT_KIND_NAMES) == ALL_KINDS
