"""Edge-closure resolution for reacting YAML networks (``_resolve_edge_models``).

All-``auto`` reacting networks defer to the orientation-proof burnt-marker closure (every edge
``None`` -> ``build_problem`` marker-gates), so a flame drawn any which way is handled correctly
with no warning.  The explicit hard-closure path (any ``frozen`` / ``equilibrium`` token) keeps
the declared-arrow flood-fill, whose labeling *is* load-bearing -- so a flame not drawn
flow-aligned is still warned about there.
"""

import warnings

from fns.io.yaml_in import _resolve_edge_models
from fns.elements.ids import FLAME_EQUILIBRIUM
from fns.thermo.api import EQ_FROZEN, EQ_KERNEL


class _Spec:
    def __init__(self, rid):
        self.residual_id = rid


# specs: node 1 is an equilibrium flame, flanked by two passthrough nodes
_SPECS = [_Spec(0), _Spec(FLAME_EQUILIBRIUM), _Spec(0)]


def _resolve(parsed, tokens):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        models = _resolve_edge_models(True, _SPECS, parsed, tokens)
    return models, [w for w in caught if "flow-aligned" in str(w.message)]


# -- all-auto: the orientation-proof marker closure --------------------------


def test_all_auto_defers_to_marker_no_warning():
    # flow-aligned: edge 0 (0->1 approach), edge 1 (1->2 downstream)
    parsed = [(0, 0, 1, 0.1, "a"), (1, 1, 2, 0.1, "b")]
    models, warns = _resolve(parsed, ["auto", "auto"])
    assert models == [None, None]  # defer to build_problem's marker-gating
    assert not warns


def test_all_auto_backward_flame_still_no_warning():
    # both edges point INTO the flame: the marker rides the signed flow, so no mislabel hazard
    parsed = [(0, 0, 1, 0.1, "a"), (1, 2, 1, 0.1, "b")]
    models, warns = _resolve(parsed, ["auto", "auto"])
    assert models == [None, None]
    assert not warns


# -- explicit / mixed: hard closure keeps the flood-fill + guard -------------


def test_explicit_hard_closure_labels_edges():
    parsed = [(0, 0, 1, 0.1, "a"), (1, 1, 2, 0.1, "b")]
    models, warns = _resolve(parsed, ["frozen", "equilibrium"])
    assert models == [EQ_FROZEN, EQ_KERNEL]
    assert not warns  # flow-aligned -> no warning


def test_mixed_auto_explicit_flood_fills_auto_edge():
    # one explicit token forces the hard-closure path; the auto edge is flood-filled (burnt
    # downstream of the flame along the declared arrows)
    parsed = [(0, 0, 1, 0.1, "a"), (1, 1, 2, 0.1, "b")]
    models, warns = _resolve(parsed, ["frozen", "auto"])
    assert models == [EQ_FROZEN, EQ_KERNEL]
    assert not warns


def test_explicit_closure_warns_on_bad_orientation():
    # both edges into the flame: the hard-closure flood-fill seeds nothing burnt -> mislabel.
    parsed = [(0, 0, 1, 0.1, "a"), (1, 2, 1, 0.1, "b")]
    _models, warns = _resolve(parsed, ["frozen", "equilibrium"])
    assert len(warns) == 1
    assert "in/out edges: 2/0" in str(warns[0].message)


def test_perfect_gas_no_guard():
    parsed = [(0, 0, 1, 0.1, "a"), (1, 2, 1, 0.1, "b")]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        models = _resolve_edge_models(False, _SPECS, parsed, ["auto", "auto"])
    assert models == [None, None]
    assert not [w for w in caught if "flow-aligned" in str(w.message)]
