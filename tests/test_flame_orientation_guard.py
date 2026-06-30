"""Parse-time guard against an orientation-mislabeled equilibrium flame.

The ``auto`` frozen/equilibrium edge split floods "burnt" downstream of a flame along the
*declared* tail->head arrows.  A flame whose edges are not drawn flow-aligned (no outgoing
edge, or no incoming edge) is silently mislabeled, so the parser warns.  This is the interim
safety net; the orientation-proof fix is the transported burnt-marker closure.
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
_TOKENS = ["auto", "auto"]


def _resolve(parsed):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        models = _resolve_edge_models(True, _SPECS, parsed, _TOKENS)
    return models, [w for w in caught if "flow-aligned" in str(w.message)]


def test_flow_aligned_flame_no_warning():
    # edge 0: 0 -> 1 (approach), edge 1: 1 -> 2 (downstream) -> frozen then equilibrium
    parsed = [(0, 0, 1, 0.1, "a"), (1, 1, 2, 0.1, "b")]
    models, warns = _resolve(parsed)
    assert models == [EQ_FROZEN, EQ_KERNEL]
    assert not warns


def test_backward_flame_warns():
    # both edges point INTO the flame (out-degree 0): nothing gets seeded burnt
    parsed = [(0, 0, 1, 0.1, "a"), (1, 2, 1, 0.1, "b")]
    _models, warns = _resolve(parsed)
    assert len(warns) == 1
    assert "in/out edges: 2/0" in str(warns[0].message)


def test_source_flame_warns():
    # both edges point OUT of the flame (in-degree 0): no reactant approach
    parsed = [(0, 1, 0, 0.1, "a"), (1, 1, 2, 0.1, "b")]
    _models, warns = _resolve(parsed)
    assert len(warns) == 1
    assert "in/out edges: 0/2" in str(warns[0].message)


def test_explicit_closure_still_warns_on_bad_orientation():
    # the warning is about orientation, independent of whether the closure is auto or explicit;
    # an asymmetric flame is flagged regardless so the user notices the drawing problem.
    parsed = [(0, 0, 1, 0.1, "a"), (1, 2, 1, 0.1, "b")]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _resolve_edge_models(True, _SPECS, parsed, ["frozen", "equilibrium"])
    assert any("flow-aligned" in str(w.message) for w in caught)


def test_perfect_gas_no_guard():
    # non-reacting networks have no flame closure to mislabel
    parsed = [(0, 0, 1, 0.1, "a"), (1, 2, 1, 0.1, "b")]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        models = _resolve_edge_models(False, _SPECS, parsed, _TOKENS)
    assert models == [None, None]
    assert not [w for w in caught if "flow-aligned" in str(w.message)]
