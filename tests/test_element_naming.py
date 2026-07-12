"""Element name assignment: a factory default is numbered, an explicit name is kept.

The dedup rule distinguishes a name the caller *chose* from the factory *default*:

* a factory default (the caller passed no ``name``) is always numbered, so a lone ``duct``
  reads ``duct-1`` and stays addressable unambiguously;
* an explicitly chosen name is kept bare when free, and suffixed only on an actual clash --
  even when it happens to equal the factory default (``name="flame"``).

This keeps a chosen name usable as a parameter address (``"flame.Qdot"``).
"""

import nefes
from nefes.elements import catalog as cat


def test_lone_default_is_numbered():
    net = nefes.Network(nodes=[cat.heat_release_flame(1000.0)], edges=[])
    assert net.element(0).name == "flame-1"


def test_explicit_name_equal_to_default_is_kept():
    net = nefes.Network()
    net.add(cat.heat_release_flame(1000.0, name="flame"))
    assert net.element(0).name == "flame"


def test_explicit_then_default_keeps_explicit_and_numbers_default():
    net = nefes.Network()
    net.add(cat.heat_release_flame(1000.0, name="flame"))  # chosen
    net.add(cat.heat_release_flame(1000.0))  # factory default
    assert [net.element(i).name for i in range(2)] == ["flame", "flame-1"]


def test_two_defaults_both_numbered():
    net = nefes.Network(nodes=[cat.duct(0.1), cat.duct(0.2)], edges=[])
    assert [net.element(i).name for i in range(2)] == ["duct-1", "duct-2"]


def test_two_explicit_names_clash_is_suffixed():
    net = nefes.Network()
    net.add(cat.duct(0.1, name="pipe"))
    net.add(cat.duct(0.2, name="pipe"))
    assert [net.element(i).name for i in range(2)] == ["pipe", "pipe-1"]


def test_chosen_name_is_a_usable_address():
    net = nefes.Network(
        nodes=[cat.mass_flow_inlet(1.0, 300.0), cat.heat_release_flame(5000.0, name="flame"), cat.pressure_outlet(1e5)],
        edges=[(0, 1, 0.01), (1, 2, 0.01)],
    )
    assert net.get("flame.Qdot") == 5000.0


def test_explicit_name_via_composite_is_kept():
    net = nefes.Network()
    net.add(cat.orifice(1.0e-3, name="plate"))
    assert net.element(0).name == "plate"


def test_composite_default_is_numbered():
    net = nefes.Network(nodes=[cat.orifice(1.0e-3)], edges=[])
    assert net.element(0).name == "orifice-1"
