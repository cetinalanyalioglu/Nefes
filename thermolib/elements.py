"""Standard atomic weights for the common combustion elements.

Used to convert between elemental mass fractions (the composition descriptor) and
gram-atoms per kilogram (the abundance vector ``b`` consumed by the element-potential
equilibrium solver).

Values are in kg/mol and match Cantera's defaults closely enough for validation; a
mechanism may override them via the YAML ``elements`` block.
"""

# kg/mol
ATOMIC_WEIGHTS = {
    "E": 5.4857990907e-7,  # electron (CEA element "E", for ionized species)
    "H": 1.008e-3,
    "D": 2.0141017780e-3,
    "He": 4.002602e-3,
    "C": 12.011e-3,
    "N": 14.007e-3,
    "O": 15.999e-3,
    "F": 18.998403163e-3,
    "Ne": 20.1797e-3,
    "Na": 22.98976928e-3,
    "Mg": 24.305e-3,
    "Al": 26.9815385e-3,
    "Si": 28.085e-3,
    "P": 30.973761998e-3,
    "S": 32.06e-3,
    "Cl": 35.45e-3,
    "Ar": 39.95e-3,
    "K": 39.0983e-3,
    "Ca": 40.078e-3,
    "Fe": 55.845e-3,
}


def normalize_element(symbol):
    """Canonicalize an element symbol's case (e.g. CEA's ``AR`` -> ``Ar``).

    The NASA ``thermo.inp`` database writes element symbols in upper case;
    :data:`ATOMIC_WEIGHTS` and Cantera use the conventional mixed case.  The
    electron pseudo-element ``E`` is left as-is.
    """
    s = symbol.strip()
    if not s:
        return s
    return s if len(s) == 1 else s[0].upper() + s[1:].lower()


def atomic_weight(symbol):
    """Return the standard atomic weight [kg/mol] for an element symbol."""
    try:
        return ATOMIC_WEIGHTS[symbol]
    except KeyError:
        pass
    norm = normalize_element(symbol)
    try:
        return ATOMIC_WEIGHTS[norm]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(
            f"Unknown element {symbol!r}; add it to thermolib.elements.ATOMIC_WEIGHTS "
            f"or specify 'atomic-weight' in the mechanism YAML."
        ) from exc
