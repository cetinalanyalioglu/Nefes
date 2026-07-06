"""Physical constants and standard-state conventions.

The values match Cantera's conventions so that the thermochemistry reproduces a
Cantera ideal-gas oracle bit-for-bit on the species thermodynamics (see
``tests/test_cantera_validation.py``).
"""

# Universal gas constant [J/(mol*K)] (CODATA, identical to Cantera's value
# expressed per mole: ct.gas_constant is 8314.462... J/(kmol*K)).
R_UNIVERSAL = 8.31446261815324

# Standard-state reference pressure [Pa] for the NASA polynomial thermo data.
# Cantera's ``reference_pressure`` defaults to one atmosphere.
P_REF = 101325.0

# Standard-state reference pressure [Pa] of the NASA Glenn / CEA ``thermo.inp``
# database (NASA-9 coefficients): one bar.  A species library carries its own
# ``P_ref`` so the pressure terms in entropy/equilibrium use the value the
# coefficients were actually referenced to.
P_REF_BAR = 1.0e5
