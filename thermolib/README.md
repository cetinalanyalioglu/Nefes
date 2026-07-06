# thermolib

A **standalone, lightweight thermochemistry library**: chemical equilibrium (HP) and thermodynamic properties of arbitrary gaseous mixtures, built around NASA-polynomial species data and a **CEA-style element-potential equilibrium kernel**. The reason this tool exists in the first place is to offer a **complex-step differentiable** thermo/chemistry backend.

## Install

Minimal runtime needs only `numpy` and `pyyaml`.
Cantera is optional (only required to run the full test suite).

## Capabilities

- **Chemical equilibrium** through a CEA-style element-potential (Lagrange-multiplier) kernel: `HP` (enthalpy/pressure) and `TP` (temperature/pressure) solves for arbitrary gaseous mixtures.
- **Mixture thermodynamics** from NASA polynomials: `cp`, `cv`, `gamma`, `h`, `s`, `rho`, mean molar mass, and the frozen speed of sound.
- **Equilibrium speed of sound** recovered from the converged sensitivity block, alongside the frozen value.
- **Complex-step differentiable** end to end: the whole property and equilibrium path is branch-free and complex-analytic, so a `x + i*eps` perturbation returns exact first derivatives (Jacobians for the host solver).
- **One canonical NASA-9 representation** (subsuming NASA-7), evaluated in a single vectorized expression over all species with no per-species Python loop.
- **Multiple data sources**: Cantera YAML via `from_cantera` (a file path is parsed directly with no Cantera dependency, a supported subset of the format; a live `cantera.Solution` is extracted through Cantera) and NASA-Glenn/CEA `thermo.inp` via `from_cea`; ships with vendored `h2o2.yaml` and `thermo.inp`.
- **Light footprint**: runtime needs only `numpy` and `pyyaml`; Cantera is optional and used only for offline import and validation.

## Usage

Load a library and evaluate frozen mixture properties at a thermodynamic point:

```python
import numpy as np
from thermolib import SpeciesLibrary, Thermo

gas = Thermo(SpeciesLibrary.from_cantera("thermolib/data/h2o2.yaml"))

idx = gas.library.species_index
X = np.zeros(gas.library.n_species)
X[idx["H2"]], X[idx["O2"]], X[idx["N2"]] = 2.0, 1.0, 3.76  # stoichiometric H2/air
Y = X * gas.library.molar_masses
Y /= Y.sum()

props = gas.properties(Y, T=1200.0, p=101325.0)
print(props.cp, props.gamma, props.a_frozen)
```

Adiabatic (HP) flame from the reactant state, with equilibrium composition and sound speed:

```python
Z = gas.elemental_mass_fractions(Y)      # elemental descriptor
h = gas.enthalpy_mass(Y, T=300.0)        # reactant enthalpy (conserved by HP)
res = gas.equilibrate_HP(Z, h, p=101325.0)
print(res.T, res.rho, res.a_equilibrium)
print(dict(zip(gas.library.species_index, res.Y)))
```

Exact derivatives by complex step, e.g. `dT/dh` at fixed `p`:

```python
eps = 1e-200
dTdh = gas.equilibrate_HP(Z, h + 1j * eps, p=101325.0).T.imag / eps
```

Load a subset directly from the packaged CEA `thermo.inp` database instead of a Cantera YAML file:

```python
lib = SpeciesLibrary.from_cea(species=["H2", "O2", "H2O", "OH", "H", "O", "N2"])
gas = Thermo(lib)
```

## Testing & validation

```bash
pytest  # 44 tests with Cantera; Cantera tests auto-skip if absent
```

Validation against Cantera (when installed) covers: species `cp/h/s` to machine precision, mixture properties, TP composition, HP adiabatic flame temperature, equilibrium sound speed, `K_c(T)`, a CH₄/air flame via GRI-Mech 3.0, and an H₂/air flame computed from **CEA `thermo.inp` NASA-9 data**.
Cantera-free tests cover conservation, realizability, HP↔TP round-trips, the species-library/mechanism separation, the vectorized-vs-per-species thermo, `thermo.inp` parsing, and the complex-step differentiation.
