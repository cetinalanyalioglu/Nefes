# thermolib

A **standalone, network-agnostic thermochemistry library**: chemical equilibrium (HP) and thermodynamic properties of arbitrary gaseous mixtures, built around NASA-polynomial species data and a **CEA-style element-potential equilibrium kernel**.

It has **no dependency on any flow-network concept** and **no runtime dependency on Cantera**: Cantera is used only as an offline mechanism importer and as a validation oracle in the test suite.

### Vocabulary: species library vs. mechanism

The thermochemical *material database* — a set of species, each with an element composition, molar mass, and a NASA polynomial — is a **`SpeciesLibrary`**.
That is **all chemical equilibrium needs**: the element-potential method works from species Gibbs energies, with no reactions involved.
A **`Mechanism`** is the *combination* of a species library with a set of **reactions** whose participants refer to species in that library; reactions are needed only for kinetics (the finite-rate design hook and the shared-Gibbs `K_c` route).
The word "mechanism" is reserved for that combination — it is not a synonym for the thermo database.

> Scope note: the finite-rate path is wired as a **design hook only**; this package delivers chemical equilibrium.

## What's implemented

| Area | Status |
|------|--------|
| Data ingestion (elements, species, NASA-7/9, reactions) | ✅ |
| Native format = subset of Cantera YAML (round-trips) | ✅ |
| Offline Cantera importer | ✅ `*.from_cantera` |
| NASA Glenn / CEA `thermo.inp` reader (NASA-9) | ✅ `ThermoInp` |
| Species library vs. mechanism separation | ✅ `SpeciesLibrary` / `Mechanism` |
| Vectorized, branch-free species thermo core | ✅ (complex-step kept) |
| Per-species `cp,h,s,g(T)`, complex-analytic | ✅ |
| Mixture properties | ✅ |
| Shared species Gibbs → equilibrium **and** `K_c(T)` | ✅ |
| Frozen **and** equilibrium speed of sound | ✅ |
| HP equilibrium, element-potential (CEA) formulation | ✅ native kernel |
| Pressure as ordinary input | ✅ |
| Exact derivatives (complex step / IFT through the solve) | ✅ |
| TP equilibrium for validation/reuse | ✅ |
| Finite-rate `net_rates` structure | ◐ design hook |
| Differentiation contract (modes; branch-free) | ✅ |
| Selectable backend (kernel / table) | ✅ (`table` deferred) |
| Validation oracle (skipped when Cantera absent) | ✅ |

## Install

```bash
conda env create -f environment.yml   # creates the 'thermolib' env (incl. Cantera)
conda activate thermolib
pip install -e .                        # editable install
```

Minimal runtime needs only `numpy` and `pyyaml`.
Cantera is optional (`pip install -e ".[cantera]"`) and offline-only.

## Usage

```python
import numpy as np
from thermolib import SpeciesLibrary, Thermo

lib = SpeciesLibrary.from_native("data/h2o2.yaml")  # or .from_cantera(...) / .from_cea(...)
gas = Thermo(lib, backend="kernel")                 # native equilibrium kernel

# Mixture properties at (Y, T, p)
Y = np.zeros(lib.n_species); Y[lib.species_index["H2O"]] = 1.0
props = gas.properties(Y, T=1500.0, p=101325.0)
print(props.cp, props.h, props.rho, props.a_frozen)

# HP equilibrium from an elemental composition (mass fractions), enthalpy, pressure
Z = {"H": 0.0285, "O": 0.2264, "N": 0.7451}      # e.g. stoichiometric H2/air elements
eq = gas.equilibrate_HP(Z, h=0.0, p=101325.0, T_guess=1800.0)
print(eq.T, eq.rho, eq.Y, eq.a_equilibrium)
```

`equilibrate_HP` / `equilibrate_TP` accept the elemental composition as a dict or an array aligned to `lib.elements`.
To go from a *species* state to the elemental descriptor, use `gas.elemental_mass_fractions(Y)`.

For reactions/kinetics (`equilibrium_constants_Kc`, the `net_rates` hook), build from a **`Mechanism`** instead: `gas = Thermo(Mechanism.from_native("data/h2o2.yaml"))`.

## Species libraries from `thermo.inp`

The NASA Glenn / CEA `thermo.inp` database (~2000 species, NASA-9 polynomials) is read directly.
Parse it once, search, and `select` the species you need into a `SpeciesLibrary`:

```python
from thermolib import ThermoInp, Thermo

db  = ThermoInp("data/thermo.inp")
db.search("H2O")                                  # -> ['H2O', 'H2O2', 'H2O(L)', ...]
lib = db.library(["H2", "O2", "H2O", "OH", "H", "O", "N2"])   # P_ref = 1 bar (CEA)
gas = Thermo(lib)                                  # equilibrium / properties as above
```

`SpeciesLibrary.from_cea(path, species=[...])` is the equivalent one-call form.
NASA-9 (CEA) and NASA-7 (Cantera/YAML) data share **one canonical internal representation**, so libraries from either source behave identically.
The CEA data is referenced to **one bar** and the Cantera/YAML data to **one atm**; each library carries its own `P_ref` so the entropy/equilibrium pressure terms use the value the coefficients were actually fit to.
An H₂/air flame computed from CEA data matches Cantera to within a couple of kelvin (the residual is genuine thermodynamic-data provenance, not numerics).

## Backends

* `backend="kernel"` — the native CEA-style element-potential kernel. Differentiation mode: *complex-transparent*.
* `backend="table"` — an offline table plus analytic surrogate. Not implemented; selecting it raises `NotImplementedError`.

## Differentiation contract

The whole property/equilibrium path is **complex-step differentiable**: it uses only smooth, complex-analytic operations (`thermolib.smooth` provides branch-free `abs/min/max/step`).
NASA-polynomial range selection branches only on `Re(T)` ("locate on the real part"), so derivatives propagate within a range.

This is also the form chosen for performance: all species share one canonical 9-term NASA representation and are evaluated in a **single vectorized expression** over `(n_species, 9)` coefficient arrays, with no per-species Python loop, while staying complex-step safe.
The branch-free array core is the right springboard if a JIT backend is added later.

For the equilibrium solve, derivatives are obtained by converging on the real parts and then taking **one undamped Newton step in log-variables** with the full complex inputs from the converged state.
Since the real residual is zero there, that step is exactly the implicit-function-theorem sensitivity, so e.g. `dT/dh` and `dT/dp` from a complex-step perturbation match finite differences to full precision (see `tests/test_equilibrium.py`).

## Equilibrium existence / robustness

The element-potential solve is a damped Newton iteration in log-variables (species moles, element potentials, total moles, and `ln T` for HP), following Gordon & McBride (NASA RP-1311).
Convergence is reliable within this envelope:

* **Composition**: every retained element has a strictly positive abundance (`b_i > 0`); elements/species with zero feed abundance are removed automatically, keeping the constraint matrix full rank.
* **Temperature**: within the mechanism's NASA-polynomial validity range (typically 200–3500/6000 K); the `ln T` step is damped (CEA control factor), keeping `T > 0` every iteration.
* **Pressure**: any `p > 0` (ordinary input).
* **Phase**: ideal gas only (no condensed species).

For HP, supply a reasonable `T_guess` (default 2000 K).
The solve uses log-space damping (RP-1311 eq. 3.1–3.2) so it tolerates guesses far from the solution; the converged equilibrium is also a natural warm start for nearby states.

## Testing & validation

```bash
pytest            # 44 tests with Cantera; Cantera tests auto-skip if absent
```

Validation against Cantera (when installed) covers: species `cp/h/s` to machine precision, mixture properties, TP composition, HP adiabatic flame temperature, equilibrium sound speed, `K_c(T)`, a CH₄/air flame via GRI-Mech 3.0, and an H₂/air flame computed from **CEA `thermo.inp` NASA-9 data**.
Cantera-free tests cover conservation, realizability, HP↔TP round-trips, the species-library/mechanism separation, the vectorized-vs-per-species thermo, `thermo.inp` parsing, and the complex-step differentiation contract.
