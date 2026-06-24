- [x] Capability to work with old "thermo.inp" — implemented `thermolib.cea`:
  `ThermoInp("thermo.inp")` parses the NASA Glenn / CEA NASA-9 database, supports
  `.search(...)`, `db["H2O"]`, and `db.library([...])` (also `SpeciesLibrary.from_cea`)
  to pull selected species into a working `SpeciesLibrary`. Inspired by
  `~/Projects/thermo-tools` but reworked to be complex-step safe and vectorized.
- [x] Stop calling the thermo material database a "mechanism." The thermo data is
  now a `SpeciesLibrary` (species + NASA polynomials — all equilibrium needs). A
  `Mechanism` is a `SpeciesLibrary` *plus* `Reaction`s that refer to its species,
  and is required only for kinetics (`equilibrium_constants_Kc`, the `net_rates`
  hook). `Thermo(...)` accepts either.
- [x] Performance — converted the species thermo to a vectorized form (all species
  in one array op over `(n_species, 9)` canonical NASA-9 coefficients; NASA-7 is
  embedded as the a1=a2=0 case). The complex-step differentiation contract is
  preserved (range selection branches only on `Re(T)`). A full JIT (JAX) path was
  considered and deliberately deferred: it would replace the complex-step contract
  and the IFT sensitivities that `cbnflow` depends on. The branch-free array core
  is the springboard if a numba/JAX backend is added later.
