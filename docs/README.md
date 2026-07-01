# FNS documentation

User-facing reference for the FNS flow-network solver. (The authoritative theory spec lives in
[`preliminary-study/docs/`](../preliminary-study/docs/); these pages document the shipped tool.)

## Element reference

- [Atomic elements](atomic-elements.md) — the 19 irreducible network elements (boundaries,
  area changes, losses, transport, reacting elements, manifolds), each with its residual and
  theory.
- [Composite elements](composite-elements.md) — convenience elements that expand to a graph of
  atomics at build time (orifice, lossy nozzle, sudden contraction, Helmholtz resonator, Fanno
  pipe, tapered duct).
