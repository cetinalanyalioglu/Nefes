# Parameter schema and the modification API

This document records the implementation contract behind the named-parameter machinery: where parameter truth lives, how writes are validated, and why the design is shaped for nesting (composites today, subnetworks later).
The user-facing guide is [reference/parameters](../reference/parameters.md).

## The problem being designed out

Physical parameters historically had no names at the spec level.
They live positionally inside `ElementSpec.fparams` (a flat float list whose layout depends on `residual_id`), areas live on the edge triples, and a set of object-valued fields (`composition_spec`, `perturbation_bc`, `dynamic_source`, ...) sit beside them.
Validation lives in the factory functions, so a raw `fparams` mutation bypasses every range check and can silently produce a wrong or non-converging solve; a composite patched in place can drift from its derived internals.
A pile of bespoke setters would re-encode the factories' layout knowledge once per setter and would not scale to nested structure.

## Source of truth

Each fact has exactly one authoritative home.

- **Structure, validation and packing** live in the per-kind schema (`nefes.elements.parameters`).
  For every element kind it declares, once: which parameters exist, their `fparams` slot or field, SI unit, bounds, and type.
  A consistency test (`tests/test_parameters.py`) packs named values through the schema and compares against every factory's actual output, so the declared layout cannot drift from the factories.
- **Parameter values** live where they always have: `fparams` slots and named fields on atomic specs, the `params` dict on composites, areas on the edge records, and the reference scales on the `Network`.
  The schema adds names *over* this storage without moving it (option A of the design study; the flip that makes the named dict the canonical store and `fparams` a derived cache is step 4 below and is deliberately deferred).
- **Everything derived stays derived.**
  A composite's `sub_elements` and `internal_edges` are factory outputs; the one legal write path is re-running the factory with merged parameters (`rebuild_composite`), never patching them, so the knob and its derivation cannot disagree.

## Architecture, three layers

1. **Descriptor schema** (`nefes/elements/parameters.py`).
   `ParamDescriptor` declares one parameter: name, unit, bounds (with open/closed endpoints), target (`fparams[slot]`, a named field, a composite knob, or the whole vector for the forced splitter), value kind, and an optional extra validator.
   `ELEMENT_PARAMS` keys atomic kinds by `residual_id`; `COMPOSITE_PARAMS` keys composites by `kind`.
   Object-valued fields carry a type/constructor validator instead of numeric bounds (`PerturbationBC`, `DynamicSource`, `TransferMatrix`/`UnknownTransferMatrix`, composition dictionaries).
2. **Addressing and write paths** (`nefes/shell/params.py`).
   Address resolution (`element.param`, `edge.area`, bare network references), the inventory, element-by-name lookup, the constant-area fan-out, the composite rebuild swap, and `copy_network`.
   All writes validate first and fail closed: an unknown address raises with near-match suggestions, and `update` resolves every address before writing anything.
   `Network.get/set/update/parameters/copy/with_params/builder` are thin wrappers over this module.
3. **Study driver** (`nefes/shell/study.py`).
   `parameter_study` walks an N-D grid (or zipped path) of addresses, solving `base.with_params(point)` per point with warm starts chained through `solve(x0=prev.x)`.
   Eigenvalue and Nyquist continuation reuse their existing `build(p)` contract through `Network.builder`; no parallel sweep concept is introduced.

## Invariants

- **Warm-start invariance.**
  A parameter write never changes the node count, edge count or edge order, so the compiled layout is preserved and `x0=prev.x` stays valid; `with_params` preserves the layout by construction (`copy_network` copies the edge lists verbatim).
  The one declared exception is `fanno_pipe.n_segments`, a fidelity knob that re-discretizes the composite interior.
- **Fail-closed validation.**
  Every value passes its descriptor's bounds/type check before anything is stored; the error names the parameter, the bound and the element (`mdot must be >= 0 [kg/s] (got -0.1) on MassFlowInlet 'inlet'`).
- **Fail-closed addressing.**
  Unknown names and parameters raise with suggestions; an element/edge name collision on `area` raises as ambiguous rather than picking silently.
- **Vector length is topology.**
  The forced splitter's `fractions` may change values but never count (its port count is wired into the graph).
- **Composites rebuild atomically.**
  A batch write to one composite merges all its updates and re-runs the factory once; the embedded smoothing override (`eps`) is recovered from the sub-elements and preserved.
- **The gas model is out of scope.**
  Reconfiguring the thermochemistry changes `n_solve` and the species pool; it is a model re-specification behind the explicit construction path, not a value write.

## Deferred steps

Two steps of the original plan are intentionally not in this change and are tracked in `TODO.md`:

- **Canonical-store flip.**
  Making the named-parameter dict the stored truth on every spec and `fparams` a derived, invalidatable cache.
  It is the robustness payoff for deep nesting, guarded by a projection round-trip invariant (dict to `fparams` to dict is identity per kind) plus the complex-step safety sweep, and becomes necessary when subnetworks land.
- **Subnetworks.**
  The recursive node type (named dict plus children) with hierarchical addressing (`can[3].injector.orifice.throat_area`) and template/instance overlays.
  The address grammar and the schema registry here were shaped so that this is an extension, not a rewrite.

## Testing

`tests/test_parameters.py` pins the contract:

- schema/factory packing consistency for every atomic kind, and schema/params-dict name consistency for every composite kind;
- `get` after `set` returns the set value, including composite knobs and fanned-out areas;
- out-of-range and mistyped writes raise named errors and leave the network untouched;
- unknown addresses raise with suggestions;
- `with_params` leaves the base pristine, preserves the edge layout, and its copies warm-start from the base's solution;
- modified values survive the YAML save/load round-trip (including composite parameters and the serializable perturbation BCs);
- `parameter_study` shapes, warm chaining, zip mode, and fail-closed address resolution;
- reacting composition writes validate against the species library.
