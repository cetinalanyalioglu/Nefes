---
name: nefes-dev
description: >-
  Use when developing the Nefes package itself -- editing the solver, the compiled
  element kernels, the assembly or the derivative engine; adding, changing, or removing
  an element type; touching the thermochemistry engine or the gas-model boundary;
  writing or fixing package tests; or updating the docs under `docs/`. Covers the
  kernel/residual architecture and the complex-step discipline every residual owes, the
  end-to-end recipe for adding an element, the edit-a-kernel loop (including the numba
  stale-cache trap), where each source of truth lives, and the formatting/commit gate.
  This is the counterpart of the `nefes-user` skill: reach for `nefes-user` when the task
  is to *use* Nefes (build a `Network`, solve, analyze); reach for this when the task is to
  *change* Nefes. Do NOT use it for user-facing modeling in a notebook or script -- that is
  `nefes-user`'s job.
---

# Nefes (dev): change the package without breaking its contracts

You are developing Nefes, not using it.
The package solves the steady mean flow of a compressible-flow network and the linear acoustic/entropy behavior around it; its correctness rests on a few contracts that do not announce themselves when broken, so the work is as much about honoring those contracts as writing the code.

## The authority: `docs/design/` and `docs/theory/`

`docs/design/` is the source of truth for *how the package is built*, the developer's analog of the user skill's `docs/best-practices.md`:

- `philosophy.md` — the principles the rest follows from (kernels over objects, exact derivatives, smoothness over branching).
- `kernel-architecture.md` — integer `residual_id` dispatch, and one source compiled twice (`float64` + `complex128`).
- `complex-step.qmd` and `smoothness-contract.md` — the exact-derivative engine and the discipline it demands.
- `solver.md`, `assembly.md`, `reproducibility.md` — the Newton/continuation solver, the residual/Jacobian assembly and acoustic stamps, and determinism.

`docs/theory/` is the source of truth for *the physics* (governing equations, per-element closures, choking, the perturbation network); `docs/nomenclature.md` is the symbol table.
Read the relevant document before you design, and prefer it over memory.
This skill deliberately does **not** restate the equations, signatures, or file internals: a second copy drifts and then produces confident, wrong changes.

## The hard rules live in `CLAUDE.md`, not here

`CLAUDE.md` owns the non-negotiable constraints and is always in context; this skill is the *how*, never a second copy of the *what*.
The two that govern almost every change:

- **Complex-step safety.** Every residual must be smooth and complex-analytic: no `abs`/`sign`/`min`/`max` and no branch taken on the flow state. Jacobians come from the complex step, so a non-analytic residual silently corrupts the derivative.
- **Subsonic scope.** Flowing or quiescent, at or below a sonic throat; supersonic/shock-seeding is deferred.

When a step below restates one of these, treat `CLAUDE.md` (and the linked design doc) as the authority.

## Hold the mental model

From `kernel-architecture.md` — getting one wrong produces a change that compiles and runs but is wrong where it matters:

- **Kernels, not objects.** The hot path is a flat sweep over typed edge arrays dispatched on an integer `residual_id` (`nefes/elements/kernels.py` branches `if rid == …`). The object shell builds/names/reports and owns no numerics; the kernels import no shell.
- **Written once, compiled twice.** Each kernel compiles to a `float64` specialization (the residual Newton drives to zero) and a `complex128` one (the complex-stepped derivative seed) from *one* body, so a change to a residual is automatically a change to its Jacobian — there is no separate Jacobian code. This is *why* the analyticity discipline is structural, not stylistic: a state-dependent branch would compile to two different functions and break the correspondence.
- **Smoothness is a library, not a ban.** The physics of a switch (flow reversal, a loss opposing either direction, subsonic-vs-choked, frozen-vs-burnt) is expressed through the regularized primitives in `nefes/assembly/smooth.py`; you round the corner, you do not branch on it.

## Adding an element type, end to end

The residual id is the spine; every table below is keyed on it. Touch, in order:

1. **`nefes/elements/ids.py`** — a new `residual_id` constant and its `ELEMENT_TYPE_NAMES` entry; then the tables that are keyed by id: `FIXED_NPORTS` (or, for a variable-port manifold, the rule in `port_kinds`), `_PORT_KINDS_FIXED`/`port_kinds`, `row_kind_tags`, `ALLOWS_AREA_CHANGE`, and `STREAM_INTRODUCING` / `BOUNDARY_RIDS` / `DISALLOWED_NEIGHBORS` if the type is a feed, a boundary, or has an adjacency rule.
2. **`nefes/elements/kernels.py`** — the residual branch (`if rid == YOUR_ID:`), built from the smooth primitives per the contract.
3. **`nefes/elements/catalog.py`** — the user-facing factory returning `ElementSpec(YOUR_ID, [fparams…], name, …)`.
4. **`nefes/elements/parameters.py`** — a `ParamDescriptor` in `ELEMENT_PARAMS` for each named parameter (unit, bounds, validation, `fparams` slot). `tests/test_parameters.py` checks the declared packing against the factory's actual output, so these cannot drift.
5. **`nefes/io/yaml_in.py` + `yaml_out.py`** — the YAML type-tag loader and writer; confirm a write→read→solve round-trip.
6. **`tests/test_complex_step_safety.py`** — a `_probe_<name>` and its `PROBES` entry. The roll-call `test_every_element_kernel_is_swept` fails until it exists; the per-kernel `test_kernel_complex_step_safe_across_regimes` then checks complex-step == finite-difference across forward / reverse / near-zero / near-choke flow. (This is the hard-constraint probe from `CLAUDE.md`.)
7. **Docs** — `docs/theory/elements.md` (the closure) and `docs/reference/atomic-elements.md` (parameters); add a `docs/reference/modeling-guide.md` row if the element maps a real restriction (orifice, valve, nozzle…).
8. **A behavioral test** — beyond the smoothness sweep, a test that pins the intended physics (a balance it must respect, a limit it must reproduce).

## Editing a kernel: the loop

1. Rewrite the residual smooth (contract + `smooth.py`), keeping it identical on the real and complex paths.
2. **Clear the numba cache** — see the trap below — or run with `NUMBA_DISABLE_JIT=1` to check against pure Python first.
3. Run that kernel's complex-step probe (`test_kernel_complex_step_safe_across_regimes` for its id), then the element's behavioral tests.
4. Update the theory/reference doc for the closure if its behavior changed.

### The numba stale-cache trap

Numba caches compiled kernels. After editing an `@njit` kernel a stale cache can mask the change, so if results look unchanged after an edit, clear the numba artifacts under `nefes/**/__pycache__/` (both `*.nbi` / `*.nbc` in `assembly/` and `elements/` — the assembly layer caches calls into the element kernels) or run with `NUMBA_DISABLE_JIT=1`. This is a productivity trap, not a correctness one, but it wastes real time when unrecognized.

## Test and verify

- Run the suite in the project conda env (`environment.yml`) or an editable install (`pip install -e .[dev]`): `pytest` (config in `pyproject.toml`, `testpaths = tests`).
- **Scientific claims get tests.** A new closure, a benchmark match, a conservation property: assert it, do not assert it in prose only. The roll-call makes the smoothness contract self-policing; extend that habit to the physics.
- **Verify against the code, don't recall.** Symbols, signatures, and the id-keyed tables drift; read the current `ids.py` / `catalog.py` / `parameters.py` before you assume a name or a slot. The numba caveat means a "verified" result may be stale — re-run after clearing the cache when in doubt.

## Docs are part of the change

`docs/` is the source of truth, so a feature that changes behavior is not done until its doc changes with it; create one if none fits and you judge it needed.
Keep the notation conventions of `docs/nomenclature.md` and `CLAUDE.md`, and keep the prose free of software/control-engineering jargon (the audience is combustion and acoustics).

## The commit gate

- Formatting is enforced by pre-commit (`isort` → `black`/`black-jupyter` → `flake8` → `nbstripout`); a commit that leaves unformatted, lint-failing, or output-bearing-notebook code is rejected. Install once with `pre-commit install`. Line length and lint rules are set in `pyproject.toml` / `.flake8`.
- Notebook outputs are never committed (`nbstripout` strips them); plot with plotly and the bundled theme.
- Commit-message and branch conventions (no authorship/tool references; delete merged branches) live in `CLAUDE.md`.

## Where to go for what

| Need | Go to |
| --- | --- |
| Why the package is built this way; the hot-path design | `docs/design/philosophy.md`, `docs/design/kernel-architecture.md` |
| The complex-step engine and the smoothness rules | `docs/design/complex-step.qmd`, `docs/design/smoothness-contract.md`, `nefes/assembly/smooth.py` |
| The solver, assembly, acoustic stamps, determinism | `docs/design/solver.md`, `docs/design/assembly.md`, `docs/design/reproducibility.md` |
| Physics: governing equations, per-element closures, choking, perturbation network | `docs/theory/` |
| The element-authoring touchpoints (ids, factories, parameter schema) | `nefes/elements/ids.py`, `catalog.py`, `parameters.py` |
| Element parameters and coefficients as documented | `docs/reference/atomic-elements.md`, `docs/reference/composite-elements.md` |
| Symbols and notation | `docs/nomenclature.md` |
| Validation map and literature benchmarks | `docs/validation/` |
| The non-negotiable constraints and repo conventions | `CLAUDE.md` |
