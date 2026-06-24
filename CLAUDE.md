# FNS — Flow Network Solver

FNS is a compressible-flow **network** analysis tool: it models a fluid system as a
directed graph and solves for the steady mean flow (and, by design, the acoustic
behavior around it) without resolving the full 3-D field.

This repo is the **real tool**, built fresh. `preliminary-study/` is a prior working
prototype (package `fns`) that validated the design, plus the authoritative spec.

## Read the spec before designing anything

`preliminary-study/docs/` is the source of truth — do not restate it here, read it:

- `theory.md` — full theory: framework, equations, elements, the solver, choking/shocks, and the acoustic/perturbation network (§12).
- `implementation-plan.md` — first-version plan: connectivity, Jacobian, storage, thermo, solver, acoustic layer, OO shell.
- `modeling-guide.md` — mapping catalogue restrictions (orifices, valves, nozzles) to elements.
- `reactive-flow-requirements.md` — reactive flow + the standalone thermochemistry library.

Prototype implementation: `preliminary-study/fns/`; runnable demos + validation: `preliminary-study/{examples,tests}/`.

However, do not assume everything in there is theoretically flawless, always do your sanity checks.

## Development guidelines

- Whenever applicable, create tests. Especially for scientific implementations.
- During development there is no need to keep things for backward compat.
- Development phase goes on until we have a tagged release on the repo.
- Do NOT save the output of Jupyter notebooks (filesize concerns) - except very small ones.
- For notebooks, always prefer plotly

### Pitfalls

- Be aware of numba cache

## Conventions

- Always prefer frequency over angular frequency for user input and graph axes

## Hard constraints (don't violate without explicit reason)

- All residual math must be **complex-step-safe**: smooth, complex-analytic, no `abs`/`min`/`max`/branches on the flow state. Jacobians come from complex-step differentiation. (See `preliminary-study/fns/smooth.py`.) Every new element kernel must get a probe in `tests/test_complex_step_safety.py` (`PROBES`); the roll-call test fails until it does, and the per-kernel sweep then checks complex-step == finite-difference across forward/reverse/near-zero/near-choke flow.
- v1 scope is **subsonic** (flowing or quiescent); supersonic/shock-seeding is deferred.

## Version control
- Do NOT include statements like "authored by ...", in the commit messages, and do NOT refer to the tool name.
- ALWAYS delete merged branches

## Behavior Guidelines
- Keep responses short. No preamble. Get to the output.
- If a task is ambiguous, ask one clarifying question — not five.
- Do NOT assume anything if not entirely clear - you are encouraged to ask.

## Coding style
- Use flake8 and black, line length is 120 characters.
- Numpy style docstring for all user-facing routines and classes.
- Prefer explanatory comments on dedicated lines instead of appending next to a line of code, for label-like comments inline is okay.

## TODO.md management
- Do NOT remove the titles
- Erase completed items
- When you add a new entry, be sure it is concise.
