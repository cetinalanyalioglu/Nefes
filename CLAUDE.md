# Nefes — Network solver for reacting compressible flows and thermoacoustics

Nefes is a compressible-flow **network** analysis tool: it models a fluid system as a
directed graph and solves for the steady mean flow (and, by design, the acoustic
behavior around it) without resolving the full 3-D field.

This repo is the **real tool**, built fresh.

## Read the spec before designing anything

`docs/` is the source of truth — do not restate it here, read it:

- `docs/theory/` — full theory: framework, governing equations, elements, transport, choking, and the acoustic/perturbation network.
- `docs/design/` — implementation contracts: assembly, solver, complex-step, kernel architecture, smoothness.
- `docs/reference/modeling-guide.md` — mapping catalogue restrictions (orifices, valves, nozzles) to elements.
- `docs/theory/thermochemistry.md` — reactive flow and the standalone thermochemistry library.
- `docs/validation/` — validation map, verification checks, and literature benchmarks.

However, do not assume everything in there is theoretically flawless, always do your sanity checks.

## Development guidelines

- Whenever applicable, create tests. Especially for scientific implementations.
- During development there is no need to keep things for backward compat.
- Development phase goes on until we have a tagged release on the repo.
- Do NOT save the output of Jupyter notebooks (filesize concerns) - except very small ones.
- For notebooks, always prefer plotly
- During active development and model implementation, deviations from theory docs is possible - do not blindly assume they are up-to-date.

### Pitfalls

- Be aware of numba cache

## Documentation

- Denote the mean/base (temporal-mean) state with an overbar `\overline{}` (e.g. `\overline{c}`, `\overline{\mathbf{x}}`, and `\overline{\mathbf{J}}` for the base Jacobian), never `\bar`. Do NOT use subscript-0 for mean states: it clashes with port indices (`p_0`). Section (area) averages use angle brackets `\langle\cdot\rangle`.

## Conventions

- Always prefer frequency over angular frequency for user input and graph axes.
- While preparing Jupyter notebooks split the lines from the sentences, the lines themselves don't need to respect the line length criterion. This rule applies to LaTeX as well.
- Inline comments should appear after two whitespaces.
- Avoid excessive usage of "--" to connect sentences.

## Hard constraints (don't violate without explicit reason)

- All residual math must be **complex-step-safe**: smooth, complex-analytic, no `abs`/`min`/`max`/branches on the flow state. Jacobians come from complex-step differentiation. (See `nefes/assembly/smooth.py`.) Every new element kernel must get a probe in `tests/test_complex_step_safety.py` (`PROBES`); the roll-call test fails until it does, and the per-kernel sweep then checks complex-step == finite-difference across forward/reverse/near-zero/near-choke flow.
- v1 scope is **subsonic** (flowing or quiescent); supersonic/shock-seeding is deferred.

## Version control
- Do NOT include statements like "authored by ...", in the commit messages, and do NOT refer to the tool name.
- ALWAYS delete merged branches

## Behavior Guidelines
- Keep responses short. No preamble. Get to the output.
- If a task is ambiguous, ask one clarifying question — not five.
- Do NOT assume anything if not entirely clear - you are encouraged to ask.
- Do NOT use the word such as "first-class" to describe anything related to this codebase.

## Coding style
- Use flake8 and black, line length is 120 characters.
- Numpy style docstring for all user-facing routines and classes.
- NEVER include in the docstring the historical context: e.g. previous state of codebase, an earlier bug, an improvement etc.
- Prefer explanatory comments on dedicated lines instead of appending next to a line of code, for label-like comments inline is okay.
- Use capital letters for absolute constants and define them before any function/class definitions in a file.
- Where appropriate (you judge), include brief "Examples" and "See also" in the docstrings.
- Mention the routines/classes that are ment to be public in the docstring of a submodule.

## Markdown files

- Let all sentences appear in a single line, no matter how many columns it takes. The 120 column rule does not apply here.
- The documents marked as *internal documents* stay local, you do not push them to the repository.

## TODO.md management
- Do NOT remove the titles
- Erase completed items
- When you add a new entry, be sure it is concise.
