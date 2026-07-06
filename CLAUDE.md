# Nefes — Network solver for reacting compressible flows and thermoacoustics

Nefes is a compressible-flow **network** analysis tool: it models a fluid system as a
directed graph and solves for the steady mean flow (and, by design, the acoustic
behavior around it) without resolving the full 3-D field.

## Repository layout

- The repository directory is `Nefes`; the distribution/package name is `nefes`.
- Installable packages: `nefes` (the solver) and `thermolib` (standalone NASA-Glenn/CEA thermochemistry, ships with vendored `.inp`/`.yaml` data).
- `docs/` is the source of truth; `tests/` holds the suite; `examples/` holds runnable samples.

## Commands

- Environment: `conda env create -f environment.yml` (conda env `nefes`, Python 3.12) or `pip install -e .[dev]`.
- Tests: `pytest` (config in `pyproject.toml`, `testpaths = tests`).
- Format: `black .` (line length 120, configured in `pyproject.toml`).
- Lint: `flake8` (config in `.flake8`).

## Documentation

Source of truth lives under `docs/`; read it before designing, and sanity-check rather than assume it is current:

- `docs/theory/` — framework, governing equations, elements, transport, choking, and the acoustic/perturbation network.
- `docs/design/` — implementation contracts: assembly, solver, complex-step, kernel architecture, smoothness.
- `docs/reference/` — modeling guide mapping restrictions (orifices, valves, nozzles) to elements.
- `docs/validation/` — validation map, verification checks, and literature benchmarks.

When you implement, modify, or remove a feature, update the associated documentation; create one if none exists and you judge it necessary.

## Notation

Canonical symbols live in `docs/nomenclature.md`; the high-use conventions:

- Mean/base (temporal-mean) state: overbar `\overline{X}` (e.g. `\overline{c}`, `\overline{\mathbf{x}}`, `\overline{\mathbf{J}}` for the base Jacobian). Never `\bar`, and never subscript-0: numeric subscripts denote port indices (`p_0`) and would collide.
- Time-domain fluctuation: prime `X' = X - \overline{X}`.
- Complex (frequency-domain) amplitude: hat `\widehat{X}`, with `X'(t) = \Re\{\widehat{X}\,e^{\mathrm{i}\omega t}\}`.
- Section (area) averages: angle brackets `\langle X\rangle`.
- Density is `\varrho`, not `\rho`.
- Prefer frequency over angular frequency for user input and graph axes.

## Hard constraints (don't violate without explicit reason)

- All residual math must be **complex-step-safe**: smooth, complex-analytic, no `abs`/`min`/`max`/branches on the flow state. Jacobians come from complex-step differentiation. (See `nefes/assembly/smooth.py`.) Every new element kernel must get a probe in `tests/test_complex_step_safety.py` (`PROBES`); the roll-call test fails until it does, and the per-kernel sweep then checks complex-step == finite-difference across forward/reverse/near-zero/near-choke flow.
- Present scope is **subsonic** (flowing or quiescent); supersonic/shock-seeding is deferred.

## Development

- Whenever applicable, create tests. Especially for scientific implementations and claims.
- Do NOT save the output of Jupyter notebooks (filesize concerns).
- For notebooks, always prefer plotly for plotting.

### Pitfalls

- Numba caches compiled kernels. After editing an `njit` kernel a stale cache can mask the change, so if results look unchanged after an edit, clear the numba cache (delete the `__pycache__`/`*.nbi`/`*.nbc` artifacts) or run with `NUMBA_DISABLE_JIT=1` to sanity-check against pure Python.

## Coding style

- Use flake8 and black, line length is 120 characters.
- Numpy style docstring for all user-facing routines and classes.
- NEVER include in the docstring the historical context: e.g. previous state of codebase, an earlier bug, an improvement etc.
- Prefer explanatory comments on dedicated lines instead of appending next to a line of code, for label-like comments inline is okay.
- Inline comments should appear after two whitespaces.
- Use capital letters for absolute constants and define them before any function/class definitions in a file.
- Where appropriate (you judge), include brief "Examples" and "See also" in the docstrings.
- Mention the main routines/classes that a module aims to export in the top docstring.
- Prefer type hints where applicable; public APIs are always typed unless there is a strong reason against.

## Prose and Markdown

Applies to responses, code comments, docstrings, and Markdown (including Jupyter Markdown cells).

- Avoid excessive usage of "--" to connect sentences.
- Avoid marketing language like "first-class" to describe the codebase.
- In Markdown and LaTeX, let each sentence occupy a single line no matter how many columns it takes; the 120-column rule does not apply there.
- Documents marked as *internal documents* stay local; do not push them to the repository.

## Version control

- Do NOT include statements like "authored by ...", in the commit messages, and do NOT refer to the tool name.
- ALWAYS delete merged branches.

## Working with the user

- Keep responses short. No preamble. Get to the output.
- If a task is ambiguous, ask one clarifying question — not five.
- Do NOT assume anything if not entirely clear; you are encouraged to ask.

## TODO.md management

- Do NOT remove the titles.
- Erase completed items.
- When you add a new entry, be sure it is concise.
