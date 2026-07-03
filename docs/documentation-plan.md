# Nefes Documentation Plan
*Internal document*

Working plan for a publication-grade, open-source documentation set covering the **theory**
(scientifically defensible) and the **software design philosophy** (the numerical and
architectural decisions — *not* the API, which is still in flux).

Status: living plan; the decisions in §6 are settled and this file is the standalone starting point for building the docs. It is a planning artifact, not itself part of the reference.

---

## 1. Goals and audiences

The documentation must serve three readerships at once, from a single authoritative source:

1. **The reviewer / independent researcher** — can reconstruct every equation, check every
   assumption, and reproduce every validation figure. This is the bar for the manuscript(s).
2. **The user / modeler** — knows fluid mechanics, wants to build networks and trust the
   results, and needs the modeling assumptions stated plainly.
3. **The contributor** — needs to understand *why* the code is shaped the way it is (kernels,
   complex-step, smoothness contract) before touching it.

Design constraint: **the docs are the substrate for the papers, not a rewrite of them.**
A methods manuscript draws from the Theory track; a software paper (JOSS-style) draws from the
Design track; the Validation track supplies the results. Writing the docs well once should make
the papers assembly, not authorship.

Non-goal (for now): API reference. Names and signatures are unstable until the tagged release,
so we document *concepts and contracts*, and only name a routine when the concept is inseparable
from where it lives (e.g. the smoothness primitives in `nefes/assembly/smooth.py`).

---

## 2. Documentation architecture

Authoritative source stays **Markdown in-repo** (portable, diff-reviewable, math via `$…$`),
under a small tree. The render target is **Quarto** (KaTeX math), layered over the same source
without moving it (§6). Three tracks plus a thin supporting layer:

```
docs/
  theory/         Track I  — the physics and mathematics (methods-paper substrate)
  design/         Track II — numerical and architectural philosophy (software-paper substrate)
  validation/     Track III— verification & validation evidence (results substrate)
  reference/      element reference, modeling guide, examples index (user-facing; mostly exists)
  nomenclature.md single-source symbol & term table (loaded/linked by every doc)
  references.bib  single bibliography; every scientific claim cites or derives
```

The existing `docs/atomic-elements.md`, `docs/composite-elements.md`, `docs/README.md` move
under `reference/` and are cross-linked, not rewritten.

### Track I — Theory

| # | Document | Scope | Provenance |
|---|----------|-------|------------|
| T0 | `overview.md` | What the method computes, the four design decisions, scope boundaries (subsonic v1). | theory §1 (refresh) |
| T1 | `framework.md` | Graph model, edges-own-state, orientation/sign convention, notation, perfect-gas closure. | theory §2 |
| T2 | `governing-equations.md` | Integral laws → steady balances / jump conditions (steadiness, not zero-volume, gives them); the zero-volume limit as a per-element convenience with the lumped source surviving; edge closure. | theory §3 |
| T3 | `state-and-recovery.qmd` | Unknown set (ṁ, p, hₜ), the implicit density/enthalpy recovery, derived state, why not (pₜ, Tₜ); figures embedded as executable Quarto cells. | theory §4 |
| T4 | `equation-structure.md` | Square-system bookkeeping (per-port rows, per-edge transport), the fixed split, direction discovery. | theory §5 |
| T5 | `transport.qmd` | Total-enthalpy transport, donor/upwind form, generalization to carried scalars, why mass is not one. Inline figure: smooth upwind weight vs. hard Heaviside + quadratic error decay. | theory §6 |
| T6 | `elements.md` | Constitutive residual of each element class (area change, loss, junction/splitter, boundaries, stabilization term); the blackbox **transfer-matrix 2-port** — mean-flow-passive (an isentropic area change), acoustically a prescribed/measured matrix (its stamp lives in T11/T13). | theory §7 + code |
| T7 | `well-posedness.md` | Why naive flux-form / hard-switch formulations fail; zero-flow stationarity. | theory §8 |
| T8 | `characteristics.md` | 1-D Euler decomposition, exact maps to network variables, Newton-invariance. | theory §9 |
| T9 | `choking.qmd` | Emergent choking via smoothed complementarity; the validated operating map; what stays non-emergent. Inline figure: solver-computed converging-nozzle operating map (back-pressure sweep). | theory §11 |
| **T10** | `thermochemistry.md` | **NEW.** Mixture thermodynamics, chemical equilibrium (HP), the frozen/equilibrium/marker closures, kinetic-energy coupling, transported mixture fractions, absolute-enthalpy datum. | reactive-flow-requirements + code |
| **T11** | `perturbation-network.md` | Linearization about the mean flow; the operator `A(ω)=J_alg+iωM+P+S`; lossless-duct phase; storage/effective length; regularization at singular points. | theory §12 (expand) |
| **T12** | `dynamic-sources.qmd` | **NEW.** Flame response `S(ω)`: FTF / n–τ, equivalence-ratio coupling, compositional (indirect) noise, the reacting-acoustics caveat. | memory + code |
| **T13** | `analyses.qmd` | **NEW.** The forward analyses made rigorous: forced response / scattering, eigenmodes via contour integration, Nyquist real-frequency stability; acoustic-power energy budget & passivity bounds. Includes the **frequency-domain matrix descriptors** — the transfer- vs scattering-matrix representation, the variable *flavors* (characteristic / primitive / Riemann bases) and the transfer↔scattering conversion, and analytic continuation of tabulated data off the real axis (the `TransferMatrix` / `ScatteringMatrix` / `FreqMatrix` containers). | theory §12.7 (expand) + code |
| **T14** | `identification.md` | **NEW.** The inverse analysis: de-embedding an element's dynamic response from a measured network transfer/scattering matrix, given a Nefes model of the rest of the setup. Both unknowns enter the perturbation operator linearly as a low-rank update, so a single factorization of the known operator recovers, per frequency by least squares: a blackbox 2-port's **transfer matrix** (`identify_transfer_matrix`) or a flame/mass-source **transfer function** (`identify_transfer_function`, single- and multi-input). Covers the Woodbury de-embed, cascade vs branched topologies, the **isentropic** (acoustics-only, classic 2×2 acoustic TM) vs full acoustic+entropy workflows, the per-frequency **conditioning** as the identifiability diagnostic, and the rational continuation that drops the recovered response back into the element / dynamic source and the eigensolver. | code (`nefes/perturbation/identify`, `matrix.py`) + theory §12.7 |
| T15 | `limitations.md` | Scope boundaries, the supersonic/shock fold problem, singular-point acoustics, honest open ends. | theory §15 |

### Track II — Software design philosophy (not API)

| # | Document | Scope |
|---|----------|-------|
| D0 | `philosophy.md` | The governing principles: smoothness over branching, exact derivatives over approximate, discovery over prescription, kernels over objects. How each maps to a hard constraint. |
| D1 | `kernel-architecture.md` | The `@njit` residual kernels doing the heavy lifting; integer `residual_id` dispatch; dtype-generic (float64/complex128) dual compilation from one source; the numba-cache pitfall. |
| **D2** | `complex-step.md` | The exact-Jacobian engine: the complex-step derivative, why it beats finite differences, the analyticity requirement, and the seeded-path discipline (recompute, never cache). |
| D3 | `smoothness-contract.md` | The complex-step-safety contract; the regularized primitive library (`smooth_abs/pos/step`, `smooth_sign_sq`, Fischer–Burmeister, `marker_gate`); error order `O(δ²/x²)`; the per-kernel probe roll-call. |
| D4 | `assembly.md` | Edge-state recovery DAG; the band-1 / algebraic row split; sparse Jacobian structure; the perturbation-operator stamps (`M`, `P`, `S`, terminals, the transfer-matrix element). How the source `S` and transfer-matrix stamps enter the operator as a **low-rank update** — the structure the identification de-embed (T14) exploits for a one-factorization inverse. |
| D5 | `solver.md` | Nondimensionalization/scaling; Newton with Levenberg–Marquardt damping; the vanishing-friction homotopy; warm-start caches. |
| D6 | `reproducibility.md` | Determinism, pinned environments (the split `nefes`/`thermolib` test envs), provenance capture on I/O, and the testing philosophy (complex-step==FD, oracle comparisons, validation gating). |

### Track III — Validation & verification

| # | Document | Scope |
|---|----------|-------|
| V0 | `validation-map.md` | Master table: every physical claim → the analytic/literature case → the test/example that checks it. Supersedes theory §14, extended for reacting + acoustic cases. |
| V1 | `verification.md` | Internal consistency checks: complex-step vs finite-difference, edge-direction-flip invariance, characteristic Newton-invariance, thermolib vs Cantera oracle, transfer↔scattering round-trip, and the **identification round-trip** (a de-embedded element matches the response it was measured from; conditioning flags an unidentifiable measurement). |
| V2 | `benchmarks.md` | Named literature cases (e.g. Greyvenstein & Laurie network; Rijke-tube stability; entropy/indirect-noise cases; expansion-chamber transmission), each with quantitative agreement. |

### Supporting / reference (mostly exists)

- `reference/atomic-elements.md`, `reference/composite-elements.md` — keep, cross-link to Track I.
- `reference/modeling-guide.md` — refresh from `preliminary-study/docs/modeling-guide.md`.
- `reference/examples.md` — annotated index of `examples/*.ipynb`, each tagged with the theory
  sections and validation cases it demonstrates (including `flame_identification.ipynb` → T14).

---

## 3. Scientific standards (the defensibility contract)

Every Theory/Validation document adheres to:

1. **Assumptions ledger.** Each doc opens with an explicit, enumerated list of the assumptions
   and their validity range. Approximations are stated with their error order (e.g. the
   regularization bias is `O(δ²/x²)`), never buried.
2. **Complete derivations.** No step is asserted that a competent reader could not reconstruct.
   Where the prototype docs hand-wave, we derive or cite. **We do not inherit claims from the
   preliminary docs on trust** — each is re-checked against the current code and the literature
   before it is restated (per the repo's own caution).
3. **Claim → evidence traceability.** Every falsifiable claim carries a `(test: …)` or
   `(example: …)` pointer, and appears in the `validation-map`. A claim with no check is flagged
   as such, not left ambiguous.
4. **Single-source notation.** All symbols live in `nomenclature.md`; documents link rather than
   redefine. Sign and orientation conventions are stated once and reused.
5. **Citations.** `references.bib` is the one bibliography. Standard results are cited to
   primary sources (complex-step: Lyness–Moler 1967, Squire–Trapp 1998, Martins et al. 2003;
   choked-nozzle acoustics: Marble–Candel; indirect noise: Magri; through-throat response:
   Stow–Dowling, Duran–Moreau; complementarity: Fischer–Burmeister; loss models:
   Borda–Carnot; etc.). Novel contributions are explicitly marked as such.
6. **Figures are reproducible.** Every figure is generated inline by an executable Quarto code
   cell embedded in its document (plotly, per repo convention), regenerable from a pinned
   environment; the code is folded by default (`code-fold`). Standalone runnable demos remain as
   notebooks under `examples/`, but a figure that belongs to a document lives in that document.

---

## 4. Style guide

The prose voice is owned by the **`technical-author` skill** (`.claude/skills/technical-author/`):
it is the authority for register, the equation-introduction pattern (lead-in colon → display
equation → "where" symbol gloss → physical interpretation), scope/assumption discipline, and
diction. The worked exemplar is `scratch/demo-governing-equations-and-linearization.md`. The
points below are the project-specific conventions that layer on top of the skill.

- **Intuition is woven inline**, never boxed. Introduce it with "Intuitively, …" / "It can be
  interpreted as …" next to the mathematics it explains; a reader following the prose straight
  through gets the physical story, and the equations state it precisely. (The earlier
  `> **In plain terms:**` two-voices convention is dropped.)
- **Term-on-first-use.** Every technical term (residual, Jacobian, characteristic, choking,
  stamp, …) is defined where it first appears.
- **Math conventions.** Mean/base (temporal-mean) state carries an overbar `\overline{}` (e.g.
  `\overline{c}`, `\overline{\mathbf{x}}`, and `\overline{\mathbf{J}}` for the base Jacobian),
  never `\bar`; subscript-0 is **not** used for mean states, as it clashes with port indices
  (`p_0`). Section (area) averages use `\langle\cdot\rangle`; density is `\varrho` (not `\rho`).
  **Frequency, not angular frequency**, for user-facing statements and graph axes; `ω` is
  permissible inside derivations, with `ω = 2πf` stated where the reader crosses from one to the other.
- **Prose wrapping**: one sentence per line, no column wrapping (per the repo Markdown rule).
- **Cross-references** use the `(test: …)` / `(example: …)` form and relative Markdown links
  between docs; `file.py`/`symbol` references are backtick-quoted.
- **Comments in embedded code** follow the repo style (two spaces before an inline comment;
  explanatory comments on their own line).

---

## 5. Sequencing

1. **Foundation.** `nomenclature.md` (fix the symbol table — `\varrho`, `\overline`, the wave and
   operator symbols), `references.bib` skeleton, `overview.md`. The voice and a worked exemplar are
   already set by the `technical-author` skill and `scratch/demo-governing-equations-and-linearization.md`.
2. **Theory core.** T1–T9 — refreshed and re-verified from `theory.md`; the mature, stable
   physics first.
3. **Theory frontier.** T10–T14 — the reacting, acoustic, and identification material that
   outran the prototype docs; written against the code, not the old plan. T14 (identification)
   depends on T13's matrix descriptors and the D4 low-rank stamp view, so it is written after both.
4. **Design track.** D0–D6 — the philosophy pieces; D2/D3 (complex-step + smoothness) are the
   signature and go first.
5. **Validation track.** V0–V2 — consolidate the scattered `(test:)` pointers into the master map.
6. **Integration pass.** Cross-linking, figure regeneration, nomenclature reconciliation, a
   read-through for the assumptions-ledger and citation completeness.

---

## 6. Decisions (settled)

- **Doc topology:** modular tree (§2), not a monolith — it scales with the codebase and maps
  cleanly onto paper sections.
- **Publication targets:** one **software paper** (Design track) and one **scientific paper**
  (Theory + Validation tracks). This planning document and anything paper-related are **internal
  documents**, not pushed to the repository.
- **Render target:** **Quarto** (KaTeX math) over the in-repo Markdown source, adopted
  provisionally — kept only if it stays low-maintenance to manage; the Markdown remains the
  authoritative source either way.
- **Prose register:** owned by the `technical-author` skill; intuition woven inline, no
  plain-terms boxes (§4).
- **Density symbol:** `\varrho`.
- **Mean-state notation:** overbar for the mean/base state (`\overline{c}`, and `\overline{\mathbf{J}}`
  for the base Jacobian, formerly `J_alg`); subscript-0 was tried and reverted because it clashes
  with port indices (`p_0`). Section averages use `\langle\cdot\rangle`.
