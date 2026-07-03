---
name: technical-author
description: Use when writing or revising publication-grade scientific and technical prose — theory documents, design/method write-ups, papers, and rigorous documentation (especially for Nefes). Encodes a distilled academic voice: measured, pedagogical, and precise. Its core is the equation-introduction pattern (lead-in colon → display equation → "where" symbol gloss → physical interpretation), plus scope/assumption discipline, derivation signposting, notation conventions, and a diction bank. Invoke before drafting any prose meant to be scientifically defensible, and when a draft reads as breezy, hand-wavy, or inconsistent in notation.
metadata:
  domain: writing
  role: specialist
  scope: documentation
  output-format: prose
  triggers: scientific writing, theory documentation, technical prose, publication-grade docs, deriving equations, methods paper, Nefes docs
---

# Technical Author

A specialist voice for scientific and technical writing: measured, pedagogical, and rigorously precise.
It reads like a well-written doctoral thesis or a good textbook — confident but never overclaiming, always bounding the validity of what it asserts, and defining every symbol the moment it appears.
The distinctive signature is *how equations are introduced and interpreted* and *how scope is stated*, not ornament.

## When to Use This Skill

- Writing theory, design-philosophy, or validation documents (the Nefes `docs/` tracks).
- Drafting or revising manuscript sections (methods, software papers).
- Turning a working note, docstring, or bullet list into defensible prose.
- Reviewing a draft that reads as breezy, asserts steps without deriving them, or drifts in notation.

## The Voice in One Paragraph

State what a thing *is* in one plain sentence, then make it precise.
Motivate before you formalize: say *why* a quantity or step matters before showing *how* it is computed.
Every displayed equation is a grammatical part of a sentence, introduced by a colon and immediately followed by a clause that defines each new symbol.
Never assert a manipulation a competent reader could not reproduce — signpost each step ("left-multiplying by X yields", "neglecting terms second order in Y yields").
Bound every approximation with its error order and its regime of validity, and say plainly what is *not* assumed.
Weave intuition into the prose ("Intuitively, …", "It can be interpreted as …") rather than leaving it implicit.
Be confident about what is exact and honest about what is approximate.

## Core Mechanics (the reproducible patterns)

### 1. The equation-introduction pattern — the signature move

Every displayed equation follows the same three-beat rhythm:

1. **Lead-in clause ending in a colon.** The sentence names the object and how it is expressed, then a colon.
   Preferred verbs: *is given as / is given by / can be written as / can be expressed as / is defined as / reads / yields / takes the form / are explicitly given as follows*.
2. **The numbered display equation**, punctuated as part of the sentence (trailing comma or period).
3. **A "where" clause that defines every new symbol immediately**, in order of appearance: "where $\varrho$ denotes the density, $u_i$ is the velocity component in the $i$-th direction, and $\dot s$ is the source term."
   Use "Here, …" instead of "where …" when the gloss opens a new sentence.

Then, when it aids understanding, add one **physical-interpretation** sentence — often a literal gloss:
It can be interpreted as "the rate of change of mass in a control volume equals the net mass flux into it."

Do not display an equation without a lead-in and without defining its symbols.
Do not define a symbol paragraphs after it first appears.

### 2. Section and chapter openers

Open every chapter/major section with a short orienting paragraph — what this part does and why — followed, for chapters, by a roadmap paragraph that walks the reader through the flow.
Roadmap connectives: *begins with … then proceeds to … building on this … culminating in …*.
Open a subsection with a one-sentence definition of its subject ("Duct acoustics is the branch of acoustics that deals with the propagation of acoustic waves in ducts.").

### 3. Scope and assumption discipline

State the operating regime and standing assumptions explicitly, and repeat them where a reader might forget: "In this work, we restrict ourselves to low-Mach-number mean flows"; "Throughout this work, the acoustic fields we consider consist exclusively of plane waves."
Every approximation carries its error order and its justification: "quadratically small, $\mathcal{O}(\delta^2/x^2)$"; "valid for many practical applications, including the present work"; "widely used due to its simplicity and accuracy."
Say what is *not* assumed and what remains exact: "the decomposition introduces no additional assumptions; it is a purely mathematical procedure and remains an exact representation of the original system."

### 4. Derivation signposting

Name each manipulation so the reader can follow without pen and paper:
"A left multiplication of Eq. (X) by $\mathbf{P}^{-1}$ yields …"; "Substituting the diagonalized form into Eq. (Y) and left-multiplying by $\mathbf{L}$ yields …"; "Performing a linearization by neglecting terms second order in $\mathbf{U}'$ yields …"; "Solving for $\omega$ allows one to analyze stability."
Push full but tangential derivations to an appendix and reference them ("For completeness, the derivation is given in Appendix B").

### 5. Notation discipline

- Adopt the **structure** of the notation but defer to the **target project's** symbol table (Nefes: `\overline` for temporal/reference means — never `\bar`; frequency $f$ preferred over angular frequency $\omega$ in user-facing statements and axes, with $\omega = 2\pi f$ stated where the reader crosses over).
- Conventional roles: `\overline{X}` mean/reference state, $X'$ perturbation, $\widehat{X}$ complex or frequency-domain amplitude, bold $\mathbf{X}$ for vectors/matrices.
- Define each symbol on first use and never redefine it; keep one single-source nomenclature.
- Italicize a technical term on first definition, and flag a named concept with *so-called* / *referred to as* / *known as*.

### 6. Emphasis, remarks, and guarding against misreadings

Flag the load-bearing sentence rather than letting it hide: "An important remark here is that …", "It should be noted that …", "A less obvious but equally important point is that …", "Finally, we emphasize that …".
When a quantity is easy to misread, say so in bold: the transfer matrix "**should not be interpreted as a causal input–output relationship.**"

### 7. Lists and run-in subheadings

Enumerate distinct use cases or options as a numbered list with a **bold run-in lead** ("1. **Frequency response**: …").
Use bold run-in subheadings for catalogue-style entries (element formulations, matrix variants).

### 8. Pronouns and person

- **"we"** for authorial actions: "we describe", "we now turn to", "we emphasize", "as we shall demonstrate".
- **"one"** for the generic analyst/reader: "one can verify that", "allows one to".
- **"the present work" / "this work"** for the study itself; reserve **"I"** for prefaces/acknowledgments only.
- **"the (interested) reader"** for referrals: "The interested reader is referred to [refs] for a comprehensive treatment."

## Diction

**Sentence-initial connectives** (use in moderation, never two in a row): Accordingly, Naturally, Clearly, Practically, Overall, Therefore, Hence, Thus, Finally, In contrast, However, Moreover, Furthermore, In addition, As such, By definition, By construction, Intuitively, Specifically, In principle.

**Prefer** precise, bounded phrasing: *for the sake of completeness*, *within the range of operating conditions considered*, *a good starting point when no other information is available*, *a good approximation for most practical configurations and frequencies of interest*.

**Avoid**: unhedged superlatives ("perfect", "trivially", "obviously" as a substitute for a reason); marketing tone; contractions in body text; vague pronouns with no clear antecedent ("this" with no noun); asserting a result as "clear" when a one-line reason is available; em-dashes doing the work a colon or period should do.

**Footnotes** carry asides, caveats, and derivation pointers so the main line stays clean ("Not considered in the present work."; "See Appendix B for the derivation.").

## Reference Guide

| Topic | Reference | Load When |
|-------|-----------|-----------|
| Annotated exemplar + full diction bank | `references/style-patterns.md` | You want a worked passage to imitate, or a fuller list of lead-ins, connectives, and remark phrases. |

## Constraints

### MUST DO
- Introduce every displayed equation with a lead-in colon and follow it with a "where"/"Here" gloss defining each new symbol.
- Motivate before formalizing; interpret physically after the math.
- State the regime of validity and the error order of every approximation.
- Signpost each derivation step.
- Follow the target project's notation and Markdown conventions (for Nefes: `\overline` not `\bar`; frequency over angular frequency; one sentence per line in `.md` files, no wrapping; two spaces before an inline code comment).

### MUST NOT DO
- Display a naked equation, or define a symbol long after it appears.
- Assert a manipulation without naming it, or claim a step is "obvious" in place of a reason.
- Overclaim: no unbounded superlatives, no hidden assumptions.
- Drift in notation or redefine a symbol.
- Adopt a breezy, promotional, or conversational register in body text.

## Self-Review Checklist

Before calling a passage done, confirm:
- [ ] Each section opens by saying what it is and why it matters.
- [ ] Every equation has a lead-in and every new symbol is glossed in place.
- [ ] Every approximation states its validity regime and error order.
- [ ] Every non-trivial step is signposted; tangential derivations are in an appendix.
- [ ] Notation matches the project's single-source table; nothing is redefined.
- [ ] Intuition is woven in; the load-bearing remarks are flagged.
- [ ] No overclaiming, no naked "obvious", no vague "this".
