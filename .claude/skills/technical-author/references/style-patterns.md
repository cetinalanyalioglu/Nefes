# Style Patterns — Annotated Exemplar and Diction Bank

This reference supplements `SKILL.md` with a worked passage to imitate and a fuller vocabulary.
The exemplar is written on a neutral, project-relevant topic so it can be copied for rhythm, not content.

## Annotated exemplar

The passage below is written in the target voice.
The bracketed tags — `[rule N]` — point to the numbered mechanics in `SKILL.md` and are **not** part of the prose; strip them when imitating.

---

**Regularized loss term.** `[rule 2: subsection opens with a one-sentence definition]`
A loss element removes total pressure from the flow in proportion to the local dynamic pressure, and must do so regardless of the flow direction.
The naive form $\propto u|u|$ satisfies the second law but is not differentiable at $u = 0$, which is fatal for a solver that obtains its Jacobian by complex-step differentiation. `[rule 6: flags the load-bearing constraint]`
We therefore replace the absolute value by a smooth surrogate.

The direction-aware dynamic-pressure term is defined as: `[rule 1: lead-in ending in a colon]`

$$
\Phi(u) \;=\; u\,\sqrt{u^2 + \delta^2},
$$

where $u$ is the edge velocity and $\delta$ is a regularization scale chosen relative to a problem reference value, e.g. $\delta = 10^{-3}\,\overline{u}_\text{ref}$. `[rule 1: "where" gloss defines each new symbol in place; rule 5: \overline for the reference state]`
Intuitively, $\Phi$ behaves as $u|u|$ away from the origin and rounds off the corner within a neighborhood of width $\delta$. `[rule 1/8: physical interpretation, woven in]`

The surrogate is analytic on the real axis, since the radicand $u^2 + \delta^2$ is strictly positive, so the complex-step derivative propagates through it exactly. `[rule 3: states why the property holds]`
The price of the regularization is a bias in the converged residual.
Expanding $\Phi$ for $|u| \gg \delta$ yields: `[rule 4: signposts the manipulation]`

$$
\Phi(u) \;=\; u|u|\left(1 + \frac{\delta^2}{2u^2} + \mathcal{O}\!\left(\frac{\delta^4}{u^4}\right)\right),
$$

so the relative error is $\mathcal{O}(\delta^2/u^2)$ — quadratically small at any converged state with $|u| \gg \delta$. `[rule 3: approximation carries its error order]`
In this work, $\delta$ is held fixed across a solve rather than annealed, which keeps the residual smooth and the Newton iteration well-behaved; the resulting bias is negligible for the operating conditions considered here. `[rule 3: scope + justification]`
The interested reader is referred to the smoothness-contract document for the full family of regularized primitives. `[rule 8: referral]`

---

Notice what the passage does *not* do: it never displays an equation without a lead-in, never leaves a symbol undefined, never calls the smoothness "obvious", and never claims the bias is zero — it bounds it.

## Diction bank

### Equation lead-ins (end with a colon)
is given as · is given by · can be written as · can be expressed as · is defined as · reads · yields · takes the form · are explicitly given as follows · can be written in the general form · can be recast as

### Symbol-gloss openers
where $X$ denotes … · where $X$ is … · Here, $X$ denotes … · in which $X$ represents …

### Physical-interpretation openers
It can be interpreted as "…" · It can be understood as … · This represents … · Intuitively, … · Physically, …

### Derivation signposts
The first step is to … · A left multiplication by $\cdot$ yields … · Substituting … into … yields … · Performing a linearization by neglecting terms second order in $\cdot$ yields … · Rearranging gives … · Solving for $\cdot$ allows one to … · The above can be reformulated in terms of $\cdot$ as follows · Taking the imaginary part yields … · Integrating over $\cdot$ gives …

### Scope and hedging (always with a reason)
In this work, we restrict ourselves to … · Throughout this work, … · Within the range of operating conditions considered … · in the context of … · valid for many practical applications, including the present work · widely used due to its simplicity and accuracy · a good approximation for most practical configurations and frequencies of interest · provides significant simplification without substantial loss of accuracy · serves as a good starting point when no other information is available

### Remarks and emphasis
An important remark here is that … · An important observation is that … · It is important to note that … · It should be noted that … · A distinguishing feature of … is … · A less obvious but equally important point is that … · Finally, we emphasize that … · A final clarification is necessary regarding …

### Naming and referral
the so-called … · referred to as … · known as … · denoted as … · The interested reader is referred to [refs] for a comprehensive treatment. · For further reading, … · For the sake of completeness, the derivation is given in Appendix B.

### Sentence-initial connectives
Accordingly · Naturally · Clearly · Practically · Overall · Therefore · Hence · Thus · Finally · In contrast · However · Moreover · Furthermore · In addition · As such · By definition · By construction · Specifically · In principle

### Exactness / honesty markers
introduces no additional assumptions · a purely mathematical procedure · remains an exact representation of the original system · this is the only place where … is assumed · this approximation is dropped in Section X

## Intuition is woven inline, never boxed

Intuition belongs in the prose, next to the mathematics it illuminates — not in a separate "plain terms" box.
Introduce it with "Intuitively, …", "It can be interpreted as …", or a literal gloss, and place it immediately after the object it explains, as in the exemplar above.
A reader should be able to follow the physical story by reading the prose straight through, with the equations as the precise statement of what the prose asserts.
Do not split the intuitive and rigorous accounts into parallel tracks; a single, well-motivated line of argument carries both.
