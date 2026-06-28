"""Central LaTeX-label toggle for FNS figures.

Plotly renders MathJax (``$...$``) only where a MathJax runtime is present.  In a
plain notebook kernel, a static HTML/PNG export, or some IDE viewers the math does
not typeset and the user is left staring at the raw source -- ``$f_{0}$``,
``\\rho``, stray braces.  This module is the single switch that decides, globally,
whether FNS labels are emitted as MathJax or as a Unicode plain-text fallback.

Usage::

    from fns.plotting import use_latex
    use_latex(False)        # all subsequent FNS figures use plain Unicode labels
    use_latex(True)         # back to MathJax (the default)

Every FNS plotting routine routes its labels through :func:`mathify` (for a bare
LaTeX *fragment*) or :func:`tex` (for a complete, possibly ``$``-delimited label),
so flipping the toggle restyles all of them.  When LaTeX is on both are essentially
no-ops (the current behaviour); when off they run :func:`detex`, a best-effort
LaTeX-to-Unicode converter that guarantees no ``$``, backslash or stray brace
survives into the figure.
"""

import re

# Process-wide default: MathJax on, matching the historical behaviour.  Flip with
# use_latex(False) for environments where MathJax does not render.
_USE_LATEX = True


def use_latex(enabled=True):
    """Enable or disable MathJax (``$...$``) labels on FNS figures process-wide.

    Parameters
    ----------
    enabled : bool, optional
        ``True`` (default) keeps the MathJax labels; ``False`` switches every FNS
        figure to the Unicode plain-text fallback produced by :func:`detex`.

    Returns
    -------
    bool
        The new state, so a caller can log or assert on it.
    """
    global _USE_LATEX
    _USE_LATEX = bool(enabled)
    return _USE_LATEX


def latex_enabled():
    """Whether FNS figures currently emit MathJax labels (see :func:`use_latex`)."""
    return _USE_LATEX


def mathify(fragment):
    """Render a bare LaTeX *fragment* as a figure label honoring the global toggle.

    ``fragment`` is a LaTeX body without the surrounding ``$`` (e.g. ``f_{0}`` or
    ``p'/\\rho c``).  Returns ``$fragment$`` when LaTeX is enabled, else the Unicode
    fallback from :func:`detex`.
    """
    return f"${fragment}$" if _USE_LATEX else detex(fragment)


def tex(label):
    """Pass through a complete label (possibly ``$``-delimited), honoring the toggle.

    Use for fixed axis titles such as ``r"$f\\;(\\mathrm{Hz})$"``: returns the label
    unchanged when LaTeX is enabled, else its :func:`detex` Unicode form.
    """
    if label is None:
        return None
    return label if _USE_LATEX else detex(label)


# -- LaTeX -> Unicode fallback ---------------------------------------------------

# Named control sequences (Greek and a handful of operators/relations) mapped to a
# single Unicode glyph.  Anything unlisted keeps its name verbatim (the backslash is
# dropped), so an unknown ``\foo`` degrades to ``foo`` rather than leaking a slash.
_NAMED = {
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ε",
    "varepsilon": "ε",
    "zeta": "ζ",
    "eta": "η",
    "theta": "θ",
    "vartheta": "ϑ",
    "iota": "ι",
    "kappa": "κ",
    "lambda": "λ",
    "mu": "μ",
    "nu": "ν",
    "xi": "ξ",
    "omicron": "ο",
    "pi": "π",
    "rho": "ρ",
    "varrho": "ϱ",
    "sigma": "σ",
    "varsigma": "ς",
    "tau": "τ",
    "upsilon": "υ",
    "phi": "φ",
    "varphi": "φ",
    "chi": "χ",
    "psi": "ψ",
    "omega": "ω",
    "Gamma": "Γ",
    "Delta": "Δ",
    "Theta": "Θ",
    "Lambda": "Λ",
    "Xi": "Ξ",
    "Pi": "Π",
    "Sigma": "Σ",
    "Upsilon": "Υ",
    "Phi": "Φ",
    "Psi": "Ψ",
    "Omega": "Ω",
    "angle": "∠",
    "to": "→",
    "rightarrow": "→",
    "leftarrow": "←",
    "cdot": "·",
    "times": "×",
    "partial": "∂",
    "infty": "∞",
    "pm": "±",
    "mp": "∓",
    "approx": "≈",
    "propto": "∝",
    "langle": "⟨",
    "rangle": "⟩",
    "nabla": "∇",
    "leq": "≤",
    "geq": "≥",
    "neq": "≠",
    "ell": "ℓ",
    "Re": "Re",
    "Im": "Im",
}

# Subscript / superscript Unicode coverage.  A group converts only when *every*
# character maps; otherwise it inlines verbatim (dropping the ``_``/``^`` marker) so
# a name-bearing subscript like ``_{0:inlet}`` reads as plain ``0:inlet``.
_SUB = {c: u for c, u in zip("0123456789+-=()aeoxhklmnpstijruv", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₒₓₕₖₗₘₙₚₛₜᵢⱼᵣᵤᵥ")}
_SUP = {c: u for c, u in zip("0123456789+-=()ni", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ")}

_WRAP = re.compile(r"\\(?:text|mathrm|mathbf|mathit|mathsf|operatorname|boldsymbol)\{([^{}]*)\}")
_SPACING = (("\\;", " "), ("\\,", " "), ("\\:", " "), ("\\>", " "), ("\\quad", "  "), ("\\qquad", "   "), ("\\!", ""))


def _script(inner, table):
    """Unicode sub/superscript for ``inner`` if fully mappable, else ``inner`` itself."""
    if inner and all(ch in table for ch in inner):
        return "".join(table[ch] for ch in inner)
    return inner


def detex(s):
    """Best-effort conversion of a LaTeX string to a Unicode plain-text label.

    Handles the constructs FNS actually emits -- Greek letters, ``\\dot``/``\\hat``
    accents, ``\\text``/``\\mathrm`` wrappers, spacing commands, ``\\to``/``\\angle``
    and friends, and ``_``/``^`` scripts -- and strips anything left so the result
    never contains ``$``, a backslash, or a stray brace.  Not a general LaTeX engine;
    it is a readability fallback, not a typesetter.
    """
    s = str(s).replace("$", "")
    # accents first (they own their brace group)
    s = re.sub(r"\\dot\{([^{}]*)\}", lambda m: m.group(1) + "̇", s)
    s = re.sub(r"\\hat\{([^{}]*)\}", lambda m: m.group(1) + "̂", s)
    s = re.sub(r"\\bar\{([^{}]*)\}", lambda m: m.group(1) + "̅", s)
    s = re.sub(r"\\overline\{([^{}]*)\}", lambda m: m.group(1) + "̅", s)
    # text/font wrappers -> their contents (repeat to unwrap any nesting)
    prev = None
    while prev != s:
        prev = s
        s = _WRAP.sub(r"\1", s)
    for tok, rep in _SPACING:
        s = s.replace(tok, rep)
    s = re.sub(r"\\([A-Za-z]+)", lambda m: _NAMED.get(m.group(1), m.group(1)), s)
    s = re.sub(r"\^\{([^{}]*)\}", lambda m: _script(m.group(1), _SUP), s)
    s = re.sub(r"\^([^\s{}_^])", lambda m: _script(m.group(1), _SUP), s)
    s = re.sub(r"_\{([^{}]*)\}", lambda m: _script(m.group(1), _SUB), s)
    s = re.sub(r"_([^\s{}_^])", lambda m: _script(m.group(1), _SUB), s)
    s = s.replace("{", "").replace("}", "").replace("\\", "")
    return re.sub(r"\s+", " ", s).strip()
