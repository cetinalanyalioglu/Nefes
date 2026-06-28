"""The central LaTeX-label toggle (`fns.plotting.use_latex`) and its `detex` fallback."""

import numpy as np
import pytest

from fns.plotting import use_latex, latex_enabled, mathify, tex, detex
from fns.plotting.labels import _SUB
import fns.plotting.complex_matrix as cm
import fns.plotting.transfer_function as tf


@pytest.fixture(autouse=True)
def _restore_latex_state():
    """Each test runs in isolation; always restore the process-wide default (on)."""
    prev = latex_enabled()
    yield
    use_latex(prev)


# -- detex: the Unicode fallback ------------------------------------------------


@pytest.mark.parametrize(
    "latex, plain",
    [
        (r"$f\;(\mathrm{Hz})$", "f (Hz)"),
        (r"$\angle\;(\mathrm{rad})$", "∠ (rad)"),
        (r"$\angle\;(\mathrm{deg})$", "∠ (deg)"),
        (r"$|\cdot|$", "|·|"),
        (r"$|F|$", "|F|"),
        (r"$\mathrm{Re}\,F$", "Re F"),
        ("f_{0}", "f₀"),
        ("g_{12}", "g₁₂"),
        (r"\dot{m}'", "ṁ'"),
        (r"p'/\rho c", "p'/ρ c"),
        ("h_t'", "hₜ'"),
        ("s'/c_p", "s'/cₚ"),
        ("P^+", "P⁺"),
        ("P^-", "P⁻"),
        (r"\sigma", "σ"),
        (r"\rho'", "ρ'"),
        (r"f_{1} \to f_{2}", "f₁ → f₂"),
    ],
)
def test_detex_known_fragments(latex, plain):
    assert detex(latex) == plain


def test_detex_never_leaks_latex_syntax():
    """No matter how exotic the input, the fallback emits no $, backslash or brace."""
    samples = [
        r"$f_{0:\text{MassFlowInlet1}}$",
        r"\frac{\partial p}{\partial t}",
        r"P^{+}_{\omega}",
        r"\unknownmacro{x}",
    ]
    for s in samples:
        out = detex(s)
        assert "$" not in out and "\\" not in out and "{" not in out and "}" not in out


def test_detex_name_subscript_inlines():
    """A subscript that is not fully sub-script-mappable inlines as plain text."""
    out = detex(r"f_{0:\text{pt-inlet}}")
    assert out == "f0:pt-inlet"
    assert ":" in out  # the colon (no subscript glyph) forced the inline path


def test_sub_table_covers_digits():
    assert all(d in _SUB for d in "0123456789")


# -- mathify / tex: the toggle --------------------------------------------------


def test_mathify_and_tex_follow_the_toggle():
    use_latex(True)
    assert latex_enabled()
    assert mathify("f_{0}") == "$f_{0}$"
    assert tex(r"$|F|$") == r"$|F|$"

    use_latex(False)
    assert not latex_enabled()
    assert mathify("f_{0}") == "f₀"
    assert tex(r"$|F|$") == "|F|"


def test_use_latex_returns_state():
    assert use_latex(False) is False
    assert use_latex(True) is True


# -- end to end: a built figure honors the toggle -------------------------------


def _has_dollar(fig):
    """True if any axis title or subplot annotation in `fig` carries a `$`."""
    texts = []
    for ax in fig.layout:
        if (ax.startswith("xaxis") or ax.startswith("yaxis")) and fig.layout[ax].title.text:
            texts.append(fig.layout[ax].title.text)
    for ann in fig.layout.annotations or ():
        if ann.text:
            texts.append(ann.text)
    for tr in fig.data:
        if tr.name:
            texts.append(tr.name)
    return any("$" in t for t in texts)


def test_complex_matrix_figure_respects_toggle():
    M = np.random.default_rng(0).standard_normal((8, 2, 2)) + 1j * np.random.default_rng(1).standard_normal((8, 2, 2))
    freqs = np.linspace(10.0, 100.0, 8)

    use_latex(True)
    assert _has_dollar(cm.plot_complex_matrix(M, freqs))

    use_latex(False)
    assert not _has_dollar(cm.plot_complex_matrix(M, freqs))


def test_transfer_function_figure_respects_toggle():
    F = np.linspace(0.0, 1.0, 16) + 0.2j
    freqs = np.linspace(10.0, 100.0, 16)

    use_latex(True)
    assert _has_dollar(tf.plot_transfer_function(F, freqs, nyquist=True))

    use_latex(False)
    assert not _has_dollar(tf.plot_transfer_function(F, freqs, nyquist=True))
    assert not _has_dollar(tf.plot_transfer_function(F, freqs))
