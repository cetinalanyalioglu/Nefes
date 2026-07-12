"""Mean-flow Newton solver (control layer, above the @njit line)."""

from .control import SolveResult, auto_initial_guess, initial_guess, solve
from .report import (
    format_residuals,
    format_states,
    format_states_html,
    print_residuals,
    print_states,
    residual_breakdown,
    residual_groups,
    residual_labels,
    states_table,
)

__all__ = [
    "solve",
    "SolveResult",
    "initial_guess",
    "auto_initial_guess",
    "states_table",
    "format_states",
    "format_states_html",
    "print_states",
    "residual_labels",
    "residual_groups",
    "residual_breakdown",
    "format_residuals",
    "print_residuals",
]
