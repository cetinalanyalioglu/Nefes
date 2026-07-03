"""Mean-flow Newton solver (control layer, above the @njit line)."""

from .control import solve, SolveResult, initial_guess, auto_initial_guess
from .report import (
    states_table,
    format_states,
    format_states_html,
    print_states,
    residual_labels,
    residual_groups,
    residual_breakdown,
    format_residuals,
    print_residuals,
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
