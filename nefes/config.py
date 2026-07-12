"""Global runtime configuration, exposed as ``nefes.config``.

Attributes
----------
enforce_subsonic : bool
    When ``True`` (the default), a steady solve that reaches a spurious supersonic
    branch is automatically re-solved onto the physical subsonic branch, so a bare
    ``solve()`` always returns a subsonic mean flow (the present modeling scope).
    Set it to ``False`` to accept whatever branch the solve reaches -- for the
    deferred supersonic/shock-bearing work, or to inspect the branch behavior.

Examples
--------
>>> import nefes
>>> nefes.config.enforce_subsonic = False  # opt out of the subsonic-scope guard

The flag can also be overridden for a single solve: ``net.solve(enforce_subsonic=False)``.
"""


class _Config:
    """Mutable holder for the global runtime flags (see the module docstring)."""

    __slots__ = ("enforce_subsonic",)

    def __init__(self):
        self.enforce_subsonic = True

    def __repr__(self) -> str:
        return f"nefes.config(enforce_subsonic={self.enforce_subsonic})"


config = _Config()
