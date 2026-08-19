"""forecast-lab: reproducible research on short-horizon market direction.

The package root stays empty by design. Any module placed here would sit outside the
layering guard, which only inspects modules inside a layer - and an unguarded module
is exactly where a dependency leak hides (ADR-001 sec. 4).
"""

__version__ = "0.1.0"
