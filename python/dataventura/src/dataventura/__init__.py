"""Connect your own Python code to Data Ventura.

Decorate a function with ``@dv.transform`` and Data Ventura can run it on a table, in your own
environment, through a warm worker process. The function's type hint for its first parameter
chooses what it receives:

- ``pyarrow.Table`` (the default when there's no hint), zero-copy
- ``polars.DataFrame``, zero-copy for most column types
- ``pandas.DataFrame``, converted (copies)
- ``numpy.ndarray``, for a single-column input, zero-copy when it has no nulls

It may return any of those, or a ``pyarrow.Array`` / ``polars.Series`` / ``pandas.Series``, which
becomes a one-column table named after the function.

The worker protocol is documented in ``PROTOCOL.md`` next to this package. ``@dv.view`` and
``@dv.reader`` are planned.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable

__version__ = "0.0.0"

__all__ = ["Transform", "transform", "registry"]


@dataclass(frozen=True)
class Transform:
    """A function registered with ``@dv.transform``."""

    name: str
    func: Callable[..., Any]
    input_type: str
    columns: tuple[str, ...] | None


_registry: dict[str, Transform] = {}


def registry() -> dict[str, Transform]:
    """The transforms registered so far, by name."""
    return dict(_registry)


def transform(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    columns: list[str] | None = None,
) -> Any:
    """Registers ``func`` as a transform.

    ``columns`` names the input columns the function reads. Data Ventura sends only those, so
    leaving out columns the function doesn't need keeps the worker's memory and the overhead low.
    """

    def register(f: Callable[..., Any]) -> Callable[..., Any]:
        t = Transform(
            name=name or f.__name__,
            func=f,
            input_type=_input_type(f),
            columns=tuple(columns) if columns is not None else None,
        )
        _registry[t.name] = t
        return f

    return register(func) if func is not None else register


def _input_type(f: Callable[..., Any]) -> str:
    """The input type named by the first parameter's hint, as a short tag."""
    params = list(inspect.signature(f).parameters.values())
    if not params:
        raise TypeError(f"transform {f.__name__} must take its input as the first parameter")
    hint = params[0].annotation
    if hint is inspect.Parameter.empty:
        return "arrow"
    text = hint if isinstance(hint, str) else f"{getattr(hint, '__module__', '')}.{getattr(hint, '__qualname__', hint)}"
    for tag, needles in _HINTS:
        if any(n in text for n in needles):
            return tag
    raise TypeError(
        f"transform {f.__name__}: unsupported input type {text!r}; use pyarrow.Table, "
        "polars.DataFrame, pandas.DataFrame or numpy.ndarray"
    )


_HINTS = (
    ("arrow", ("pyarrow", "pa.Table", "Table")),
    ("polars", ("polars", "pl.DataFrame")),
    ("pandas", ("pandas", "pd.DataFrame")),
    ("numpy", ("numpy", "np.ndarray", "ndarray")),
)
