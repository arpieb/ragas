"""Every deprecation warning must name something that actually exists.

``ragas.metrics`` tells callers to migrate to ``ragas.metrics.collections``, and
that warning is the main migration signal users get. It was wrong for 22 of the
53 deprecated names: 11 classes had been renamed in collections, and the 10
pre-built singletons had no counterpart at all -- the matching lowercase name in
collections is a *module*, so the suggested import silently handed back the wrong
kind of object instead of failing.

These tests parse the suggested import out of the warning text and run it, which
is the only check that cannot drift from what users are actually told.
"""

from __future__ import annotations

import re
import warnings

import pytest

from ragas.metrics import (
    _COLLECTIONS_CLASS_FOR_INSTANCE,
    _COLLECTIONS_RENAMES,
    _DEPRECATED_METRICS,
    _NO_COLLECTIONS_EQUIVALENT,
)

IMPORT_RE = re.compile(r"from ragas\.metrics\.collections import (\w+)")


def warning_for(name: str) -> str:
    import ragas.metrics

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        getattr(ragas.metrics, name)
    assert len(caught) == 1, f"{name} emitted {len(caught)} warnings, expected 1"
    return str(caught[0].message)


@pytest.mark.parametrize("name", sorted(_DEPRECATED_METRICS))
def test_suggested_import_resolves(name):
    """If the warning tells you to import something, that import must work."""
    from ragas.metrics import collections

    match = IMPORT_RE.search(warning_for(name))
    if match is None:
        # Only the un-ported metrics are allowed to suggest nothing.
        assert name in _NO_COLLECTIONS_EQUIVALENT, (
            f"{name} suggests no migration target but is not recorded as un-ported"
        )
        return

    target = match.group(1)
    assert hasattr(collections, target), (
        f"{name}'s warning says `from ragas.metrics.collections import {target}`, "
        f"but collections has no attribute {target!r}"
    )
    assert isinstance(getattr(collections, target), type), (
        f"{name}'s warning points at collections.{target}, which is not a class "
        f"(it is {type(getattr(collections, target)).__name__}) -- the import would "
        f"succeed and hand back the wrong kind of object"
    )


@pytest.mark.parametrize("name", sorted(_NO_COLLECTIONS_EQUIVALENT))
def test_unported_metrics_do_not_promise_a_replacement(name):
    """Skipped while every metric has a port -- the set is empty, not broken."""
    msg = warning_for(name)
    assert IMPORT_RE.search(msg) is None, (
        f"{name} has no collections port, but its warning suggests an import"
    )
    assert "no equivalent" in msg


@pytest.mark.parametrize("name", sorted(_COLLECTIONS_CLASS_FOR_INSTANCE))
def test_singletons_point_at_a_constructible_class(name):
    """The pre-built instances have no drop-in; the warning must say so."""
    msg = warning_for(name)
    assert "(llm=llm)" in msg, f"{name} should show how to construct the metric"
    assert IMPORT_RE.search(msg).group(1) == _COLLECTIONS_CLASS_FOR_INSTANCE[name]


def test_renames_are_actually_renames():
    """Guards against a rename entry that is really a different metric."""
    from ragas.metrics import collections

    for old, new in _COLLECTIONS_RENAMES.items():
        assert not hasattr(collections, old), (
            f"{old} now exists in collections; drop it from _COLLECTIONS_RENAMES"
        )
        assert hasattr(collections, new), f"rename target {new} does not exist"


def test_nothing_is_left_unported():
    """Records the state the tier 1-3 ports reached.

    An empty set is why `test_unported_metrics_do_not_promise_a_replacement`
    skips. If a metric ever lands in ragas.metrics with no collections
    equivalent, add it here -- that un-skips the test and restores the honest
    "no migration target" warning for it.
    """
    assert _NO_COLLECTIONS_EQUIVALENT == frozenset(), (
        f"un-ported metrics: {sorted(_NO_COLLECTIONS_EQUIVALENT)}. Update this "
        f"test deliberately when that is intended."
    )


def test_every_deprecated_name_is_classified():
    """No name may fall through into a stale generic message."""
    classified = (
        set(_COLLECTIONS_RENAMES)
        | set(_COLLECTIONS_CLASS_FOR_INSTANCE)
        | set(_NO_COLLECTIONS_EQUIVALENT)
    )
    from ragas.metrics import collections

    for name in _DEPRECATED_METRICS:
        if name in classified:
            continue
        # Everything else must resolve under its own name.
        assert hasattr(collections, name), (
            f"{name} is unclassified and absent from collections -- it needs an "
            f"entry in _COLLECTIONS_RENAMES or _NO_COLLECTIONS_EQUIVALENT"
        )
