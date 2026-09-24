"""Tests for the testset Document protocol.

The point of ``DocumentLike`` being a ``Protocol`` is that ragas stopped
depending on LangChain to express the type *without* breaking anyone whose
documents come out of a LangChain loader. These tests pin both halves.
"""

from __future__ import annotations

import typing as t

import pytest

from ragas.testset.document import Document, DocumentLike


def read(docs: t.Sequence[DocumentLike]) -> list[tuple[str, dict]]:
    """Stand-in for what testset generation actually does with a document."""
    return [(d.page_content, d.metadata) for d in docs]


def test_concrete_document_defaults_metadata():
    doc = Document(page_content="hello")
    assert doc.page_content == "hello"
    assert doc.metadata == {}


def test_concrete_document_metadata_is_not_shared():
    """A mutable default would leak between instances."""
    a, b = Document(page_content="a"), Document(page_content="b")
    a.metadata["x"] = 1
    assert b.metadata == {}


def test_accepts_ragas_document():
    assert read([Document(page_content="y", metadata={"k": "v"})]) == [
        ("y", {"k": "v"})
    ]


def test_accepts_arbitrary_duck_typed_object():
    class Duck:
        page_content = "z"
        metadata: dict = {}

    assert read([Duck()]) == [("z", {})]


def test_protocol_is_not_runtime_checkable():
    """Guards the gotcha: a Protocol with non-method members cannot isinstance.

    If someone makes this ``@runtime_checkable`` and adds an ``isinstance``
    check, it raises TypeError at runtime instead of failing a type check.
    """
    with pytest.raises(TypeError):
        isinstance(Document(page_content="x"), DocumentLike)  # type: ignore[misc]


def test_accepts_a_real_langchain_document():
    """LangChain documents must keep working -- users need change nothing.

    Skipped once langchain-core is no longer installed (stage 8); the structural
    guarantee is covered by the duck-typed test above regardless.
    """
    lc = pytest.importorskip("langchain_core.documents")
    doc = lc.Document(page_content="x", metadata={"a": 1})
    assert read([doc]) == [("x", {"a": 1})]
