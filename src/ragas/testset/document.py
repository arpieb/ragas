"""Document types for testset generation.

Testset generation previously took ``langchain_core.documents.Document``, but
only ever reads two attributes off it: ``page_content`` and ``metadata``.

``DocumentLike`` captures exactly that, structurally. Because it is a
``Protocol``, **LangChain ``Document`` objects still satisfy it verbatim** --
callers loading documents with a LangChain loader need to change nothing, while
ragas itself no longer depends on LangChain to express the type.

``Document`` is the concrete counterpart, for the places that need to *build* a
document rather than accept one (for example converting llama-index documents).
"""

from __future__ import annotations

import typing as t
from dataclasses import dataclass, field


@dataclass
class Document:
    """A source document for testset generation."""

    page_content: str
    """The document's text."""

    metadata: t.Dict[str, t.Any] = field(default_factory=dict)
    """Arbitrary document metadata, carried through onto the generated nodes."""


class DocumentLike(t.Protocol):
    """Anything exposing ``page_content`` and ``metadata``.

    Deliberately *not* ``@runtime_checkable``: a ``Protocol`` with non-method
    members raises ``TypeError`` under ``isinstance()``. Where a runtime check is
    needed, test for the attribute instead.
    """

    page_content: str
    metadata: t.Dict[str, t.Any]
