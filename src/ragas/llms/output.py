"""Native result types for the ragas LLM interface.

These replace ``langchain_core.outputs.LLMResult`` / ``Generation`` in the
``BaseRagasLLM`` contract.

The field names deliberately match LangChain's (``generations``, ``text``,
``llm_output``) so that the many call sites reading
``response.generations[0][i].text`` are unchanged. Before this module existed,
wrappers with no other LangChain involvement -- the Haystack and OCI GenAI ones
-- had to import LangChain purely to build a return value.

``LLMResult.flatten()`` is intentionally not reimplemented: its only caller was
``LangchainLLMWrapper``, removed in stage 3.
"""

from __future__ import annotations

import typing as t

from pydantic import BaseModel, Field


class Generation(BaseModel):
    """A single completion returned by an LLM."""

    text: str
    """The generated text."""

    generation_info: t.Optional[t.Dict[str, t.Any]] = None
    """Raw provider metadata for this completion (finish reason, logprobs, ...)."""


class LLMResult(BaseModel):
    """The result of an LLM call.

    ``generations`` is a list of prompts, each holding a list of completions for
    that prompt. A single prompt asked for ``n`` completions therefore looks like
    ``[[gen_1, ..., gen_n]]``.
    """

    generations: t.List[t.List[Generation]] = Field(default_factory=list)
    """Per-prompt lists of completions."""

    llm_output: t.Optional[t.Dict[str, t.Any]] = None
    """Raw provider-level metadata for the call as a whole (token usage, model, ...)."""
