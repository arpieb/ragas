"""Native prompt-value types for the ragas LLM interface.

These replace ``langchain_core.prompt_values.PromptValue`` / ``StringPromptValue``
and ``langchain_core.messages.BaseMessage``.

``StringPromptValue`` keeps the field name ``text`` so existing construction
sites -- ``StringPromptValue(text=...)`` in ``prompt/base.py`` and
``metrics/_nv_metrics.py`` -- and reads of ``prompt_value.text`` are unchanged.

``Message`` carries an explicit ``role`` rather than encoding it in the class
name the way LangChain's ``HumanMessage``/``AIMessage``/``SystemMessage`` do.
Consumers that previously had to sniff ``m.__class__.__name__`` can just read
``m.role`` (see ``llms/oci_genai_wrapper.py``).
"""

from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod

from pydantic import BaseModel

MessageRole = t.Literal["system", "user", "assistant"]


class Message(BaseModel):
    """A single chat message.

    ``content`` is usually a string. It may also be a list of content parts, as
    used for multimodal prompts that interleave text and images.
    """

    role: MessageRole = "user"
    content: t.Union[str, t.List[t.Dict[str, t.Any]]]


def coerce_role(value: str) -> MessageRole:
    """Map an arbitrary role string onto a known role, defaulting to ``user``.

    Provider payloads and user-supplied message dicts carry free-form role
    strings; this keeps ``Message`` construction total rather than raising.
    """
    if value == "system":
        return "system"
    if value == "assistant":
        return "assistant"
    return "user"


class PromptValue(BaseModel, ABC):
    """A prompt that can be rendered either as plain text or as chat messages."""

    @abstractmethod
    def to_string(self) -> str:
        """Render the prompt as a single string, for completion-style APIs."""

    @abstractmethod
    def to_messages(self) -> t.List[Message]:
        """Render the prompt as chat messages, for chat-style APIs."""


class StringPromptValue(PromptValue):
    """A plain-text prompt."""

    text: str

    def to_string(self) -> str:
        return self.text

    def to_messages(self) -> t.List[Message]:
        return [Message(role="user", content=self.text)]


class ChatPromptValue(PromptValue):
    """A prompt expressed as a sequence of chat messages."""

    messages: t.List[Message]

    def to_string(self) -> str:
        """Flatten to text, for completion-style APIs that cannot take roles."""
        return "\n".join(
            f"{m.role}: {m.content}"
            if isinstance(m.content, str)
            else f"{m.role}: {m.content!r}"
            for m in self.messages
        )

    def to_messages(self) -> t.List[Message]:
        return list(self.messages)
