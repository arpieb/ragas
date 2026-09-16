"""Tests for the ragas-native LLM result and prompt value types.

These replaced the LangChain types in the ``BaseRagasLLM`` contract. The field
names were deliberately kept identical to LangChain's so that call sites reading
``response.generations[0][i].text`` did not have to change -- these tests pin
that access pattern so a future refactor cannot quietly break it.
"""

from __future__ import annotations

import pytest

from ragas.llms.output import Generation, LLMResult
from ragas.prompt.value import (
    ChatPromptValue,
    Message,
    StringPromptValue,
    coerce_role,
)


class TestLLMResult:
    def test_access_pattern_used_across_the_codebase(self):
        """``resp.generations[0][i].text`` is what every caller reads."""
        resp = LLMResult(
            generations=[[Generation(text="first"), Generation(text="second")]]
        )
        assert resp.generations[0][0].text == "first"
        assert resp.generations[0][1].text == "second"

    def test_batched_shape(self):
        """One inner list per prompt, one entry per completion."""
        resp = LLMResult(generations=[[Generation(text="a")], [Generation(text="b")]])
        assert len(resp.generations) == 2
        assert resp.generations[1][0].text == "b"

    def test_round_trips_through_json(self):
        resp = LLMResult(
            generations=[[Generation(text="hi", generation_info={"finish": "stop"})]],
            llm_output={"model": "gpt-4o-mini"},
        )
        restored = LLMResult.model_validate_json(resp.model_dump_json())
        assert restored == resp
        assert restored.generations[0][0].generation_info == {"finish": "stop"}
        assert restored.llm_output == {"model": "gpt-4o-mini"}

    def test_defaults(self):
        assert LLMResult().generations == []
        assert Generation(text="x").generation_info is None


class TestStringPromptValue:
    def test_to_string_and_messages(self):
        pv = StringPromptValue(text="hello")
        assert pv.to_string() == "hello"
        assert pv.to_messages() == [Message(role="user", content="hello")]

    def test_text_attribute_is_read_directly(self):
        """``prompt_value.text`` is read in pydantic_prompt.py."""
        assert StringPromptValue(text="abc").text == "abc"

    def test_round_trips_through_json(self):
        pv = StringPromptValue(text="hello")
        assert StringPromptValue.model_validate_json(pv.model_dump_json()) == pv


class TestChatPromptValue:
    def test_preserves_roles(self):
        pv = ChatPromptValue(
            messages=[
                Message(role="system", content="be terse"),
                Message(role="user", content="hi"),
            ]
        )
        assert [m.role for m in pv.to_messages()] == ["system", "user"]

    def test_to_string_flattens_with_roles(self):
        pv = ChatPromptValue(
            messages=[
                Message(role="system", content="be terse"),
                Message(role="user", content="hi"),
            ]
        )
        assert pv.to_string() == "system: be terse\nuser: hi"

    def test_to_messages_returns_a_copy(self):
        """Mutating the result must not corrupt the prompt value."""
        pv = ChatPromptValue(messages=[Message(role="user", content="hi")])
        pv.to_messages().append(Message(role="user", content="injected"))
        assert len(pv.messages) == 1


class TestMessage:
    def test_defaults_to_user(self):
        assert Message(content="hi").role == "user"

    def test_accepts_multimodal_list_content(self):
        parts = [{"type": "text", "text": "describe"}, {"type": "image_url"}]
        assert Message(content=parts).content == parts

    @pytest.mark.parametrize(
        "given,expected",
        [
            ("system", "system"),
            ("assistant", "assistant"),
            ("user", "user"),
            ("human", "user"),
            ("", "user"),
            ("something-odd", "user"),
        ],
    )
    def test_coerce_role(self, given, expected):
        assert coerce_role(given) == expected


def test_image_text_prompt_value_emits_ragas_messages():
    """The multimodal prompt value re-parented onto the ragas types."""
    from ragas.prompt.multi_modal_prompt import ImageTextPromptValue

    messages = ImageTextPromptValue(items=["just some text"]).to_messages()
    assert len(messages) == 1
    assert isinstance(messages[0], Message)
    assert messages[0].role == "user"
