"""SimpleCriteriaScore prompt classes and models."""

import typing as t

from pydantic import BaseModel, Field

from ragas.prompt.metrics.base_prompt import BasePrompt

BASE_INSTRUCTION = "Evaluate the input based on the criteria defined."


class SimpleCriteriaInput(BaseModel):
    """Input model for criteria-based scoring."""

    user_input: t.Optional[str] = Field(
        default=None, description="The input to the llm system"
    )
    response: t.Optional[str] = Field(
        default=None, description="The response from the llm system"
    )
    retrieved_contexts: t.Optional[t.List[str]] = Field(
        default=None, description="The retrieved contexts from the llm system"
    )
    reference_contexts: t.Optional[t.List[str]] = Field(
        default=None, description="The reference contexts for the evaluation"
    )
    reference: t.Optional[str] = Field(
        default=None, description="The reference answer for evaluation"
    )


class SimpleCriteriaOutput(BaseModel):
    """Output model for criteria-based scoring."""

    reason: str = Field(..., description="Reason for the scoring")
    score: int = Field(..., description="The score for the submission")


class SimpleCriteriaPrompt(BasePrompt[SimpleCriteriaInput, SimpleCriteriaOutput]):
    """Prompt for scoring a submission against a user-supplied criterion."""

    input_model = SimpleCriteriaInput
    output_model = SimpleCriteriaOutput
    instruction = BASE_INSTRUCTION

    examples = [
        (
            SimpleCriteriaInput(
                user_input="Who was the first president of the United States?",
                response="Thomas Jefferson was the first president of the United States.",
                reference="George Washington was the first president of the United States.",
            ),
            SimpleCriteriaOutput(
                reason="The response names the wrong president.",
                score=0,
            ),
        ),
        (
            SimpleCriteriaInput(
                user_input="Who was the first president of the United States?",
                response="George Washington was the first president of the United States.",
                reference="George Washington was the first president of the United States.",
            ),
            SimpleCriteriaOutput(
                reason="The response matches the reference exactly.",
                score=5,
            ),
        ),
    ]
