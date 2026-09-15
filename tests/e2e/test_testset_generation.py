import os

import pytest

from ragas.testset import TestsetGenerator


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_testset_generation_e2e():
    # generate kg -- plain file walk, no langchain document loader needed
    from pathlib import Path

    import openai
    from langchain_core.documents import Document

    from ragas.embeddings import embedding_factory
    from ragas.llms import llm_factory

    docs = [
        Document(page_content=p.read_text(), metadata={"source": str(p)})
        for p in sorted(Path("./docs").rglob("*.md"))
    ]

    # llm_factory requires an explicit client
    generator_llm = llm_factory("gpt-4o", client=openai.OpenAI())
    generator_embeddings = embedding_factory()

    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,  # type: ignore
    )
    dataset = generator.generate_with_langchain_docs(docs, testset_size=3)
    assert dataset is not None
