=== "OpenAI"
    Install the OpenAI SDK:

    ```bash
    pip install openai
    ```

    Set your API key:

    ```python
    import os
    os.environ["OPENAI_API_KEY"] = "your-openai-key"
    ```

    Build the evaluator LLM with `llm_factory`:

    ```python
    from openai import OpenAI
    from ragas.llms import llm_factory

    evaluator_llm = llm_factory("gpt-4o", client=OpenAI())
    ```

=== "Anthropic"
    Install the Anthropic SDK:

    ```bash
    pip install anthropic
    ```

    Set your API key:

    ```python
    import os
    os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"
    ```

    ```python
    import anthropic
    from ragas.llms import llm_factory

    evaluator_llm = llm_factory(
        "claude-sonnet-4-5",
        provider="anthropic",
        client=anthropic.Anthropic(),
    )
    ```

=== "Google"
    Install the Google GenAI SDK:

    ```bash
    pip install google-genai
    ```

    ```python
    import os
    os.environ["GOOGLE_API_KEY"] = "your-google-key"
    ```

    ```python
    from google import genai
    from ragas.llms import llm_factory

    evaluator_llm = llm_factory(
        "gemini-2.0-flash",
        provider="google",
        client=genai.Client(),
    )
    ```

    For Vertex AI, or to avoid the Google SDK entirely, use the LiteLLM tab.

=== "Azure"
    Install the OpenAI SDK, which ships the Azure client:

    ```bash
    pip install openai
    ```

    ```python
    import os
    os.environ["AZURE_OPENAI_API_KEY"] = "your-azure-key"
    ```

    ```python
    from openai import AzureOpenAI
    from ragas.llms import llm_factory

    client = AzureOpenAI(
        api_version="2024-02-01",
        azure_endpoint="https://<your-resource>.openai.azure.com",
    )
    evaluator_llm = llm_factory("gpt-4o", provider="azure", client=client)
    ```

    `gpt-4o` here is your **deployment name**, which need not match the model name.

=== "AWS Bedrock"
    Bedrock is reached through LiteLLM:

    ```bash
    pip install litellm boto3
    ```

    Credentials come from the usual AWS chain — environment, profile or instance role:

    ```python
    import os
    os.environ["AWS_REGION_NAME"] = "us-east-1"
    ```

    ```python
    import instructor
    import litellm
    from ragas.llms import llm_factory

    client = instructor.from_litellm(litellm.completion)
    evaluator_llm = llm_factory(
        "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
        provider="litellm",
        client=client,
    )
    ```

=== "Others (LiteLLM)"
    LiteLLM reaches 100+ providers — Ollama, vLLM, Groq, Mistral, Cohere, Together,
    Vertex AI and more — behind one interface, so ragas is not tied to any single
    vendor's SDK.

    ```bash
    pip install litellm
    ```

    ```python
    import instructor
    import litellm
    from ragas.llms import llm_factory

    client = instructor.from_litellm(litellm.completion)

    # any model string LiteLLM understands
    evaluator_llm = llm_factory("ollama/llama3", provider="litellm", client=client)
    ```

    See the [LiteLLM provider list](https://docs.litellm.ai/docs/providers) for the
    model string to use. Credentials are read from the environment variables each
    provider expects.

!!! note "Migrating from `LangchainLLMWrapper`"
    ragas no longer depends on LangChain, and `LangchainLLMWrapper` has been
    removed. Replace `LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))` with
    `llm_factory("gpt-4o", client=OpenAI())` as above. See the
    **Migrating off LangChain** guide under How-to Guides > Migrations.
