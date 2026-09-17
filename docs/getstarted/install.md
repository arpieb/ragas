# Installation

To get started, install Ragas using `pip` with the following command:

```bash
pip install ragas
```

If you'd like to experiment with the latest features, install the most recent version from the main branch:

```bash
pip install git+https://github.com/vibrantlabsai/ragas.git
```

If you're planning to contribute and make modifications to the code, ensure that you clone the repository and set it up as an [editable install](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs).

```bash
git clone https://github.com/vibrantlabsai/ragas.git 
pip install -e .
```

!!! note "Choosing a provider"
    ragas reaches providers through **LiteLLM**, which is installed with it — so
    100+ providers, including Bedrock, Ollama and vLLM, work out of the box with
    credentials read from the environment:

    ```python
    from ragas.llms import llm_factory

    evaluator_llm = llm_factory("ollama/llama3")
    ```

    ragas ships no vendor SDK of its own. To hand `llm_factory` a provider client
    directly instead, install that SDK yourself:

    ```bash
    pip install openai        # OpenAI / Azure OpenAI
    pip install anthropic     # Anthropic
    pip install google-genai  # Google
    ```

    ragas no longer depends on LangChain. If you are coming from a version that
    did, see the [migration guide](../howtos/migrations/migrate_off_langchain.md).
