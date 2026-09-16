"""OpenTelemetry emission from the ragas run tree.

ragas depends on ``opentelemetry-api`` only, so in production spans are a no-op
unless the user configures an SDK. These tests configure one (via the test-only
``opentelemetry-sdk`` dependency) and assert that the span tree mirrors the
native run tree.
"""

from __future__ import annotations

import pytest

from ragas.callbacks import ChainType, RagasTracer, new_group

trace_api = pytest.importorskip("opentelemetry.trace")
sdk_trace = pytest.importorskip("opentelemetry.sdk.trace")
export = pytest.importorskip("opentelemetry.sdk.trace.export")
in_memory = pytest.importorskip(
    "opentelemetry.sdk.trace.export.in_memory_span_exporter"
)


@pytest.fixture
def spans(monkeypatch):
    """Route ragas' module-level tracer at a fresh in-memory provider."""
    import ragas.callbacks as cb

    exporter = in_memory.InMemorySpanExporter()
    provider = sdk_trace.TracerProvider()
    provider.add_span_processor(export.SimpleSpanProcessor(exporter))
    monkeypatch.setattr(cb, "_tracer", provider.get_tracer("ragas-test"))
    return exporter


def by_name(exporter):
    return {s.name: s for s in exporter.get_finished_spans()}


def test_spans_mirror_the_run_tree(spans):
    tracer = RagasTracer()
    root_rm, root_group = new_group(
        name="ragas evaluation",
        inputs={},
        callbacks=[tracer],
        metadata={"type": ChainType.EVALUATION},
    )
    row_rm, row_group = new_group(
        name="row 0",
        inputs={"user_input": "q"},
        callbacks=root_group,
        metadata={"type": ChainType.ROW},
    )
    metric_rm, _ = new_group(
        name="faithfulness",
        inputs={},
        callbacks=row_group,
        metadata={"type": ChainType.METRIC},
    )
    metric_rm.on_chain_end({"output": 0.75})
    row_rm.on_chain_end({"faithfulness": 0.75})
    root_rm.on_chain_end({"scores": []})

    finished = by_name(spans)
    assert set(finished) == {"ragas evaluation", "row 0", "faithfulness"}

    # nesting matches the ChainRun tree
    root, row, metric = (
        finished["ragas evaluation"],
        finished["row 0"],
        finished["faithfulness"],
    )
    assert root.parent is None
    assert row.parent.span_id == root.context.span_id
    assert metric.parent.span_id == row.context.span_id
    # ...and they share one trace
    assert root.context.trace_id == row.context.trace_id == metric.context.trace_id


def test_span_attributes_use_openinference_keys(spans):
    tracer = RagasTracer()
    rm, _ = new_group(
        name="faithfulness",
        inputs={"question": "q"},
        callbacks=[tracer],
        metadata={"type": ChainType.METRIC},
    )
    rm.on_chain_end({"output": 0.75})

    span = by_name(spans)["faithfulness"]
    assert span.attributes["openinference.span.kind"] == "EVALUATOR"
    assert span.attributes["ragas.chain.type"] == "metric"
    assert span.attributes["ragas.run.id"] == str(rm.run_id)
    assert "question" in span.attributes["input.value"]
    assert "0.75" in span.attributes["output.value"]


def test_prompt_spans_are_llm_kind(spans):
    tracer = RagasTracer()
    rm, _ = new_group(
        name="StatementGenerator",
        inputs={},
        callbacks=[tracer],
        metadata={"type": ChainType.RAGAS_PROMPT},
    )
    rm.on_chain_end({"output": []})
    assert (
        by_name(spans)["StatementGenerator"].attributes["openinference.span.kind"]
        == "LLM"
    )


def test_unserialisable_payloads_do_not_raise(spans):
    """Chain payloads hold arbitrary objects; rendering must never break a run."""

    class Opaque:
        def __repr__(self):
            return "<opaque>"

    tracer = RagasTracer()
    rm, _ = new_group(
        name="x", inputs={"obj": Opaque()}, callbacks=[tracer], metadata={}
    )
    rm.on_chain_end({"obj": Opaque()})

    span = by_name(spans)["x"]
    assert "opaque" in span.attributes["output.value"]


def test_long_payloads_are_truncated(spans):
    tracer = RagasTracer()
    rm, _ = new_group(name="x", inputs={}, callbacks=[tracer], metadata={})
    rm.on_chain_end({"big": "x" * 50_000})

    value = by_name(spans)["x"].attributes["output.value"]
    assert value.endswith("...[truncated]")
    assert len(value) < 50_000


def test_error_is_recorded_on_the_span(spans):
    tracer = RagasTracer()
    rm, _ = new_group(name="x", inputs={}, callbacks=[tracer], metadata={})
    rm.on_chain_error(ValueError("boom"))

    span = by_name(spans)["x"]
    assert span.status.status_code is trace_api.StatusCode.ERROR
    assert any(e.name == "exception" for e in span.events)


def test_spans_are_a_noop_without_a_configured_provider():
    """The shipped behaviour: opentelemetry-api only, no SDK, nothing breaks.

    Crucially the native run tree is still populated, which is why ragas can get
    away with not depending on the SDK.
    """
    import ragas.callbacks as cb

    noop = trace_api.NoOpTracer()
    original, cb._tracer = cb._tracer, noop
    try:
        tracer = RagasTracer()
        rm, _ = new_group(name="x", inputs={"a": 1}, callbacks=[tracer], metadata={})
        rm.on_chain_end({"b": 2})
    finally:
        cb._tracer = original

    assert list(tracer.traces.values())[0].outputs == {"b": 2}
