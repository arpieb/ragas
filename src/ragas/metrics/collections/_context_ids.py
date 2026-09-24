"""ID-based context metrics - compare retrieved context IDs against reference IDs.

Both metrics work on identifiers rather than text, so they need no LLM and no
embeddings. IDs are compared as strings, which lets a caller mix integer and
string IDs without surprises.
"""

import logging
import typing as t

import numpy as np

from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult

logger = logging.getLogger(__name__)


def _as_id_set(ids: t.Sequence[t.Any]) -> t.Set[str]:
    """Normalise IDs to strings so int and str IDs compare equal."""
    return {str(i) for i in ids}


class IDBasedContextPrecision(BaseMetric):
    """
    Proportion of retrieved context IDs that are relevant.

    Relevance is exact membership in the reference IDs -- no text comparison is
    involved, so this is a cheap alternative to the LLM-judged context precision
    metrics when your retriever returns stable identifiers.

    Usage:
        >>> from ragas.metrics.collections import IDBasedContextPrecision
        >>>
        >>> metric = IDBasedContextPrecision()
        >>>
        >>> result = await metric.ascore(
        ...     retrieved_context_ids=["doc1", "doc2", "doc3"],
        ...     reference_context_ids=["doc1", "doc2"],
        ... )
        >>> print(result.value)  # 0.666...

    Attributes:
        name: The metric name
        allowed_values: Score range (0.0 to 1.0)
    """

    def __init__(self, name: str = "id_based_context_precision", **base_kwargs):
        super().__init__(name=name, **base_kwargs)

    async def ascore(
        self,
        retrieved_context_ids: t.List[t.Union[str, int]],
        reference_context_ids: t.List[t.Union[str, int]],
    ) -> MetricResult:
        """
        Args:
            retrieved_context_ids: IDs the retriever returned
            reference_context_ids: IDs known to be relevant

        Returns:
            MetricResult with the proportion of retrieved IDs that are relevant,
            or NaN when nothing was retrieved.
        """
        retrieved = _as_id_set(retrieved_context_ids)
        reference = _as_id_set(reference_context_ids)

        if not retrieved:
            logger.warning(
                "No retrieved context IDs provided, cannot calculate precision."
            )
            return MetricResult(value=float(np.nan))

        hits = len(retrieved & reference)
        return MetricResult(value=hits / len(retrieved))


class IDBasedContextRecall(BaseMetric):
    """
    Proportion of reference context IDs that were retrieved.

    The mirror of :class:`IDBasedContextPrecision`: precision asks how much of
    what you retrieved was useful, recall asks how much of what was useful you
    retrieved.

    Usage:
        >>> from ragas.metrics.collections import IDBasedContextRecall
        >>>
        >>> metric = IDBasedContextRecall()
        >>>
        >>> result = await metric.ascore(
        ...     retrieved_context_ids=["doc1", "doc2"],
        ...     reference_context_ids=["doc1", "doc2", "doc3"],
        ... )
        >>> print(result.value)  # 0.666...

    Attributes:
        name: The metric name
        allowed_values: Score range (0.0 to 1.0)
    """

    def __init__(self, name: str = "id_based_context_recall", **base_kwargs):
        super().__init__(name=name, **base_kwargs)

    async def ascore(
        self,
        retrieved_context_ids: t.List[t.Union[str, int]],
        reference_context_ids: t.List[t.Union[str, int]],
    ) -> MetricResult:
        """
        Args:
            retrieved_context_ids: IDs the retriever returned
            reference_context_ids: IDs known to be relevant

        Returns:
            MetricResult with the proportion of reference IDs that were
            retrieved, or NaN when there are no reference IDs.
        """
        retrieved = _as_id_set(retrieved_context_ids)
        reference = _as_id_set(reference_context_ids)

        if not reference:
            logger.warning(
                "No reference context IDs provided, cannot calculate recall."
            )
            return MetricResult(value=float(np.nan))

        hits = len(reference & retrieved)
        return MetricResult(value=hits / len(reference))
