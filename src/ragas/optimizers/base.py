from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ragas.dataset_schema import SingleMetricAnnotation
from ragas.llms.base import BaseRagasLLM
from ragas.losses import Loss
from ragas.metrics.base import MetricWithLLM
from ragas.run_config import RunConfig

if t.TYPE_CHECKING:
    from ragas.callbacks import Callbacks


@dataclass
class Optimizer(ABC):
    """
    Abstract base class for all optimizers.
    """

    metric: t.Optional[MetricWithLLM] = None
    llm: t.Optional[BaseRagasLLM] = None

    @abstractmethod
    def optimize(
        self,
        dataset: SingleMetricAnnotation,
        loss: Loss,
        config: t.Dict[t.Any, t.Any],
        run_config: t.Optional[RunConfig] = None,
        batch_size: t.Optional[int] = None,
        callbacks: t.Optional[Callbacks] = None,
        with_debugging_logs=False,
        raise_exceptions: bool = True,
    ) -> t.Dict[str, str]:
        """
        Optimizes the prompts for the given metric.

        Parameters
        ----------
        dataset : SingleMetricAnnotation
            The annotated samples to optimize against.
        loss : Loss
            The loss used to score candidate prompts.
        config : Dict[Any, Any]
            The optimizer configuration.
        run_config : Optional[RunConfig], optional
            Execution limits for the underlying runs, by default None.
        batch_size : Optional[int], optional
            The number of samples per batch, by default None.
        callbacks : Optional[Callbacks], optional
            The callbacks to use during optimization, by default None.
        with_debugging_logs : bool, optional
            Whether to emit debugging logs, by default False.
        raise_exceptions : bool, optional
            Whether to raise instead of swallowing errors, by default True.

        Returns
        -------
        Dict[str, str]
            The optimized prompts for given chain.
        """
        raise NotImplementedError("The method `optimize` must be implemented.")
