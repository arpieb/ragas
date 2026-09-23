"""Average precision over a ranked list of binary verdicts.

Shared by the context-precision metrics. It lived as two identical copies inside
``collections/context_precision/metric.py``; the non-LLM variant would have made
a third.
"""

import typing as t


def average_precision(verdicts: t.Sequence[int]) -> float:
    """Mean of the precision@k values taken at each relevant position.

    Rewards relevant contexts appearing early in the ranking. The ``1e-10`` guard
    keeps an all-zero verdict list at 0.0 rather than dividing by zero.
    """
    cumsum = 0
    numerator = 0.0
    for i, v in enumerate(verdicts):
        cumsum += v
        if v:
            numerator += cumsum / (i + 1)

    denominator = cumsum + 1e-10
    return numerator / denominator
