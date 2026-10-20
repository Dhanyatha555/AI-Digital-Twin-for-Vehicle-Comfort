"""
Ride comfort evaluation and ISO 2631-1 human vibration weighting filter.
"""

from .iso2631 import ISO2631Filter, RideComfortEvaluation, ComfortCategory
from .metrics import (
    SuspensionMetrics,
    compute_metrics,
    build_comparison_dataframe,
)

# Alias for backwards compatibility
evaluate_suspension_performance = compute_metrics

__all__ = [
    "ISO2631Filter",
    "RideComfortEvaluation",
    "ComfortCategory",
    "SuspensionMetrics",
    "compute_metrics",
    "build_comparison_dataframe",
    "evaluate_suspension_performance",
]
