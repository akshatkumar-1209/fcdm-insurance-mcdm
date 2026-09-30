"""
FCDM: A Robust Hybrid MCDM Framework
===================================
Implementation of the hybrid multi-criteria decision making framework
based on Fuzzy LBWA and Modified ARTASI with Decision Stability Intervals
and Monte Carlo Robustness Analysis.

Reference Paper:
"A robust hybrid MCDM framework with emphasis on decision stability intervals:
Performance evaluation of global insurance brokers using fuzzy LBWA and modified ARTASI"
Applied Soft Computing Journal 190 (2026) 114557.
"""

from .flbwa import TriangularFuzzyNumber, FuzzyLBWA
from .modified_artasi import ModifiedARTASI
from .dsi_sensitivity import DecisionStabilityIntervals
from .robustness_montecarlo import MonteCarloRobustness, TOPSIS
from .dataset import (
    CRITERIA_METADATA,
    EXPERT_PREFERENCES_TABLE5,
    BASELINE_WEIGHTS_TABLE7,
    INSURANCE_BROKERS,
    DECISION_MATRIX_2024,
    get_dataset
)

__all__ = [
    "TriangularFuzzyNumber",
    "FuzzyLBWA",
    "ModifiedARTASI",
    "DecisionStabilityIntervals",
    "MonteCarloRobustness",
    "TOPSIS",
    "CRITERIA_METADATA",
    "EXPERT_PREFERENCES_TABLE5",
    "BASELINE_WEIGHTS_TABLE7",
    "INSURANCE_BROKERS",
    "DECISION_MATRIX_2024",
    "get_dataset"
]
