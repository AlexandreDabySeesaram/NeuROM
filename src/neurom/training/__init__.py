"""Training strategies for tensor decompositions."""

from neurom.training.history import StageRecord, TrainingHistory
from neurom.training.criteria import (
    StageCriterion,
    EnrichmentCriterion,
    RelativeChange,
    Stagnation,
    FixedIterations,
    RelativeGain,
    Usefulness,
    MaxStages,
    AnyOf,
)
from neurom.training.progress import ProgressReporter, ProgressLog
from neurom.training.base import Phase, PGDTrainer
from neurom.training.greedy import GreedyTrainer

__all__ = [
    "StageRecord",
    "TrainingHistory",
    "StageCriterion",
    "EnrichmentCriterion",
    "RelativeChange",
    "Stagnation",
    "FixedIterations",
    "RelativeGain",
    "Usefulness",
    "MaxStages",
    "AnyOf",
    "Phase",
    "PGDTrainer",
    "GreedyTrainer",
    "ProgressReporter",
    "ProgressLog",
]
