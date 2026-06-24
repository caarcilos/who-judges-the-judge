from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from persuasion_judge_audit.models import Example, JudgePrediction


class RunnerError(RuntimeError):
    """Raised when a provider runner cannot produce a valid prediction."""


class JudgeRunner(ABC):
    """Interface implemented by all external judge providers."""

    @abstractmethod
    def predict(self, examples: Sequence[Example]) -> list[JudgePrediction]:
        """Return exactly one structured prediction for every example."""

