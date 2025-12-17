from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class TypeArtifacts:
    type_value: str
    beta: Dict[str, float]
    mean_: Dict[str, float]
    baseline_survival: List[float]
    baseline_ratio: List[float]
    sample_size: int
    event_count: int
    fitted_at: str
    penalizer: Optional[float]
    l1_ratio: Optional[float]
    feature_cols: List[str]
    baseline_method: str

    def survival_at(self, t: int) -> float:
        if t < 0:
            return 1.0
        if t < len(self.baseline_survival):
            return float(self.baseline_survival[t])
        return float(self.baseline_survival[-1])

    def ratio_at(self, t: int) -> float:
        if t < 1:
            return 1.0
        if t < len(self.baseline_ratio):
            return float(self.baseline_ratio[t])
        return float(self.baseline_ratio[-1])

    def extend_baseline(self, max_time: int, extend_fn=None, tail_k: int = 6) -> None:
        current_T = len(self.baseline_survival) - 1
        if max_time <= current_T:
            return
        tail_start = max(1, len(self.baseline_ratio) - tail_k)
        tail_values = np.array(self.baseline_ratio[tail_start:])
        if extend_fn is None:
            fill_values = float(np.mean(tail_values)) if len(tail_values) else 1.0
            new_ratios = np.full(max_time - current_T, fill_values, dtype=float)
        else:
            new_times = np.arange(current_T + 1, max_time + 1, dtype=int)
            new_ratios = np.array(extend_fn(tail_values, new_times), dtype=float)
        last_survival = self.baseline_survival[-1]
        cumulative = last_survival * np.cumprod(new_ratios)
        self.baseline_ratio.extend(new_ratios.tolist())
        self.baseline_survival.extend(cumulative.tolist())


@dataclass
class TrainingResult:
    artifacts_by_type: Dict[str, TypeArtifacts] = field(default_factory=dict)
    skipped: Dict[str, str] = field(default_factory=dict)

    def add(self, type_value: str, artifacts: Optional[TypeArtifacts], reason: Optional[str]) -> None:
        if artifacts is not None:
            self.artifacts_by_type[type_value] = artifacts
        elif reason:
            self.skipped[type_value] = reason
