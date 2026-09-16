"""Prototype controller for reversible, validation-calibrated test-time updates."""
from dataclasses import dataclass
import numpy as np


class BenefitCalibrator:
    """Ridge predictor with a split-conformal lower bound on Dice change."""
    def __init__(self, alpha=0.1, ridge=1e-6):
        self.alpha, self.ridge = alpha, ridge

    def fit(self, x_fit, delta_fit, x_cal, delta_cal):
        rows = [[1.0, *map(float, row)] for row in x_fit]
        target = list(map(float, delta_fit))
        width = len(rows[0])
        normal = [[sum(row[i] * row[j] for row in rows) + (self.ridge if i == j and i else 0.0)
                   for j in range(width)] for i in range(width)]
        rhs = [sum(row[i] * y for row, y in zip(rows, target)) for i in range(width)]
        # Small dependency-free Gaussian elimination avoids BLAS runtime conflicts.
        aug = [normal[i] + [rhs[i]] for i in range(width)]
        for col in range(width):
            pivot = max(range(col, width), key=lambda r: abs(aug[r][col]))
            aug[col], aug[pivot] = aug[pivot], aug[col]
            scale = aug[col][col]
            aug[col] = [v / scale for v in aug[col]]
            for row in range(width):
                if row != col:
                    factor = aug[row][col]
                    aug[row] = [a - factor * b for a, b in zip(aug[row], aug[col])]
        self.coef_ = [row[-1] for row in aug]
        pred = self.predict(x_cal)
        residual = pred - np.asarray(delta_cal, float)
        n = len(residual)
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.margin_ = float(np.quantile(residual, level, method="higher"))
        return self

    def predict(self, x):
        return np.asarray([self.coef_[0] + sum(c * float(v) for c, v in zip(self.coef_[1:], row)) for row in x])

    def lower_bound(self, x):
        return self.predict(x) - self.margin_


@dataclass
class RiskBudgetController:
    max_commits: int
    max_latency_ms: float
    commits: int = 0

    def decide(self, benefit_lower_bound: float, candidate_latency_ms: float) -> bool:
        accept = benefit_lower_bound > 0 and candidate_latency_ms <= self.max_latency_ms and self.commits < self.max_commits
        if accept:
            self.commits += 1
        return accept


def reversible_step(snapshot, candidate, accept):
    """Return candidate state only after controller acceptance; otherwise roll back."""
    return candidate if accept else snapshot
