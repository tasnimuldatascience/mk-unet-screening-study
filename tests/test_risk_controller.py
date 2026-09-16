import unittest
import numpy as np
from screening.risk_controller import BenefitCalibrator, RiskBudgetController, reversible_step


class RiskControllerTests(unittest.TestCase):
    def test_lower_bound_is_conservative_on_calibration_points(self):
        x = np.arange(20, dtype=float)[:, None]
        y = 0.02 * x[:, 0] - 0.1
        c = BenefitCalibrator(alpha=0.1).fit(x[:10], y[:10], x[10:], y[10:])
        self.assertTrue(np.all(c.lower_bound(x[10:]) <= y[10:] + 1e-8))

    def test_commit_budget_and_rollback(self):
        ctl = RiskBudgetController(max_commits=1, max_latency_ms=20)
        self.assertFalse(ctl.decide(-0.01, 10))
        self.assertFalse(ctl.decide(0.01, 30))
        self.assertTrue(ctl.decide(0.01, 10))
        self.assertFalse(ctl.decide(0.02, 10))
        self.assertEqual(reversible_step({"w": 1}, {"w": 2}, False), {"w": 1})


if __name__ == "__main__":
    unittest.main()
