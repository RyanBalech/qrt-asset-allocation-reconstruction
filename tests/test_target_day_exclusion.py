"""Check the key distinction between overlap evidence and ridge fitting."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
from v4_equity_imputation import infer
from v5_adaptive_ridge import adaptive


class TargetDayExclusionTest(unittest.TestCase):
    def test_later_target_day_changes_direct_evidence_but_not_ridge_fit(self):
        rng = np.random.default_rng(91)
        calendar = pd.bdate_range("2020-01-01", periods=45)
        returns = rng.normal(0, 0.02, size=(2, 5, 20))
        stocks = rng.normal(0, 1, size=(len(calendar), 8))
        allocations = pd.DataFrame({"group": [1] * 5})
        dates = {0: calendar[20], 1: calendar[21]}

        first, direct, covered = infer(
            returns,
            np.array([0, 1]),
            allocations,
            dates,
            {1: calendar},
            {1: stocks},
            alpha=0.7,
            radius=260,
            tau=100,
        )
        changed = returns.copy()
        changed[1, :, 0] += 1.0
        second, changed_direct, changed_covered = infer(
            changed,
            np.array([0, 1]),
            allocations,
            dates,
            {1: calendar},
            {1: stocks},
            alpha=0.7,
            radius=260,
            tau=100,
        )

        np.testing.assert_allclose(first[0], second[0], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(changed_direct[0] - direct[0], np.ones(5))
        self.assertTrue(covered[0].all())
        self.assertTrue(changed_covered[0].all())

    def test_adaptive_skips_last_calendar_session(self):
        rng = np.random.default_rng(17)
        calendar = pd.bdate_range("2020-01-01", periods=30)
        result = adaptive(
            rng.normal(size=(1, 5, 20)),
            np.array([0]),
            pd.DataFrame({"group": [1] * 5}),
            {0: calendar[-1]},
            {1: calendar},
            {1: rng.normal(size=(30, 8))},
        )
        self.assertTrue(np.isnan(result).all())

    def test_adaptive_excludes_target_day_from_fitting(self):
        rng = np.random.default_rng(91)
        calendar = pd.bdate_range("2020-01-01", periods=45)
        returns = rng.normal(0, 0.02, size=(2, 5, 20))
        args = (
            np.array([0, 1]),
            pd.DataFrame({"group": [1] * 5}),
            {0: calendar[20], 1: calendar[21]},
            {1: calendar},
            {1: rng.normal(size=(45, 8))},
        )
        first = adaptive(returns, *args)
        changed = returns.copy()
        changed[1, :, 0] += 1
        second = adaptive(changed, *args)
        np.testing.assert_allclose(first[0], second[0], rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
