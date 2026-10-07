"""Feature caches must distinguish inference mode and date/pool inputs."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
import v5_pipeline as pipeline


class FeatureCacheTests(unittest.TestCase):
    def test_changed_mode_pool_or_calendar_rebuilds_features(self):
        for change in ("adaptive", "pool", "grid", "config"):
            with (
                self.subTest(change=change),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                (root / "v5_frozen_config.json").write_text("{}")
                frame = pd.DataFrame({"external": [0.1], "previous_external": [0.2]})
                pool, grid = np.array([0]), pd.bdate_range("2020-01-01", periods=2)
                with (
                    patch.object(pipeline, "ROOT", root),
                    patch.object(
                        pipeline, "build", return_value=(frame, {}, frame)
                    ) as build,
                ):
                    pipeline.cached_build({}, pool, grid, "test", False)
                    if change == "adaptive":
                        pipeline.cached_build({}, pool, grid, "test", True)
                    elif change == "pool":
                        pipeline.cached_build({}, np.array([1]), grid, "test", False)
                    elif change == "grid":
                        pipeline.cached_build(
                            {}, pool, grid + pd.Timedelta(days=1), "test", False
                        )
                    else:
                        (root / "v5_frozen_config.json").write_text('{"changed":true}')
                        pipeline.cached_build({}, pool, grid, "test", False)
                    self.assertEqual(build.call_count, 2)

    def test_identical_inputs_reuse_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "v5_frozen_config.json").write_text("{}")
            frame = pd.DataFrame({"external": [0.1], "previous_external": [0.2]})
            pool, grid = np.array([0]), pd.bdate_range("2020-01-01", periods=2)
            with (
                patch.object(pipeline, "ROOT", root),
                patch.object(
                    pipeline, "build", return_value=(frame, {}, frame)
                ) as build,
            ):
                pipeline.cached_build({}, pool, grid, "test", True)
                pd.DataFrame({"q": [0], "real_date": [str(grid[0])]}).to_csv(
                    root / "v5_calendar_test.csv", index=False
                )
                cached, dates, prior = pipeline.cached_build(
                    {}, pool, grid, "test", True
                )
                self.assertEqual(build.call_count, 1)
                pd.testing.assert_frame_equal(cached, frame)
                self.assertEqual(dates[0], grid[0])
                self.assertEqual(prior.external.iloc[0], 0.2)


if __name__ == "__main__":
    unittest.main()
