# Evaluation notes

The published v5 comparison uses three reserved offset pools (0, 7 and 14) from a 21-session sparse grid. They contain 47,192 labeled allocation rows across 180 dates. The v5 correction models were trained on 15 other pools, and settings were chosen on offsets 3, 10 and 17. V4 and v5 predictions were scored on the same reserved rows.

| Measure | Value |
| --- | ---: |
| v4 accuracy | 0.745571 |
| v5 accuracy | 0.762121 |
| Difference | 0.016549 |
| v5 95% date bootstrap interval | 0.750803–0.773856 |
| Difference 95% date bootstrap interval | 0.011705–0.021985 |

The bootstrap resamples dates, not individual rows, because allocations observed on the same date share market conditions. The result is a paired historical reconstruction comparison. Earlier v4 development inspected some of these dates, and the baseline CatBoost prior is inherited from an earlier evaluation. Treat the confidence interval as descriptive of the comparison, not a clean estimate of live trading accuracy.

The v5 public score recorded in the local research notes is 0.774082. That number was reported by the participant after upload; no independent leaderboard export is included in this repository. The local submission audit records 31,870 test rows and a SHA-256 digest of `37d1fb06220be7cae045a93a2bda5fe946845d1dbf4c69f67b15e21a893f2d3a` for the submitted CSV.

The subsequent v6 chronological comparison used 56,180 rows across 216 dates. It measured v6 accuracy of 0.718868 and v5 accuracy of 0.718281, a 0.000587 difference. Its date-bootstrap interval for the difference was -0.001637 to 0.002674. The comparison is also not entirely untouched because earlier work examined some 2015 labels. It does not support presenting v6 as a material improvement.

Most importantly, direct overlap and public target-day stock returns are historical-batch information. The reported accuracy must never be read as the performance of a deployable next-day trading system.
