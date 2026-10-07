# Reconstructing historical allocation returns | QRT Challenge

An independent follow-up to the [QRT Asset Allocation Performance Forecasting challenge](https://challengedata.ens.fr/challenges/167). The challenge asks for the sign of the next-day return of an anonymized allocation, given 20 days of return and signed-volume history. This repository documents my later investigation into what can be recovered when the evaluation set is a historical batch rather than a live stream.

The central observation is that nearby rows contain overlapping return histories. When a later row is available, one of its lagged returns may reveal a queried row's target day. For remaining dates, I align anonymized sessions with public market calendars, estimate allocation returns from public equity returns using local ridge regression, and use CatBoost to correct the resulting directional probabilities.

**This is historical reconstruction, not a prospective trading model.** It uses later rows from the supplied batch and public prices on the historical target dates. Neither source would be available when making a live next-day decision. The distinction is central to interpreting the result.

## Results

| Evaluation | Accuracy | Scope |
| --- | ---: | --- |
| v4 reference | 74.56% | Same 47,192 reserved rows as v5 |
| v5 reconstruction | **76.21%** | 47,192 rows across 180 reserved dates |
| v5 public leaderboard | **77.41%** | Participant-reported score for the submitted CSV; not independently verified here |

The v5 minus v4 paired difference on the reserved dates is **+1.65 percentage points**. A date-level bootstrap interval for that difference is **+1.17 to +2.20 points**. These reserved dates were excluded from fitting the v5 correction models and from the current selection of settings. Earlier research inspected some of the same labels, and the inherited CatBoost prior was not trained in a fully nested evaluation. The interval therefore describes this comparison; it is not a claim of untouched out-of-sample trading performance. See [evaluation notes](docs/evaluation.md).

I also tried a later v6 correction model. Its chronological check improved over v5 by only **0.06 percentage points** on 56,180 rows, with an interval spanning zero. I kept v5 as the presented submission because the extra complexity was not supported by that comparison.

## How the method works

1. **Recover session order.** Compare overlapping 20-day histories across allocations, then constrain candidate dates using market calendars and public index/stock moves.
2. **Use direct overlap when available.** A later snapshot can contain the return of an earlier snapshot's target day. This path is tracked separately from statistical reconstruction.
3. **Estimate missing returns.** Fit local, weighted ridge regressions from public equity returns to observed allocation histories. The queried target day is removed from each local fit; adaptive equity weights share information across allocations.
4. **Correct the sign.** Build features from direct returns, reconstructed returns, historical volatility, turnover, overlap quality and a CatBoost prior. Average two CatBoost correction models trained on disjoint date pools.

The research implementation is under [`research/`](research). [`v5_pipeline.py`](research/v5_pipeline.py) builds the validation and inference features; [`v5_adaptive_ridge.py`](research/v5_adaptive_ridge.py) implements the adaptive regression; [`v5_finalize.py`](research/v5_finalize.py) refits the frozen models and writes the submission. The frozen settings are in [`v5_frozen_config.json`](research/v5_frozen_config.json).

## Reproduction and data

The source code, frozen configuration and aggregate evaluation are public. The challenge CSVs, cached public quotes, intermediate arrays and fitted models are deliberately excluded. The v5 scripts are the original artifact-dependent research pipeline, not a one-command clean-room reproduction. They run after the intermediate files listed in [reproduction notes](docs/reproduction.md) have been prepared. This keeps the repository honest about what a reviewer can run from a fresh clone.

Challenge participants can obtain the original CSVs from the [challenge page](https://challengedata.ens.fr/challenges/167), subject to its dataset terms. No challenge row, submission label, model checkpoint or scraped market-data cache is included here. `QRT_TEST_CSV` points to the participant's local test CSV for the final integrity check.

## Context and credit

The initial ENS Data Camp project was carried out with Omar Karim, Hitaishi Dhoowooah, Gabriel Dreik and Korouhanba Khuman Laikhuram. This later reconstruction study and its v5 research code are my follow-up work. QRT provided the challenge; this repository is an independent participant project and is not affiliated with or endorsed by QRT.
