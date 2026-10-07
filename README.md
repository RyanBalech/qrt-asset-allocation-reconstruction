# QRT Asset Allocation — Historical Reconstruction

Independent follow-up research on the [QRT asset-allocation challenge](https://challengedata.ens.fr/challenges/167): recover next-day return direction from overlapping allocation histories, public market calendars and local equity-factor models.

The key observation is structural: adjacent snapshots share most of their 20-day return histories. A later snapshot can expose an earlier snapshot's target return. I combine this direct evidence with weighted ridge reconstruction and CatBoost probability correction.

**Scope:** historical batch reconstruction. The method uses later snapshots and public prices on historical target dates; these inputs are unavailable for a live next-day forecast.

## Results

| Reserved-date evaluation | Accuracy |
| --- | ---: |
| v4 reference | 74.56% |
| v5 adaptive reconstruction | **76.21%** |

Both methods are evaluated on **47,192 rows across 180 dates**. The paired gain is **1.65 percentage points**, with a date-bootstrap 95% interval of **[1.17, 2.20] points**. The v5 submission recorded a **77.41% public leaderboard score**.

[Evaluation notes](docs/evaluation.md) document the submission hash, leaderboard provenance, split construction and earlier development exposure. The reserved comparison is not a fully nested evaluation. A later v6 chronological comparison did not establish a reliable improvement, so v5 remains the presented method.

## Method

1. **Session alignment:** cross-allocation cosine matching of overlapping histories, constrained by exchange calendars and public market moves.
2. **Direct reconstruction:** identify later snapshots containing a query's target return and track coverage separately.
3. **Equity-factor reconstruction:** local weighted ridge fits, excluding the query's target day from every fit; adaptive factor weights share information across allocations.
4. **Directional correction:** combine reconstruction, volatility, turnover, overlap quality and a CatBoost prior; average corrections fitted on disjoint date pools.

The [frozen configuration](research/v5_frozen_config.json), [pipeline](research/v5_pipeline.py), [adaptive regression](research/v5_adaptive_ridge.py) and [submission finalization](research/v5_finalize.py) expose the experimental choices rather than hiding them behind a notebook.

## Run the checks

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The synthetic tests check target-day exclusion in both ridge implementations, direct-overlap behavior, calendar boundaries and feature-cache invalidation. They require no challenge data.

Full experiments require challenge CSVs, public-price caches, intermediate arrays and fitted priors. These are excluded from Git. Follow the [reproduction guide](docs/reproduction.md) for the required inputs and execution order; the original research pipeline is artifact-dependent.

## Research context

The initial ENS Data Camp project was completed with Omar Karim, Hitaishi Dhoowooah, Gabriel Dreik and Korouhanba Khuman Laikhuram. This reconstruction study and v5 code are my independent follow-up. QRT provided the challenge; this participant project is not affiliated with QRT.
