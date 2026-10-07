"""Recover absent portfolio-return days using public equity returns.

Local regressions see only histories in the current sparse feature pool.
The queried target day is removed from every fit, including donor histories.
Direct return reconstruction is evaluated separately.
"""

from overlap_model import load, ROOT
from v4_exchange_sessions import sessions
from probabilistic_reconstruction import predict_candidates
import numpy as np
import pandas as pd


def mapping():
    a = pd.read_csv(ROOT / "v4_training_calendar.csv").drop_duplicates("idx")
    test_path = ROOT / "v4_test_calendar_stock.csv"
    if not test_path.exists():
        test_path = ROOT / "v4_test_calendar.csv"
    b = pd.read_csv(test_path).rename(columns={"q": "idx"})
    return {
        int(r.idx): pd.Timestamp(r.real_date) for r in pd.concat([a, b]).itertuples()
    }


def stock_matrices(cals, expanded=False):
    matrices = {}
    for g, cal in cals.items():
        suffix = "expanded_" if expanded else ""
        p = pd.read_parquet(ROOT / f"v4_stock_prices_{suffix}g{g}.parquet").reindex(cal)
        x = p.pct_change(fill_method=None).clip(-0.25, 0.25)
        scale = x.loc[:"2015-12-31"].std().fillna(x.std()).clip(lower=0.005)
        matrices[g] = np.nan_to_num(x.to_numpy() / scale.to_numpy())
    return matrices


def infer(R, pool, meta, real, cals, xs, alpha=0.1, radius=0, tau=40.0):
    ext = np.full((len(pool), R.shape[1]), np.nan)
    direct = ext.copy()
    coverage = np.zeros_like(ext, bool)
    for g, cal in cals.items():
        ids = np.flatnonzero(meta.group.values == g)
        pos = np.array(
            [cal.get_indexer([real[q]])[0] if q in real else -1 for q in pool]
        )
        present = np.isfinite(R[pool][:, ids, 0]).any(1) & (pos >= 20)
        for i, q in enumerate(pool):
            t = pos[i]
            if not present[i] or t + 1 >= len(cal):
                continue
            future = np.flatnonzero(present & (pos > t) & (pos <= t + 20))
            if len(future):
                j = future[np.argmin(pos[future])]
                h = pos[j] - t
                direct[i, ids] = R[pool[j], ids, h - 1]
                coverage[i, ids] = np.isfinite(direct[i, ids])
            neighbors = (
                np.flatnonzero(present & (abs(pos - t) <= radius))
                if radius
                else np.array([i])
            )
            ts = []
            ys = []
            weights = []
            for j in neighbors:
                obs = pos[j] - np.arange(20)
                ok = (obs != t + 1) & (obs >= 0)
                ts.extend(obs[ok])
                ys.extend(R[pool[j], ids][:, ok].T)
                # Prefer weights implied by snapshots near the query and data
                # near the queried day. No label is used to select observations.
                w = np.exp(-abs(pos[j] - t) / tau) * np.exp(
                    -abs(obs[ok] - t) / (tau * 2)
                )
                weights.extend(w)
            ts = np.array(ts)
            b = np.nan_to_num(np.array(ys))
            w = np.array(weights)
            a = xs[g][ts]
            future_x = xs[g][t + 1]
            available = (abs(a).sum(0) > 1e-8) & (abs(future_x) > 1e-12)
            if available.sum() < 5:
                continue
            a = a[:, available]
            future_x = future_x[available]
            root = np.sqrt(w)
            a = a * root[:, None]
            b = b * root[:, None]
            if len(a) > a.shape[1]:
                p = future_x @ np.linalg.solve(
                    a.T @ a + np.eye(a.shape[1]) * alpha * a.shape[1], a.T @ b
                )
            else:
                gram = a @ a.T / a.shape[1]
                p = (future_x @ a.T / a.shape[1]) @ np.linalg.solve(
                    gram + np.eye(len(a)) * alpha, b
                )
            ext[i, ids] = p
    return ext, direct, coverage


if __name__ == "__main__":
    R, Y, dates, allocs, d, meta, order = load()
    cals = sessions()
    real = mapping()
    xs = stock_matrices(cals)
    z = np.load(ROOT / "baseline_predictions.npz")
    panel = np.load(ROOT / "panel.npz")
    base = np.zeros_like(Y)
    base[panel["di"], panel["ai"]] = np.r_[z["oof"], z["test"]] - 0.5
    reports = []
    for seed in [42, 2026, 123]:
        pool = np.load(ROOT / f"fusion_scores_{seed}.npz")["pool"]
        yy = Y[pool]
        c = pd.read_parquet(ROOT / f"v3_candidates_{seed}.parquet")
        old, _ = predict_candidates(c, c.prob.values, R, pool, meta, base[pool])
        scale = np.nanstd(R[pool], axis=2) + 1e-6
        for radius in [0, 30, 65]:
            for alpha in [0.03, 0.15, 0.7]:
                ext, direct, cover = infer(
                    R, pool, meta, real, cals, xs, alpha=alpha, radius=radius
                )
                ok = np.isfinite(yy) & np.isfinite(ext)
                miss = ok & ~cover
                # Weak baseline probability supplies a small directional prior.
                signal = ext / scale + 0.3 * base[pool]
                pred = np.where(cover, direct / scale, signal)
                rec = {
                    "seed": seed,
                    "radius": radius,
                    "alpha": alpha,
                    "n": int(ok.sum()),
                    "old": float(np.mean((old[ok] > 0) == (yy[ok] > 0))),
                    "new": float(np.mean((pred[ok] > 0) == (yy[ok] > 0))),
                    "imputation_only": float(np.mean((signal[ok] > 0) == (yy[ok] > 0))),
                    "uncovered_accuracy": float(
                        np.mean((signal[miss] > 0) == (yy[miss] > 0))
                    ),
                    "direct_coverage": float(cover[ok].mean()),
                }
                reports.append(rec)
                print(rec, flush=True)
    pd.DataFrame(reports).to_csv(
        ROOT / "v4_equity_imputation_development.csv", index=False
    )
