"""Iteratively reweight public-stock factors using multi-allocation exposures."""

from v4_equity_imputation import *
from v5_broad_probe import broad_matrices
from scipy.linalg import cho_factor, cho_solve


def adaptive(
    R, pool, meta, real, cals, xs, alpha=0.7, radius=260, tau=100, steps=2, power=0.5
):
    out = np.full((len(pool), R.shape[1]), np.nan)
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
            ts = []
            ys = []
            weights = []
            for j in np.flatnonzero(present & (abs(pos - t) <= radius)):
                obs = pos[j] - np.arange(20)
                ok = (obs != t + 1) & (obs >= 0)
                ts.extend(obs[ok])
                ys.extend(R[pool[j], ids][:, ok].T)
                weights.extend(
                    np.exp(-abs(pos[j] - t) / tau)
                    * np.exp(-abs(obs[ok] - t) / (tau * 2))
                )
            ts = np.asarray(ts)
            b = np.nan_to_num(np.asarray(ys))
            w = np.asarray(weights)
            a = xs[g][ts]
            future = xs[g][t + 1]
            available = (abs(a).sum(0) > 1e-8) & (abs(future) > 1e-12)
            if available.sum() < 5:
                continue
            a = a[:, available]
            future = future[available]
            root = np.sqrt(w)
            a = a * root[:, None]
            b = b * root[:, None]
            p = a.shape[1]
            factor = np.ones(p)
            y_scale = np.std(b, axis=0) + 1e-5
            for step in range(steps + 1):
                aa = a * np.sqrt(factor)[None, :]
                ff = future * np.sqrt(factor)
                gram = aa @ aa.T / p
                gram.flat[:: len(gram) + 1] += alpha
                coeff = (
                    aa.T
                    @ cho_solve(
                        cho_factor(gram, check_finite=False), b, check_finite=False
                    )
                    / p
                )
                prediction = ff @ coeff
                # Common feature relevance across portfolios; shrink to a
                # uniform prior to avoid discarding weak but useful stocks.
                energy = np.mean((coeff / y_scale[None, :]) ** 2, axis=1)
                energy = (energy / (energy.mean() + 1e-12) + 0.1) ** power
                factor = np.clip(energy / energy.mean(), 0.1, 10.0)
            out[i, ids] = prediction
    return out


if __name__ == "__main__":
    R, Y, dates, allocs, d, meta, order = load()
    real = mapping()
    cals = sessions()
    xs = broad_matrices(cals, groups=[1, 2, 4])
    xs[3] = stock_matrices(cals, expanded=True)[3]
    rows = []
    for seed in [42, 2026, 123]:
        z = np.load(ROOT / f"v4_expanded_{seed}_260_0.7.npz")
        pool = z["pool"]
        yy = Y[pool]
        cover = z["coverage"]
        for alpha, power in [(0.7, 0.5), (0.7, 1.0), (0.15, 0.5)]:
            out = adaptive(R, pool, meta, real, cals, xs, alpha=alpha, power=power)
            ok = np.isfinite(yy) & np.isfinite(out)
            miss = ok & ~cover
            pred = np.where(cover, z["direct"], out)
            row = {
                "seed": seed,
                "alpha": alpha,
                "power": power,
                "accuracy": float(np.mean((pred[ok] > 0) == (yy[ok] > 0))),
                "uncovered": float(np.mean((out[miss] > 0) == (yy[miss] > 0))),
            }
            rows.append(row)
            print(row, flush=True)
            np.savez_compressed(
                ROOT / f"v5_adaptive_{seed}_{alpha}_{power}.npz",
                pool=pool,
                external=out,
            )
        pd.DataFrame(rows).to_csv(ROOT / "v5_adaptive_development.csv", index=False)
