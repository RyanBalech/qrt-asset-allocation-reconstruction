"""Use leave-one-day-out equity reconstruction to disambiguate real dates."""

from v4_equity_imputation import *


def evidence(R, V, meta, cals, xs, queries, days, alphas=(0.1, 0.7, 3.0)):
    scores = np.zeros((len(queries), len(days), len(alphas)))
    for g, cal in cals.items():
        ids = np.flatnonzero(
            (meta.group.values == g) & (meta.n.values > 500) & (meta.sv1.values < 0.9)
        )
        pos = cal.get_indexer(days)
        present = pos >= 20
        banks = np.zeros((len(days), len(alphas), 20, 20))
        for k, t in enumerate(pos):
            if t < 20:
                continue
            a = xs[g][t - np.arange(20)].copy()
            a -= a.mean(0)
            a /= np.sqrt(np.mean(a * a)) + 1e-12
            gram = a @ a.T / a.shape[1]
            for j, alpha in enumerate(alphas):
                inv = np.linalg.inv(gram + np.eye(20) * alpha)
                cv = inv / np.diag(inv)[:, None]
                banks[k, j] = cv.T @ cv
        for i, q in enumerate(queries):
            exists = np.isfinite(R[q, ids, 0]).any()
            valid = present == exists
            if exists:
                monday = np.isfinite(V[q, ids, 0]).mean() > 0.5
                valid &= (days.weekday == 0) == monday
                y = np.nan_to_num(R[q, ids].T.astype(float))
                y -= y.mean(0)
                y /= y.std(0) + 1e-9
                gram = y @ y.T / y.shape[1]
                scores[i] += 1 - np.einsum("djkl,kl->dj", banks, gram) / 20
            scores[i, ~valid] = -100
    return scores


if __name__ == "__main__":
    R, Y, dates, allocs, d, meta, order = load()
    V = np.load(ROOT / "panel.npz")["V"]
    cals = sessions()
    xs = stock_matrices(cals)
    real = mapping()
    known = [
        q for q, dt in real.items() if q < 2522 and dt >= pd.Timestamp("2011-01-01")
    ]
    rng = np.random.default_rng(29092026)
    queries = np.sort(rng.choice(known, 120, replace=False))
    days = pd.DatetimeIndex([real[q] for q in queries])
    s = evidence(R, V, meta, cals, xs, queries, days)
    for j, alpha in enumerate([0.1, 0.7, 3.0]):
        print(
            "STOCK DATE ACC",
            alpha,
            np.mean(s[:, :, j].argmax(1) == np.arange(120)),
            flush=True,
        )
    test = pd.read_csv(ROOT / "v4_test_calendar.csv").sort_values("q")
    grid = pd.DatetimeIndex(test.sort_values("grid_position").real_date)
    ss = evidence(R, V, meta, cals, xs, test.q.to_numpy(), grid)
    np.savez_compressed(
        ROOT / "v4_stock_calendar_scores.npz",
        scores=ss,
        queries=test.q.to_numpy(),
        grid=grid.to_numpy(),
    )
    for j, alpha in enumerate([0.1, 0.7, 3.0]):
        chosen = ss[:, :, j].argmax(1)
        previous = test.grid_position.to_numpy()
        print(
            "TEST agree with old calendar",
            alpha,
            np.mean(chosen == previous),
            "previous avg",
            ss[np.arange(120), previous, j].mean(),
            "best avg",
            ss[:, :, j].max(1).mean(),
            flush=True,
        )
