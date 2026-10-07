from v4_equity_imputation import *
import argparse


def broad_matrices(cals, groups=(1, 2, 3, 4)):
    xs = {}
    for g in groups:
        p = pd.read_parquet(ROOT / f"v5_stock_prices_g{g}.parquet").reindex(cals[g])
        x = p.pct_change(fill_method=None).clip(-0.25, 0.25)
        scale = x.loc[:"2015-12-31"].std().fillna(x.std()).clip(lower=0.005)
        xs[g] = np.nan_to_num(x.to_numpy() / scale.to_numpy())
    return xs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", type=int, default=[1, 2, 3, 4])
    args = ap.parse_args()
    R, Y, dates, allocs, d, meta, order = load()
    real = mapping()
    cals = sessions()
    xs = broad_matrices(cals, args.groups)
    reports = []
    for g in args.groups:
        ids = np.flatnonzero(meta.group.values == g)
        for seed in [42, 2026, 123]:
            z = np.load(ROOT / f"v4_expanded_{seed}_260_0.7.npz")
            pool = z["pool"]
            yy = Y[pool]
            cover = z["coverage"]
            for alpha in [0.15, 0.7]:
                ext, _, _ = infer(
                    R,
                    pool,
                    meta,
                    real,
                    {g: cals[g]},
                    {g: xs[g]},
                    alpha=alpha,
                    radius=260,
                    tau=100,
                )
                ok = np.isfinite(yy[:, ids]) & np.isfinite(ext[:, ids]) & ~cover[:, ids]
                rec = {
                    "group": g,
                    "seed": seed,
                    "alpha": alpha,
                    "n": int(ok.sum()),
                    "old": float(
                        np.mean((z["external"][:, ids][ok] > 0) == (yy[:, ids][ok] > 0))
                    ),
                    "new": float(
                        np.mean((ext[:, ids][ok] > 0) == (yy[:, ids][ok] > 0))
                    ),
                }
                reports.append(rec)
                print(rec, flush=True)
                np.savez_compressed(
                    ROOT / f"v5_broad_g{g}_{seed}_{alpha}.npz", pool=pool, external=ext
                )
        pd.DataFrame([x for x in reports if x["group"] == g]).to_csv(
            ROOT / f"v5_broad_g{g}_development.csv", index=False
        )
