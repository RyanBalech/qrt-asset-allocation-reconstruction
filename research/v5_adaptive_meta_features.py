from v5_broad_meta_features import *
from v5_adaptive_ridge import adaptive

if __name__ == "__main__":
    R, Y, dates, allocs, d, meta, order = load()
    panel = np.load(ROOT / "panel.npz")
    V = panel["V"]
    real = mapping()
    cals = sessions()
    xs = broad_matrices(cals, groups=[1, 2, 4])
    xs[3] = stock_matrices(cals, expanded=True)[3]
    zz = np.load(ROOT / "baseline_predictions.npz")
    base = np.zeros_like(Y)
    base[panel["di"], panel["ai"]] = np.r_[zz["oof"], zz["test"]] - 0.5
    turn = np.full_like(Y, np.nan)
    turn[panel["di"], panel["ai"]] = d.MEDIAN_DAILY_TURNOVER.to_numpy()
    manifest = json.loads((ROOT / "v5_pool_manifest.json").read_text())
    for name, entry in manifest.items():
        if entry["role"] == "validation":
            continue
        cache = ROOT / f"v5_adaptive_meta_pool_{name}.parquet"
        if cache.exists():
            continue
        pool = np.asarray(entry["pool"])
        old = pd.read_parquet(ROOT / f"v5_meta_pool_{name}.parquet")
        z = np.load(
            ROOT
            / (
                f"v4_final_{name}.npz"
                if name in ["3", "10", "17"]
                else f"v5_reconstruction_{name}.npz"
            )
        )
        direct = z["direct"]
        cover = z["coverage"]
        ext = adaptive(R, pool, meta, real, cals, xs, alpha=0.7, power=0.5)
        np.savez_compressed(
            ROOT / f"v5_adaptive_reconstruction_{name}.npz",
            pool=pool,
            external=ext,
            direct=direct,
            coverage=cover,
        )
        x = features(R, V, pool, meta, real, cals, ext, direct, cover, base, turn)
        x["q"] = np.repeat(pool, len(meta))
        x["target"] = Y[pool].reshape(-1)
        x["pool"] = int(name)
        x = x[np.isfinite(x.target)].reset_index(drop=True)
        assert np.array_equal(x.q, old.q) and np.array_equal(
            x.allocation, old.allocation
        )
        x = add_prior_features(x, old)
        x.to_parquet(cache, index=False)
        acc = np.mean(
            (np.where(x.covered > 0.5, x.direct, x.external + 0.3 * x.baseline) > 0)
            == (x.target > 0)
        )
        print(
            "ADAPTIVE FEATURES",
            name,
            entry["role"],
            len(x),
            "accuracy",
            acc,
            flush=True,
        )
