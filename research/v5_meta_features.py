"""Feature construction for supervised reconstruction-error correction."""

from v4_equity_imputation import *


def features(
    R, V, pool, meta, real, cals, external, direct, coverage, base, turnover=None
):
    n, a = external.shape
    scale = np.nanstd(R[pool], axis=2) + 1e-6
    f = {
        "external": external / scale,
        "direct": direct / scale,
        "covered": coverage.astype(float),
        "baseline": base[pool],
        "volatility": scale,
        "turnover": np.broadcast_to(meta.turn.values, (n, a))
        if turnover is None
        else turnover[pool],
        "group": np.broadcast_to(meta.group.values, (n, a)),
        "allocation": np.broadcast_to(np.arange(a), (n, a)),
        "special": np.broadcast_to((meta.sv1.values > 0.9).astype(float), (n, a)),
    }
    for w in [1, 3, 5, 10, 20]:
        f[f"ret_mean_{w}"] = np.nanmean(R[pool, :, :w], axis=2) / scale
        if w > 1:
            f[f"ret_std_{w}"] = np.nanstd(R[pool, :, :w], axis=2) / scale
        f[f"volume_mean_{w}"] = np.nanmean(V[pool, :, :w], axis=2)
    for key in [
        "gap",
        "overlap_corr",
        "overlap_scale",
        "overlap_bias",
        "donor_vol_ratio",
        "donor_ret_mean",
        "market_external_mean",
        "market_direct_mean",
        "direct_local_rank",
        "external_local_rank",
    ]:
        f[key] = np.full((n, a), np.nan)
    for g, cal in cals.items():
        ids = np.flatnonzero(meta.group.values == g)
        pos = np.array(
            [cal.get_indexer([real[q]])[0] if q in real else -1 for q in pool]
        )
        present = np.isfinite(R[pool][:, ids, 0]).any(1) & (pos >= 20)
        for i, q in enumerate(pool):
            for nm, values in [("direct", direct), ("external", external)]:
                x = values[i, ids] / scale[i, ids]
                f[f"market_{nm}_mean"][i, ids] = np.nanmean(x)
                f[f"{nm}_local_rank"][i, ids] = pd.Series(x).rank(pct=True).to_numpy()
            if not present[i]:
                continue
            future = np.flatnonzero(present & (pos > pos[i]))
            if not len(future):
                continue
            j = future[np.argmin(pos[future])]
            h = pos[j] - pos[i]
            f["gap"][i, ids] = h
            f["donor_vol_ratio"][i, ids] = scale[j, ids] / scale[i, ids]
            f["donor_ret_mean"][i, ids] = np.nanmean(R[pool[j], ids], 1) / scale[i, ids]
            if h < 20:
                x = R[q, ids, : 20 - h] / scale[i, ids, None]
                y = R[pool[j], ids, h:] / scale[i, ids, None]
                f["overlap_bias"][i, ids] = np.nanmean(x - y, axis=1)
                f["overlap_scale"][i, ids] = np.nanmean(x * y, axis=1) / (
                    np.nanmean(y * y, axis=1) + 0.05
                )
                f["overlap_corr"][i, ids] = np.nanmean(x * y, axis=1) / (
                    np.sqrt(np.nanmean(x * x, axis=1) * np.nanmean(y * y, axis=1))
                    + 0.05
                )
    return pd.DataFrame(
        {
            k: np.nan_to_num(v.reshape(-1), nan=-999, posinf=999, neginf=-999).astype(
                np.float32
            )
            for k, v in f.items()
        }
    )


if __name__ == "__main__":
    import json
    from v4_joint_dates import assign

    R, Y, dates, allocs, d, meta, order = load()
    panel = np.load(ROOT / "panel.npz")
    V = panel["V"]
    real = mapping()
    cals = sessions()
    xs = stock_matrices(cals, expanded=True)
    zz = np.load(ROOT / "baseline_predictions.npz")
    base = np.zeros_like(Y)
    base[panel["di"], panel["ai"]] = np.r_[zz["oof"], zz["test"]] - 0.5
    turn = np.full_like(Y, np.nan)
    turn[panel["di"], panel["ai"]] = d.MEDIAN_DAILY_TURNOVER.to_numpy()
    union = pd.DatetimeIndex(sorted(set().union(*[set(c) for c in cals.values()])))
    known = {dt: q for q, dt in real.items() if q < 2522}
    lo = union.get_indexer([pd.Timestamp("2011-01-04")])[0]
    manifest = {}
    for offset in range(21):
        grid = union[lo + offset :: 21]
        grid = grid[grid <= "2015-10-30"]
        pool = np.sort([known[dt] for dt in grid])
        manifest[str(offset)] = {
            "pool": pool.tolist(),
            "grid": grid.astype(str).tolist(),
            "role": "validation"
            if offset in [0, 7, 14]
            else "development"
            if offset in [3, 10, 17]
            else "train",
        }
    (ROOT / "v5_pool_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        "Reserved final validation offsets 0,7,14, excluded from model fitting.",
        flush=True,
    )
    for name, entry in manifest.items():
        # Validation will be generated and scored only after model selection.
        if entry["role"] == "validation":
            continue
        cache = ROOT / f"v5_meta_pool_{name}.parquet"
        if cache.exists():
            continue
        pool = np.asarray(entry["pool"])
        oldcache = ROOT / f"v4_final_{name}.npz"
        if oldcache.exists():
            z = np.load(oldcache)
            assert np.array_equal(pool, z["pool"])
            ext, direct, cover = z["external"], z["direct"], z["coverage"]
        else:
            ext, direct, cover = infer(
                R, pool, meta, real, cals, xs, alpha=0.7, radius=260, tau=100
            )
            np.savez_compressed(
                ROOT / f"v5_reconstruction_{name}.npz",
                pool=pool,
                external=ext,
                direct=direct,
                coverage=cover,
            )
        x = features(R, V, pool, meta, real, cals, ext, direct, cover, base, turn)
        x["q"] = np.repeat(pool, len(meta))
        x["target"] = Y[pool].reshape(-1)
        x["pool"] = int(name)
        x = x[np.isfinite(x.target)]
        x.to_parquet(cache, index=False)
        print("FEATURES", name, entry["role"], len(x), flush=True)
