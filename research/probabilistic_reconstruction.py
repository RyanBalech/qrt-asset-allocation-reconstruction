"""Learn overlap confidence from feature-derived order; never from test labels.

Unlike the v1 hard switch, retain several reference/gap possibilities and
combine their sign evidence. Training labels are reconstructed calendar gaps.
"""

from overlap_model import load, ROOT
from whiten_matches import whiten
from fused_reconstruction import tensor_scores
from catboost import CatBoostClassifier
from scipy.special import ndtr
import numpy as np
import pandas as pd
import json

SEEDS = [42, 2026, 123, 789, 314]


def make_pool(order, seed):
    rng = np.random.default_rng(seed)
    keep = []
    for _, chain in order[order.GROUP.eq(3)].groupby("component"):
        chain = chain.sort_values("position")
        pos = int(rng.integers(0, 22))
        while pos < len(chain):
            keep.append(int(chain.iloc[pos].idx))
            pos += int(rng.integers(18, 25))
    return np.sort(keep)


def candidates(scores, pool, R, V, meta, volume_clock=None):
    scores = np.nan_to_num(scores, nan=-2)
    sim = scores.max(3)
    gap = scores.argmax(3) + 1
    quality = np.maximum(sim - 0.45, 0)
    support = (
        quality.max(2)
        + 0.5 * np.sort(quality, axis=2)[:, :, 2]
        + 0.25 * np.sort(quality, axis=2)[:, :, 1]
    )
    clock = None
    if volume_clock is not None:
        clock = volume_clock[np.ix_(pool, pool)]
        clock_mean = clock.mean(2)
    rows = []
    records = []
    # Top consensus references plus each group's own strongest reference.
    for i, q in enumerate(pool):
        js = set(np.argsort(support[i])[-2:].tolist())
        js.update(np.argmax(sim[i], axis=0).tolist())
        js.discard(i)
        if clock is not None:
            js.update(np.argsort(clock_mean[i])[-4:].tolist())
            js.discard(i)
        for j in sorted(js):
            r = pool[j]
            pair = []
            pair.extend(sim[i, j])
            pair.extend(gap[i, j])
            pair.extend(sim[j, i])
            pair.extend(gap[j, i])
            pair.extend([support[i, j], support[j, i]])
            if clock is not None:
                pair.extend(clock[i, j])
                pair.extend(clock[j, i])
                pair.extend(
                    [
                        clock_mean[i, j],
                        clock_mean[j, i],
                        float((clock_mean[i] > clock_mean[i, j]).sum()),
                    ]
                )
                for gg in range(1, 5):
                    ids = np.flatnonzero(
                        (meta.group.values == gg)
                        & (meta.n.values > 500)
                        & (meta.sv1.values < 0.9)
                    )
                    pair.extend(
                        [
                            float(np.isfinite(V[q, ids, 0]).mean()),
                            float(np.isfinite(V[r, ids, 0]).mean()),
                            float(np.isfinite(R[q, ids, 0]).mean()),
                            float(np.isfinite(R[r, ids, 0]).mean()),
                        ]
                    )
            # Changes in the volume cross section and its time differences
            # identify nearby snapshots even when return windows do not overlap.
            for gg in range(1, 5):
                ids = np.flatnonzero(
                    (meta.group.values == gg)
                    & (meta.n.values > 500)
                    & (meta.sv1.values < 0.9)
                )
                a = V[q, ids, 1:]
                b = V[r, ids, 1:]
                for x, z in [
                    (np.nanmean(a, 1), np.nanmean(b, 1)),
                    (a[:, -1], b[:, 0]),
                    (a[:, 0], b[:, -1]),
                    (np.diff(a, axis=1).ravel(), np.diff(b, axis=1).ravel()),
                ]:
                    ok = np.isfinite(x) & np.isfinite(z)
                    x = x[ok]
                    z = z[ok]
                    cos = float(x @ z / (np.linalg.norm(x) * np.linalg.norm(z) + 1e-12))
                    pair.append(cos)
            for g in range(4):
                ids = np.flatnonzero(meta.group.values == g + 1)
                if not np.isfinite(R[q, ids, 0]).any():
                    continue
                hs = set([20, int(gap[i, j, g])])
                hs.update([int(v) for v in gap[i, j] if 14 <= v <= 19])
                # Close alternatives matter when only one overlap day is visible.
                hs.update([18, 19])
                for h in sorted(hs):
                    f = pair + [
                        g + 1,
                        h,
                        float(sim[i, :, g].max()),
                        float(support[i].max()),
                        float(sim[:, j, g].max()),
                    ]
                    f.extend(scores[i, j, :, h - 1] if h < 20 else [-2.0] * 4)
                    f.extend(scores[i, j, :, h - 2] if h > 1 else [-2.0] * 4)
                    f.extend(scores[i, j, :, h] if h < 19 else [-2.0] * 4)
                    f.extend((gap[i, j] - h).tolist())
                    rows.append(f)
                    records.append((q, r, g + 1, h))
    return np.array(rows, dtype=np.float32), pd.DataFrame(
        records, columns=["q", "ref", "GROUP", "gap"]
    )


def true_labels(c, order):
    n = 2642
    pos = np.full((n, 4), np.nan)
    comp = np.full((n, 4), -1)
    for a in order.itertuples():
        pos[a.idx, a.GROUP - 1] = a.position
        comp[a.idx, a.GROUP - 1] = a.component
    q = c.q.values
    r = c.ref.values
    g = c.GROUP.values - 1
    delta = pos[r, g] - pos[q, g]
    return (
        (comp[q, g] >= 0) & (comp[q, g] == comp[r, g]) & (delta == c.gap.values)
    ).astype(int)


def predict_candidates(c, prob, R, pool, meta, base, noise=0.3, power=1.0):
    result = base.copy()
    coverage = np.zeros_like(base)
    scale = np.nanstd(R[:2522], axis=(0, 2))
    idx = {q: i for i, q in enumerate(pool)}
    for (q, g), a in c.assign(prob=prob).groupby(["q", "GROUP"]):
        ids = np.flatnonzero(meta.group.values == g)
        p = a.prob.values**power
        total = p.sum()
        if total < 1e-6:
            continue
        # At most one date-gap hypothesis is correct in these sparse pools.
        if total > 1:
            p = p / total
        values = R[a.ref.values, :, a.gap.values - 1][:, ids]
        evidence = ndtr(values / (scale[ids][None, :] * noise)) - 0.5
        evidence = np.nan_to_num(evidence)
        result[idx[q], ids] = (p[:, None] * evidence).sum(0) + (
            1 - min(total, 1)
        ) * base[idx[q], ids]
        coverage[idx[q], ids] = min(total, 1)
    return result, coverage


def main():
    R, Y, dates, allocs, d, meta, order = load()
    V = np.load(ROOT / "panel.npz")["V"]
    panel = np.load(ROOT / "panel.npz")
    di = panel["di"]
    ai = panel["ai"]
    z = np.load(ROOT / "baseline_predictions.npz")
    base = np.zeros_like(Y)
    base[di, ai] = np.r_[z["oof"], z["test"]] - 0.5
    As = [
        whiten(
            R,
            np.flatnonzero(
                (meta.group.values == g)
                & (meta.n.values > 500)
                & (meta.sv1.values < 0.9)
            ),
            0.1,
        )
        for g in range(1, 5)
    ]
    # Reserve all five existing validation pools before fitting confidence.
    reserved = np.unique(np.concatenate([make_pool(order, s) for s in SEEDS]))
    features = []
    labels = []
    queries = []
    for seed in range(1000, 1020):
        cache = ROOT / f"v2_candidates_{seed}.npz"
        if cache.exists():
            zz = np.load(cache)
            x = zz["x"]
            y = zz["y"]
            q = zz["q"]
        else:
            pool = make_pool(order, seed)
            scores = np.stack([tensor_scores(A, pool) for A in As], axis=2)
            x, c = candidates(scores, pool, R, V, meta)
            y = true_labels(c, order)
            q = c.q.values
            np.savez_compressed(cache, x=x, y=y, q=q)
        keep = ~np.isin(q, reserved)
        features.append(x[keep])
        labels.append(y[keep])
        queries.append(q[keep])
        print(
            "training pool",
            seed,
            "candidates",
            keep.sum(),
            "positive",
            y[keep].sum(),
            flush=True,
        )
    x = np.concatenate(features)
    y = np.concatenate(labels)
    model = CatBoostClassifier(
        iterations=550,
        depth=5,
        learning_rate=0.04,
        l2_leaf_reg=15,
        loss_function="Logloss",
        thread_count=5,
        random_seed=20260920,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(x, y)
    model.save_model(str(ROOT / "v2_gap_confidence.cbm"))
    print("fit", x.shape, "positives", y.sum(), flush=True)
    metrics = []
    for name in [str(s) for s in SEEDS] + ["test"]:
        zz = np.load(ROOT / f"fusion_scores_{name}.npz")
        pool = zz["pool"]
        scores = zz["scores"]
        x, c = candidates(scores, pool, R, V, meta)
        prob = model.predict_proba(x)[:, 1]
        c["prob"] = prob
        c.to_parquet(ROOT / f"v2_candidates_{name}.parquet", index=False)
        for power in [1.0, 0.75, 1.25]:
            pred, coverage = predict_candidates(
                c, prob, R, pool, meta, base[pool], power=power
            )
            if name == "test":
                mask = d.split.eq("test").values
                p = pred[di[mask] - 2522, ai[mask]]
                pd.DataFrame(
                    {"ROW_ID": d.ROW_ID[mask].values, "target": (p > 0).astype(np.int8)}
                ).to_csv(ROOT / f"submission_v2_p{power}.csv", index=False)
            else:
                ok = np.isfinite(Y[pool])
                acc = np.mean((pred[ok] > 0) == (Y[pool][ok] > 0))
                print(
                    "RESULT",
                    name,
                    power,
                    acc,
                    "coverage",
                    coverage[ok].mean(),
                    flush=True,
                )
                metrics.append(
                    dict(seed=name, power=power, accuracy=acc, n=int(ok.sum()))
                )
    pd.DataFrame(metrics).to_csv(ROOT / "v2_validation.csv", index=False)


if __name__ == "__main__":
    main()
