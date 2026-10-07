"""Transductive overlap reconstruction. All date matching uses X only.

For a successor h trading sessions ahead, its RET_h approximates the
current row's next-day return. This is a batch-dataset reconstruction,
not a model that can operate prospectively on a single new timestamp.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

threadpool_limits(5)
ROOT = Path(__file__).resolve().parent


def match(R, ids, query, reference, scale, max_gap=19):
    """Return candidate successor per gap, using cross-sectional cosine."""
    records = []
    for h in range(1, max_gap + 1):
        q = R[query][:, ids, : 20 - h] / scale[None, :, None]
        r = R[reference][:, ids, h:] / scale[None, :, None]
        q = np.nan_to_num(q).reshape(len(query), -1)
        r = np.nan_to_num(r).reshape(len(reference), -1)
        q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
        r /= np.linalg.norm(r, axis=1, keepdims=True) + 1e-12
        sim = q @ r.T
        sim[np.asarray(query)[:, None] == np.asarray(reference)[None, :]] = -2
        best = np.argsort(sim, axis=1)[:, -2:][:, ::-1]
        for i, t in enumerate(query):
            for rank, j in enumerate(best[i]):
                records.append((t, reference[j], h, float(sim[i, j]), rank))
    return pd.DataFrame(records, columns=["q", "ref", "gap", "sim", "rank"])


def load():
    z = np.load(ROOT / "panel.npz")
    R, Y, dates, allocs = [z[k] for k in ["R", "Y", "dates", "allocs"]]
    d = pd.read_parquet(ROOT / "data.parquet")
    meta = (
        d[d.split.eq("train")]
        .groupby("ALLOCATION")
        .agg(
            group=("GROUP", "first"),
            n=("target", "count"),
            turn=("MEDIAN_DAILY_TURNOVER", "median"),
            sv1=("SIGNED_VOLUME_1", lambda x: x.notna().mean()),
        )
        .reindex(allocs)
    )
    order = pd.read_csv(ROOT / "training_order.csv")
    order["idx"] = pd.Index(dates).get_indexer(order.TS)
    return R, Y, dates, allocs, d, meta, order


if __name__ == "__main__":
    R, Y, dates, allocs, d, meta, order = load()
    out = []
    for g in range(1, 5):
        allids = np.flatnonzero(meta.group.values == g)
        ids = np.flatnonzero(
            (meta.group.values == g) & (meta.n.values > 500) & (meta.sv1.values < 0.9)
        )
        scale = np.nanstd(R[:2522, ids, :], axis=(0, 2))
        tt = np.flatnonzero((dates > "DATE_2522") & np.isfinite(R[:, ids[0], 0]))
        c = match(R, ids, tt, tt, scale)
        c["GROUP"] = g
        out.append(c)
        best = c.loc[c.groupby("q").sim.idxmax()].copy()
        print(
            "TEST",
            g,
            "good counts",
            [(s, int((best.sim > s).sum())) for s in [0.7, 0.8, 0.85, 0.9, 0.95]],
            "n",
            len(tt),
            flush=True,
        )
        print(
            best[best.sim > 0.85].gap.value_counts().sort_index().to_string(),
            flush=True,
        )
        # Oracle chain evaluation assesses how returns change at distant snapshots.
        o = order[order.GROUP.eq(g)]
        for h in [1, 5, 10, 15, 18, 19, 20]:
            a = o.copy()
            b = o.copy()
            b.position -= h
            joined = a.merge(
                b, on=["GROUP", "component", "position"], suffixes=("_q", "_r")
            )
            ys = Y[joined.idx_q.values][:, allids]
            ps = R[joined.idx_r.values][:, allids, h - 1]
            ok = np.isfinite(ys) & np.isfinite(ps)
            print(
                "ORACLE",
                g,
                h,
                "n",
                ok.sum(),
                "acc",
                np.mean((ys[ok] > 0) == (ps[ok] > 0)),
                flush=True,
            )
        # Unseen, widely spaced validation feature pool; labels are not used in matching.
        for seed in [42, 2026, 123]:
            rng = np.random.default_rng(seed)
            keep = []
            for _, chain in o.groupby("component"):
                chain = chain.sort_values("position")
                pos = int(rng.integers(0, 20))
                while pos < len(chain):
                    keep.append(int(chain.iloc[pos].idx))
                    pos += int(rng.integers(16, 24))
            keep = np.array(keep)
            cc = match(R, ids, keep, keep, scale)
            cc["GROUP"] = g
            cc["seed"] = seed
            cc.to_parquet(
                ROOT / f"validation_candidates_g{g}_{seed}.parquet", index=False
            )
            bb = cc.loc[cc.groupby("q").sim.idxmax()]
            ys = Y[bb.q.values][:, allids]
            ps = R[bb.ref.values[:, :, None] if False else bb.ref.values][:, allids]
            ps = ps[np.arange(len(bb)), :, bb.gap.values - 1]
            for threshold in [0.8, 0.85, 0.9, 0.95]:
                ok = (
                    np.isfinite(ys)
                    & np.isfinite(ps)
                    & (bb.sim.values[:, None] > threshold)
                )
                allok = np.isfinite(ys)
                print(
                    "VALID",
                    g,
                    seed,
                    threshold,
                    "coverage",
                    ok.sum() / allok.sum(),
                    "acc",
                    np.mean((ys[ok] > 0) == (ps[ok] > 0)),
                    flush=True,
                )
    out = pd.concat(out, ignore_index=True)
    out["TS"] = dates[out.q]
    out["SUCCESSOR"] = dates[out.ref]
    out.to_parquet(ROOT / "test_candidates_full.parquet", index=False)
