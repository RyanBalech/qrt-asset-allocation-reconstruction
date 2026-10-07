"""Fuse date evidence across markets and validate on sparse snapshot pools."""

from overlap_model import load, ROOT
from whiten_matches import whiten
import numpy as np
import pandas as pd
from scipy.special import expit


def tensor_scores(A, pool):
    n = len(pool)
    scores = np.full((n, n, 19), np.nan)
    for h in range(1, 20):
        a = A[pool, : 20 - h].reshape(n, -1).copy()
        b = A[pool, h:].reshape(n, -1).copy()
        a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
        b /= np.linalg.norm(b, axis=1, keepdims=True) + 1e-12
        scores[:, :, h - 1] = a @ b.T
    for i in range(n):
        scores[i, i, :] = -2
    return scores


def build_gap_examples(order):
    parts = []
    # Pair every anchor with all plausible future monthly anchors, across each market.
    for g in range(1, 5):
        a = order[order.GROUP.eq(g)]
        for h in range(14, 26):
            b = a.copy()
            b.position -= h
            links = a.merge(
                b, on=["GROUP", "component", "position"], suffixes=("_q", "_r")
            )
            parts.append(links[["idx_q", "idx_r"]])
    pairs = pd.concat(parts).drop_duplicates()
    n = 2642
    pos = np.full((n, 4), np.nan)
    comp = np.full((n, 4), -1.0)
    for row in order.itertuples():
        pos[row.idx, row.GROUP - 1] = row.position
        comp[row.idx, row.GROUP - 1] = row.component
    qi = pairs.idx_q.values
    ri = pairs.idx_r.values
    gaps = pos[ri] - pos[qi]
    gaps[(comp[ri] != comp[qi]) | (comp[ri] < 0)] = np.nan
    return gaps, pairs


def fused_predict(scores, R, pool, meta, gap_examples, base, mode="fused"):
    """Use strong group agreement, then infer a possible gap of exactly 20.

    Smaller gaps are visible directly; gaps beyond 20 cannot expose the target.
    The empirical prior is learned solely from feature-derived training order.
    """
    n = len(pool)
    sim = np.nan_to_num(scores, nan=-2).max(axis=3)  # n,n,4
    gap = np.nan_to_num(scores, nan=-2).argmax(axis=3) + 1
    # Independent markets corroborate the same date relationship.
    quality = np.maximum(sim - 0.45, 0)
    support = (
        quality.max(axis=2)
        + 0.50 * np.sort(quality, axis=2)[:, :, 2]
        + 0.25 * np.sort(quality, axis=2)[:, :, 1]
    )
    ref = support.argmax(axis=1)
    pred = base.copy()
    audit = []
    for i, q in enumerate(pool):
        j = int(ref[i])
        r = pool[j]
        known = sim[i, j] > 0.65
        evidence = float(support[i, j])
        for g in range(4):
            ids = np.flatnonzero(meta.group.values == g + 1)
            if not np.isfinite(R[q, ids, 0]).any():
                continue
            local_s = sim[i, j, g]
            h = int(gap[i, j, g])
            p20 = 0.0
            # A reliable individual match remains usable without cross-group consensus.
            bestj, besth = np.unravel_index(
                np.argmax(scores[i, :, g, :]), scores[i, :, g, :].shape
            )
            bests = scores[i, bestj, g, besth]
            rlocal = pool[bestj]
            if mode == "local":
                if bests > 0.55:
                    value = R[rlocal, ids, besth]
                    pred[i, ids] = np.where(np.isfinite(value), value, pred[i, ids])
                    audit.append(
                        (q, g + 1, rlocal, besth + 1, float(bests), "direct", 1.0)
                    )
                continue
            if local_s > 0.52 and evidence > 0.18:
                value = R[r, ids, h - 1]
                pred[i, ids] = np.where(np.isfinite(value), value, pred[i, ids])
                audit.append((q, g + 1, r, h, float(local_s), "direct", 1.0))
            elif bests > 0.65:
                value = R[rlocal, ids, besth]
                pred[i, ids] = np.where(np.isfinite(value), value, pred[i, ids])
                audit.append((q, g + 1, rlocal, besth + 1, float(bests), "direct", 1.0))
            elif mode == "fused" and known.any() and evidence > 0.30:
                use = known.copy()
                use[g] = False
                if not use.any():
                    continue
                # Kernel prior for this group's gap, given the observed gaps in others.
                diff = gap_examples[:, use] - gap[i, j, use][None, :]
                ok = np.isfinite(diff).all(axis=1) & np.isfinite(gap_examples[:, g])
                w = np.exp(-0.5 * np.sum(diff[ok] ** 2 / 0.65**2, axis=1))
                hg = gap_examples[ok, g]
                # Failure to detect an overlap makes gaps below 20 implausible.
                w[hg < 20] *= 0.02
                if w.sum() < 2:
                    continue
                p20 = float(w[hg == 20].sum() / w.sum())
                if p20 > 0.24:
                    value = R[r, ids, 19]
                    pred[i, ids] = np.where(np.isfinite(value), value, pred[i, ids])
                    audit.append((q, g + 1, r, 20, float(local_s), "inferred20", p20))
    return pred, pd.DataFrame(
        audit, columns=["q", "GROUP", "ref", "gap", "sim", "method", "p20"]
    )


if __name__ == "__main__":
    R, Y, dates, allocs, d, meta, order = load()
    As = []
    for g in range(1, 5):
        ids = np.flatnonzero(
            (meta.group.values == g) & (meta.n.values > 500) & (meta.sv1.values < 0.9)
        )
        As.append(whiten(R, ids, 0.1))
    gap_examples, _ = build_gap_examples(order)
    np.save(ROOT / "gap_examples.npy", gap_examples)
    z = np.load(ROOT / "baseline_predictions.npz")
    base = np.zeros_like(Y)
    di = np.load(ROOT / "panel.npz")["di"]
    ai = np.load(ROOT / "panel.npz")["ai"]
    # Return-scale fallback has same sign as the probability model.
    base[di, ai] = np.r_[z["oof"], z["test"]] - 0.5
    results = []
    pools = {}
    for seed in [42, 2026, 123, 789, 314]:
        rng = np.random.default_rng(seed)
        keep = []
        for _, chain in order[order.GROUP.eq(3)].groupby("component"):
            chain = chain.sort_values("position")
            pos = int(rng.integers(0, 22))
            while pos < len(chain):
                keep.append(int(chain.iloc[pos].idx))
                pos += int(rng.integers(18, 25))
        pools[str(seed)] = np.sort(keep)
    pools["test"] = np.arange(2522, len(R))
    for name, pool in pools.items():
        scores = np.stack([tensor_scores(A, pool) for A in As], axis=2)
        np.savez_compressed(
            ROOT / f"fusion_scores_{name}.npz", scores=scores, pool=pool
        )
        for mode in ["local", "consensus", "fused"]:
            pred, audit = fused_predict(
                scores, R, pool, meta, gap_examples, base[pool], mode=mode
            )
            if name == "test":
                mask = d.split.eq("test").values
                p = pred[di[mask] - 2522, ai[mask]]
                pd.DataFrame(
                    {"ROW_ID": d.ROW_ID[mask].values, "target": (p > 0).astype(np.int8)}
                ).to_csv(ROOT / f"submission_{mode}.csv", index=False)
                audit["TS"] = dates[audit.q]
                audit["SUCCESSOR"] = dates[audit.ref]
                audit.to_csv(ROOT / f"audit_{mode}_test.csv", index=False)
            else:
                valid = np.isfinite(Y[pool])
                acc = float(np.mean((pred[valid] > 0) == (Y[pool][valid] > 0)))
                baseline = float(
                    np.mean((base[pool][valid] > 0) == (Y[pool][valid] > 0))
                )
                print(
                    name,
                    mode,
                    "accuracy",
                    acc,
                    "baseline",
                    baseline,
                    "n",
                    valid.sum(),
                    "matched",
                    len(audit),
                    "methods",
                    audit.method.value_counts().to_dict(),
                    flush=True,
                )
                results.append(
                    {
                        "seed": name,
                        "mode": mode,
                        "accuracy": acc,
                        "baseline": baseline,
                        "n": int(valid.sum()),
                    }
                )
                audit.to_csv(ROOT / f"audit_{mode}_{name}.csv", index=False)
    pd.DataFrame(results).to_csv(ROOT / "fusion_validation.csv", index=False)
