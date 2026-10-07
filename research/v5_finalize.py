"""Refit locked correction models on all labeled pools, then create submission."""

from v5_pipeline import *
from v5_train_meta import prepare
import hashlib
import os

if __name__ == "__main__":
    cfg = json.loads((ROOT / "v5_frozen_config.json").read_text())
    manifest = json.loads((ROOT / "v5_pool_manifest.json").read_text())
    prefix = "adaptive" if cfg.get("adaptive_weights") else "broad"
    frames = []
    for name, entry in manifest.items():
        path = ROOT / (
            f"v5_holdout_features_{name}.parquet"
            if entry["role"] == "validation"
            else f"v5_{prefix}_meta_pool_{name}.parquet"
        )
        x = pd.read_parquet(path)
        if prefix == "broad":
            x = selected_features(x)
        frames.append(x)
    train = pd.concat(frames, ignore_index=True)
    assert not train[["q", "allocation"]].duplicated().any()
    y = (train.target > 0).astype(int)
    for name in cfg["models"]:
        previous = CatBoostClassifier()
        previous.load_model(str(ROOT / f"v5_meta_{name}.cbm"))
        params = previous.get_params()
        params.update(thread_count=5, verbose=False, allow_writing_files=False)
        model = CatBoostClassifier(**params)
        model.fit(train[previous.feature_names_], y)
        model.save_model(str(ROOT / f"v5_final_{name}.cbm"))
        print("REFIT", name, "rows", len(train), "dates", train.q.nunique(), flush=True)
    ctx = context()
    cal = pd.read_csv(ROOT / "v4_calendar_final_test.csv").sort_values("q")
    pool = cal.q.to_numpy()
    grid = cal.sort_values("grid_position").real_date.tolist()
    x, real, prior = cached_build(
        ctx, pool, grid, "test", adaptive_weights=cfg.get("adaptive_weights", False)
    )
    x.to_parquet(ROOT / "v5_test_features.parquet", index=False)
    p = predict(x, cfg, final=True).reshape(len(pool), len(ctx["meta"]))
    mask = ctx["data"].split.eq("test").to_numpy()
    idx = {q: i for i, q in enumerate(pool)}
    rr = np.array([idx[q] for q in ctx["panel"]["di"][mask]])
    aa = ctx["panel"]["ai"][mask]
    prob = p[rr, aa]
    assert np.isfinite(prob).all()
    result = pd.DataFrame(
        {
            "ROW_ID": ctx["data"].ROW_ID[mask].to_numpy(),
            "target": (prob > 0.5).astype(np.int8),
        }
    )
    original = pd.read_csv(os.environ["QRT_TEST_CSV"])
    assert len(result) == len(original) == 31870 and result.ROW_ID.is_unique
    assert (
        np.array_equal(result.ROW_ID, original.ROW_ID)
        and result.target.isin([0, 1]).all()
    )
    pd.testing.assert_frame_equal(
        ctx["data"].loc[mask, original.columns].reset_index(drop=True),
        original,
        check_dtype=False,
        check_exact=False,
        rtol=1e-5,
        atol=1e-7,
    )
    path = ROOT / "submission_improved_v5.csv"
    result.to_csv(path, index=False)
    pd.DataFrame({"ROW_ID": result.ROW_ID, "probability": prob}).to_csv(
        ROOT / "v5_test_probabilities.csv", index=False
    )
    old = pd.read_csv(ROOT / "submission_improved_v4.csv")
    assert np.array_equal(old.ROW_ID, result.ROW_ID)
    audit = {
        "rows": len(result),
        "positive_fraction": float(result.target.mean()),
        "changed_from_v4": float((result.target != old.target).mean()),
        "training_rows": len(train),
        "training_dates": int(train.q.nunique()),
        "public_score": None,
        "submission": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "config": cfg,
    }
    (ROOT / "v5_submission_audit.json").write_text(json.dumps(audit, indent=2))
    print("SUBMISSION", json.dumps(audit), flush=True)
