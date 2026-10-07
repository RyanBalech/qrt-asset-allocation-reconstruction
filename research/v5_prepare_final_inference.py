"""Compute reserved and test features without inspecting target outcomes."""

from v5_pipeline import *

cfg = json.loads((ROOT / "v5_frozen_config.json").read_text())
ctx = context()
del ctx["Y"]
manifest = json.loads((ROOT / "v5_pool_manifest.json").read_text())
for name, entry in manifest.items():
    if entry["role"] != "validation":
        continue
    cached_build(
        ctx,
        entry["pool"],
        entry["grid"],
        "holdout_" + name,
        adaptive_weights=cfg["adaptive_weights"],
    )
    print("Reserved inference prepared without scoring:", name, flush=True)
cal = pd.read_csv(ROOT / "v4_calendar_final_test.csv").sort_values("q")
cached_build(
    ctx,
    cal.q.to_numpy(),
    cal.sort_values("grid_position").real_date.tolist(),
    "test",
    adaptive_weights=cfg["adaptive_weights"],
)
print("Test inference prepared.", flush=True)
