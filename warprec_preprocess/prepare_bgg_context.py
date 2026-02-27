"""
BGG Context Feature Extraction for WarpRec
==========================================
Input:  datasets/bgg/train_df.tsv, valid_df.tsv, test_df.tsv, context_info.tsv
Output: warp_output/bgg_context_ready.tsv
        warp_output/bgg_context_info.tsv
"""

import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("warp_output")
OUTPUT_DIR.mkdir(exist_ok=True)

CONTEXT_FEATURES = [
    "playing_time_very_short", "playing_time_short", "playing_time_moderate",
    "playing_time_long", "playing_time_very_long",
    "gaming_mood_party", "gaming_mood_easy-going", "gaming_mood_expert",
    "gaming_mood_intense", "gaming_mood_cooperative", "gaming_mood_competitive",
    "gaming_mood_thematic", "gaming_mood_story-based",
    "social_companion_1-player", "social_companion_2-players",
    "social_companion_large-group", "social_companion_toddlers",
    "social_companion_preschoolers", "social_companion_children",
    "social_companion_family", "social_companion_friends",
]

FEATURE_GROUPS = {
    "playing_time": [0, 1, 2, 3, 4],
    "gaming_mood":  [5, 6, 7, 8, 9, 10, 11, 12],
    "social":       [13, 14, 15, 16, 17, 18, 19, 20],
}

# ── Step 1: Load the three splits WITHOUT index_col ───────────────────────────────
print("STEP 1 — Loading BGG splits...")

df_train = pd.read_csv("datasets/bgg/train_df.tsv", sep="\t")
df_valid = pd.read_csv("datasets/bgg/valid_df.tsv", sep="\t")
df_test  = pd.read_csv("datasets/bgg/test_df.tsv",  sep="\t")

# Drop unnamed index column if present
for df in [df_train, df_valid, df_test]:
    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    df.drop(columns=unnamed, inplace=True)

print(f"  Train: {len(df_train):,} | Valid: {len(df_valid):,} | Test: {len(df_test):,}")
print(f"  Columns: {df_train.columns.tolist()}")

# ── Step 2: Concat and rename ─────────────────────────────────────────────────
print("STEP 2 — Concatenating and renaming columns...")

df_all = pd.concat([df_train, df_valid, df_test], ignore_index=True)

df_all = df_all.rename(columns={
    "user_id:token":   "user_id",
    "game_id:token":   "item_id",
    "rating:float":    "rating",
    "timestamp:float": "timestamp",
})

print(f"  Columns after rename: {df_all.columns.tolist()}")

# ── Step 3: Load context_info and join ─────────────────────────────────────
print("STEP 3 — Loading context_info and joining...")

ctx_info = pd.read_csv("datasets/bgg/context_info.tsv", sep="\t")
unnamed = [c for c in ctx_info.columns if c.startswith("Unnamed")]
ctx_info.drop(columns=unnamed, inplace=True)
ctx_info.columns = [c.replace(":float", "") for c in ctx_info.columns]

print(f"  context_info columns: {ctx_info.columns.tolist()[:5]}...")

df_all = df_all.merge(ctx_info, on="context_id", how="left")
print(f"  After join: {len(df_all):,} rows, {df_all.shape[1]} columns")

# ── Step 4: Filter by rating threshold ───────────────────────────────────────
print("STEP 4 — Filtering by rating threshold (>=7)...")
df_pos = df_all[df_all["rating"] >= 7.0].copy()
print(f"  Positive interactions: {len(df_pos):,} / {len(df_all):,}")

# ── Step 5: Save context_ready ───────────────────────────────────────────────
print("STEP 5 — Saving bgg_context_ready.tsv...")

output_cols = ["user_id", "item_id", "rating", "timestamp"] + CONTEXT_FEATURES
df_out = df_pos[output_cols].dropna()
df_out = df_out.astype({"user_id": str, "item_id": str})

out_path = OUTPUT_DIR / "bgg_context_ready.tsv"
df_out.to_csv(out_path, sep="\t", index=False)
print(f"  Saved: {out_path} | Shape: {df_out.shape}")

# ── Step 6: Save context_info for items ───────────────────────────────────────
print("STEP 6 — Saving bgg_context_info.tsv (aggregated by item)...")

df_item_ctx = (
    df_all[["item_id"] + CONTEXT_FEATURES]
    .rename(columns={"item_id": "item_id:token"})
    .groupby("item_id:token")
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
    .reset_index()
)

ctx_path = OUTPUT_DIR / "bgg_context_info.tsv"
df_item_ctx.to_csv(ctx_path, sep="\t", index=False)
print(f"  Saved: {ctx_path} | Shape: {df_item_ctx.shape}")

# ── Step 7: Summary ───────────────────────────────────────────────────────────
print("\nSUMMARY")
print(f"  Users: {df_out['user_id'].nunique():,}")
print(f"  Items: {df_out['item_id'].nunique():,}")
print(f"  Interactions (positive): {len(df_out):,}")
for feat in CONTEXT_FEATURES:
    print(f"  {feat:<40} unique={df_out[feat].nunique()}")
print("\nDone.")