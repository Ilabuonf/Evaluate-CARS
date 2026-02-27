"""
Frappe Context Feature Extraction for WarpRec
==============================================
Input:  datasets/frappe/frappe_train.csv, frappe_valid.csv, frappe_test.csv
Output: warp_output/frappe_context_ready.tsv
        warp_output/frappe_context_info.tsv
"""

import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("warp_output")
OUTPUT_DIR.mkdir(exist_ok=True)

CONTEXT_FEATURES = [
    "daytime", "weekday", "isweekend",
    "homework", "cost",
    "weather", "country", "city"
]

# ── Step 1: Load and merge the three splits ──────────────────────────────────────
print("STEP 1 — Loading Frappe splits...")

df_train = pd.read_csv("datasets/frappe/frappe_train.csv", sep=",")
df_valid = pd.read_csv("datasets/frappe/frappe_valid.csv", sep=",")
df_test  = pd.read_csv("datasets/frappe/frappe_test.csv",  sep=",")

print(f"  Train: {len(df_train):,} | Valid: {len(df_valid):,} | Test: {len(df_test):,}")

df_all = pd.concat([df_train, df_valid, df_test], ignore_index=True)

# ── Step 2: Rename columns for WarpRec ──────────────────────────────────────
print("STEP 2 — Renaming columns...")

df_all = df_all.rename(columns={
    "user":  "user_id",
    "item":  "item_id",
    "label": "rating",
})

# Frappe has no timestamp — we create a synthetic one (progressive index)
# to allow WarpRec to use temporal_leave_k_out
df_all["timestamp"] = range(len(df_all))

# ── Step 3: Filter only positive interactions (label=1) for ranking ─────────
# WarpRec for leave-one-out needs positive interactions
print("STEP 3 — Filtering positive interactions...")
df_pos = df_all[df_all["rating"] == 1].copy()
print(f"  Positive interactions: {len(df_pos):,} / {len(df_all):,}")

# ── Step 4: Save context_ready ───────────────────────────────────────────────
print("STEP 4 — Saving frappe_context_ready.tsv...")

output_cols = ["user_id", "item_id", "rating", "timestamp"] + CONTEXT_FEATURES
df_out = df_pos[output_cols].dropna()

out_path = OUTPUT_DIR / "frappe_context_ready.tsv"
df_out.to_csv(out_path, sep="\t", index=False)
print(f"  Saved: {out_path} | Shape: {df_out.shape}")

# ── Step 5: Save context_info for items ───────────────────────────────────────
print("STEP 5 — Saving frappe_context_info.tsv...")

df_item_ctx = (
    df_all[["item_id"] + CONTEXT_FEATURES]
    .groupby("item_id")
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
    .reset_index()
)
df_item_ctx = df_item_ctx.rename(columns={"item_id": "item_id:token"})

ctx_path = OUTPUT_DIR / "frappe_context_info.tsv"
df_item_ctx.to_csv(ctx_path, sep="\t", index=False)
print(f"  Saved: {ctx_path} | Shape: {df_item_ctx.shape}")

# ── Step 6: Summary ───────────────────────────────────────────────────────────
print("\nSUMMARY")
print(f"  Users: {df_out['user_id'].nunique():,}")
print(f"  Items: {df_out['item_id'].nunique():,}")
print(f"  Interactions (positive): {len(df_out):,}")
for feat in CONTEXT_FEATURES:
    print(f"  {feat:<15} unique={df_out[feat].nunique()}")
print("\nDone.")