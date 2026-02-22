"""
Yelp Context Feature Extraction
=====================================

Feature estratte:
    - Temporali (dalla review date): hour_of_day, day_of_week, is_weekend, season, time_slot
    - Utente (dal JSON user):        user_elite, user_experience
    - Business (dal JSON business):  city, category, price_range, alcohol, outdoor_seating

Output:
    - warp_output/yelp_context_ready.tsv  → dataset completo per WarpRec
    - warp_output/yelp_context_info.tsv   → context_info item con TUTTE le feature

Usage:
    python prepare_yelp_context.py
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
JSON_BIZ  = "datasets/yelp_json/yelp_academic_dataset_business.json"
JSON_REV  = "datasets/yelp_json/yelp_academic_dataset_review.json"
JSON_USER = "datasets/yelp_json/yelp_academic_dataset_user.json"
OUTPUT_DIR = Path("warp_output")
OUTPUT_DIR.mkdir(exist_ok=True)

ROW_LIMIT = 200_000

CONTEXT_FEATURES = [
    "hour_of_day", "day_of_week", "is_weekend", "season", "time_slot",
    "user_elite", "user_experience",
    "city", "category", "price_range", "alcohol", "outdoor_seating",
]

# ─── Step 1: Business context ─────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — Extracting business context features...")

def parse_attributes(attrs):
    alcohol = "none"
    outdoor = 0
    if not attrs or not isinstance(attrs, dict):
        return alcohol, outdoor
    alc = str(attrs.get("Alcohol", "none")).replace("'", "").replace('"', "").lower().strip()
    if "full" in alc:
        alcohol = "full"
    elif "beer" in alc or "wine" in alc:
        alcohol = "beer_wine"
    out = attrs.get("OutdoorSeating", False)
    outdoor = 1 if str(out).lower() in ("true", "1", "yes") else 0
    return alcohol, outdoor

biz_data = []
with open(JSON_BIZ, "r", encoding="utf-8") as f:
    for line in f:
        b = json.loads(line)
        attrs = b.get("attributes") or {}
        price_range = attrs.get("RestaurantsPriceRange2", 2)
        try:
            price_range = int(str(price_range).strip("'\" "))
        except:
            price_range = 2
        alcohol, outdoor = parse_attributes(attrs)
        main_cat = b["categories"].split(",")[0].strip() if b.get("categories") else "Other"
        biz_data.append({
            "business_id_orig": b["business_id"],
            "city":             b.get("city", "Unknown"),
            "category":         main_cat,
            "price_range":      price_range,
            "alcohol":          alcohol,
            "outdoor_seating":  outdoor,
        })

df_biz = pd.DataFrame(biz_data)
print(f"  Business loaded: {len(df_biz):,}")

# ─── Step 2: Review temporal features ────────────────────────────────────────
print("\nSTEP 2 — Extracting temporal context from reviews...")

def hour_to_slot(h):
    if 6 <= h < 12:  return "morning"
    if 12 <= h < 17: return "afternoon"
    if 17 <= h < 21: return "evening"
    return "night"

rev_data = []
with open(JSON_REV, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        r = json.loads(line)
        rev_data.append({
            "user_id_orig":     r["user_id"],
            "business_id_orig": r["business_id"],
            "rating":           r["stars"],
            "date_str":         r["date"],
        })
        if i >= ROW_LIMIT:
            break

df_rev = pd.DataFrame(rev_data)
df_rev["date"]        = pd.to_datetime(df_rev["date_str"], errors="coerce")
df_rev["hour_of_day"] = df_rev["date"].dt.hour.fillna(12).astype(int)
df_rev["day_of_week"] = df_rev["date"].dt.dayofweek.fillna(0).astype(int)
df_rev["is_weekend"]  = (df_rev["day_of_week"] >= 5).astype(int)
df_rev["season"]      = df_rev["date"].dt.quarter.fillna(1).astype(int)
df_rev["time_slot"]   = df_rev["hour_of_day"].apply(hour_to_slot)
print(f"  Reviews loaded: {len(df_rev):,}")

# ─── Step 3: User features ───────────────────────────────────────────────────
print("\nSTEP 3 — Extracting user context features...")

needed_users = set(df_rev["user_id_orig"].unique())
user_data = []
with open(JSON_USER, "r", encoding="utf-8") as f:
    for line in f:
        u = json.loads(line)
        uid = u["user_id"]
        if uid not in needed_users:
            continue
        elite_years = u.get("elite", "")
        is_elite = 1 if elite_years and elite_years != "None" else 0
        rc = u.get("review_count", 0)
        if rc < 10:       exp = "novice"
        elif rc < 50:     exp = "intermediate"
        elif rc < 200:    exp = "experienced"
        else:             exp = "expert"
        user_data.append({"user_id_orig": uid, "user_elite": is_elite, "user_experience": exp})

df_user = pd.DataFrame(user_data)
print(f"  Users loaded: {len(df_user):,}")

# ─── Step 4: Merge ───────────────────────────────────────────────────────────
print("\nSTEP 4 — Merging all features...")

df_full = df_rev.merge(df_biz, on="business_id_orig", how="left")
df_full = df_full.merge(df_user, on="user_id_orig", how="left")

df_full["city"]            = df_full["city"].fillna("Unknown")
df_full["category"]        = df_full["category"].fillna("Other")
df_full["price_range"]     = df_full["price_range"].fillna(2).astype(int)
df_full["alcohol"]         = df_full["alcohol"].fillna("none")
df_full["outdoor_seating"] = df_full["outdoor_seating"].fillna(0).astype(int)
df_full["user_elite"]      = df_full["user_elite"].fillna(0).astype(int)
df_full["user_experience"] = df_full["user_experience"].fillna("novice")
print(f"  Merged: {len(df_full):,} rows")

# ─── Step 5: Label Encoding ───────────────────────────────────────────────────
print("\nSTEP 5 — Label encoding IDs...")

rev_raw = []
with open(JSON_REV, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        r = json.loads(line)
        rev_raw.append({"user_id_orig": r["user_id"], "business_id_orig": r["business_id"]})
        if i >= ROW_LIMIT:
            break

df_raw = pd.DataFrame(rev_raw)
user_codes = dict(zip(
    pd.Categorical(df_raw["user_id_orig"]).categories,
    range(len(pd.Categorical(df_raw["user_id_orig"]).categories))
))
item_codes = dict(zip(
    pd.Categorical(df_raw["business_id_orig"]).categories,
    range(len(pd.Categorical(df_raw["business_id_orig"]).categories))
))

df_full["user_id"] = df_full["user_id_orig"].map(user_codes)
df_full["item_id"] = df_full["business_id_orig"].map(item_codes)

# Encoding categoriche
for col in ["city", "category", "alcohol", "user_experience", "time_slot"]:
    df_full[col] = pd.Categorical(df_full[col]).codes

# ─── Step 6: Salva dataset completo ──────────────────────────────────────────
print("\nSTEP 6 — Saving complete context dataset...")

output_cols = ["user_id", "item_id", "rating"] + CONTEXT_FEATURES
df_out = df_full[output_cols].dropna(subset=["user_id", "item_id"])
df_out = df_out.astype({"user_id": int, "item_id": int})

out_path = OUTPUT_DIR / "yelp_context_ready.tsv"
df_out.to_csv(out_path, sep="\t", index=False)
print(f"  Saved: {out_path} | Shape: {df_out.shape}")

# ─── Step 7: Salva context_info item con TUTTE le feature ────────────────────
# include anche feature temporali e utente (aggregate per mode per item)
# Così user context e item context hanno le stesse feature
print("\nSTEP 7 — Saving item context info (ALL features)...")

df_item_ctx = (
    df_full[["item_id"] + CONTEXT_FEATURES]
    .groupby("item_id")
    .agg(lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0])
    .reset_index()
)
df_item_ctx = df_item_ctx.rename(columns={"item_id": "item_id:token"})

ctx_path = OUTPUT_DIR / "yelp_context_info.tsv"
df_item_ctx.to_csv(ctx_path, sep="\t", index=False)
print(f"  Saved: {ctx_path} | Shape: {df_item_ctx.shape}")
print(f"  Columns: {list(df_item_ctx.columns)}")

# ─── Step 8: Summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CONTEXT FEATURE SUMMARY")
print("=" * 60)
print(f"  {'Feature':<22} {'Unique (interactions)':>22} {'Unique (items)':>16}")
print(f"  {'-'*60}")
for feat in CONTEXT_FEATURES:
    n_int  = df_out[feat].nunique() if feat in df_out.columns else "-"
    n_item = df_item_ctx[feat].nunique() if feat in df_item_ctx.columns else "-"
    print(f"  {feat:<22} {str(n_int):>22} {str(n_item):>16}")

print(f"\n  {'dataset':>10}: {out_path}")
print(f"  {'context_info':>10}: {ctx_path}")
print("\n Done. Now re-run WarpRec and then compute_cars_metrics.py")