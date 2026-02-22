"""
Calcolo Metriche CARS su Predizioni WarpRec

Usage:
    python compute_cars_metrics.py
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.metrics.context_consistency import compute_acc
from src.metrics.context_satisfaction import compute_cs_wcs
from src.metrics.similarity_metrics import compute_similarity_metrics
from src.metrics.advanced_metrics import (
    compute_context_recall,
    compute_context_ranking_correlation,
    compute_context_group_balance,
)
from src.metrics.weighted_ranking import (
    compute_context_weighted_ndcg,
    compute_context_weighted_map,
)

# ─── Config ───────────────────────────────────────────────────────────────────
RECS_DIR     = Path("results/Yelp/recs")
CONTEXT_DATA = Path("warp_output/yelp_context_ready.tsv")
CONTEXT_INFO = Path("warp_output/yelp_context_info.tsv")
OUTPUT_DIR   = Path("results/Yelp/cars_metrics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONTEXT_FEATURES = [
    "hour_of_day", "day_of_week", "is_weekend", "season",
    "user_elite", "user_experience",
    "city", "category", "price_range", "alcohol", "outdoor_seating",
]

FEATURE_GROUPS = {
    "temporal":      ["hour_of_day", "day_of_week", "is_weekend", "season"],
    "social":        ["user_elite", "user_experience"],
    "business_info": ["city", "category", "price_range", "alcohol", "outdoor_seating"],
}

K_VALUES = [5, 10]
RELEVANCE_THRESHOLD = 4.0

# ─── Utility ──────────────────────────────────────────────────────────────────
def print_partial(metrics: dict, label: str):
    """Stampa i risultati parziali di un gruppo di metriche."""
    if not metrics:
        return
    print(f"\n     ┌─ {label}")
    for k, v in sorted(metrics.items()):
        if not k.endswith("@all"):
            print(f"     │  {k:<25} {v:.4f}")
    print(f"     └{'─'*35}")

def elapsed(start: datetime) -> str:
    secs = (datetime.now() - start).seconds
    return f"{secs//60}m {secs%60}s"

def get_latest_predictions(recs_dir: Path) -> dict:
    model_files = {}
    for f in recs_dir.glob("*.tsv"):
        if "_adapted" in f.name:
            continue
        model_name = f.stem.split("_")[0]
        if model_name not in model_files or f.stat().st_mtime > model_files[model_name].stat().st_mtime:
            model_files[model_name] = f
    return model_files

# ─── Step 1: Carica contesto utenti ───────────────────────────────────────────
print("=" * 65)
print("STEP 1 — Loading context dataset...")
ctx_df = pd.read_csv(CONTEXT_DATA, sep="\t")
ctx_df["user_id"] = ctx_df["user_id"].astype(str)
ctx_df["item_id"] = ctx_df["item_id"].astype(str)

available_features = [f for f in CONTEXT_FEATURES if f in ctx_df.columns]
user_ctx = (
    ctx_df.groupby("user_id")[available_features]
    .agg(lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0])
    .reset_index()
    .rename(columns={"user_id": "user_id:token"})
)
print(f"  Users: {len(user_ctx):,} | Features: {available_features}")

ctx_df["label"] = (ctx_df["rating"] >= RELEVANCE_THRESHOLD).astype(float)
ground_truth_df = ctx_df[["user_id", "item_id", "label"]].rename(
    columns={"user_id": "user_id:token", "item_id": "item_id:token"}
)

# ─── Step 2: Carica context_info item ─────────────────────────────────────────
print("\nSTEP 2 — Loading item context info...")
item_ctx = pd.read_csv(CONTEXT_INFO, sep="\t")
item_ctx["item_id:token"] = item_ctx["item_id:token"].astype(str)
item_features = [f for f in available_features if f in item_ctx.columns]
print(f"  Items: {len(item_ctx):,} | Features: {item_features}")

# ─── Step 3: Adattamento predizioni ───────────────────────────────────────────
def adapt_predictions(pred_path: Path) -> pd.DataFrame:
    pred_df = pd.read_csv(pred_path, sep="\t")
    pred_df = pred_df.rename(columns={"user_id": "user_id:token", "item_id": "item_id:token", "rating": "prediction"})
    pred_df["user_id:token"] = pred_df["user_id:token"].astype(str)
    pred_df["item_id:token"] = pred_df["item_id:token"].astype(str)
    pred_df = pred_df.merge(user_ctx, on="user_id:token", how="left")
    ctx_cols = [f for f in available_features if f in pred_df.columns]
    pred_df["q_context_id"] = pred_df[ctx_cols].astype(str).apply("_".join, axis=1)
    pred_df = pred_df.merge(ground_truth_df, on=["user_id:token", "item_id:token"], how="left")
    pred_df["label"] = pred_df["label"].fillna(0.0)
    pred_df = pred_df.sort_values(["user_id:token", "q_context_id", "prediction"], ascending=[True, True, False])
    pred_df["rank"] = pred_df.groupby(["user_id:token", "q_context_id"]).cumcount() + 1
    return pred_df

# ─── Step 4: Valuta modelli ────────────────────────────────────────────────────
print("\nSTEP 3 — Selecting latest prediction file per model...")
model_files = get_latest_predictions(RECS_DIR)
for model, path in sorted(model_files.items()):
    mtime = datetime.fromtimestamp(path.stat().st_mtime).strftime("%d/%m/%Y %H:%M:%S")
    print(f"  {model:<10} → {path.name}  (modified: {mtime})")

print("\nSTEP 4 — Computing CARS metrics...")
print("=" * 65)

all_results = {}
filtered_groups = {
    g: [f for f in feats if f in item_features]
    for g, feats in FEATURE_GROUPS.items()
    if any(f in item_features for f in feats)
}

for model_name, pred_path in sorted(model_files.items()):
    model_start = datetime.now()
    print(f"\n{'='*65}")
    print(f"  MODEL: {model_name}  |  {pred_path.name}")
    print(f"{'='*65}")
    metrics = {}

    try:
        pred_df = adapt_predictions(pred_path)
        print(f"  Rows: {len(pred_df):,} | Users: {pred_df['user_id:token'].nunique():,}\n")

        # ACC
        t = datetime.now()
        print(f"  [1/9] ACC... ", end="", flush=True)
        res = compute_acc(pred_df, item_ctx, item_features, K_VALUES)
        metrics.update(res)
        print(f"done ({elapsed(t)})")
        print_partial(res, "ACC")

        # CS + WCS
        t = datetime.now()
        print(f"  [2/9] CS / WCS... ", end="", flush=True)
        res = compute_cs_wcs(pred_df, item_ctx, item_features, alpha=0.5, k_values=K_VALUES)
        metrics.update(res)
        print(f"done ({elapsed(t)})")
        print_partial({k: v for k, v in res.items() if k.startswith(("CS@", "WCS@"))}, "CS / WCS")

        # WCA + Friction
        t = datetime.now()
        print(f"  [3/9] WCA / Friction... ", end="", flush=True)
        res = compute_similarity_metrics(pred_df, item_ctx, item_features, K_VALUES)
        metrics.update(res)
        print(f"done ({elapsed(t)})")
        print_partial({k: v for k, v in res.items() if not k.endswith("@all")}, "WCA / Friction")

        # CR
        t = datetime.now()
        print(f"  [4/9] CR... ", end="", flush=True)
        res = compute_context_recall(pred_df, item_ctx, item_features, K_VALUES)
        metrics.update(res)
        print(f"done ({elapsed(t)})")
        print_partial({k: v for k, v in res.items() if not k.endswith("@all")}, "CR")

        # CRC
        t = datetime.now()
        print(f"  [5/9] CRC... ", end="", flush=True)
        res = compute_context_ranking_correlation(pred_df, item_ctx, item_features, K_VALUES)
        metrics.update(res)
        print(f"done ({elapsed(t)})")
        print_partial({k: v for k, v in res.items() if not k.endswith("@all")}, "CRC")

        # CGB
        t = datetime.now()
        print(f"  [6/9] CGB... ", end="", flush=True)
        res = compute_context_group_balance(pred_df, item_ctx, item_features, filtered_groups, K_VALUES)
        metrics.update(res)
        print(f"done ({elapsed(t)})")
        print_partial({k: v for k, v in res.items() if not k.endswith("@all")}, "CGB")

        # WCS per gruppi
        t = datetime.now()
        print(f"  [7/9] WCS per gruppi... ", end="", flush=True)
        group_res = {}
        for group_name, group_feats in filtered_groups.items():
            if group_feats:
                wcs_g = compute_cs_wcs(pred_df, item_ctx, group_feats, alpha=0.5, k_values=K_VALUES)
                for k in K_VALUES:
                    group_res[f"WCS_{group_name}@{k}"] = wcs_g.get(f"WCS@{k}", np.nan)
        metrics.update(group_res)
        print(f"done ({elapsed(t)})")
        print_partial(group_res, "WCS per gruppi")

        # CW-nDCG
        t = datetime.now()
        print(f"  [8/9] CW-nDCG... ", end="", flush=True)
        res = compute_context_weighted_ndcg(pred_df, item_ctx, item_features, K_VALUES)
        metrics.update(res)
        print(f"done ({elapsed(t)})")
        print_partial(res, "CW-nDCG")

        # CW-MAP
        t = datetime.now()
        print(f"  [9/9] CW-MAP... ", end="", flush=True)
        res = compute_context_weighted_map(pred_df, item_ctx, item_features, K_VALUES)
        metrics.update(res)
        print(f"done ({elapsed(t)})")
        print_partial(res, "CW-MAP")

        all_results[model_name] = metrics
        print(f"\n  ✅ {model_name} completed in {elapsed(model_start)}")

    except Exception as e:
        print(f"\n  Error: {e}")
        import traceback; traceback.print_exc()

# ─── Step 5: Risultati finali ──────────────────────────────────────────────────
print("\n" + "=" * 65)
print("FINAL RESULTS — All Models")
print("=" * 65)

if all_results:
    df_results = pd.DataFrame(all_results).T.round(4)
    # Escludi colonne @all per la stampa finale
    df_print = df_results[[c for c in df_results.columns if not c.endswith("@all")]]
    df_print = df_print.reindex(sorted(df_print.columns), axis=1)
    print(df_print.to_string())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"cars_metrics_{timestamp}.csv"
    df_results.to_csv(out_path)
    print(f"\n✅ Full results saved to: {out_path}")
else:
    print("No results computed.")