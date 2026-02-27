# src/cars_callback.py
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from warprec.utils.callback import WarpRecCallback
from warprec.utils.logger import logger

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

class CARSCallback(WarpRecCallback):

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

    def __init__(self,
                 context_data_path: str = "warp_output/yelp_context_ready.tsv",
                 context_info_path: str = "warp_output/yelp_context_info.tsv",
                 recs_dir: str = "results/Yelp/recs",
                 output_dir: str = "results/Yelp/cars_metrics"):

        self.context_data_path = Path(context_data_path)
        self.context_info_path = Path(context_info_path)
        self.recs_dir = Path(recs_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Carica contesto una sola volta
        self._load_context()

    def _load_context(self):
        ctx_df = pd.read_csv(self.context_data_path, sep="\t")
        ctx_df["user_id"] = ctx_df["user_id"].astype(str)
        ctx_df["item_id"] = ctx_df["item_id"].astype(str)

        available = [f for f in self.CONTEXT_FEATURES if f in ctx_df.columns]

        self.user_ctx = (
            ctx_df.groupby("user_id")[available]
            .agg(lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0])
            .reset_index()
            .rename(columns={"user_id": "user_id:token"})
        )

        ctx_df["label"] = (ctx_df["rating"] >= 4.0).astype(float)
        self.ground_truth = ctx_df[["user_id", "item_id", "label"]].rename(
            columns={"user_id": "user_id:token", "item_id": "item_id:token"}
        )

        self.item_ctx = pd.read_csv(self.context_info_path, sep="\t")
        self.item_ctx["item_id:token"] = self.item_ctx["item_id:token"].astype(str)
        self.item_features = [f for f in available if f in self.item_ctx.columns]
        self.available = available

    def _adapt_predictions(self, pred_path: Path) -> pd.DataFrame:
        pred_df = pd.read_csv(pred_path, sep="\t")
        pred_df = pred_df.rename(columns={
            "user_id": "user_id:token",
            "item_id": "item_id:token",
            "rating":  "prediction"
        })
        pred_df["user_id:token"] = pred_df["user_id:token"].astype(str)
        pred_df["item_id:token"] = pred_df["item_id:token"].astype(str)
        pred_df = pred_df.merge(self.user_ctx, on="user_id:token", how="left")
        ctx_cols = [f for f in self.available if f in pred_df.columns]
        pred_df["q_context_id"] = pred_df[ctx_cols].astype(str).apply("_".join, axis=1)
        pred_df = pred_df.merge(self.ground_truth, on=["user_id:token", "item_id:token"], how="left")
        pred_df["label"] = pred_df["label"].fillna(0.0)
        pred_df = pred_df.sort_values(
            ["user_id:token", "q_context_id", "prediction"],
            ascending=[True, True, False]
        )
        pred_df["rank"] = pred_df.groupby(["user_id:token", "q_context_id"]).cumcount() + 1
        return pred_df

    def on_evaluation_complete(self, model, params, results, **kwargs):
        model_name = model.name
        logger.msg(f"Computing CARS metrics for {model_name}...")

        # Trova il file di predizioni più recente per questo modello
        pred_files = sorted(
            self.recs_dir.glob(f"{model_name}_*.tsv"),
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )
        if not pred_files:
            logger.negative(f"No prediction file found for {model_name}, skipping CARS metrics.")
            return

        pred_df = self._adapt_predictions(pred_files[0])
        filtered_groups = {
            g: [f for f in feats if f in self.item_features]
            for g, feats in self.FEATURE_GROUPS.items()
            if any(f in self.item_features for f in feats)
        }

        metrics = {}
        metrics.update(compute_acc(pred_df, self.item_ctx, self.item_features, self.K_VALUES))
        metrics.update(compute_cs_wcs(pred_df, self.item_ctx, self.item_features, alpha=0.5, k_values=self.K_VALUES))
        metrics.update(compute_similarity_metrics(pred_df, self.item_ctx, self.item_features, self.K_VALUES))
        metrics.update(compute_context_recall(pred_df, self.item_ctx, self.item_features, self.K_VALUES))
        metrics.update(compute_context_ranking_correlation(pred_df, self.item_ctx, self.item_features, self.K_VALUES))
        metrics.update(compute_context_group_balance(pred_df, self.item_ctx, self.item_features, filtered_groups, self.K_VALUES))
        for g_name, g_feats in filtered_groups.items():
            if g_feats:
                wcs_g = compute_cs_wcs(pred_df, self.item_ctx, g_feats, alpha=0.5, k_values=self.K_VALUES)
                for k in self.K_VALUES:
                    metrics[f"WCS_{g_name}@{k}"] = wcs_g.get(f"WCS@{k}", float("nan"))
        metrics.update(compute_context_weighted_ndcg(pred_df, self.item_ctx, self.item_features, self.K_VALUES))
        metrics.update(compute_context_weighted_map(pred_df, self.item_ctx, self.item_features, self.K_VALUES))

        # Stampa risultati
        logger.msg(f"CARS metrics for {model_name}:")
        for k, v in sorted(metrics.items()):
            if not k.endswith("@all"):
                logger.msg(f"  {k}: {v:.4f}")

        # Salva CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = self.output_dir / f"cars_{model_name}_{timestamp}.csv"
        pd.Series(metrics).to_csv(out, header=["value"])
        logger.positive(f"CARS metrics saved to {out}")

        # Log su wandb se attivo
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({f"cars/{k}": v for k, v in metrics.items()
                          if not k.endswith("@all")})
        except ImportError:
            pass