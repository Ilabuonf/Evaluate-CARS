"""
load_cars_context.py — Carica i tensori di contesto CARS nello stash del dataset WarpRec,
allineando gli indici agli indici interni di WarpRec (non all'ordinamento del CSV).

Problema risolto: il CSV originale ha 143686 utenti, ma WarpRec filtra e rimappa
a 10202 utenti interni (0-10201). Senza allineamento, user_context_lookup[internal_idx]
restituiva il contesto dell'utente sbagliato.
"""

import torch
import pandas as pd
import numpy as np
from typing import Optional


CONTEXT_FEATURES = [
    "hour_of_day", "day_of_week", "is_weekend", "season",
    "user_elite", "user_experience",
    "city", "category", "price_range", "alcohol", "outdoor_seating",
]

FEATURE_GROUPS = {
    "temporal":      [0, 1, 2, 3],
    "social":        [4, 5],
    "business_info": [6, 7, 8, 9, 10],
}
FEATURE_GROUPS_FRAPPE = {
    "temporal":     [0, 1, 2],   # daytime, weekday, isweekend
    "activity":     [3, 4],      # homework, cost
    "environment":  [5, 6, 7],   # weather, country, city
}


def compute_idf_weights(
    ctx_df: pd.DataFrame,
    features: list,
    num_items: int,
    device: str = "cpu",
) -> torch.Tensor:
    weights = []
    for feat in features:
        if feat not in ctx_df.columns:
            weights.append(1.0)
            continue
        n_unique = ctx_df[feat].nunique()
        idf = np.log(num_items / max(n_unique, 1))
        weights.append(max(idf, 0.01))
    return torch.tensor(weights, dtype=torch.float32).to(device)


def load_cars_context_to_dataset(
    dataset,
    context_data_path: str = "warp_output/yelp_context_ready.tsv",
    context_info_path: str = "warp_output/yelp_context_info.tsv",
    features: Optional[list] = None,
    feature_groups: Optional[dict] = None,
    device: str = "cuda",
):
    """
    Carica i tensori di contesto CARS nello stash del dataset WarpRec.

    I tensori vengono costruiti usando le mappature interne WarpRec
    (dataset.get_mappings()) per garantire che user_context_lookup[internal_idx]
    restituisca il contesto corretto per ogni utente/item.

    Args:
        dataset: Istanza di warprec.data.Dataset.
        context_data_path: Path al TSV delle transazioni con features di contesto.
        context_info_path: Path al TSV delle features aggregate per item.
        features: Lista di features da usare (default: CONTEXT_FEATURES).
        feature_groups: Gruppi di features per CGB (default: FEATURE_GROUPS).
        device: Device PyTorch ("cuda" o "cpu").
    """
    if features is None:
        features = CONTEXT_FEATURES
    if feature_groups is None:
        feature_groups = FEATURE_GROUPS

    print(f"[CARS] Loading context tensors into dataset stash on device: {device}...")

    # ── Mappature interne WarpRec ─────────────────────────────────────────────
    # user_mapping: {user_id_str -> internal_idx (0-based)}
    # item_mapping: {item_id_str -> internal_idx (0-based)}
    user_mapping, item_mapping = dataset.get_mappings()
    num_users = len(user_mapping)
    num_items = len(item_mapping)

    # ── Carica CSV ────────────────────────────────────────────────────────────
    ctx_df = pd.read_csv(context_data_path, sep="\t", dtype={"user_id": str, "item_id": str})
    item_ctx_df = pd.read_csv(context_info_path, sep="\t", dtype={"item_id:token": str})

    available_user_features = [f for f in features if f in ctx_df.columns]
    available_item_features = [f for f in features if f in item_ctx_df.columns]

    # ── User context: mode per utente, allineata agli indici interni ──────────
    user_ctx_agg = (
        ctx_df.groupby("user_id")[available_user_features]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
    )
    # user_ctx_agg.index sono user_id_str

    user_ctx_tensor = torch.zeros(num_users, len(available_user_features), dtype=torch.float32)
    missing_users = 0
    for user_id_str, internal_idx in user_mapping.items():
        if user_id_str in user_ctx_agg.index:
            row = user_ctx_agg.loc[user_id_str, available_user_features].values.astype(np.float32)
            user_ctx_tensor[internal_idx] = torch.from_numpy(row)
        else:
            missing_users += 1

    if missing_users > 0:
        print(f"[CARS] Warning: {missing_users}/{num_users} users have no context (filled with zeros).")

    user_ctx_tensor = user_ctx_tensor.to(device)

    # ── Item context: allineata agli indici interni ───────────────────────────
    item_ctx_indexed = item_ctx_df.set_index("item_id:token")

    item_ctx_tensor = torch.zeros(num_items, len(available_item_features), dtype=torch.float32)
    missing_items = 0
    for item_id_str, internal_idx in item_mapping.items():
        if item_id_str in item_ctx_indexed.index:
            row = item_ctx_indexed.loc[item_id_str, available_item_features].values.astype(np.float32)
            item_ctx_tensor[internal_idx] = torch.from_numpy(row)
        else:
            missing_items += 1

    if missing_items > 0:
        print(f"[CARS] Warning: {missing_items}/{num_items} items have no context (filled with zeros).")

    item_ctx_tensor = item_ctx_tensor.to(device)

    # ── IDF weights ───────────────────────────────────────────────────────────
    idf_weights = compute_idf_weights(
        item_ctx_df, available_item_features,
        num_items=num_items, device=device,
    )

    # ── Feature groups (filtra indici fuori range) ────────────────────────────
    adjusted_groups = {}
    for group_name, feat_indices in feature_groups.items():
        adjusted = [i for i in feat_indices if i < len(available_item_features)]
        if adjusted:
            adjusted_groups[group_name] = adjusted

    # ── Aggiungi allo stash ───────────────────────────────────────────────────
    dataset.add_to_stash("item_context_lookup", item_ctx_tensor)
    dataset.add_to_stash("user_context_lookup", user_ctx_tensor)
    dataset.add_to_stash("context_feature_weights", idf_weights)
    dataset.add_to_stash("feature_groups", adjusted_groups)

    print(
        f"[CARS] Context tensors loaded successfully on {device}.\n"
        f"       users={num_users}, items={num_items}, features={len(available_item_features)}\n"
        f"       user_ctx_tensor: {user_ctx_tensor.shape}, item_ctx_tensor: {item_ctx_tensor.shape}\n"
    )