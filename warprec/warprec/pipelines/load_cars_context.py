"""
load_cars_context.py — Loads CARS context tensors into the WarpRec dataset stash,
aligning indices with WarpRec internal mappings (instead of CSV order).
"""

import torch
import pandas as pd
import numpy as np
from typing import Optional

# Default features for Yelp dataset
CONTEXT_FEATURES = [
    "hour_of_day", "day_of_week", "is_weekend", "season",
    "user_elite", "user_experience",
    "city", "category", "price_range", "alcohol", "outdoor_seating",
]

# Feature group indices for specific metrics like CGB
FEATURE_GROUPS = {
    "temporal":      [0, 1, 2, 3],
    "social":        [4, 5],
    "business_info": [6, 7, 8, 9, 10],
}

# Feature group indices for Frappe dataset
FEATURE_GROUPS_FRAPPE = {
    "temporal":     [0, 1, 2],
    "activity":     [3, 4],
    "environment":  [5, 6, 7],
}


def compute_idf_weights(
    ctx_df: pd.DataFrame,
    features: list,
    num_items: int,
    device: str = "cpu",
) -> torch.Tensor:
    """Computes IDF-like weights for each context feature."""
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
    if features is None:
        features = CONTEXT_FEATURES
    if feature_groups is None:
        feature_groups = FEATURE_GROUPS

    print(f"[CARS] Loading context tensors into dataset stash on device: {device}...")

    user_mapping, item_mapping = dataset.get_mappings()
    num_users = len(user_mapping)
    num_items = len(item_mapping)

    ctx_df = pd.read_csv(context_data_path, sep="\t", dtype={"user_id": str, "item_id": str})
    item_ctx_df = pd.read_csv(context_info_path, sep="\t", dtype={"item_id:token": str})

    available_user_features = [f for f in features if f in ctx_df.columns]
    available_item_features = [f for f in features if f in item_ctx_df.columns]

    user_ctx_agg = (
        ctx_df.groupby("user_id")[available_user_features]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
    )

    # Use -1 as sentinel for "no context data" (works for both binary and
    # categorical features since real values are always >= 0)
    user_ctx_tensor = torch.full(
        (num_users, len(available_user_features)), -1, dtype=torch.float32
    )
    missing_users = 0
    for user_id_str, internal_idx in user_mapping.items():
        if user_id_str in user_ctx_agg.index:
            row = user_ctx_agg.loc[user_id_str, available_user_features].values.astype(np.float32)
            user_ctx_tensor[internal_idx] = torch.from_numpy(row)
        else:
            missing_users += 1

    if missing_users > 0:
        print(f"[CARS] Warning: {missing_users}/{num_users} users have no context (filled with -1).")

    user_ctx_tensor = user_ctx_tensor.to(device)

    item_ctx_indexed = item_ctx_df.set_index("item_id:token")

    item_ctx_tensor = torch.full(
        (num_items, len(available_item_features)), -1, dtype=torch.float32
    )
    missing_items = 0
    for item_id_str, internal_idx in item_mapping.items():
        if item_id_str in item_ctx_indexed.index:
            row = item_ctx_indexed.loc[item_id_str, available_item_features].values.astype(np.float32)
            item_ctx_tensor[internal_idx] = torch.from_numpy(row)
        else:
            missing_items += 1

    if missing_items > 0:
        print(f"[CARS] Warning: {missing_items}/{num_items} items have no context (filled with -1).")

    item_ctx_tensor = item_ctx_tensor.to(device)

    idf_weights = compute_idf_weights(
        item_ctx_df, available_item_features,
        num_items=num_items, device=device,
    )

    adjusted_groups = {}
    for group_name, feat_indices in feature_groups.items():
        adjusted = [i for i in feat_indices if i < len(available_item_features)]
        if adjusted:
            adjusted_groups[group_name] = adjusted

    dataset.add_to_stash("item_context_lookup", item_ctx_tensor)
    dataset.add_to_stash("user_context_lookup", user_ctx_tensor)
    dataset.add_to_stash("context_feature_weights", idf_weights)
    dataset.add_to_stash("feature_groups", adjusted_groups)

    print(
        f"[CARS] Context tensors loaded successfully on {device}.\n"
        f"       users={num_users}, items={num_items}, features={len(available_item_features)}\n"
        f"       user_ctx_tensor: {user_ctx_tensor.shape}, item_ctx_tensor: {item_ctx_tensor.shape}\n"
    )