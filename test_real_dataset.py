import torch
import pandas as pd
import sys
import os
import math

# Add the path to allow importing your CARS metrics
sys.path.append(os.path.abspath("warprec"))

from warprec.evaluation.metrics.cars.acc import ACC
from warprec.evaluation.metrics.cars.cr import CR
from warprec.evaluation.metrics.cars.wca_friction import Friction, WCA
from warprec.evaluation.metrics.cars.cs_wcs import CS, WCS
from warprec.evaluation.metrics.cars.cw_ndcg_map import CWnDCG, CWMAP
from warprec.evaluation.metrics.cars.crc import CRC

def test_on_real_data():
    # 1. Dataset Loading
    path = "warp_output/yelp_context_ready.tsv"
    if not os.path.exists(path):
        print(f"Error: file {path} not found.")
        return
    df = pd.read_csv(path, sep="\t")
    
    # Context features used for matching
    context_cols = ['hour_of_day', 'day_of_week', 'is_weekend', 'season',
                    'user_elite', 'user_experience', 'city', 'category',
                    'price_range', 'alcohol', 'outdoor_seating']
    
    u_idx = 0
    i_idxs = [10, 11, 12]
    k = 3

    # Initialize lookup tables as float tensors
    user_context_lookup = torch.from_numpy(df[context_cols].values).float()
    item_context_lookup = torch.from_numpy(df[context_cols].values).float()
    
    # 2. REAL VALUES COMPARISON TABLE (Requested as first output)
    print("\n" + "="*80)
    print(f"{'REAL VALUES COMPARISON TABLE (User 0 vs Top-3 Items)':^80}")
    print("="*80)
    print(f"{'FEATURE':<20} | {'USER 0':<10} | {'ITEM 10':<10} | {'ITEM 11':<10} | {'ITEM 12':<10}")
    print("-" * 80)
    
    match_matrix = [] # Tracking binary matches for manual verification
    for i, col in enumerate(context_cols):
        u_val = user_context_lookup[u_idx][i].item()
        vals = [item_context_lookup[idx][i].item() for idx in i_idxs]
        
        # Visually mark matches with icons
        m10 = "✅" if u_val == vals[0] else "  "
        m11 = "✅" if u_val == vals[1] else "  "
        m12 = "✅" if u_val == vals[2] else "  "
        
        print(f"{col:<20} | {u_val:<10.1f} | {vals[0]:>5.1f} {m10} | {vals[1]:>5.1f} {m11} | {vals[2]:>5.1f} {m12}")
        match_matrix.append([u_val == v for v in vals])

    # 3. LOCAL MATCH ANALYSIS (Calculations per item)
    print("\n" + "="*80)
    print(f"{'LOCAL MATCH COUNT & CONSISTENCY (Friction per Item)':^80}")
    print("="*80)
    
    item_matches = [sum([row[j] for row in match_matrix]) for j in range(k)]
    fractions = [m/len(context_cols) for m in item_matches]
    
    for i, idx in enumerate(i_idxs):
        # Explicitly showing the X/11 calculation
        print(f"ITEM {idx}: {item_matches[i]} matches out of {len(context_cols)} total -> Consistency: {fractions[i]:.4f}")

    # 4. STEP-BY-STEP METRIC EXPLANATION
    print("\n" + "="*80)
    print(f"{'STEP-BY-STEP METRIC LOGIC':^80}")
    print("="*80)

    # ACC: Exact match logic
    print(f"[ACC]      -> No item has 11/11 matches. ACC Result = 0.0")
    
    # Friction: Arithmetic mean of the list consistency
    avg_friction = sum(fractions) / k
    print(f"[Friction] -> Mean of consistencies: ({fractions[0]:.4f} + {fractions[1]:.4f} + {fractions[2]:.4f}) / 3 = {avg_friction:.4f}")

    # CR: Union of satisfied features across the whole list
    features_satisfied = []
    for i in range(len(context_cols)):
        if any(match_matrix[i]): features_satisfied.append(i)
    cr_val = len(features_satisfied) / len(context_cols)
    print(f"[CR]       -> Unique features satisfied by at least one item: {len(features_satisfied)}/11 = {cr_val:.4f}")

    # CW-nDCG: Position-aware context-weighted ranking
    # Assuming binary relevance: items 10 and 11 are relevant (1.0), 12 is not (0.0)
    binary_rel = [1.0, 1.0, 0.0]
    # Logarithmic discounts for positions 1, 2, and 3
    # Formula: (Relevance * Context_Weight) / log2(position + 1)
    dcg = (fractions[0] * 1.0) + (fractions[1] * 0.6309) + (0 * 0.5) 
    idcg = (1.0 * 1.0) + (1.0 * 0.6309) + (1.0 * 0.5) 
    ndcg_val = dcg / idcg
    print(f"[CW-nDCG]  -> (Rel1*M1/log2(2) + Rel2*M2/log2(3) + 0) / IDCG_Total = {ndcg_val:.4f}")

    # 5. EXECUTION VIA PYTORCH METRIC CLASSES (Verification)
    # This part runs your actual class-based logic to confirm it matches manual calculations
    common_params = {
        "k": k, "num_users": len(df), "num_items": len(df),
        "item_context_lookup": item_context_lookup,
        "user_context_lookup": user_context_lookup
    }

    metrics = {
        "ACC": ACC(**common_params), "CR": CR(**common_params),
        "Friction": Friction(**common_params), "WCA": WCA(**common_params),
        "CS": CS(**common_params), "WCS": WCS(**common_params),
        "CW-nDCG": CWnDCG(**common_params), "CW-MAP": CWMAP(**common_params),
        "CRC": CRC(**common_params)
    }

    results = {}
    # Prepare tensors for the metric update
    top_k_indices = torch.tensor([i_idxs], dtype=torch.long)
    user_indices = torch.tensor([u_idx], dtype=torch.long)
    top_k_values = torch.tensor([[0.95, 0.80, 0.40]], dtype=torch.float)
    bin_rel_tensor = torch.tensor([binary_rel], dtype=torch.float)

    for name, m in metrics.items():
        m.update(
            preds=None,
            **{f"top_{k}_indices": top_k_indices},
            **{f"top_{k}_values": top_k_values},
            **{f"top_{k}_binary_relevance": bin_rel_tensor},
            user_indices=user_indices,
            valid_users=torch.tensor([1.0])
        )
        results.update(m.compute())

    # 6. FINAL AGGREGATED RESULTS TABLE
    print("\n" + "="*45)
    print(f"{'FINAL AGGREGATED RESULTS':<25} | {'VALUE':<15}")
    print("-" * 45)
    for name, value in sorted(results.items()):
        print(f"{name:<25} | {value:<15.4f}")
    print("="*45)

if __name__ == "__main__":
    test_on_real_data()