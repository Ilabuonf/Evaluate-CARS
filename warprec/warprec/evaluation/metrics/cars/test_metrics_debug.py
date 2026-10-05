import numpy as np
import torch


def test_cars_full_suite():
  print("--- START SUITE TEST CARS ---")

  # 1. DATA SETUP (1 User, Top-2 recommendations, 3 Features)
  k = 2
  alpha = 0.5

  # User Context: [1, 1, 0]
  user_ctx = torch.tensor([[1, 1, 0]], dtype=torch.float)

  # Item 1: Perfect match [1, 1, 0] | Item 2: Partial match [1, 0, 1]
  item_ctx = torch.tensor(
      [[1, 1, 0], [1, 0, 1]], dtype=torch.float
  ).unsqueeze(0)

  # Binary relevance (both relevant in Ground Truth)
  binary_rel = torch.tensor([[1.0, 1.0]])

  # Model scores (item 1 has a higher score than item 2)
  top_k_scores = torch.tensor([[0.9, 0.1]])

  # IDF weights (e.g., feature 2 is very rare/important)
  feature_weights = torch.tensor([1.0, 2.0, 1.0])

  # --- CORE LOGIC ---
  match = (item_ctx == user_ctx.unsqueeze(1)).float()  # [1, 2, 3]

  # 2. SIMPLE METRICS TEST
  acc = match.all(dim=-1).float().mean().item()
  friction = match.mean(dim=-1).mean().item()
  cr = match.any(dim=1).float().mean().item()

  print(f"ACC:      {acc:.2f} (Atteso: 0.50)")
  print(f"Friction: {friction:.2f} (Atteso: 0.67)")
  print(f"CR:       {cr:.2f} (Atteso: 1.00)")

  # 3. CS & WCS TEST (Satisfaction with penalty)
  def calculate_cs(m, w):
    inter = (m * w).sum(dim=-1)
    union = w.sum()
    mismatch = ((1 - m) * w).sum(dim=-1)
    penalty = alpha * mismatch / union
    return (inter / (union + penalty)).mean().item()

  cs_unweighted = calculate_cs(match, torch.ones(3))
  cs_weighted = calculate_cs(match, feature_weights)
  print(f"CS:       {cs_unweighted:.2f} (Atteso: ~0.65)")
  print(f"WCS:      {cs_weighted:.2f} (Atteso: ~0.59 - influenzato dai pesi)")

  # 4. CW-nDCG TEST (Context Weighted)
  ctx_weights = match.mean(dim=-1)  # [1.0, 0.33]
  cw_rel = binary_rel * ctx_weights

  positions = torch.arange(1, k + 1, dtype=torch.float)
  discount = 1.0 / torch.log2(positions + 1)
  dcg = (cw_rel * discount).sum().item()
  idcg = (torch.ones_like(cw_rel) * discount).sum().item()
  cw_ndcg = dcg / idcg
  print(f"CW-nDCG:  {cw_ndcg:.2f} (Atteso: 0.74)")

  # 5. CRC TEST (Correlation)
  # Item 1 (Pos 1): Match 1.0 | Item 2 (Pos 2): Match 0.33
  # Since score order (0.9, 0.1) matches alignment order, correlation is 1.0
  ctx_scores = match.mean(dim=-1)  # [1.0, 0.33]

  def pearson(a, b):
    a_m, b_m = a.mean(), b.mean()
    num = ((a - a_m) * (b - b_m)).sum()
    den = torch.sqrt(((a - a_m) ** 2).sum() * ((b - b_m) ** 2).sum())
    return (num / den).item()

  crc = pearson(top_k_scores[0], ctx_scores[0])
  print(f"CRC:      {crc:.2f} (Atteso: 1.00)")

  print("--- END SUITE TEST ---")


if __name__ == "__main__":
  test_cars_full_suite()
