import torch
import numpy as np

def test_cars_full_suite():
    print("--- INIZIO SUITE COMPLETA TEST CARS ---")
    
    # 1. SETUP DATI (1 Utente, Top-2 raccomandazioni, 3 Feature)
    k = 2
    alpha = 0.5
    
    # Contesto Utente: [1, 1, 0]
    user_ctx = torch.tensor([[1, 1, 0]], dtype=torch.float) 
    
    # Item 1: Match perfetto [1, 1, 0] | Item 2: Match parziale [1, 0, 1]
    item_ctx = torch.tensor([[1, 1, 0], [1, 0, 1]], dtype=torch.float).unsqueeze(0)
    
    # Rilevanza binaria (entrambi rilevanti per il Ground Truth)
    binary_rel = torch.tensor([[1.0, 1.0]]) 
    
    # Punteggi del modello (l'item 1 ha score più alto dell'item 2)
    top_k_scores = torch.tensor([[0.9, 0.1]]) 
    
    # Pesi IDF (es. la feature 2 è molto rara/importante)
    feature_weights = torch.tensor([1.0, 2.0, 1.0])

    # --- LOGICA CORE ---
    match = (item_ctx == user_ctx.unsqueeze(1)).float() # [1, 2, 3]
    
    # 2. TEST METRICHE SEMPLICI
    acc = match.all(dim=-1).float().mean().item()
    friction = match.mean(dim=-1).mean().item()
    cr = match.any(dim=1).float().mean().item()
    
    print(f"ACC:      {acc:.2f} (Atteso: 0.50)")
    print(f"Friction: {friction:.2f} (Atteso: 0.67)")
    print(f"CR:       {cr:.2f} (Atteso: 1.00)")

    # 3. TEST CS & WCS (Satisfaction con penalità)
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

    # 4. TEST CW-nDCG (Context Weighted)
    ctx_weights = match.mean(dim=-1) # [1.0, 0.33]
    cw_rel = binary_rel * ctx_weights
    
    positions = torch.arange(1, k + 1, dtype=torch.float)
    discount = 1.0 / torch.log2(positions + 1)
    dcg = (cw_rel * discount).sum().item()
    idcg = (torch.ones_like(cw_rel) * discount).sum().item()
    cw_ndcg = dcg / idcg
    print(f"CW-nDCG:  {cw_ndcg:.2f} (Atteso: 0.74)")

    # 5. TEST CRC (Correlation)
    # Item 1 (Pos 1): Match 1.0 | Item 2 (Pos 2): Match 0.33
    # Poiché l'ordine degli score (0.9, 0.1) segue l'ordine dei match, la corr è 1.0
    ctx_scores = match.mean(dim=-1) # [1.0, 0.33]
    
    def pearson(a, b):
        a_m, b_m = a.mean(), b.mean()
        num = ((a - a_m) * (b - b_m)).sum()
        den = torch.sqrt(((a - a_m)**2).sum() * ((b - b_m)**2).sum())
        return (num / den).item()

    crc = pearson(top_k_scores[0], ctx_scores[0])
    print(f"CRC:      {crc:.2f} (Atteso: 1.00)")

    print("--- FINE SUITE TEST ---")

if __name__ == "__main__":
    test_cars_full_suite()