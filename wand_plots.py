"""
wandb_plots.py — Learning curves per dataset, publication quality
Metriche reali: nDCG@10, loss  |  X: training_iteration
Ogni run = un trial HPE (set di iperparametri), più run per modello
Il grafico mostra: media ± std tra trial dello stesso modello
"""

import wandb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────────────────────────
ENTITY  = "i-buonfrate-politecnico-di-bari"
PROJECT = "EvaluateCars"

GROUPS = {
    "BGG":    "bgg_experiments",
    "Frappe": "frappe_experiments",
    "Yelp":   "yelp_experiments",
}

METRIC = "nDCG@10"
X_KEY  = "training_iteration"

MODELS_NEURAL = ["FM", "DeepFM", "NFM", "AFM", "xDeepFM"]

MODEL_STYLE = {
    "FM":      {"color": "#2166ac", "ls": "-",  "lw": 1.8, "zorder": 3},
    "DeepFM":  {"color": "#d6604d", "ls": "-",  "lw": 1.8, "zorder": 3},
    "NFM":     {"color": "#4dac26", "ls": "-",  "lw": 1.8, "zorder": 3},
    "AFM":     {"color": "#b2182b", "ls": "-",  "lw": 2.2, "zorder": 4},
    "xDeepFM": {"color": "#762a83", "ls": "--", "lw": 1.8, "zorder": 3},
}

SMOOTHING = 0.6
OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def get_model(run):
    return run.name.split("_")[0]

def ema(values, alpha=SMOOTHING):
    if alpha == 0:
        return np.array(values)
    out, last = [], values[0]
    for v in values:
        last = alpha * last + (1 - alpha) * v
        out.append(last)
    return np.array(out)

def fetch_group(group_name):
    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"group": group_name})

    data = {}
    for run in runs:
        model = get_model(run)
        if model not in MODELS_NEURAL:
            continue

        hist = run.history(keys=[X_KEY, METRIC], samples=2000, pandas=True)
        hist = hist.dropna(subset=[METRIC])

        if len(hist) < 3:
            print(f"  skip {run.name}: solo {len(hist)} righe")
            continue

        hist = hist.sort_values(X_KEY)
        y = hist[METRIC].values.astype(float)

        if model not in data:
            data[model] = []
        data[model].append(y)
        print(f"  ok {run.name}: {len(y)} epoch, best={y.max():.4f}")

    return data

def align_and_aggregate(trials):
    max_len = max(len(t) for t in trials)
    x_grid = np.arange(1, max_len + 1)
    aligned = []
    for t in trials:
        x_t = np.arange(1, len(t) + 1)
        y_interp = np.interp(x_grid, x_t, t)
        aligned.append(y_interp)
    arr = np.array(aligned)
    return x_grid, arr.mean(axis=0), arr.std(axis=0)

# ─── PLOT ─────────────────────────────────────────────────────────────────────

def set_style():
    plt.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "DejaVu Serif"],
        "font.size":          11,
        "axes.titlesize":     13,
        "axes.titleweight":   "bold",
        "axes.labelsize":     11,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "legend.fontsize":    9,
        "legend.framealpha":  0.92,
        "legend.edgecolor":   "#cccccc",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.alpha":         0.3,
        "grid.linestyle":     ":",
        "grid.color":         "#999999",
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.08,
    })

def plot_dataset(ds_name, group_name):
    print(f"\n{'─'*55}")
    print(f"Fetching {ds_name} ({group_name})...")
    data = fetch_group(group_name)

    if not data:
        print(f"  Nessun dato per {ds_name}.")
        return

    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax_ndcg, ax_bar = axes[0], axes[1]

    # ── Pannello sinistro: learning curves ───────────────────────────────
    for model in MODELS_NEURAL:
        if model not in data:
            continue
        style = MODEL_STYLE[model]
        trials = data[model]

        if len(trials) == 1:
            y_raw = trials[0]
            x = np.arange(1, len(y_raw) + 1)
            y = ema(y_raw)
            ax_ndcg.plot(x, y, label=model,
                         color=style["color"], ls=style["ls"],
                         lw=style["lw"], zorder=style["zorder"])
        else:
            x, mean_y, std_y = align_and_aggregate([ema(t) for t in trials])
            ax_ndcg.plot(x, mean_y, label=model,
                         color=style["color"], ls=style["ls"],
                         lw=style["lw"], zorder=style["zorder"])
            ax_ndcg.fill_between(x,
                                 mean_y - std_y, mean_y + std_y,
                                 color=style["color"], alpha=0.12,
                                 zorder=style["zorder"] - 1)

    ax_ndcg.set_xlabel("Epoch")
    ax_ndcg.set_ylabel("Validation nDCG@10")
    ax_ndcg.set_title(f"{ds_name} — Validation nDCG@10")
    ax_ndcg.legend(loc="lower right")
    ax_ndcg.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
    ax_ndcg.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=8))

    # ── Pannello destro: bar chart best nDCG@10 ──────────────────────────
    model_names, best_vals, best_stds = [], [], []
    for model in MODELS_NEURAL:
        if model not in data:
            continue
        bests = [t.max() for t in data[model]]
        model_names.append(model)
        best_vals.append(np.mean(bests))
        best_stds.append(np.std(bests) if len(bests) > 1 else 0)

    colors = [MODEL_STYLE[m]["color"] for m in model_names]
    bars = ax_bar.bar(model_names, best_vals,
                      yerr=best_stds, capsize=4,
                      color=colors, alpha=0.85,
                      edgecolor="white", linewidth=0.8)

    y_offset = (max(best_vals) - min(best_vals)) * 0.05 + 0.0002
    for bar, val in zip(bars, best_vals):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + y_offset,
                    f"{val:.4f}", ha="center", va="bottom",
                    fontsize=8, color="#333333")

    ax_bar.set_ylabel("Best nDCG@10 (mean ± std)")
    ax_bar.set_title(f"{ds_name} — Best nDCG@10 per Model")
    ax_bar.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
    span = max(best_vals) - min(best_vals)
    ax_bar.set_ylim(max(0, min(best_vals) - span * 0.5),
                    max(best_vals) + span * 0.8 + 0.003)

    fig.tight_layout(pad=1.5)

    out_pdf = OUTPUT_DIR / f"wandb_{ds_name.lower()}.pdf"
    out_png = OUTPUT_DIR / f"wandb_{ds_name.lower()}.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)
    print(f"\n  Salvato: {out_pdf}  |  {out_png}")

# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for ds_name, group_name in GROUPS.items():
        plot_dataset(ds_name, group_name)
    print("\nFatto. File in ./figures/")