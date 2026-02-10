# Evaluate-CARS
### Comprehensive Evaluation Framework for Context-Aware Recommendation Systems

A complete pipeline for training and evaluating context-aware recommendation models on multiple datasets (BoardGameGeek, Frappe, Yelp) with extensive context-aware metrics.

---

## Overview
This framework provides:
* **End-to-end training pipelines** for context-aware recommender systems.
* **Comprehensive evaluation metrics** including traditional, context-specific, and advanced metrics.
* **Support for multiple datasets**: BoardGameGeek (BGG), Frappe, Yelp.
* **Multiple CTR models**: FM, AFM, FFM, FwFM, DeepFM, and baseline models (Random, Popularity).
* **Context-weighted ranking metrics**: CW-nDCG, CW-MAP.
* **Automated evaluation scripts** with parallel execution support.

---

## Key Features

### Training Pipeline
* Automatic data preprocessing and RecBole format conversion.
* Support for 6+ CTR models with hyperparameter configurations.
* Baseline model implementations (Random, Popularity).
* GPU acceleration support.
* Automatic checkpoint saving.

### Evaluation Metrics
**Traditional Ranking Metrics:**
* AUC, LogLoss
* nDCG@5/10, MAP@10, MRR@10
* Precision@5, Recall@10

**Context-Aware Metrics:**
* **Context Consistency (ACC@K)**: Measures feature-level matching.
* **Context Satisfaction (CS@K, WCS@K)**: Modified Jaccard similarity with IDF weighting.
* **Similarity Metrics**: WCA (Weighted Cosine), Friction (Inverted Hamming).
* **Advanced Metrics**: Context Recall (CR@K), Context Ranking Correlation (CRC@K), Context Group Balance (CGB@K).
* **Context-Weighted Ranking**: CW-nDCG@K, CW-MAP@K.
* **Dimensional Analysis**: Feature group-wise evaluation.

---

## Repository Structure

```text
Evaluate-CARS/
├── configs/                  # YAML configuration files
│   ├── bgg_config.yaml      # BoardGameGeek settings
│   ├── frappe_config.yaml   # Frappe settings
│   └── yelp_config.yaml     # Yelp settings
│
├── datasets/                 # Dataset storage
│   ├── bgg/                 # BoardGameGeek data
│   ├── frappe/              # Frappe data
│   └── yelp/                # Yelp data
│
├── src/                      # Core source code
│   ├── metrics/             # Metric implementations
│   │   ├── context_consistency.py
│   │   ├── context_satisfaction.py
│   │   ├── similarity_metrics.py
│   │   ├── advanced_metrics.py
│   │   └── weighted_ranking.py
│   │
│   ├── models/              # Model implementations
│   │   └── baselines.py    # Random & Popularity
│   │
│   ├── pipelines/           # Training pipelines
│   │   ├── base_pipeline.py
│   │   ├── bgg_pipeline.py
│   │   ├── frappe_pipeline.py
│   │   └── yelp_pipeline.py
│   │
│   └── utils/               # Utility functions
│
├── evaluators/              # Complete evaluators
│   ├── evaluate_bgg.py     # BGG evaluator
│   ├── evaluate_frappe.py  # Frappe evaluator
│   └── evaluate_yelp.py    # Yelp evaluator
│
├── scripts/                 # Automation scripts
│   ├── run_bgg_pipeline.sh
│   ├── run_frappe_pipeline.sh
│   ├── run_yelp_pipeline.sh
│   └── evaluate_all.sh
│
├── outputs/                 # Model outputs & predictions
└── results/                 # Evaluation results & visualizations
```

### Installation
# Clone repository
git clone [https://github.com/Ilabuonf/Evaluate-CARS.git](https://github.com/Ilabuonf/Evaluate-CARS.git)
cd Evaluate-CARS

### 1. Install dependencies
pip install pandas numpy scipy scikit-learn
pip install recbole  # For CTR models
pip install ranx     # For ranking metrics
pip install tqdm matplotlib seaborn  # Optional: progress bars & visualization

### 2. Data Preparation
Place your datasets in the appropriate directories:
```text
datasets/
├── bgg/
│   ├── train_df.tsv
│   ├── test_df.tsv
│   └── context_info.tsv
├── frappe/
│   ├── frappe_train.csv
│   └── frappe_test.csv
└── yelp/
    ├── yelp_train.csv
    └── yelp_test.csv
```
### 3. Training Models

**Option A: Single Dataset (via module execution)**
```bash
# Train all models on BGG
python -m src.pipelines.bgg_pipeline

# Train on Frappe
python -m src.pipelines.frappe_pipeline

# Train on Yelp
python -m src.pipelines.yelp_pipeline
```

**Option B: Using Automation Scripts**
```bash
# Full pipeline (train + evaluate)
./scripts/run_bgg_pipeline.sh

# Training only
./scripts/run_bgg_pipeline.sh --train-only

# Evaluation only (if models are already trained)
./scripts/run_bgg_pipeline.sh --eval-only
```
### 4. Evaluation
Run evaluation for a specific dataset:
```bash
python -m evaluators.evaluate_bgg
python -m evaluators.evaluate_frappe
python -m evaluators.evaluate_yelp
```

## Datasets

### 1. BoardGameGeek (BGG) - Board Game Recommendations
Primary dataset focusing on entertainment with rich contextual constraints.
* **Statistics:** 43,660 Users | 901 Items | 1,113,609 Interactions | 410 Unique Contexts.
* **Context Features (21 binary features):**
    * *Playing Time:* very_short, short, moderate, long, very_long.
    * *Gaming Mood:* party, easy-going, expert, intense, cooperative, competitive, thematic, story-based.
    * *Social Companion:* 1-player, 2-players, large-group, toddlers, preschoolers, children, family, friends.
      
Dataset published by professors at the Autonomous University of Madrid.

### 2. Frappe - Mobile Application Recommendations
Mobile app usage dataset collected through crowdsourcing.
* **Statistics:** 957 Users | 4,082 Items | 96,203 Interactions | 5,382 Unique Contexts.
* **Context Groups:** Temporal (daytime, weekday, weekend), Activity (homework, cost), Environment (weather, country, city).
  
Public dataset taken from https://huggingface.co/datasets/reczoo/Frappe_x1

### 3. Yelp - Local Business Recommendations
The Yelp Dataset is a comprehensive collection of data related to businesses, reviews, users, tips, and check-ins.
* **Statistics:** 4,803 Users | 22,233 Businesses | 100k Interactions | 38,855 Unique Contexts (7× Frappe).
* **Context Groups:** Temporal (hour, day, weekend), Social (review length, elite status), Spatial (614 cities, 56 categories, 4 price ranges).
  
Public dataset taken from https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset/data

## Evaluation Metrics

### 1. Standard Ranking Metrics (Context-Agnostic)
* **Precision@K / Recall@K**: Retrieval quality.
* **MAP@K / MRR@K**: Average precision and rank of first relevant item.
* **nDCG@K**: Position-aware ranking quality.

### 2. Context Similarity and Consistency
* **ACC@K (Average Context Consistency)**: Percentage of exact feature matches.
* **CS (Context Satisfaction)**: Modified Jaccard index with penalties for missing features:
  $$CS = \frac{|C_q \cap C_i|}{|C_q \cup C_i| + \alpha \cdot \frac{|C_q \setminus C_i|}{|C_q|}}$$
* **WCS (Weighted Context Satisfaction)**: IDF-weighted version of CS prioritizing rare features.
* **WCA (Weighted Context Alignment)**: IDF-weighted cosine similarity.
* **Friction**: Inverted Weighted Hamming Distance measuring the "cost" of mismatch.

### 3. Advanced Context-Centric Metrics
* **CR@K (Context Recall)**: Percentage of requested context features covered by top-K items.
* **CRC (Context Ranking Correlation)**: Spearman correlation between ranking position and context quality.
* **CGB (Context Group Balance)**: Measures balance across context dimensions (Temporal, Social, Spatial):
  $$CGB = 1 - \min\left(\frac{\sigma_{groups}}{0.5}, 1\right)$$

### 4. Context-Weighted (CW) Ranking Metrics
Integrates context similarity directly into traditional relevance:
* **CW-nDCG@K**: 
  $$CW\text{-}DCG = \sum_{k=1}^{K} \frac{(2^{rel_k} - 1) \cdot \text{sim}(C_q, C_k)}{\log_2(k+1)}$$
* **CW-MAP@K**: MAP weighted by context similarity (CS, WCA, or Friction).