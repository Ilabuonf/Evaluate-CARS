# Evaluate-CARS
### Comprehensive Evaluation Framework for Context-Aware Recommendation Systems

A complete pipeline for training and evaluating context-aware recommendation models on multiple datasets (**BoardGameGeek**, **Frappe**, **Yelp**) using **WarpRec** as the underlying recommendation framework, with an extensive suite of context-aware metrics.

---

## Overview
This framework provides:
* **End-to-end training pipelines** for context-aware recommender systems via WarpRec.
* **Distributed hyperparameter optimisation** using Ray Tune with early stopping.
* **Comprehensive evaluation metrics** including traditional ranking, context-specific, and advanced context-dynamics metrics.
* **Support for three datasets**: BoardGameGeek (BGG), Frappe, Yelp.
* **Five CTR models**: FM, DeepFM, NFM, AFM, xDeepFM — plus baseline models (Random, Popularity).
* **Context-weighted ranking metrics**: CW-nDCG, CW-MAP.
* **Experiment tracking** with Weights & Biases and energy monitoring via CodeCarbon.

---

## Repository Structure
```text
Evaluate-CARS/
├── configs/                        # WarpRec YAML configuration files
│   ├── bgg_warp_config.yml
│   ├── frappe_warp_config.yml
│   └── yelp_warp_config.yml
│
├── datasets/                       # Raw dataset storage
│   ├── bgg/
│   ├── frappe/
│   └── yelp/
│
├── warprec_preprocess/             # Dataset preprocessing for WarpRec
│   ├── prepare_bgg_context.py      # BGG context tensor construction
│   ├── prepare_frappe_context.py   # Frappe context tensor construction
│   ├── prepare_yelp_context.py     # Yelp context tensor construction
│   ├── cars_callback.py            # WarpRec CARS evaluation callback
│   └── json_to_csv.py              # Format conversion utility
│
├── warprec/                        # WarpRec framework (local)
│
├── outputs/                        # Model predictions (top-K per dataset)
│   ├── bgg/
│   ├── frappe/
│   └── yelp/
│
├── results/                        # Evaluation metric outputs
│   ├── bgg/   
│   ├── frappe/ 
│   ├── yelp/   
│   └── carbon/                     # CodeCarbon emissions logs
│
├── figures/                        # Generated plots (W&B learning curves)
├── warp_output/                    # WarpRec training artefacts & time reports
├── log/                            # Run logs
│
├── wand_plots.py                   # W&B learning curve generation
├── test_real_dataset.py            # Dataset validation script
└── requirements.txt
```

### Installation
# Clone repository
git clone [https://github.com/Ilabuonf/Evaluate-CARS.git](https://github.com/Ilabuonf/Evaluate-CARS.git)
cd Evaluate-CARS

### 1. Install dependencies
pip install -r requirements.txt

### 2. Data Preparation
```bash
python warprec_preprocess/prepare_bgg_context.py
python warprec_preprocess/prepare_frappe_context.py
python warprec_preprocess/prepare_yelp_context.py
```


### 3. Run training and evaluation
Run evaluation for a specific dataset:
```bash
# BoardGameGeek
python -m warprec --config configs/bgg_warp_config.yml

# Frappe
python -m warprec --config configs/frappe_warp_config.yml

# Yelp
python -m warprec --config configs/yelp_warp_config.yml
```

WarpRec handles hyperparameter search via Ray Tune, per-epoch validation, early stopping (patience 10, grace 20), and final test evaluation. Predictions are written to outputs/<dataset>/.


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
The Yelp Dataset is a comprehensive collection of data related to businesses, reviews, users, tips, and check-ins, specifically filtered for the restaurant and food domain.

* **Statistics:** 45,651 Users | 16,237 Businesses | 604,498 Interactions | ~150k Unique Contexts.
* **Context Groups:**
    * **Temporal:** hour_of_day, day_of_week, is_weekend.
    * **Social & User:** review_length, user_elite, user_experience, alcohol, outdoor_seating.
    * **Spatial:** city (637 unique), category (340 unique), price_range.

Public dataset taken from: https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset/data

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

## Acknowledgments
This evaluation framework is built upon **WarpRec**, an open-source recommendation framework developed by **SisInfLab** (Politecnico di Bari). WarpRec provides the core infrastructure for model training, dataset splitting, and hyperparameter optimization used in this project.
For more information, visit the [WarpRec Repository](https://github.com/sisinflab/warprec).

## Citation
If you use this framework or the proposed metrics, please cite the following:

**This Thesis:**
```bibtex
@mastersthesis{buonfrate2026evaluate,
  author  = {Ilaria Buonfrate},
  title   = {Evaluate-CARS: A Comprehensive Evaluation Framework for
             Context-Aware Recommendation Systems},
  school  = {Politecnico di Bari},
  year    = {2026}
}
