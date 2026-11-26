# ReviewSense Analytics — Scalable Product Review Sentiment & Buying Intelligence

**Course:** `23RAI351` – Big Data Analytics (End-Semester Project)  
**Faculty Guide:** Dr. Ravi Kumar V.  
**Team No. 1:** Sreenidhi and Lakshya Yarlagadda  
**Repository:** [Lakshya-dot/Product-Review-BigData-Analytics](https://github.com/Lakshya-dot/Product-Review-BigData-Analytics)

---

## Overview

**ReviewSense Analytics** is an end-to-end big data NLP and consumer decision platform built on **1,140,000 genuine Amazon Reviews 2023 records** (**1,139,264** after cleaning & deduplication) spanning **543,496 unique products** (`parent_asin`) across **6 product categories** and **4 yearly cohorts (2023–2026)**.

The project combines:
- **Large-Scale Data Engineering (Python + Polars + PyArrow)** to sample, clean, date-shift, and join `meta_*.jsonl` catalog details without exceeding 16 GB laptop RAM.
- **Databricks Delta Lakehouse & PySpark MLlib (`Data_Bricks(files)/`)** for distributed ingestion, text preprocessing, TF-IDF vectorization, multi-class sentiment modeling, Gold aggregations, and Spark Structured Streaming.
- **Explainable Product Buying Advisor** computing a transparent `0–100` score and **BUY / DON'T BUY / INSUFFICIENT DATA** verdict for every product.
- **6-View Multi-Page Streamlit + Plotly Web Application (`dashboard/`)** designed for both end consumers and academic evaluation.

---

## Dataset Design & Methodology

- **Source:** [McAuley Lab Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) (`raw/review_categories/` + `raw/meta_categories/`).
- **6 Categories:** `Electronics`, `Home and Kitchen`, `Health and Household`, `Beauty and Personal Care`, `Sports and Outdoors`, and `Books`.
- **Temporal Prototype Mapping (`+3 Year Shift`):** Because Amazon Reviews 2023 ends around September 2023, genuine 2020–2023 reviews were shifted forward by exactly 3 years (`2020 -> 2023`, `2021 -> 2024`, `2022 -> 2025`, `2023 -> 2026`), preserving real text, ratings, seasonal spikes, `original_timestamp`, `date_shifted = True`, and **`is_synthetic = False`** (0% synthetic data).
- **Rating-Derived Proxy Labels:** `1–2 Stars -> Negative (0)`, `3 Stars -> Neutral (1)`, `4–5 Stars -> Positive (2)`.
- **Variant Grouping:** Products are grouped strictly by Amazon `parent_asin` so color/size variants stay unified without mixing distinct models.

### Yearly Cohort Breakdown (`results/year_summary.csv`)

| Year | Source Year | Cleaned Reviews | Positive % | Neutral % | Negative % | Avg Rating |
|---|---|---:|---:|---:|---:|---:|
| **2023** | 2020 | 219,790 | 79.20% | 6.41% | 14.39% | 4.24 |
| **2024** | 2021 | 259,807 | 76.00% | 6.93% | 17.07% | 4.13 |
| **2025** | 2022 | 309,846 | 74.34% | 7.22% | 18.44% | 4.07 |
| **2026** | 2023 (Jan–Sep) | 349,821 | 75.59% | 6.97% | 17.45% | 4.10 |
| **All** | 2020–2023 | **1,139,264** | **76.04%** | **6.92%** | **17.04%** | **4.12** |

---

## 5-Stage / 15-Phase System Architecture

| Stage | Phases | Description | Key Files |
|---|---|---|---|
| **A. Data Engineering** | 1–4 | Inspect raw JSONL files, reservoir-sample 4 yearly datasets, join `meta_*.jsonl` titles/brands/prices/features, validate integrity rules | `scripts/inspect_dataset.py`, `scripts/build_yearly_datasets.py`, `scripts/add_product_info.py`, `scripts/validate_datasets.py` |
| **B. Databricks Lakehouse** | 5–6 | Upload Parquet datasets to Unity Catalog Volumes, write Delta tables (`reviews_all`, `products`), regex clean text & label (`reviews_clean`) | `Data_Bricks(files)/01_setup_and_load.ipynb`, `Data_Bricks(files)/02_clean_and_label.ipynb` |
| **C. NLP & ML Training** | 7–8 | Deterministic 80/20 `review_id` hash split (`911,272` train / `227,992` test), negation-preserving stop-word removal, TF-IDF + class-weighted classifiers | `Data_Bricks(files)/03_train_models.py`, `scripts/build_local_gold.py` |
| **D. Gold Analytics, Advisor & Streaming** | 9–12 | Historical Gold tables, lift-filtered (`> 1.3`) keywords, 9 complaint themes, 7-factor BUY/DON'T BUY engine, Spark Structured Streaming micro-batches | `Data_Bricks(files)/04_historical_analytics.py`, `05_search_and_recommend.py`, `06_streaming_reviews.py` |
| **E. Web Platform** | 13–15 | 6-view Midnight Carbon Streamlit + Plotly website with top navbar, side-by-side radar comparison, live predictor, stream monitor & 16-run scalability benchmark | `dashboard/app.py`, `dashboard/utils.py`, `dashboard/views/*.py` |

---

## ML Model Benchmark (`results/model_metrics.csv`)

Evaluated on a deterministic 80/20 `review_id` hash split (**911,272** training reviews / **227,992** held-out test reviews) using TF-IDF (`max_features=80,000`, `ngram_range=(1, 2)`, `min_df=5`, balanced class weights):

| Model | Status | Accuracy | Macro F1 | Weighted F1 | Negative Recall | Neutral Recall | Positive Recall | Train Rows |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Logistic Regression** | **Selected** | 79.97% | **0.6601** | 0.8323 | **74.34%** | 61.49% | 82.93% | 911,272 |
| **Naive Bayes** | Baseline | **80.05%** | 0.6597 | **0.8335** | 73.87% | **62.73%** | **83.02%** | 911,272 |
| **Random Forest** | Baseline | 77.28% | 0.6025 | 0.7965 | 67.78% | 42.52% | 82.59% | 150,000 |

---

## Explainable BUY / DON'T BUY Scoring Formula

Every product (`parent_asin`) receives a `0–100` score combining 7 weighted signals:
- **20%** Rating-based Positive % (`4–5` stars)
- **15%** ML-Predicted Positive % (Logistic Regression)
- **20%** Average Rating scaled (`(avg_rating - 1) / 4 * 100`)
- **15%** Recent-Year Positive %
- **10%** Year-over-Year Sentiment Trend (`recent_year - first_year`)
- **10%** Cross-Year Consistency (`100 - 2 * std`)
- **10%** Complaint Inversion (`100 - negative_pct`)

**Decision Rules:**
- `< 20` reviews → **`INSUFFICIENT DATA`** (`539,202` products)
- `>= 20` reviews & score `>= 65` → **`BUY`** (`3,018` products)
- `>= 20` reviews & score `< 65` → **`DON'T BUY`** (`1,276` products)

---

## 16-Run Scalability Benchmark (`results/scalability_benchmark_16_runs.csv`)

Measured across 4 dataset scales (`10,000`, `100,000`, `250,000`, `500,000` reviews) × 4 partition counts (`2`, `4`, `8`, `16`):
- **500,000 reviews @ 2 partitions:** `77.68s` (`6,436.3 rev/s`)
- **500,000 reviews @ 4 partitions:** `70.04s` (`7,139.2 rev/s`)
- **500,000 reviews @ 8 partitions:** `81.32s` (`6,148.3 rev/s`)
- **500,000 reviews @ 16 partitions:** **`55.44s`** (**`9,019.1 rev/s`** — fastest 500K runtime; peak overall `9,039.9 rev/s` at 10K @ 2 partitions)

---

## Website Views (`dashboard/views/`)

1. **Home (`home.py`):** Website hero, instant product buying search, live scale KPIs, high-volume `BUY` vs. `DON'T BUY` catalog highlights, and module directory.
2. **Market Analytics (`historical.py`):** Year & category filters, 2-row Avg Rating + Sentiment Mix chart, grouped/heatmap category comparison, 1–5 star rating distribution, monthly trends, word lift butterfly chart, complaint heatmap, brand volume-vs-sentiment scatter, and 15-phase architecture view.
3. **Buying Advisor (`search.py`):** Search 543,496 products by title, brand, category, verdict, confidence, and max price; inspect BUY Score gauge, 7-factor breakdown, aspect sentiment, complaints, lazy-loaded product specifications, verified customer review quotes, and **Side-by-Side 6-Axis Radar Comparison**.
4. **ML & Review Checker (`ml_models.py`):** Model comparison table & radar, row-normalized/raw confusion matrix, TF-IDF feature polarity butterfly, **Compare Both** live review analyzer with token-level attribution, and **Batch CSV Scorer**.
5. **Live Stream (`streaming.py`):** Micro-batch stream simulator over 5,000 held-out 2026 reviews with live positive pulse gauge, rolling sentiment chart, `NEGATIVE ALERT` triage cards, and CSV log export.
6. **Scalability & Scope (`scalability.py`):** 16-run runtime curves, throughput & relative speedup heatmaps, stage time breakdown (`Seconds` vs. `%`), and academic limitations & future roadmap.

---

## Repository Structure

```text
Product-Review-BigData-Analytics/
├── .streamlit/config.toml                      # Midnight Carbon dark theme
├── requirements.txt                            # Python dependencies (local & Streamlit Cloud)
├── requirements_dashboard.txt                  # Dashboard dependency list
├── run_dashboard.bat / run_dashboard.ps1       # Local & same-Wi-Fi launchers
├── share_public_link.bat                       # Public HTTPS tunnel launcher (Cloudflare)
├── Data_Bricks(files)/                         # Databricks notebooks 01 through 06
├── scripts/                                    # Data prep & build_local_gold.py
├── models/                                     # Fitted TF-IDF, Logistic Regression & Naive Bayes (.joblib)
├── results/                                    # Exported CSV tables for report & slides
├── data/gold/
│   ├── *.parquet                               # Aggregated Gold tables & stream pool
│   └── product_recommendations_shards/         # 4 zstd Parquet shards (<49 MB each, 543,496 products)
└── dashboard/
    ├── app.py                                  # Top-navbar st.navigation router
    ├── utils.py                                # Theme, lazy shard loader & ML inference
    └── views/                                  # home.py, historical.py, search.py, ml_models.py, streaming.py, scalability.py
```

---

## Quick Start

### 1. Run Locally
```bash
git clone https://github.com/Lakshya-dot/Product-Review-BigData-Analytics.git
cd Product-Review-BigData-Analytics
pip install -r requirements.txt
streamlit run dashboard/app.py
```
On Windows with the included launchers:
- **Local / Same-Wi-Fi:** Double-click `run_dashboard.bat` and open `http://localhost:8501`.
- **Public Shareable Link:** Double-click `share_public_link.bat` (requires `tools\cloudflared.exe`) to generate an instant `https://*.trycloudflare.com` URL.

### 2. Deploy 24/7 on Streamlit Community Cloud
1. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub (`Lakshya-dot`).
2. Select repository **`Lakshya-dot/Product-Review-BigData-Analytics`**, branch **`main`**, and main file path **`dashboard/app.py`**.
3. Click **Deploy**. All Gold tables, sharded product recommendations, and fitted ML models are already included in the repo.

---

## Limitations & Future Enhancements

- **Proxy Labels:** Labels are derived from 1–5 star ratings rather than manual sentiment annotation.
- **Sarcasm & Context:** Bag-of-words TF-IDF struggles with subtle sarcasm; upgrading to fine-tuned transformers (DistilBERT/RoBERTa) is planned.
- **Compute Environment:** Built on Databricks Free Edition + single-node workstation; deploying on a multi-node Spark cluster with Kafka ingestion is the natural production extension.
