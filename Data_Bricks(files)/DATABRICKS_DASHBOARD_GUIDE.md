# Databricks Execution & Dashboard Guide (`workspace.review_analytics`)

This guide documents the complete Databricks workflow for **Team No. 1** (`23RAI351` Big Data Analytics), including all **8 Databricks assets**, verified execution outputs on **Databricks Free Edition Serverless compute**, interactive dashboard demo steps, and optional SQL Warehouse streaming into the Streamlit website.

---

## 1. Completed Databricks Pipeline (`01` through `07`)

All notebooks reside under `/Users/lakshyayarlagadda2008@gmail.com/Review_Sentiment_BigData` and operate on Unity Catalog schema **`workspace.review_analytics`**:

1. **`01_setup_and_load.ipynb` (Phase 5 — Bronze Ingestion)**
   - Creates `workspace.review_analytics` and volume `/Volumes/workspace/review_analytics/raw_files`.
   - Loads uploaded Parquet files into Delta tables: `reviews_2023` (`220,000`), `reviews_2024` (`260,000`), `reviews_2025` (`310,000`), `reviews_2026` (`350,000`), `reviews_all` (`1,140,000`), and `products` (`543,748`).
2. **`02_clean_and_label.ipynb` (Phase 6 — Silver ETL)**
   - Cleans review text, normalizes fields, deduplicates, and derives sentiment proxy labels (`1–2` Neg, `3` Neu, `4–5` Pos) -> `reviews_clean` (`1,139,264` rows).
3. **`03_train_models.py` (Phases 7–8 — Distributed PySpark MLlib)**
   - Deterministic `review_id` split (`912,260` train rows); trains balanced TF-IDF (`HashingTF` + `IDF`) classifiers:
     - **`logistic_regression` (Best):** `82.53%` accuracy, `0.6644` Macro F1, `0.8466` Weighted F1 (`27.71s`).
     - **`naive_bayes`:** `77.69%` accuracy, `0.6174` Macro F1.
     - **`random_forest`:** `75.46%` accuracy, `0.5866` Macro F1.
   - Persists best pipeline to `/Volumes/workspace/review_analytics/models/best_sentiment_model` and metrics to `workspace.review_analytics.model_metrics`.
4. **`04_historical_analytics.py` (Phase 9 — Gold Summary Tables)**
   - Scores all `1,139,264` reviews (`reviews_scored`) and builds `agg_year_summary` (`5`), `agg_year_category` (`24`), `agg_monthly` (`258`), `agg_brand` (`3,955`), `agg_product_year` (`709,300`), `agg_top_words` (`750`), and `agg_complaints` (`216`).
5. **`05_search_and_recommend.py` (Phases 10–11 — BUY / DON'T BUY Advisor)**
   - Joins aspect sentiment and complaints with weighted 7-factor scoring across `543,496` products into `product_recommendations` (`BUY`: `3,074`, `DON'T BUY`: `1,220`, `INSUFFICIENT DATA`: `539,202`).
6. **`06_streaming_reviews.py` (Phase 12 — Spark Structured Streaming)**
   - Simulates micro-batch ingestion over 2026 JSON batches using the saved PySpark MLlib model, writing `999` scored records and negative review alerts into `stream_scored`.
7. **`07_databricks_dashboard.py` (Phases 13–14 — Interactive Databricks Native Dashboard)**
   - Interactive 13-cell dashboard with 4 live filter widgets (`1. Year`, `2. Category`, `3. Product Search`, `4. Verdict Filter`), 8 Midnight Carbon Matplotlib PNG visuals, and 4 interactive `display()` Delta grids designed for zero-error **Focus/Present** presentation.
8. **`08_databricks_sql_queries.sql` (Optional AI/BI SQL Datasets)**
   - 9 ready-to-paste Spark SQL queries targeting `workspace.review_analytics.*` for Databricks SQL Editor or AI/BI Lakeview Dashboards.

---

## 2. How to Use the Databricks Dashboard During Presentation

1. Open **`07_databricks_dashboard`** in Databricks and switch to **Present** mode (`View → Dashboard → Review Sentiment Analytics Dashboard`).
2. Use the 4 filter widgets at the top of the canvas:
   - **`1. Year`**: `All`, `2023`, `2024`, `2025`, `2026`
   - **`2. Category`**: `All`, `Electronics`, `Home and Kitchen`, `Health and Household`, `Beauty and Personal Care`, `Sports and Outdoors`, `Books`
   - **`3. Product Search`**: any keyword (default: `echo dot`, e.g. `anker`, `headphone`, `blender`)
   - **`4. Verdict Filter`**: `All`, `BUY`, `DON'T BUY`, `INSUFFICIENT DATA`
3. Click the blue **Update** button in the top-right corner to execute Spark queries across Delta tables and refresh every visual & table tile.

---

## 3. Optional: Build a Drag-and-Drop Databricks AI/BI Dashboard (SQL Editor)

If you also want a standalone Lakeview SQL dashboard under **Analytics → Dashboards**:
1. Click **Dashboards → Create dashboard**.
2. In the **Data** tab, paste queries `1`–`9` from `Data_Bricks(files)/08_databricks_sql_queries.sql`.
3. In the **Canvas** tab, bind counters (`year_summary`), stacked/grouped bars (`year_category`), line trends (`monthly_trend`), model benchmark bars (`model_metrics`), searchable table (`product_advisor`), and streaming alert feed (`live_stream`).

---

## 4. Optional: Connect the Streamlit Website to Live Databricks SQL Warehouse

Your Streamlit app (`dashboard/app.py`) reads local Gold Parquet shards by default so it runs offline with zero latency. To query **live Databricks Unity Catalog Delta tables** instead:

1. Ensure the optional connector is installed:
   ```bash
   pip install databricks-sql-connector
   ```
2. Set the 3 workspace environment variables (from **SQL Warehouses → Connection details** and **User Settings → Access tokens**):
   ```powershell
   $env:DATABRICKS_HOST = "dbc-ce6da86c-6085.cloud.databricks.com"
   $env:DATABRICKS_HTTP_PATH = "/sql/1.0/warehouses/<warehouse-id>"
   $env:DATABRICKS_TOKEN = "<your-personal-access-token>"
   ```
3. Launch `streamlit run dashboard/app.py`. When those 3 variables are present, `dashboard/utils.py` queries `SELECT * FROM workspace.review_analytics.<table>` live; otherwise it automatically falls back to local Gold Parquet.
