# Databricks Dashboard & Live Connection Guide

This guide explains how to present the **complete project inside Databricks** (so your faculty can inspect both your PySpark Delta tables and a native Databricks dashboard) while keeping your standalone **ReviewSense Streamlit website** working too.

---

## Step 1: Finish Running Phases 10–12 in Databricks

You already ran notebooks `01`–`04` in Databricks (`workspace.review_analytics`). Now import and run these two notebooks in your Databricks Workspace:

1. **`05_search_and_recommend.py`** (Phases 10–11)
   - Creates `product_stats`, `product_aspect_sentiment`, `product_complaints`, and `product_recommendations`.
2. **`06_streaming_reviews.py`** (Phase 12)
   - Simulates Spark Structured Streaming over `reviews_2026` and creates `stream_scored`.

---

## Step 2: Option A — Interactive Databricks Notebook Dashboard (Fastest, 2 Minutes)

1. Import **`Data_Bricks(files)/07_databricks_dashboard.py`** into your Databricks Workspace.
2. Attach compute and click **Run all**.
3. Use the 4 **Widgets** at the top (`1. Year`, `2. Category`, `3. Product Search`, `4. Verdict Filter`) to filter live.
4. Click **View → Dashboard** (in the Databricks notebook toolbar) to switch into a clean **Databricks Dashboard** layout for your viva demo!

---

## Step 3: Option B — Native Databricks AI/BI Dashboard (Lakeview)

For a drag-and-drop SQL dashboard inside Databricks:

1. In Databricks left menu, click **Dashboards → Create dashboard** and name it `ReviewSense Big Data Analytics`.
2. Go to the **Data** tab and paste queries `1` through `9` from **`Data_Bricks(files)/08_databricks_sql_queries.sql`** as datasets (`year_summary`, `year_category`, `monthly_trend`, `rating_dist`, `complaints`, `brands`, `model_metrics`, `product_advisor`, `live_stream`).
3. Switch to the **Canvas** tab and add widgets:
   - **Counters (from `year_summary`, filtered `year = 'All'`):** `reviews`, `positive_pct`, `avg_rating`, `products`
   - **Stacked Bar Chart (`year_summary`):** X = `year`, Y = `positive_pct / neutral_pct / negative_pct`
   - **Grouped Bar Chart (`year_category`):** X = `category`, Y = `positive_pct`, Group = `year`
   - **Line Chart (`monthly_trend`):** X = `review_month`, Y = `positive_pct`
   - **Table (`product_advisor`):** Show `product_title`, `brand`, `reviews`, `avg_rating`, `buy_score`, `recommendation`, `reason_summary`
   - **Bar Chart (`model_metrics`):** X = `model`, Y = `macro_f1` and `accuracy_pct`
   - **Global Filter Widget:** Filter by `category` and `year` across datasets.
4. Click **Publish** to share a native Databricks Dashboard link with your faculty!

---

## Step 4 (Optional): Link Your Streamlit Website Directly to Databricks

Your Streamlit website (`dashboard/app.py`) reads local Gold Parquet shards by default so it never breaks offline. If you want it to query **live Databricks Unity Catalog Delta tables** via SQL Warehouse instead:

1. Install the optional Databricks SQL connector:
   ```bash
   pip install databricks-sql-connector
   ```
2. Set these 3 environment variables (from your Databricks SQL Warehouse → **Connection details** and User Settings → **Access tokens**) before launching `streamlit run dashboard/app.py`:
   ```powershell
   $env:DATABRICKS_HOST = "<your-workspace-host>.cloud.databricks.com"
   $env:DATABRICKS_HTTP_PATH = "/sql/1.0/warehouses/<warehouse-id>"
   $env:DATABRICKS_TOKEN = "<your-personal-access-token>"
   ```
3. When those 3 variables are set, `dashboard/utils.py` automatically switches to live `SELECT * FROM workspace.review_analytics.<table>` queries! When unset, it uses the fast local Gold Parquet tables.
