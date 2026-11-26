# ReviewSense Analytics — Scalable Product Review Sentiment & Buying Intelligence

**Course:** 23RAI351 – Big Data Analytics  
**Faculty:** Dr. Ravi Kumar V.  
**Team No. 1:** Sreenidhi and Lakshya Yarlagadda  
**Dataset:** Amazon Reviews 2023 (McAuley Lab) — 1,140,000 genuine reviews (1,139,264 cleaned) across 6 categories & 543,496 unique products (`parent_asin`), date-shifted +3 years to **2023–2026**.

---

## Complete Project Architecture (Phases 1–15)

| Stage | Phases | Description | Location |
|---|---|---|---|
| **A. Data Engineering** | 1–4 | Inspect raw files, sample 4 yearly Parquet datasets (`2023–2026`), join `meta_*.jsonl` product info, validate | [scripts/](file:///E:/Projects/Product_Review_BigData/scripts/) (`inspect_dataset.py`, `build_yearly_datasets.py`, `add_product_info.py`, `validate_datasets.py`) |
| **B. Databricks Lakehouse** | 5–6 | Upload to Unity Catalog Volumes, write Delta tables, clean text & assign proxy labels (`1-2 Neg`, `3 Neu`, `4-5 Pos`) | [01_setup_and_load.ipynb](file:///E:/Projects/Product_Review_BigData/Data_Bricks(files)/01_setup_and_load.ipynb), [02_clean_and_label.ipynb](file:///E:/Projects/Product_Review_BigData/Data_Bricks(files)/02_clean_and_label.ipynb) |
| **C. PySpark MLlib Training** | 7–8 | Deterministic 80/20 `review_id` hash split, TF-IDF vectorization, class-weighted Naive Bayes, Logistic Regression & Random Forest | [03_train_models.py](file:///E:/Projects/Product_Review_BigData/Data_Bricks(files)/03_train_models.py) |
| **D. Analytics, Search & Streaming** | 9–12 | Gold tables, lift word analysis, complaint themes, 7-factor BUY / DON'T BUY engine, Structured Streaming, & local Gold pipeline | [04_historical_analytics.py](file:///E:/Projects/Product_Review_BigData/Data_Bricks(files)/04_historical_analytics.py), [05_search_and_recommend.py](file:///E:/Projects/Product_Review_BigData/Data_Bricks(files)/05_search_and_recommend.py), [06_streaming_reviews.py](file:///E:/Projects/Product_Review_BigData/Data_Bricks(files)/06_streaming_reviews.py), [build_local_gold.py](file:///E:/Projects/Product_Review_BigData/scripts/build_local_gold.py) |
| **E. Executive Dashboard** | 13–15 | 5-page Midnight Carbon Streamlit + Plotly platform (Overview, Buying Advisor, ML & Predictor, Live Stream, Scalability) | [dashboard/app.py](file:///E:/Projects/Product_Review_BigData/dashboard/app.py) & `dashboard/views/*.py` |

---

## Folder Structure

```text
E:\Projects\Product_Review_BigData\
├── .streamlit\config.toml          # Midnight Carbon dark theme & server port 8501
├── run_dashboard.bat               # Double-click Windows launcher
├── run_dashboard.ps1               # PowerShell launcher
├── requirements_dashboard.txt      # Dashboard & ML packages
├── data\
│   ├── processed\                  # Cleaned yearly Parquet files + products.parquet
│   └── gold\                       # 16 precomputed Gold analytics & recommendation tables
├── models\                         # Fitted TF-IDF vectorizer, Logistic Regression & Naive Bayes (.joblib)
├── results\                        # Exported CSV tables for final report & presentation
├── Data_Bricks(files)\             # Databricks notebooks 01 through 06
├── scripts\                        # Data prep + build_local_gold.py
└── dashboard\
    ├── app.py                      # st.navigation router
    ├── utils.py                    # Shared styling, loaders & ML inference
    └── views\
        ├── historical.py           # Page 1: Overview & Market Analytics
        ├── search.py               # Page 2: Product Buying Advisor & Radar Comparison
        ├── ml_models.py            # Page 3: ML Benchmark, Feature Butterfly & Predictor
        ├── streaming.py            # Page 4: Live Review Stream Telemetry (Phase 12)
        └── scalability.py          # Page 5: 16-Run Scalability Benchmark & Scope
```

---

## How to Run & Share on Demo Day

1. **Local / Same-Wi-Fi Launch:** Double-click `run_dashboard.bat` (or run `.\run_dashboard.ps1`), then open **http://localhost:8501**.
2. **Share With Teammates Anywhere (Public HTTPS Link):** Double-click `share_public_link.bat` — it starts the Streamlit server and creates an instant `https://*.trycloudflare.com` URL using `tools\cloudflared.exe`.
2. **Rebuild Gold Tables & Models (Optional):**
   ```powershell
   C:\Users\laksh\AppData\Local\Programs\Python\Python312\python.exe -u scripts\build_local_gold.py --force
   ```
