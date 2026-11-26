# 23RAI351 End-Semester Viva & Live Demo Guide

**Project:** ReviewSense Analytics — Scalable Product Review Sentiment & Buying Intelligence  
**Team No. 1:** Sreenidhi and Lakshya Yarlagadda  
**Faculty Guide:** Dr. Ravi Kumar V.  

---

## 1. Live Demo Flow (4-Minute End-to-End Walkthrough)

### Part A: Databricks Lakehouse & Spark Pipeline (First 2 Minutes — Your Primary Evaluation Surface)
1. **Show Unity Catalog Data Architecture (`Catalog -> workspace -> review_analytics`):**
   - Explain the **Bronze -> Silver -> Gold** medallion layout:
     - `/Volumes/workspace/review_analytics/raw_files` (5 uploaded Parquet datasets)
     - Bronze (`reviews_all`: `1,140,000` rows, `products`: `543,748` rows)
     - Silver (`reviews_clean`: `1,139,264` rows after deduplication & validation)
     - Gold (`reviews_scored`, `agg_*`, `product_recommendations`, `stream_scored`)
2. **Walk Through PySpark MLlib & Structured Streaming (`03_train_models` & `06_streaming_reviews`):**
   - Point out distributed TF-IDF (`HashingTF` + `IDF`) + class-weighted `LogisticRegression` training on `912,260` rows in `27.71s` (`82.53%` accuracy, `0.6644` Macro F1, `0.8466` Weighted F1).
   - Point out `06_streaming_reviews` reading streaming batches with `readStream` and writing alerts to Delta table `stream_scored` (`999` streamed rows + negative alerts).
3. **Present Interactive Databricks Dashboard (`07_databricks_dashboard` -> Focus View):**
   - Show the **4 Interactive Filter Widgets** (`1. Year`, `2. Category`, `3. Product Search`, `4. Verdict Filter`).
   - Change `2. Category` from `All` -> `Electronics` or search a different keyword in `3. Product Search`, click **Update**, and show the KPIs (`1,139,264` total, `76.0%` positive, `4.12 / 5.0` avg rating) and `BUY` / `DON'T BUY` product table refreshing live from Delta Lake.

### Part B: ReviewSense Web Application & Scalability (Next 2 Minutes)
1. **Open Streamlit App (`http://localhost:8501`):**
   - Highlight the **Top Navigation Bar** across **Home**, **Market Analytics**, **Buying Advisor**, **ML & Review Checker**, **Live Stream**, and **Scalability & Scope**.
   - Search **`Echo Dot`** in **Buying Advisor** (`BUY`, confidence, 7-factor score breakdown, complaint themes, and **Side-by-Side Comparison Radar**).
   - Test a custom review in **ML & Review Checker -> Compare Both** to show positive/negative token contributions (`not`, `broken`, `great`, etc.).
   - Show the **16-Run Scalability Benchmark** chart peaking at **`9,019.1 rev/s`** (`55.44s`) for `500,000` reviews at `16` partitions.

---

## 2. Quick Metric Reference Sheet (Memorize These Numbers)

| Item | Verified Value |
|---|---|
| Raw / Cleaned Reviews | `1,140,000` raw -> **`1,139,264`** cleaned (`736` dropped) |
| Unique Products (`parent_asin`) | **`543,496`** recommendations (`543,748` raw catalog entries) |
| Overall Sentiment Split | **`76.04%`** Positive / **`6.92%`** Neutral / **`17.04%`** Negative |
| Overall Average Star Rating | **`4.12 / 5.0`** (`2023`: `4.24` -> `2025`: `4.07` -> `2026`: `4.10`) |
| Best Databricks MLlib Model | **`logistic_regression`** (`82.53%` Acc, `0.6644` Macro F1, `0.8466` Weighted F1, `27.71s`) |
| Best Local TF-IDF Model | **`Logistic Regression`** (`79.97%` Acc, **`0.6601`** Macro F1, `74.34%` Neg Recall) |
| Recommendation Verdicts (Databricks) | **`BUY`: `3,074` \| `DON'T BUY`: `1,220` \| `INSUFFICIENT DATA`: `539,202`** |
| Recommendation Verdicts (Local Gold) | **`BUY`: `3,018` \| `DON'T BUY`: `1,276` \| `INSUFFICIENT DATA`: `539,202`** |
| Structured Streaming Output | **`999`** micro-batch scored rows across 6 categories (`stream_scored`) |
| Peak 500K Throughput | **`9,019.1 rev/s`** (`55.44s` at `16` partitions) |

---

## 3. Anticipated Viva Questions & Crisp Answers

1. **Q: Why did you map 2020–2023 reviews forward to 2023–2026?**  
   *A:* McAuley Lab's Amazon Reviews 2023 ends around September 2023. Shifting timestamps forward by exactly `+3 years` keeps **100% genuine review text, star ratings, and seasonal patterns** (`is_synthetic = False`) while letting us demonstrate a contemporary 2023–2026 longitudinal pipeline.

2. **Q: Why use Apache Spark + Delta Lake instead of plain Pandas?**  
   *A:* Raw JSONL files exceed single-machine Pandas memory limits cleanly without chunking/columnar pruning. Spark + Delta Lake gives distributed parallel transforms across partitions, Unity Catalog table governance, ACID updates/overwrites, MLlib distributed training (`912,260` rows in `27.71s`), and unified batch + streaming writes into Delta.

3. **Q: Why did you choose Logistic Regression over Naive Bayes even when local Naive Bayes had slightly higher raw accuracy (`80.05%` vs `79.97%`)?**  
   *A:* Reviews are heavily class-imbalanced (`76.04%` Positive vs. `6.92%` Neutral and `17.04%` Negative). Raw accuracy favors the majority class. We selected **`Logistic Regression`** on **Macro F1** (`0.6601` locally; **`0.6644` Macro F1 & `82.53%` Acc** on Databricks Spark MLlib) plus stronger negative-class recall (`74.34%`–`74.64%`), which matters most for catching defective products and complaint alerts.

4. **Q: Why do you keep `"not"`, `"no"`, `"never"` during stop-word removal?**  
   *A:* Standard English stop-word lists strip negation words, flipping *"not good"* into *"good"*! Preserving negation tokens with bigrams (`ngram_range=(1, 2)`) lets TF-IDF + Logistic Regression learn negative polarity for *"not work"*, *"not worth"*, etc.

5. **Q: Why do you group products by `parent_asin` instead of `asin`?**  
   *A:* On Amazon, color/size variations of the same product share a `parent_asin` but have distinct child `asin`s. Grouping by `parent_asin` consolidates all customer feedback for a single model cleanly.

6. **Q: Why does a product need `>= 20` reviews before getting `BUY` or `DON'T BUY`?**  
   *A:* Statistical reliability — products with 1–5 reviews suffer from extreme variance (`539,202` long-tail products fall under `INSUFFICIENT DATA`), so the threshold protects buyers against fluke 5.0 scores on 2 reviews.

7. **Q: Explain how the BUY Score (`0–100`) works.**  
   *A:* Seven interpretable weighted signals: `20%` positive rating share, `15%` ML-predicted positive share, `20%` scaled rating (`(avg-1)/4*100`), `15%` recent-year positivity, `10%` multi-year trend, `10%` cross-year consistency, and `10%` inverted negative rate (`BUY >= 65`).

8. **Q: Why did Positive distinctive words show smaller Lift (`~1.3x`) than Negative words (`~2.0x+`) in the word chart?**  
   *A:* Mathematical upper bound! Since `76%` of all reviews are Positive, the maximum possible lift `P(pos|word) / P(pos)` for any word is `1 / 0.76 ≈ 1.32x`. By contrast, Negative reviews are `17%` of the corpus, allowing Negative lift up to `1 / 0.17 ≈ 5.88x` (e.g., `"not"` appears `221K+` times with ~`2.09x` lift).

9. **Q: How did partition count affect your 16-run Scalability Benchmark?**  
   *A:* At small scale (`10,000` reviews), extra partitions (`8` or `16`) add scheduling overhead, so `2` partitions is fastest (`1.106s`, `9,039.9 rev/s`). At large scale (`500,000` reviews), parallelism dominates: scaling from `2` to `16` partitions cuts total runtime from `77.684s` down to **`55.438s`** (**`9,019.1 rev/s`**).

10. **Q: How does the Streamlit site interact with Databricks without breaking offline?**  
    *A:* `dashboard/utils.py` checks for `DATABRICKS_HOST`, `DATABRICKS_HTTP_PATH`, and `DATABRICKS_TOKEN`. When configured, it queries `workspace.review_analytics.*` live via `databricks-sql-connector`; when unset (e.g. classroom Wi-Fi or Streamlit Cloud), it falls back transparently to zstd-compressed Gold Parquet shards (`<49 MB` each).
