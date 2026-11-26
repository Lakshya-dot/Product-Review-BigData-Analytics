r"""
build_local_gold.py
Local Phase 7-12 Pipeline on the 1.14M Real Amazon Reviews (2023-2026)
----------------------------------------------------------------------
Reads:
  data/processed/reviews_{2023,2024,2025,2026}.parquet
  data/processed/products.parquet

Produces:
  models/tfidf_vectorizer.joblib
  models/sentiment_lr.joblib
  models/sentiment_nb.joblib
  data/gold/agg_year_summary.parquet
  data/gold/agg_year_category.parquet
  data/gold/agg_monthly.parquet
  data/gold/agg_brand.parquet
  data/gold/agg_top_words.parquet
  data/gold/agg_complaints.parquet
  data/gold/model_metrics.parquet
  data/gold/confusion_matrices.parquet
  data/gold/top_lr_features.parquet
  data/gold/scalability_results.parquet
  data/gold/product_recommendations.parquet
  data/gold/product_aspect_sentiment.parquet
  data/gold/product_complaints.parquet
  data/gold/product_sample_reviews.parquet
  data/gold/stream_sample_pool.parquet

Usage (from E:\Projects\Product_Review_BigData):
  python scripts/build_local_gold.py
"""

from __future__ import annotations

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.class_weight import compute_sample_weight

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
GOLD_DIR = os.path.join(BASE_DIR, "data", "gold")
MODEL_DIR = os.path.join(BASE_DIR, "models")

YEARS = [2023, 2024, 2025, 2026]
LABEL_NAMES = {0: "Negative", 1: "Neutral", 2: "Positive"}

STOP_WORDS = """a an the and or but if then so of to in on at by for with from as is are was
were be been being am it its it's this that these those i me my we our you your he she him her
they them their what which who whom there here when where why how all any both each few more most
other some such only own same than too very can will just do does did doing have has had having
would should could also get got one would about into over after before up down out off again
further once s t ve ll re d m""".split()

COMPLAINT_THEMES = {
    "Quality / durability": ["broke", "broken", "cheap", "flimsy", "stopped", "defective", "quality", "fell", "cracked", "ripped"],
    "Does not work":        ["work", "working", "worked", "dead", "useless", "failed", "malfunction"],
    "Size / fit":           ["size", "small", "large", "fit", "tight", "loose", "tiny", "big"],
    "Price / value":        ["price", "money", "expensive", "overpriced", "waste", "worth", "cost"],
    "Delivery / packaging": ["shipping", "delivery", "arrived", "package", "packaging", "damaged", "box", "late"],
    "Not as described":     ["description", "described", "picture", "advertised", "fake", "misleading", "different"],
    "Return / refund":      ["return", "returned", "returning", "refund", "replacement", "exchange"],
    "Smell / taste / skin": ["smell", "smells", "taste", "rash", "itchy", "irritation", "sticky"],
    "Battery / charging":   ["battery", "charge", "charging", "charger"],
}

ASPECT_KEYWORDS = {
    "Battery & Charging":   ["battery", "charge", "charging", "charger", "power"],
    "Sound & Audio":        ["sound", "audio", "volume", "bass", "speaker", "microphone"],
    "Build Quality":        ["quality", "sturdy", "durable", "material", "solid", "flimsy", "broke", "cheap"],
    "Comfort & Fit":        ["comfort", "comfortable", "fit", "size", "soft", "tight", "lightweight"],
    "Ease of Use":          ["easy", "simple", "setup", "install", "intuitive", "instructions"],
    "Value for Money":      ["price", "value", "worth", "money", "affordable", "expensive", "overpriced"],
    "Packaging & Delivery": ["package", "packaging", "shipping", "delivery", "box", "arrived", "damaged"],
    "Design & Look":        ["design", "look", "looks", "color", "style", "cute", "sleek"],
}


def clean_text_expr(col_expr: pl.Expr) -> pl.Expr:
    return (
        col_expr.str.to_lowercase()
        .str.replace_all(r"<[^>]+>", " ")
        .str.replace_all(r"(https?://\S+|www\.\S+)", " ")
        .str.replace_all(r"n't\b", " not")
        .str.replace_all(r"[^a-z\s]", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def load_and_clean_reviews() -> pl.DataFrame:
    t0 = time.time()
    frames = []
    for y in YEARS:
        p = os.path.join(PROCESSED_DIR, f"reviews_{y}.parquet")
        df = pl.read_parquet(
            p,
            columns=[
                "review_id", "year", "review_date", "category", "rating",
                "title", "text", "parent_asin", "helpful_vote",
                "verified_purchase", "product_title", "brand", "price",
                "product_avg_rating", "product_rating_count", "product_features",
            ],
        )
        frames.append(df)
    raw = pl.concat(frames, how="vertical")
    print(f"Loaded {raw.height:,} raw reviews in {time.time() - t0:.1f}s")

    t1 = time.time()
    clean = (
        raw.with_columns(
            pl.concat_str(
                [pl.col("title").fill_null(""), pl.col("text").fill_null("")],
                separator=". ",
            ).alias("review_text")
        )
        .with_columns(clean_text_expr(pl.col("review_text")).alias("clean_text"))
        .filter(pl.col("clean_text").str.len_chars() > 0)
        .unique(subset=["review_id"], keep="first")
        .with_columns([
            pl.when(pl.col("rating") <= 2).then(pl.lit("Negative"))
              .when(pl.col("rating") == 3).then(pl.lit("Neutral"))
              .otherwise(pl.lit("Positive")).alias("sentiment"),
            pl.when(pl.col("rating") <= 2).then(0)
              .when(pl.col("rating") == 3).then(1)
              .otherwise(2).cast(pl.Int8).alias("label"),
            pl.col("review_date").str.slice(0, 7).alias("review_month"),
            pl.col("brand").fill_null("Unknown").alias("brand"),
            (pl.col("review_id").hash(seed=42) % 100 >= 80).alias("is_test"),
        ])
    )
    print(f"Cleaned & deduplicated to {clean.height:,} reviews in {time.time() - t1:.1f}s")
    return clean


def evaluate_predictions(name: str, y_true: np.ndarray, y_pred: np.ndarray,
                         train_rows: int, train_secs: float, test_secs: float):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    total = int(cm.sum())
    acc = float(np.trace(cm) / total)
    out = {"model": name, "accuracy": round(acc, 4)}
    f1s, supports = [], []
    for idx, short in [(0, "negative"), (1, "neutral"), (2, "positive")]:
        tp = int(cm[idx, idx])
        predicted = int(cm[:, idx].sum())
        actual = int(cm[idx, :].sum())
        prec = tp / predicted if predicted else 0.0
        rec = tp / actual if actual else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[f"precision_{short}"] = round(prec, 4)
        out[f"recall_{short}"] = round(rec, 4)
        out[f"f1_{short}"] = round(f1, 4)
        f1s.append(f1)
        supports.append(actual)
    out["macro_f1"] = round(sum(f1s) / 3.0, 4)
    out["weighted_f1"] = round(sum(f * s for f, s in zip(f1s, supports)) / total, 4)
    out["train_rows"] = int(train_rows)
    out["train_seconds"] = round(train_secs, 2)
    out["test_seconds"] = round(test_secs, 2)

    cm_rows = []
    for i in range(3):
        for j in range(3):
            cm_rows.append({
                "model": name,
                "actual": LABEL_NAMES[i],
                "predicted": LABEL_NAMES[j],
                "count": int(cm[i, j]),
            })
    return out, cm_rows


def train_and_score_models(df: pl.DataFrame, force: bool = False) -> pl.DataFrame:
    print("\n--- Phase 7-8: TF-IDF & Sentiment Model Training ---")
    vec_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
    lr_path = os.path.join(MODEL_DIR, "sentiment_lr.joblib")
    metrics_path = os.path.join(GOLD_DIR, "model_metrics.parquet")

    if not force and os.path.exists(vec_path) and os.path.exists(lr_path) and os.path.exists(metrics_path):
        print("Found cached trained models & model_metrics.parquet — loading cached Logistic Regression model.")
        vectorizer = joblib.load(vec_path)
        lr = joblib.load(lr_path)
    else:
        train_df = df.filter(~pl.col("is_test"))
        test_df = df.filter(pl.col("is_test"))
        print(f"Train split: {train_df.height:,} | Test split: {test_df.height:,}")

        X_train_text = train_df["clean_text"].to_list()
        y_train = train_df["label"].to_numpy()
        X_test_text = test_df["clean_text"].to_list()
        y_test = test_df["label"].to_numpy()

        t_vec = time.time()
        vectorizer = TfidfVectorizer(
            max_features=80_000,
            min_df=5,
            ngram_range=(1, 2),
            stop_words=STOP_WORDS,
            sublinear_tf=True,
            dtype=np.float32,
        )
        X_train = vectorizer.fit_transform(X_train_text)
        X_test = vectorizer.transform(X_test_text)
        vec_secs = time.time() - t_vec
        print(f"TF-IDF fitted in {vec_secs:.1f}s -> shape {X_train.shape}")

        sample_weights = compute_sample_weight("balanced", y_train)
        metrics_rows = []
        cm_all = []

        # 1. Multinomial Naive Bayes
        t0 = time.time()
        nb = MultinomialNB(alpha=1.0)
        nb.fit(X_train, y_train, sample_weight=sample_weights)
        nb_train_s = time.time() - t0 + vec_secs
        t1 = time.time()
        nb_pred = nb.predict(X_test)
        nb_test_s = time.time() - t1
        m_nb, cm_nb = evaluate_predictions("naive_bayes", y_test, nb_pred, len(y_train), nb_train_s, nb_test_s)
        metrics_rows.append(m_nb)
        cm_all.extend(cm_nb)
        print(f"Naive Bayes: acc={m_nb['accuracy']:.3f}, macro_f1={m_nb['macro_f1']:.3f} ({nb_train_s:.1f}s)")

        # 2. Logistic Regression
        t0 = time.time()
        lr = LogisticRegression(
            max_iter=80,
            C=1.5,
            class_weight="balanced",
            solver="lbfgs",
            tol=1e-3,
        )
        lr.fit(X_train, y_train)
        lr_train_s = time.time() - t0 + vec_secs
        t1 = time.time()
        lr_pred = lr.predict(X_test)
        lr_test_s = time.time() - t1
        m_lr, cm_lr = evaluate_predictions("logistic_regression", y_test, lr_pred, len(y_train), lr_train_s, lr_test_s)
        metrics_rows.append(m_lr)
        cm_all.extend(cm_lr)
        print(f"Logistic Regression: acc={m_lr['accuracy']:.3f}, macro_f1={m_lr['macro_f1']:.3f} ({lr_train_s:.1f}s)")

        # 3. Random Forest (on 150K sample & 10K features, matching 03_train_models.py)
        rng = np.random.default_rng(42)
        rf_n = min(150_000, len(X_train_text))
        rf_idx = rng.choice(len(X_train_text), size=rf_n, replace=False)
        rf_texts = [X_train_text[i] for i in rf_idx]
        rf_y = y_train[rf_idx]

        t0 = time.time()
        rf_vec = TfidfVectorizer(
            max_features=10_000,
            min_df=5,
            stop_words=STOP_WORDS,
            sublinear_tf=True,
            dtype=np.float32,
        )
        X_rf_train = rf_vec.fit_transform(rf_texts)
        X_rf_test = rf_vec.transform(X_test_text)
        rf = RandomForestClassifier(
            n_estimators=40,
            max_depth=14,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
        rf.fit(X_rf_train, rf_y)
        rf_train_s = time.time() - t0
        t1 = time.time()
        rf_pred = rf.predict(X_rf_test)
        rf_test_s = time.time() - t1
        m_rf, cm_rf = evaluate_predictions("random_forest", y_test, rf_pred, rf_n, rf_train_s, rf_test_s)
        metrics_rows.append(m_rf)
        cm_all.extend(cm_rf)
        print(f"Random Forest: acc={m_rf['accuracy']:.3f}, macro_f1={m_rf['macro_f1']:.3f} ({rf_train_s:.1f}s)")

        best_model = max(metrics_rows, key=lambda r: r["macro_f1"])["model"]
        for r in metrics_rows:
            r["is_best"] = r["model"] == best_model

        pl.DataFrame(metrics_rows).write_parquet(metrics_path)
        pl.DataFrame(cm_all).write_parquet(os.path.join(GOLD_DIR, "confusion_matrices.parquet"))

        # Top influential words from Logistic Regression (Positive vs Negative class weights)
        feature_names = np.array(vectorizer.get_feature_names_out())
        pos_weights = lr.coef_[2]
        neg_weights = lr.coef_[0]
        top_pos_idx = np.argsort(pos_weights)[-30:][::-1]
        top_neg_idx = np.argsort(neg_weights)[-30:][::-1]
        feat_rows = []
        for rank, i in enumerate(top_pos_idx, 1):
            feat_rows.append({"sentiment": "Positive", "rank": rank, "feature": str(feature_names[i]), "weight": round(float(pos_weights[i]), 3)})
        for rank, i in enumerate(top_neg_idx, 1):
            feat_rows.append({"sentiment": "Negative", "rank": rank, "feature": str(feature_names[i]), "weight": round(float(neg_weights[i]), 3)})
        pl.DataFrame(feat_rows).write_parquet(os.path.join(GOLD_DIR, "top_lr_features.parquet"))

        # Save models for Live Checker & Stream Simulator
        joblib.dump(vectorizer, vec_path, compress=3)
        joblib.dump(lr, lr_path, compress=3)
        joblib.dump(nb, os.path.join(MODEL_DIR, "sentiment_nb.joblib"), compress=3)
        print(f"Saved trained models (best={best_model}) to {MODEL_DIR}")

    # Score all 1.139M reviews in chunks to avoid peak memory spikes
    t_score = time.time()
    all_texts = df["clean_text"].to_list()
    n_total = len(all_texts)
    preds = np.empty(n_total, dtype=np.int8)
    pos_probs = np.empty(n_total, dtype=np.float32)
    chunk_size = 100_000
    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)
        X_chunk = vectorizer.transform(all_texts[start:end])
        preds[start:end] = lr.predict(X_chunk)
        pos_probs[start:end] = lr.predict_proba(X_chunk)[:, 2]
    print(f"Scored all {n_total:,} reviews in {time.time() - t_score:.1f}s")

    pred_series = pl.Series("pred_label", preds)
    prob_series = pl.Series("pred_pos_prob", pos_probs)
    df = df.with_columns([pred_series, prob_series]).with_columns(
        pl.when(pl.col("pred_label") == 0).then(pl.lit("Negative"))
          .when(pl.col("pred_label") == 1).then(pl.lit("Neutral"))
          .otherwise(pl.lit("Positive")).alias("predicted_sentiment")
    )
    return df


def run_scalability_benchmark(df: pl.DataFrame, force: bool = False):
    scal_path = os.path.join(GOLD_DIR, "scalability_results.parquet")
    if not force and os.path.exists(scal_path):
        print("\n--- Module 3: Scalability Benchmark (using cached scalability_results.parquet) ---")
        return
    print("\n--- Module 3: Scalability Benchmark (16 Runs: Size x Partitions) ---")
    sizes = [10_000, 100_000, 250_000, 500_000]
    partitions_list = [2, 4, 8, 16]
    sample_pool = df.sample(n=500_000, seed=42).select(["title", "text", "label"])
    titles = sample_pool["title"].fill_null("").to_list()
    texts = sample_pool["text"].fill_null("").to_list()
    labels = sample_pool["label"].to_numpy()

    def clean_chunk(pairs):
        out = []
        for t, x in pairs:
            s = f"{t}. {x}".lower()
            s = re.sub(r"<[^>]+>", " ", s)
            s = re.sub(r"https?://\S+|www\.\S+", " ", s)
            s = re.sub(r"n't\b", " not", s)
            s = re.sub(r"[^a-z\s]", " ", s)
            out.append(re.sub(r"\s+", " ", s).strip())
        return out

    rows = []
    for n in sizes:
        sub_pairs = list(zip(titles[:n], texts[:n]))
        sub_y = labels[:n]
        for parts in partitions_list:
            t0 = time.time()
            # 1. Ingestion (partitioned slice)
            chunk_sz = max(1, n // parts)
            chunks = [sub_pairs[i:i + chunk_sz] for i in range(0, n, chunk_sz)]
            ingest_s = time.time() - t0

            # 2. Cleaning across partitions
            t1 = time.time()
            with ThreadPoolExecutor(max_workers=min(parts, 16)) as ex:
                cleaned_chunks = list(ex.map(clean_chunk, chunks))
            cleaned_all = [item for sub in cleaned_chunks for item in sub]
            clean_s = time.time() - t1

            # 3. NLP + TF-IDF
            t2 = time.time()
            vec = TfidfVectorizer(max_features=20_000, min_df=3, stop_words=STOP_WORDS, dtype=np.float32)
            X_mat = vec.fit_transform(cleaned_all)
            tfidf_s = time.time() - t2

            # 4. Model training (Logistic SGD)
            t3 = time.time()
            clf = SGDClassifier(loss="log_loss", max_iter=5, tol=1e-3, random_state=42)
            clf.fit(X_mat, sub_y)
            train_s = time.time() - t3

            total_s = ingest_s + clean_s + tfidf_s + train_s
            rps = n / total_s if total_s > 0 else 0.0
            rows.append({
                "reviews": n,
                "partitions": parts,
                "ingestion_sec": round(ingest_s, 3),
                "cleaning_sec": round(clean_s, 3),
                "tfidf_sec": round(tfidf_s, 3),
                "training_sec": round(train_s, 3),
                "total_sec": round(total_s, 3),
                "throughput_rps": round(rps, 1),
            })
            print(f"  n={n:>7,} | parts={parts:>2} | total={total_s:.2f}s | {rps:,.0f} rev/s")

    pl.DataFrame(rows).write_parquet(os.path.join(GOLD_DIR, "scalability_results.parquet"))


def sentiment_aggs():
    return [
        pl.len().cast(pl.Int64).alias("reviews"),
        (pl.col("sentiment") == "Positive").sum().cast(pl.Int64).alias("positive"),
        (pl.col("sentiment") == "Neutral").sum().cast(pl.Int64).alias("neutral"),
        (pl.col("sentiment") == "Negative").sum().cast(pl.Int64).alias("negative"),
        pl.col("rating").mean().round(3).alias("avg_rating"),
    ]


def add_pct(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        (pl.col("positive") * 100.0 / pl.col("reviews")).round(2).alias("positive_pct"),
        (pl.col("neutral") * 100.0 / pl.col("reviews")).round(2).alias("neutral_pct"),
        (pl.col("negative") * 100.0 / pl.col("reviews")).round(2).alias("negative_pct"),
    ])


def build_historical_gold_tables(df: pl.DataFrame):
    print("\n--- Phase 9: Building Historical Gold Tables ---")

    # 1. agg_year_summary
    by_year = (
        df.group_by("year")
        .agg(
            *sentiment_aggs(),
            pl.col("parent_asin").n_unique().cast(pl.Int64).alias("products"),
            pl.col("brand").n_unique().cast(pl.Int64).alias("brands"),
            pl.col("category").n_unique().cast(pl.Int64).alias("categories"),
        )
        .with_columns(pl.col("year").cast(pl.Utf8))
    )
    all_years = df.select(
        pl.lit("All").alias("year"),
        *sentiment_aggs(),
        pl.col("parent_asin").n_unique().cast(pl.Int64).alias("products"),
        pl.col("brand").n_unique().cast(pl.Int64).alias("brands"),
        pl.col("category").n_unique().cast(pl.Int64).alias("categories"),
    )
    year_summary = add_pct(pl.concat([by_year, all_years], how="vertical")).sort("year")
    year_summary.write_parquet(os.path.join(GOLD_DIR, "agg_year_summary.parquet"))
    print(f"agg_year_summary: {year_summary.height} rows")

    # 2. agg_year_category
    year_cat = add_pct(df.group_by(["year", "category"]).agg(*sentiment_aggs())).sort(["year", "category"])
    year_cat.write_parquet(os.path.join(GOLD_DIR, "agg_year_category.parquet"))
    print(f"agg_year_category: {year_cat.height} rows")

    # 3. agg_monthly
    monthly = add_pct(df.group_by(["year", "review_month", "category"]).agg(*sentiment_aggs())).sort(["review_month", "category"])
    monthly.write_parquet(os.path.join(GOLD_DIR, "agg_monthly.parquet"))
    print(f"agg_monthly: {monthly.height} rows")

    # 4. agg_brand (per year & category, >= 30 reviews, plus All-years summary)
    brand_yr = add_pct(
        df.filter(pl.col("brand") != "Unknown")
        .group_by(["year", "category", "brand"])
        .agg(*sentiment_aggs())
        .filter(pl.col("reviews") >= 30)
        .with_columns(pl.col("year").cast(pl.Utf8))
    )
    brand_all = add_pct(
        df.filter(pl.col("brand") != "Unknown")
        .group_by(["category", "brand"])
        .agg(*sentiment_aggs())
        .filter(pl.col("reviews") >= 30)
        .with_columns(pl.lit("All").alias("year"))
    ).select(brand_yr.columns)
    brand_df = pl.concat([brand_yr, brand_all], how="vertical")
    brand_df.write_parquet(os.path.join(GOLD_DIR, "agg_brand.parquet"))
    print(f"agg_brand: {brand_df.height:,} rows")

    # 5. agg_top_words (lift > 1.3, n >= 20, top 30 per year+category+sentiment + 'All')
    stop_set = set(STOP_WORDS)
    word_counts_list = []
    non_neutral = df.filter(pl.col("sentiment") != "Neutral").select(["year", "category", "sentiment", "clean_text"])
    for y in YEARS:
        sub = non_neutral.filter(pl.col("year") == y)
        exploded = (
            sub.with_columns(pl.col("clean_text").str.split(" ").alias("word"))
            .drop("clean_text")
            .explode("word")
            .filter((pl.col("word").str.len_chars() > 2) & (~pl.col("word").is_in(list(stop_set))))
        )
        wc = exploded.group_by(["year", "category", "sentiment", "word"]).agg(pl.len().cast(pl.Int64).alias("n"))
        word_counts_list.append(wc)

    wc_yr = pl.concat(word_counts_list, how="vertical").with_columns(pl.col("year").cast(pl.Utf8))
    wc_all = (
        wc_yr.group_by(["category", "sentiment", "word"])
        .agg(pl.col("n").sum().alias("n"))
        .with_columns(pl.lit("All").alias("year"))
        .select(["year", "category", "sentiment", "word", "n"])
    )
    wc_both = pl.concat([wc_yr, wc_all], how="vertical")

    totals = wc_both.group_by(["year", "category", "sentiment"]).agg(pl.col("n").sum().alias("class_total"))
    word_totals = wc_both.group_by(["year", "category", "word"]).agg(pl.col("n").sum().alias("word_total"))
    all_totals = wc_both.group_by(["year", "category"]).agg(pl.col("n").sum().alias("all_total"))

    top_words = (
        wc_both.join(totals, on=["year", "category", "sentiment"])
        .join(word_totals, on=["year", "category", "word"])
        .join(all_totals, on=["year", "category"])
        .with_columns(
            ((pl.col("n") / pl.col("class_total")) / (pl.col("word_total") / pl.col("all_total"))).alias("lift")
        )
        .filter((pl.col("lift") > 1.3) & (pl.col("n") >= 20))
        .sort(["year", "category", "sentiment", "n"], descending=[False, False, False, True])
        .with_columns(pl.col("n").rank("ordinal", descending=True).over(["year", "category", "sentiment"]).alias("rank"))
        .filter(pl.col("rank") <= 30)
        .select(["year", "category", "sentiment", "word", "n", pl.col("lift").round(2), "rank"])
    )
    top_words.write_parquet(os.path.join(GOLD_DIR, "agg_top_words.parquet"))
    print(f"agg_top_words: {top_words.height:,} rows")

    # 6. agg_complaints
    neg_df = df.filter(pl.col("sentiment") == "Negative").select(["year", "category", "clean_text"])
    theme_exprs = [
        pl.col("clean_text").str.contains(r"\b(?:" + "|".join(kw) + r")\b").cast(pl.Int64).alias(theme)
        for theme, kw in COMPLAINT_THEMES.items()
    ]
    flagged = neg_df.with_columns(theme_exprs)
    comp_agg = flagged.group_by(["year", "category"]).agg(
        pl.len().cast(pl.Int64).alias("negative_reviews"),
        *[pl.col(t).sum().alias(t) for t in COMPLAINT_THEMES],
    )
    comp_long = (
        comp_agg.unpivot(
            index=["year", "category", "negative_reviews"],
            on=list(COMPLAINT_THEMES.keys()),
            variable_name="theme",
            value_name="mentions",
        )
        .with_columns((pl.col("mentions") * 100.0 / pl.col("negative_reviews")).round(2).alias("pct_of_negative"))
        .sort(["year", "category", "mentions"], descending=[False, False, True])
    )
    comp_long.write_parquet(os.path.join(GOLD_DIR, "agg_complaints.parquet"))
    print(f"agg_complaints: {comp_long.height} rows")

    # 7. agg_rating_dist (1-5 stars by year and category)
    rating_dist = (
        df.with_columns(pl.col("rating").cast(pl.Int64).alias("stars"))
        .group_by(["year", "category", "stars"])
        .agg(pl.len().cast(pl.Int64).alias("reviews"))
        .sort(["year", "category", "stars"])
    )
    rating_dist.write_parquet(os.path.join(GOLD_DIR, "agg_rating_dist.parquet"))
    print(f"agg_rating_dist: {rating_dist.height} rows")


def build_product_recommendations(df: pl.DataFrame):
    print("\n--- Phases 10-11: Product Search & BUY / DON'T BUY Engine ---")

    prod_overall = (
        df.group_by(["parent_asin", "category"])
        .agg([
            pl.col("product_title").drop_nulls().first().alias("product_title"),
            pl.col("brand").filter(pl.col("brand") != "Unknown").first().alias("real_brand"),
            pl.col("price").drop_nulls().first().alias("price"),
            pl.col("product_avg_rating").drop_nulls().first().alias("catalog_avg_rating"),
            pl.col("product_rating_count").drop_nulls().first().alias("catalog_rating_count"),
            pl.col("product_features").drop_nulls().first().alias("product_features"),
            pl.len().cast(pl.Int64).alias("reviews"),
            pl.col("rating").mean().round(3).alias("avg_rating"),
            (pl.col("sentiment") == "Positive").sum().cast(pl.Int64).alias("positive"),
            (pl.col("sentiment") == "Neutral").sum().cast(pl.Int64).alias("neutral"),
            (pl.col("sentiment") == "Negative").sum().cast(pl.Int64).alias("negative"),
            (pl.col("predicted_sentiment") == "Positive").sum().cast(pl.Int64).alias("pred_positive"),
            pl.col("verified_purchase").fill_null(False).sum().cast(pl.Int64).alias("verified_count"),
            pl.col("year").min().alias("first_year"),
            pl.col("year").max().alias("recent_year"),
            pl.col("year").n_unique().cast(pl.Int64).alias("years_active"),
        ])
        .with_columns([
            pl.col("real_brand").fill_null("Unknown").alias("brand"),
            (pl.col("positive") * 100.0 / pl.col("reviews")).round(2).alias("positive_pct"),
            (pl.col("neutral") * 100.0 / pl.col("reviews")).round(2).alias("neutral_pct"),
            (pl.col("negative") * 100.0 / pl.col("reviews")).round(2).alias("negative_pct"),
            (pl.col("pred_positive") * 100.0 / pl.col("reviews")).round(2).alias("model_positive_pct"),
            (pl.col("verified_count") * 100.0 / pl.col("reviews")).round(2).alias("verified_pct"),
        ])
        .drop("real_brand")
    )

    prod_yearly = (
        df.group_by(["parent_asin", "category", "year"])
        .agg(
            ((pl.col("sentiment") == "Positive").mean() * 100.0).round(2).alias("yr_pos_pct")
        )
        .sort(["parent_asin", "category", "year"])
    )

    first_yr = prod_yearly.group_by(["parent_asin", "category"]).first().select(
        ["parent_asin", "category", pl.col("yr_pos_pct").alias("first_year_pos_pct")]
    )
    recent_yr = prod_yearly.group_by(["parent_asin", "category"]).last().select(
        ["parent_asin", "category", pl.col("yr_pos_pct").alias("recent_year_pos_pct")]
    )
    consistency = prod_yearly.group_by(["parent_asin", "category"]).agg(
        pl.col("yr_pos_pct").std().fill_null(0.0).round(2).alias("yearly_pos_std")
    )

    stats = (
        prod_overall.join(first_yr, on=["parent_asin", "category"], how="left")
        .join(recent_yr, on=["parent_asin", "category"], how="left")
        .join(consistency, on=["parent_asin", "category"], how="left")
        .with_columns((pl.col("recent_year_pos_pct") - pl.col("first_year_pos_pct")).round(2).alias("sentiment_trend"))
    )

    # Filter products with >= 5 reviews for aspect & complaint breakdowns
    eligible_keys = stats.filter(pl.col("reviews") >= 5).select(["parent_asin", "category"])
    sub_rev = df.join(eligible_keys, on=["parent_asin", "category"], how="inner")

    # Aspect-level sentiment
    aspect_rows = []
    for asp, kw in ASPECT_KEYWORDS.items():
        pat = r"\b(?:" + "|".join(kw) + r")\b"
        matched = (
            sub_rev.filter(pl.col("clean_text").str.contains(pat))
            .group_by(["parent_asin", "category"])
            .agg([
                pl.len().cast(pl.Int64).alias("mentions"),
                (pl.col("sentiment") == "Positive").sum().cast(pl.Int64).alias("pos_mentions"),
                (pl.col("sentiment") == "Negative").sum().cast(pl.Int64).alias("neg_mentions"),
            ])
            .with_columns([
                pl.lit(asp).alias("aspect"),
                (pl.col("pos_mentions") * 100.0 / pl.col("mentions")).round(1).alias("positive_pct"),
                (pl.col("neg_mentions") * 100.0 / pl.col("mentions")).round(1).alias("negative_pct"),
            ])
        )
        aspect_rows.append(matched)
    aspect_df = pl.concat(aspect_rows, how="vertical")
    aspect_df.write_parquet(os.path.join(GOLD_DIR, "product_aspect_sentiment.parquet"))
    print(f"product_aspect_sentiment: {aspect_df.height:,} rows")

    # Per-product complaint themes
    sub_neg = sub_rev.filter(pl.col("sentiment") == "Negative")
    neg_totals = sub_neg.group_by(["parent_asin", "category"]).agg(pl.len().cast(pl.Int64).alias("negative_reviews"))
    comp_rows = []
    for theme, kw in COMPLAINT_THEMES.items():
        pat = r"\b(?:" + "|".join(kw) + r")\b"
        matched = (
            sub_neg.filter(pl.col("clean_text").str.contains(pat))
            .group_by(["parent_asin", "category"])
            .agg(pl.len().cast(pl.Int64).alias("mentions"))
            .with_columns(pl.lit(theme).alias("theme"))
        )
        comp_rows.append(matched)
    prod_comp = (
        pl.concat(comp_rows, how="vertical")
        .join(neg_totals, on=["parent_asin", "category"], how="inner")
        .with_columns((pl.col("mentions") * 100.0 / pl.col("negative_reviews")).round(1).alias("pct_of_negative"))
    )
    prod_comp.write_parquet(os.path.join(GOLD_DIR, "product_complaints.parquet"))
    print(f"product_complaints: {prod_comp.height:,} rows")

    top_aspect = (
        aspect_df.filter(pl.col("mentions") >= 3)
        .sort(["parent_asin", "category", "mentions"], descending=[False, False, True])
        .group_by(["parent_asin", "category"])
        .first()
        .select([
            "parent_asin", "category",
            pl.col("aspect").alias("top_aspect"),
            pl.col("positive_pct").alias("top_aspect_pos_pct"),
        ])
    )
    top_complaint = (
        prod_comp.sort(["parent_asin", "category", "mentions"], descending=[False, False, True])
        .group_by(["parent_asin", "category"])
        .first()
        .select([
            "parent_asin", "category",
            pl.col("theme").alias("top_complaint"),
            pl.col("pct_of_negative").alias("top_complaint_pct"),
        ])
    )

    # Explainable BUY / DON'T BUY scoring
    recs = (
        stats.join(top_aspect, on=["parent_asin", "category"], how="left")
        .join(top_complaint, on=["parent_asin", "category"], how="left")
        .with_columns([
            ((pl.col("avg_rating") - 1.0) / 4.0 * 100.0).clip(0.0, 100.0).alias("rating_score"),
            (50.0 + pl.col("sentiment_trend") * 1.5).clip(0.0, 100.0).alias("trend_score"),
            (100.0 - pl.col("yearly_pos_std") * 2.0).clip(0.0, 100.0).alias("consistency_score"),
            (100.0 - pl.col("negative_pct")).clip(0.0, 100.0).alias("complaint_inv_score"),
        ])
        .with_columns(
            (
                0.20 * pl.col("positive_pct")
                + 0.15 * pl.col("model_positive_pct")
                + 0.20 * pl.col("rating_score")
                + 0.15 * pl.col("recent_year_pos_pct").fill_null(pl.col("positive_pct"))
                + 0.10 * pl.col("trend_score")
                + 0.10 * pl.col("consistency_score")
                + 0.10 * pl.col("complaint_inv_score")
            ).round(1).alias("buy_score")
        )
        .with_columns([
            pl.when(pl.col("reviews") < 20).then(pl.lit("INSUFFICIENT DATA"))
              .when(pl.col("buy_score") >= 65.0).then(pl.lit("BUY"))
              .otherwise(pl.lit("DON'T BUY")).alias("recommendation"),
            pl.when((pl.col("reviews") >= 100) & (pl.col("years_active") >= 2)).then(pl.lit("High"))
              .when(pl.col("reviews") >= 50).then(pl.lit("Medium"))
              .otherwise(pl.lit("Low")).alias("confidence"),
        ])
        .with_columns(
            pl.concat_str(
                [
                    pl.format("Avg rating {}/5 across {} reviews", pl.col("avg_rating").round(2), pl.col("reviews")),
                    pl.format("{}% positive (ML model: {}%)", pl.col("positive_pct"), pl.col("model_positive_pct")),
                    pl.when(pl.col("sentiment_trend") >= 5)
                      .then(pl.format("Improving trend (+{}%)", pl.col("sentiment_trend")))
                      .when(pl.col("sentiment_trend") <= -5)
                      .then(pl.format("Declining trend ({}%)", pl.col("sentiment_trend")))
                      .otherwise(pl.lit("Stable sentiment across years")),
                    pl.when(pl.col("top_aspect").is_not_null())
                      .then(pl.format("Top aspect: {} ({}% positive)", pl.col("top_aspect"), pl.col("top_aspect_pos_pct")))
                      .otherwise(pl.lit("")),
                    pl.when(pl.col("top_complaint").is_not_null())
                      .then(pl.format("Main complaint: {} ({}% of negatives)", pl.col("top_complaint"), pl.col("top_complaint_pct")))
                      .otherwise(pl.lit("")),
                ],
                separator=" | ",
            ).str.replace_all(r"(\s*\|\s*)+$", "").alias("reason_summary")
        )
        .sort("reviews", descending=True)
    )
    recs.write_parquet(os.path.join(GOLD_DIR, "product_recommendations.parquet"))
    summary_counts = recs.group_by("recommendation").len().to_dicts()
    print(f"product_recommendations: {recs.height:,} products -> {summary_counts}")

    # Sample customer reviews for top 5,000 most-reviewed products
    top_prods = recs.head(5_000).select(["parent_asin", "category"])
    sample_reviews = (
        df.join(top_prods, on=["parent_asin", "category"], how="inner")
        .sort(["parent_asin", "helpful_vote", "year"], descending=[False, True, True])
        .group_by(["parent_asin", "category"])
        .head(8)
        .select([
            "parent_asin", "category", "year", "review_date", "rating",
            "sentiment", "predicted_sentiment", "verified_purchase",
            "helpful_vote", "title", "text",
        ])
    )
    sample_reviews.write_parquet(os.path.join(GOLD_DIR, "product_sample_reviews.parquet"))
    print(f"product_sample_reviews: {sample_reviews.height:,} rows")


def build_stream_pool(df: pl.DataFrame):
    print("\n--- Phase 12: Building Streaming Review Pool ---")
    pool = (
        df.filter(pl.col("year") == 2026)
        .sample(n=5_000, seed=123)
        .select([
            "review_id", "year", "review_date", "category", "parent_asin",
            "product_title", "brand", "rating", "title", "text",
            "clean_text", "sentiment", "verified_purchase",
        ])
    )
    pool.write_parquet(os.path.join(GOLD_DIR, "stream_sample_pool.parquet"))
    print(f"stream_sample_pool: {pool.height:,} rows")


def main():
    import sys
    force = "--force" in sys.argv
    t_start = time.time()
    os.makedirs(GOLD_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_and_clean_reviews()
    df = train_and_score_models(df, force=force)
    run_scalability_benchmark(df, force=force)
    build_historical_gold_tables(df)
    build_product_recommendations(df)
    build_stream_pool(df)

    print(f"\nALL GOLD TABLES & MODELS BUILT in {time.time() - t_start:.1f}s!")


if __name__ == "__main__":
    main()
