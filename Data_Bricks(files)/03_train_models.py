# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Train and evaluate sentiment models (PySpark MLlib)
# MAGIC Naive Bayes vs Logistic Regression vs Random Forest on TF-IDF features.
# MAGIC Labels are rating-derived proxies: 1-2 stars = Negative, 3 = Neutral, 4-5 = Positive.

# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 1: Setup

# COMMAND ----------

import time
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import NaiveBayes, LogisticRegression, RandomForestClassifier

DB = "workspace.review_analytics"
MODEL_DIR = "/Volumes/workspace/review_analytics/models"
spark.sql(f"CREATE VOLUME IF NOT EXISTS {DB}.models")

data = spark.table(f"{DB}.reviews_clean").select("review_id", "year", "category", "clean_text", "label")

STOP_WORDS = """a an the and or but if then so of to in on at by for with from as is are was
were be been being am it its it's this that these those i me my we our you your he she him her
they them their what which who whom there here when where why how all any both each few more most
other some such only own same than too very can will just do does did doing have has had having
would should could also get got one would about into over after before up down out off again
further once s t ve ll re d m""".split()
LABEL_NAMES = {0.0: "Negative", 1.0: "Neutral", 2.0: "Positive"}
print("Setup done")

# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 2: Train/test split (80/20) + class weights

# COMMAND ----------

# Split by a hash of review_id: the same review always lands in the same set,
# so train and test can never overlap (no leakage), even if the cell is re-run.
bucket = F.abs(F.xxhash64("review_id")) % 100
train = data.filter(bucket < 80)
test = data.filter(bucket >= 80)

counts = {r["label"]: r["count"] for r in train.groupBy("label").count().collect()}
n_train = sum(counts.values())
# Rare classes (Negative, Neutral) get a bigger weight so the model doesn't ignore them
weights = {lab: n_train / (3 * n) for lab, n in counts.items()}
train = train.withColumn("class_weight",
                         F.when(F.col("label") == 0.0, weights[0.0])
                          .when(F.col("label") == 1.0, weights[1.0])
                          .otherwise(weights[2.0]))

print(f"Train: {n_train:,} reviews   Test: {test.count():,} reviews")
for lab in sorted(counts):
    print(f"  {LABEL_NAMES[lab]:<9} {counts[lab]:>9,} train reviews   weight = {weights[lab]:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 3: Feature pipeline + training helper

# COMMAND ----------

def feature_stages(num_features):
    """Tokenize -> remove stop words -> term frequency -> TF-IDF.
    These stages are FITTED ON TRAINING DATA ONLY (inside the Pipeline)."""
    return [
        RegexTokenizer(inputCol="clean_text", outputCol="words", pattern=r"\s+", minTokenLength=2),
        StopWordsRemover(inputCol="words", outputCol="filtered", stopWords=STOP_WORDS),
        HashingTF(inputCol="filtered", outputCol="tf", numFeatures=num_features),
        IDF(inputCol="tf", outputCol="features", minDocFreq=5),
    ]

models, timings = {}, {}

def train_and_test(name, classifier, train_df, num_features=1 << 18):
    t0 = time.time()
    model = Pipeline(stages=feature_stages(num_features) + [classifier]).fit(train_df)
    train_secs = time.time() - t0

    t1 = time.time()
    (model.transform(test)
          .select("review_id", "year", "category", "label", "prediction")
          .write.format("delta").mode("overwrite").option("overwriteSchema", "true")
          .saveAsTable(f"{DB}.pred_{name}"))
    test_secs = time.time() - t1

    models[name] = model
    timings[name] = (train_secs, test_secs, train_df.count())
    print(f"{name}: trained in {train_secs:.0f}s, predicted test set in {test_secs:.0f}s")

print("Helper ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 4: Naive Bayes

# COMMAND ----------

nb = NaiveBayes(featuresCol="features", labelCol="label", weightCol="class_weight",
                modelType="multinomial", smoothing=1.0)
train_and_test("naive_bayes", nb, train)

# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 5: Logistic Regression

# COMMAND ----------

lr = LogisticRegression(featuresCol="features", labelCol="label", weightCol="class_weight",
                        family="multinomial", maxIter=30, regParam=0.01)
train_and_test("logistic_regression", lr, train)

# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 6: Random Forest (on a ~200,000 sample, smaller feature space)

# COMMAND ----------

rf_train = train.sample(fraction=min(1.0, 200_000 / n_train), seed=42)
rf = RandomForestClassifier(featuresCol="features", labelCol="label", weightCol="class_weight",
                            numTrees=40, maxDepth=10, seed=42)
train_and_test("random_forest", rf, rf_train, num_features=1 << 14)

# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 7: Evaluate all models

# COMMAND ----------

def evaluate(name):
    rows = spark.table(f"{DB}.pred_{name}").groupBy("label", "prediction").count().collect()
    cm = {(r["label"], r["prediction"]): r["count"] for r in rows}
    labels = [0.0, 1.0, 2.0]
    total = sum(cm.values())
    out = {"model": name, "accuracy": sum(cm.get((l, l), 0) for l in labels) / total}
    f1s, supports = [], []
    for l in labels:
        tp = cm.get((l, l), 0)
        predicted = sum(cm.get((t, l), 0) for t in labels)
        actual = sum(cm.get((l, p), 0) for p in labels)
        prec = tp / predicted if predicted else 0.0
        rec = tp / actual if actual else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        short = LABEL_NAMES[l].lower()
        out[f"precision_{short}"], out[f"recall_{short}"], out[f"f1_{short}"] = prec, rec, f1
        f1s.append(f1)
        supports.append(actual)
    out["macro_f1"] = sum(f1s) / 3
    out["weighted_f1"] = sum(f * s for f, s in zip(f1s, supports)) / total
    train_secs, test_secs, n_rows = timings.get(name, (None, None, None))
    out["train_rows"], out["train_seconds"] = n_rows, train_secs
    return out, cm

results = []
for name in models:
    res, cm = evaluate(name)
    results.append(res)
    print(f"\n=== {name} ===  accuracy {res['accuracy']:.3f}   macro F1 {res['macro_f1']:.3f}")
    print("actual / predicted".ljust(20) + "Negative".rjust(10) + "Neutral".rjust(10) + "Positive".rjust(10))
    for t in [0.0, 1.0, 2.0]:
        print(f"{LABEL_NAMES[t]:<20}" + "".join(f"{cm.get((t, p), 0):>10,}" for p in [0.0, 1.0, 2.0]))

best = max(results, key=lambda r: r["macro_f1"])["model"]
metrics_df = spark.createDataFrame(results).withColumn("is_best", F.col("model") == best)
(metrics_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
           .saveAsTable(f"{DB}.model_metrics"))
display(metrics_df.select("model", "accuracy", "macro_f1", "weighted_f1",
                          "recall_negative", "recall_neutral", "recall_positive",
                          "train_rows", "train_seconds", "is_best"))
print(f"\nBest model (by macro F1): {best}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 8: Save the best model

# COMMAND ----------

models[best].write().overwrite().save(f"{MODEL_DIR}/best_sentiment_model")
print(f"Saved {best} to {MODEL_DIR}/best_sentiment_model")
