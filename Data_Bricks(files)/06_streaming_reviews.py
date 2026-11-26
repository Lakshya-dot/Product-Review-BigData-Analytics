# Databricks notebook source
# MAGIC %md
# MAGIC # 06 - Live Review Stream Simulation (Phase 12 - Spark Structured Streaming)
# MAGIC Simulates real-time incoming customer reviews, cleans text identically to training,
# MAGIC runs the saved TF-IDF + Logistic Regression PipelineModel in micro-batches, and writes
# MAGIC predictions + alerts to Delta tables.

# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 1: Setup streaming folders & schemas

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType, BooleanType
from pyspark.ml import PipelineModel

DB = "workspace.review_analytics"
MODEL_DIR = "/Volumes/workspace/review_analytics/models"
STREAM_IN = "/Volumes/workspace/review_analytics/stream_input"
CHECKPOINT = "/Volumes/workspace/review_analytics/stream_checkpoint"

spark.sql(f"CREATE VOLUME IF NOT EXISTS {DB}.stream_input")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {DB}.stream_checkpoint")

LABEL_NAMES = {0.0: "Negative", 1.0: "Neutral", 2.0: "Positive"}
model = PipelineModel.load(f"{MODEL_DIR}/best_sentiment_model")
label_map = F.create_map(*[x for k, v in LABEL_NAMES.items() for x in (F.lit(k), F.lit(v))])

STREAM_SCHEMA = StructType([
    StructField("review_id", StringType()),
    StructField("year", LongType()),
    StructField("review_date", StringType()),
    StructField("category", StringType()),
    StructField("parent_asin", StringType()),
    StructField("product_title", StringType()),
    StructField("brand", StringType()),
    StructField("rating", DoubleType()),
    StructField("title", StringType()),
    StructField("text", StringType()),
    StructField("verified_purchase", BooleanType()),
])

print("Streaming setup ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 2: Drip sample micro-batches from 2026 reviews into stream_input

# COMMAND ----------

sample_batch = (
    spark.table(f"{DB}.reviews_2026")
    .select("review_id", "year", "review_date", "category", "parent_asin",
            "product_title", "brand", "rating", "title", "text", "verified_purchase")
    .orderBy(F.rand(seed=42))
    .limit(1000)
)
(sample_batch.repartition(5).write.mode("overwrite").json(f"{STREAM_IN}/batch_01"))
print("Wrote 1,000 live-stream JSON review events in 5 micro-files")

# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 3: Structured Streaming Pipeline (readStream -> clean -> ML predict -> Delta sink)

# COMMAND ----------

raw_stream = (
    spark.readStream
    .schema(STREAM_SCHEMA)
    .option("maxFilesPerTrigger", 1)
    .json(f"{STREAM_IN}/batch_*")
)

cleaned_stream = (
    raw_stream
    .withColumn("review_text", F.concat_ws(". ", "title", "text"))
    .withColumn("clean_text", F.lower("review_text"))
    .withColumn("clean_text", F.regexp_replace("clean_text", r"<[^>]+>", " "))
    .withColumn("clean_text", F.regexp_replace("clean_text", r"(https?://\S+|www\.\S+)", " "))
    .withColumn("clean_text", F.regexp_replace("clean_text", r"n't\b", " not"))
    .withColumn("clean_text", F.regexp_replace("clean_text", r"[^a-z\s]", " "))
    .withColumn("clean_text", F.trim(F.regexp_replace("clean_text", r"\s+", " ")))
    .filter(F.length("clean_text") > 0)
    .withColumn("actual_sentiment", F.when(F.col("rating") <= 2, "Negative")
                                     .when(F.col("rating") == 3, "Neutral")
                                     .otherwise("Positive"))
    .withColumn("ingested_at", F.current_timestamp())
)

scored_stream = (
    model.transform(cleaned_stream)
    .withColumn("predicted_sentiment", label_map[F.col("prediction")])
    .withColumn("is_negative_alert", F.col("predicted_sentiment") == "Negative")
    .select("review_id", "ingested_at", "category", "parent_asin", "product_title", "brand",
            "rating", "actual_sentiment", "predicted_sentiment", "is_negative_alert",
            "verified_purchase", "title", "text")
)

query = (
    scored_stream.writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", f"{CHECKPOINT}/reviews_v1")
    .trigger(availableNow=True)
    .toTable(f"{DB}.stream_scored")
)
query.awaitTermination()
print(f"Stream batch complete. Total streamed rows: {spark.table(f'{DB}.stream_scored').count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 4: Live Stream Summary & Negative Spike Alerts

# COMMAND ----------

stream_df = spark.table(f"{DB}.stream_scored")
display(
    stream_df.groupBy("category", "predicted_sentiment")
    .count()
    .orderBy("category", "predicted_sentiment")
)
display(stream_df.filter(F.col("is_negative_alert")).orderBy(F.desc("ingested_at")).limit(20))
