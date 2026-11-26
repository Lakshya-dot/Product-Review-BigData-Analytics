# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - Four-year historical analytics (PySpark)
# MAGIC Scores every review with the best model and builds the summary (gold) Delta tables used by the dashboard.

# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 1: Setup

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.ml import PipelineModel

DB = "workspace.review_analytics"
MODEL_DIR = "/Volumes/workspace/review_analytics/models"
LABEL_NAMES = {0.0: "Negative", 1.0: "Neutral", 2.0: "Positive"}

def save(df, name):
    (df.write.format("delta").mode("overwrite")
       .option("overwriteSchema", "true").saveAsTable(f"{DB}.{name}"))
    print(f"{name}: {spark.table(f'{DB}.{name}').count():,} rows")

def sentiment_counts():
    """Reusable aggregations: counts and % of each sentiment + average rating."""
    return [
        F.count("*").alias("reviews"),
        F.sum(F.when(F.col("sentiment") == "Positive", 1).otherwise(0)).alias("positive"),
        F.sum(F.when(F.col("sentiment") == "Neutral", 1).otherwise(0)).alias("neutral"),
        F.sum(F.when(F.col("sentiment") == "Negative", 1).otherwise(0)).alias("negative"),
        F.round(F.avg("rating"), 3).alias("avg_rating"),
    ]

def add_pct(df):
    for s in ["positive", "neutral", "negative"]:
        df = df.withColumn(f"{s}_pct", F.round(F.col(s) * 100 / F.col("reviews"), 2))
    return df

print("Setup done")

# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 2: Score every review with the trained model

# COMMAND ----------

model = PipelineModel.load(f"{MODEL_DIR}/best_sentiment_model")
label_map = F.create_map(*[x for k, v in LABEL_NAMES.items() for x in (F.lit(k), F.lit(v))])

scored = (model.transform(spark.table(f"{DB}.reviews_clean"))
               .withColumn("predicted_sentiment", label_map[F.col("prediction")])
               .drop("words", "filtered", "tf", "features", "rawPrediction", "probability", "prediction"))
save(scored, "reviews_scored")
reviews = spark.table(f"{DB}.reviews_scored")

# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 3: Year summary (KPIs per year + "All years")

# COMMAND ----------

by_year = reviews.groupBy("year").agg(
    *sentiment_counts(),
    F.countDistinct("parent_asin").alias("products"),
    F.countDistinct("brand").alias("brands"),
    F.countDistinct("category").alias("categories"))
all_years = reviews.agg(
    *sentiment_counts(),
    F.countDistinct("parent_asin").alias("products"),
    F.countDistinct("brand").alias("brands"),
    F.countDistinct("category").alias("categories")).withColumn("year", F.lit("All"))
year_summary = add_pct(by_year.withColumn("year", F.col("year").cast("string"))
                       .unionByName(all_years))
save(year_summary, "agg_year_summary")
display(spark.table(f"{DB}.agg_year_summary").orderBy("year"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 4: Category-wise sentiment per year (also the 4-year comparison)

# COMMAND ----------

save(add_pct(reviews.groupBy("year", "category").agg(*sentiment_counts())), "agg_year_category")
display(spark.table(f"{DB}.agg_year_category")
        .groupBy("category").pivot("year").agg(F.first("positive_pct"))
        .orderBy("category"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 5: Monthly trends

# COMMAND ----------

save(add_pct(reviews.groupBy("year", "review_month", "category").agg(*sentiment_counts())),
     "agg_monthly")

# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 6: Brand-wise and product-wise sentiment

# COMMAND ----------

brand = (add_pct(reviews.filter(F.col("brand") != "Unknown")
                        .groupBy("year", "category", "brand").agg(*sentiment_counts()))
         .filter(F.col("reviews") >= 30))          # ignore brands with very few reviews
save(brand, "agg_brand")

product = add_pct(reviews.groupBy("year", "category", "parent_asin", "product_title", "brand")
                         .agg(*sentiment_counts()))
save(product, "agg_product_year")

# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 7: Most common positive and negative words

# COMMAND ----------

# Pure counts would put "product", "use", "one" on top of BOTH lists, so we keep words
# that are clearly more frequent in that sentiment than overall (lift > 1.3).
words = (reviews.filter(F.col("sentiment") != "Neutral")
                .select("year", "category", "sentiment", F.explode("tokens").alias("word"))
                .filter(F.length("word") > 2))
word_counts = words.groupBy("year", "category", "sentiment", "word").agg(F.count("*").alias("n"))
totals = word_counts.groupBy("year", "category", "sentiment").agg(F.sum("n").alias("class_total"))
overall = (word_counts.groupBy("year", "category", "word").agg(F.sum("n").alias("word_total"))
           .join(word_counts.groupBy("year", "category").agg(F.sum("n").alias("all_total")),
                 ["year", "category"]))

from pyspark.sql.window import Window
rank_w = Window.partitionBy("year", "category", "sentiment").orderBy(F.desc("n"))
top_words = (word_counts.join(totals, ["year", "category", "sentiment"])
             .join(overall, ["year", "category", "word"])
             .withColumn("lift", (F.col("n") / F.col("class_total")) /
                                 (F.col("word_total") / F.col("all_total")))
             .filter((F.col("lift") > 1.3) & (F.col("n") >= 20))
             .withColumn("rank", F.row_number().over(rank_w))
             .filter(F.col("rank") <= 30)
             .select("year", "category", "sentiment", "word", "n", F.round("lift", 2).alias("lift"), "rank"))
save(top_words, "agg_top_words")
display(spark.table(f"{DB}.agg_top_words")
        .filter((F.col("year") == 2026) & (F.col("category") == "Electronics") & (F.col("rank") <= 10))
        .orderBy("sentiment", "rank"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 8: Customer complaint patterns

# COMMAND ----------

COMPLAINT_THEMES = {
    "Quality / durability":   ["broke", "broken", "cheap", "flimsy", "stopped", "defective", "quality", "fell", "cracked", "ripped"],
    "Does not work":          ["work", "working", "worked", "dead", "useless", "failed", "malfunction"],
    "Size / fit":             ["size", "small", "large", "fit", "tight", "loose", "tiny", "big"],
    "Price / value":          ["price", "money", "expensive", "overpriced", "waste", "worth", "cost"],
    "Delivery / packaging":   ["shipping", "delivery", "arrived", "package", "packaging", "damaged", "box", "late"],
    "Not as described":       ["description", "described", "picture", "advertised", "fake", "misleading", "different"],
    "Return / refund":        ["return", "returned", "returning", "refund", "replacement", "exchange"],
    "Smell / taste / skin":   ["smell", "smells", "taste", "rash", "itchy", "irritation", "sticky"],
    "Battery / charging":     ["battery", "charge", "charging", "charger"],
}
negative = reviews.filter(F.col("sentiment") == "Negative")
theme_cols = [F.arrays_overlap("tokens", F.array(*[F.lit(w) for w in kw])).cast("int").alias(theme)
              for theme, kw in COMPLAINT_THEMES.items()]
flags = negative.select("year", "category", *theme_cols)
complaints = (flags.groupBy("year", "category")
                   .agg(F.count("*").alias("negative_reviews"),
                        *[F.sum(F.col(f"`{t}`")).alias(t) for t in COMPLAINT_THEMES]))
complaints_long = complaints.select(
    "year", "category", "negative_reviews",
    F.explode(F.array(*[F.struct(F.lit(t).alias("theme"), F.col(f"`{t}`").alias("mentions"))
                        for t in COMPLAINT_THEMES])).alias("x")
).select("year", "category", "negative_reviews", "x.theme", "x.mentions") \
 .withColumn("pct_of_negative", F.round(F.col("mentions") * 100 / F.col("negative_reviews"), 2))
save(complaints_long, "agg_complaints")
display(spark.table(f"{DB}.agg_complaints")
        .groupBy("theme").pivot("year").agg(F.round(F.sum("mentions") * 100 / F.sum("negative_reviews"), 1))
        .orderBy("theme"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 9: Check - all dashboard tables

# COMMAND ----------

for t in ["reviews_scored", "agg_year_summary", "agg_year_category", "agg_monthly",
          "agg_brand", "agg_product_year", "agg_top_words", "agg_complaints", "model_metrics"]:
    print(f"{t:<20} {spark.table(f'{DB}.{t}').count():>10,} rows")
