# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - Product Search & BUY / DON'T BUY Recommendation Engine (Phases 10-11)
# MAGIC Aggregates per-product sentiment, aspect-level features, complaint themes, and computes an
# MAGIC explainable BUY / DON'T BUY / INSUFFICIENT DATA recommendation score (0-100).

# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 1: Setup

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

DB = "workspace.review_analytics"
reviews = spark.table(f"{DB}.reviews_scored")
products = spark.table(f"{DB}.products")

def save(df, name):
    (df.write.format("delta").mode("overwrite")
       .option("overwriteSchema", "true").saveAsTable(f"{DB}.{name}"))
    print(f"{name}: {spark.table(f'{DB}.{name}').count():,} rows")

print("Setup done")

# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 2: Product-level statistics (overall + first-year vs recent-year trend)

# COMMAND ----------

prod_overall = (
    reviews.groupBy("parent_asin", "category")
    .agg(
        F.first("product_title", ignorenulls=True).alias("product_title"),
        F.first("brand", ignorenulls=True).alias("brand"),
        F.first("price", ignorenulls=True).alias("price"),
        F.first("product_avg_rating", ignorenulls=True).alias("catalog_avg_rating"),
        F.first("product_rating_count", ignorenulls=True).alias("catalog_rating_count"),
        F.first("product_features", ignorenulls=True).alias("product_features"),
        F.count("*").alias("reviews"),
        F.round(F.avg("rating"), 3).alias("avg_rating"),
        F.sum(F.when(F.col("sentiment") == "Positive", 1).otherwise(0)).alias("positive"),
        F.sum(F.when(F.col("sentiment") == "Neutral", 1).otherwise(0)).alias("neutral"),
        F.sum(F.when(F.col("sentiment") == "Negative", 1).otherwise(0)).alias("negative"),
        F.sum(F.when(F.col("predicted_sentiment") == "Positive", 1).otherwise(0)).alias("pred_positive"),
        F.sum(F.when(F.col("verified_purchase") == True, 1).otherwise(0)).alias("verified_count"),
        F.min("year").alias("first_year"),
        F.max("year").alias("recent_year"),
        F.countDistinct("year").alias("years_active"),
    )
    .withColumn("positive_pct", F.round(F.col("positive") * 100.0 / F.col("reviews"), 2))
    .withColumn("neutral_pct", F.round(F.col("neutral") * 100.0 / F.col("reviews"), 2))
    .withColumn("negative_pct", F.round(F.col("negative") * 100.0 / F.col("reviews"), 2))
    .withColumn("model_positive_pct", F.round(F.col("pred_positive") * 100.0 / F.col("reviews"), 2))
    .withColumn("verified_pct", F.round(F.col("verified_count") * 100.0 / F.col("reviews"), 2))
)

# Yearly positive % per product for trend & consistency
prod_yearly = (
    reviews.groupBy("parent_asin", "category", "year")
    .agg(
        F.count("*").alias("yr_reviews"),
        F.round(F.avg(F.when(F.col("sentiment") == "Positive", 100.0).otherwise(0.0)), 2).alias("yr_pos_pct"),
    )
)

w_first = Window.partitionBy("parent_asin", "category").orderBy(F.asc("year"))
w_last = Window.partitionBy("parent_asin", "category").orderBy(F.desc("year"))

first_yr_df = (
    prod_yearly.withColumn("rn", F.row_number().over(w_first))
    .filter(F.col("rn") == 1)
    .select("parent_asin", "category", F.col("yr_pos_pct").alias("first_year_pos_pct"))
)
recent_yr_df = (
    prod_yearly.withColumn("rn", F.row_number().over(w_last))
    .filter(F.col("rn") == 1)
    .select("parent_asin", "category", F.col("yr_pos_pct").alias("recent_year_pos_pct"))
)
consistency_df = (
    prod_yearly.groupBy("parent_asin", "category")
    .agg(F.coalesce(F.round(F.stddev("yr_pos_pct"), 2), F.lit(0.0)).alias("yearly_pos_std"))
)

product_stats = (
    prod_overall
    .join(first_yr_df, ["parent_asin", "category"], "left")
    .join(recent_yr_df, ["parent_asin", "category"], "left")
    .join(consistency_df, ["parent_asin", "category"], "left")
    .withColumn("sentiment_trend", F.round(F.col("recent_year_pos_pct") - F.col("first_year_pos_pct"), 2))
)

save(product_stats, "product_stats")

# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 3: Aspect / feature-level sentiment per product

# COMMAND ----------

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

aspect_flags = reviews.select(
    "parent_asin", "category", "sentiment",
    *[F.arrays_overlap("tokens", F.array(*[F.lit(w) for w in kw])).cast("int").alias(asp)
      for asp, kw in ASPECT_KEYWORDS.items()]
)

aspect_long = (
    aspect_flags.select(
        "parent_asin", "category", "sentiment",
        F.explode(F.array(*[
            F.struct(F.lit(asp).alias("aspect"), F.col(f"`{asp}`").alias("mentioned"))
            for asp in ASPECT_KEYWORDS
        ])).alias("a")
    )
    .filter(F.col("a.mentioned") == 1)
    .groupBy("parent_asin", "category", F.col("a.aspect").alias("aspect"))
    .agg(
        F.count("*").alias("mentions"),
        F.sum(F.when(F.col("sentiment") == "Positive", 1).otherwise(0)).alias("pos_mentions"),
        F.sum(F.when(F.col("sentiment") == "Negative", 1).otherwise(0)).alias("neg_mentions"),
    )
    .withColumn("positive_pct", F.round(F.col("pos_mentions") * 100.0 / F.col("mentions"), 1))
    .withColumn("negative_pct", F.round(F.col("neg_mentions") * 100.0 / F.col("mentions"), 1))
)

save(aspect_long, "product_aspect_sentiment")

# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 4: Complaint themes per product

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

neg_reviews = reviews.filter(F.col("sentiment") == "Negative")
neg_flags = neg_reviews.select(
    "parent_asin", "category",
    *[F.arrays_overlap("tokens", F.array(*[F.lit(w) for w in kw])).cast("int").alias(t)
      for t, kw in COMPLAINT_THEMES.items()]
)

prod_neg_totals = neg_reviews.groupBy("parent_asin", "category").agg(F.count("*").alias("negative_reviews"))

prod_complaints = (
    neg_flags.select(
        "parent_asin", "category",
        F.explode(F.array(*[
            F.struct(F.lit(t).alias("theme"), F.col(f"`{t}`").alias("hit"))
            for t in COMPLAINT_THEMES
        ])).alias("c")
    )
    .filter(F.col("c.hit") == 1)
    .groupBy("parent_asin", "category", F.col("c.theme").alias("theme"))
    .agg(F.count("*").alias("mentions"))
    .join(prod_neg_totals, ["parent_asin", "category"], "inner")
    .withColumn("pct_of_negative", F.round(F.col("mentions") * 100.0 / F.col("negative_reviews"), 1))
)

save(prod_complaints, "product_complaints")

# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 5: Explainable BUY / DON'T BUY / INSUFFICIENT DATA scoring

# COMMAND ----------

stats = spark.table(f"{DB}.product_stats")

# Component scores (each 0-100):
#   positive_pct (20%) + model_positive_pct (15%) + rating_score (20%) +
#   recent_year_pos_pct (15%) + trend_score (10%) + consistency_score (10%) + complaint_inv (10%)
scored_rec = (
    stats
    .withColumn("rating_score", F.least(F.lit(100.0), F.greatest(F.lit(0.0), (F.col("avg_rating") - 1.0) / 4.0 * 100.0)))
    .withColumn("trend_score", F.least(F.lit(100.0), F.greatest(F.lit(0.0), 50.0 + F.col("sentiment_trend") * 1.5)))
    .withColumn("consistency_score", F.least(F.lit(100.0), F.greatest(F.lit(0.0), 100.0 - F.col("yearly_pos_std") * 2.0)))
    .withColumn("complaint_inv_score", F.least(F.lit(100.0), F.greatest(F.lit(0.0), 100.0 - F.col("negative_pct"))))
    .withColumn(
        "buy_score",
        F.round(
            0.20 * F.col("positive_pct")
            + 0.15 * F.col("model_positive_pct")
            + 0.20 * F.col("rating_score")
            + 0.15 * F.coalesce(F.col("recent_year_pos_pct"), F.col("positive_pct"))
            + 0.10 * F.col("trend_score")
            + 0.10 * F.col("consistency_score")
            + 0.10 * F.col("complaint_inv_score"),
            1,
        ),
    )
    .withColumn(
        "recommendation",
        F.when(F.col("reviews") < 20, F.lit("INSUFFICIENT DATA"))
         .when(F.col("buy_score") >= 65.0, F.lit("BUY"))
         .otherwise(F.lit("DON'T BUY")),
    )
    .withColumn(
        "confidence",
        F.when((F.col("reviews") >= 100) & (F.col("years_active") >= 2), F.lit("High"))
         .when(F.col("reviews") >= 50, F.lit("Medium"))
         .otherwise(F.lit("Low")),
    )
)

# Attach top strength aspect and top complaint theme for plain-English explanation
w_asp = Window.partitionBy("parent_asin", "category").orderBy(F.desc("mentions"))
top_aspect = (
    spark.table(f"{DB}.product_aspect_sentiment")
    .filter(F.col("mentions") >= 3)
    .withColumn("rn", F.row_number().over(w_asp))
    .filter(F.col("rn") == 1)
    .select(
        "parent_asin", "category",
        F.col("aspect").alias("top_aspect"),
        F.col("positive_pct").alias("top_aspect_pos_pct"),
    )
)

w_cmp = Window.partitionBy("parent_asin", "category").orderBy(F.desc("mentions"))
top_complaint = (
    spark.table(f"{DB}.product_complaints")
    .withColumn("rn", F.row_number().over(w_cmp))
    .filter(F.col("rn") == 1)
    .select(
        "parent_asin", "category",
        F.col("theme").alias("top_complaint"),
        F.col("pct_of_negative").alias("top_complaint_pct"),
    )
)

final_rec = (
    scored_rec
    .join(top_aspect, ["parent_asin", "category"], "left")
    .join(top_complaint, ["parent_asin", "category"], "left")
    .withColumn(
        "reason_summary",
        F.concat_ws(
            " | ",
            F.concat(F.lit("Avg rating "), F.round("avg_rating", 2), F.lit("/5 across "), F.col("reviews"), F.lit(" reviews")),
            F.concat(F.col("positive_pct"), F.lit("% positive (model: "), F.col("model_positive_pct"), F.lit("%)")),
            F.when(F.col("sentiment_trend") >= 5, F.concat(F.lit("Improving trend (+"), F.col("sentiment_trend"), F.lit("%)")))
             .when(F.col("sentiment_trend") <= -5, F.concat(F.lit("Declining trend ("), F.col("sentiment_trend"), F.lit("%)")))
             .otherwise(F.lit("Stable sentiment across years")),
            F.when(F.col("top_aspect").isNotNull(),
                   F.concat(F.lit("Top aspect: "), F.col("top_aspect"), F.lit(" ("), F.col("top_aspect_pos_pct"), F.lit("% positive)"))),
            F.when(F.col("top_complaint").isNotNull(),
                   F.concat(F.lit("Main complaint: "), F.col("top_complaint"), F.lit(" ("), F.col("top_complaint_pct"), F.lit("% of negatives)"))),
        ),
    )
)

save(final_rec, "product_recommendations")
display(spark.table(f"{DB}.product_recommendations").groupBy("recommendation").count().orderBy("recommendation"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 6: Interactive Search Widgets

# COMMAND ----------

dbutils.widgets.text("product_name", "echo dot", "Product Name")
dbutils.widgets.text("brand", "", "Brand (optional)")
dbutils.widgets.dropdown("category", "All", ["All", "Electronics", "Home and Kitchen", "Health and Household",
                                              "Beauty and Personal Care", "Sports and Outdoors", "Books"], "Category")

q_name = dbutils.widgets.get("product_name").strip().lower()
q_brand = dbutils.widgets.get("brand").strip().lower()
q_cat = dbutils.widgets.get("category")

recs = spark.table(f"{DB}.product_recommendations")
if q_cat != "All":
    recs = recs.filter(F.col("category") == q_cat)
if q_name:
    recs = recs.filter(F.lower(F.coalesce(F.col("product_title"), F.lit(""))).contains(q_name))
if q_brand:
    recs = recs.filter(F.lower(F.coalesce(F.col("brand"), F.lit(""))).contains(q_brand))

matches = recs.orderBy(F.desc("reviews")).limit(15)
display(matches.select("parent_asin", "product_title", "brand", "category", "reviews",
                       "avg_rating", "positive_pct", "buy_score", "recommendation", "confidence"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 7: Full Report for Top Match (or set SELECTED = "<parent_asin>")

# COMMAND ----------

SELECTED = ""  # Optional: paste a specific parent_asin here
target_row = (
    spark.table(f"{DB}.product_recommendations").filter(F.col("parent_asin") == SELECTED).limit(1).collect()
    if SELECTED else matches.limit(1).collect()
)

if not target_row:
    print("No matching product found. Adjust the search widgets above and re-run Cells 6 & 7.")
else:
    p = target_row[0]
    print(f"=== {p['product_title']} ({p['parent_asin']}) ===")
    print(f"Brand: {p['brand']}  |  Category: {p['category']}  |  Price: ${p['price'] or 'N/A'}")
    print(f"Verdict: {p['recommendation']}  (Score: {p['buy_score']}/100, Confidence: {p['confidence']})")
    print(f"Reviews: {p['reviews']:,}  |  Avg Rating: {p['avg_rating']}  |  Positive: {p['positive_pct']}%  |  Model Pos: {p['model_positive_pct']}%")
    print(f"Why: {p['reason_summary']}")
    display(spark.table(f"{DB}.product_aspect_sentiment").filter(F.col("parent_asin") == p["parent_asin"]).orderBy(F.desc("mentions")))
    display(spark.table(f"{DB}.product_complaints").filter(F.col("parent_asin") == p["parent_asin"]).orderBy(F.desc("mentions")))
