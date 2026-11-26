# Databricks notebook source
# MAGIC %md
# MAGIC # 07 - Interactive Review Analytics Dashboard (Databricks Native)
# MAGIC Reads directly from the Unity Catalog Delta tables in `workspace.review_analytics`.
# MAGIC
# MAGIC **How to present as a Databricks Dashboard:**
# MAGIC 1. Run all cells once (`Run all`).
# MAGIC 2. Use the **Widgets** at the top (`year`, `category`, `product_keyword`, `verdict`) to filter.
# MAGIC 3. Click **View -> Dashboard** (or add cell outputs to a Databricks Dashboard) for a clean presentation view.

# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 1: Setup & Interactive Filter Widgets

# COMMAND ----------

from pyspark.sql import functions as F
import plotly.express as px
import plotly.graph_objects as go

DB = "workspace.review_analytics"
CATEGORIES = [
    "All",
    "Electronics",
    "Home and Kitchen",
    "Health and Household",
    "Beauty and Personal Care",
    "Sports and Outdoors",
    "Books",
]

dbutils.widgets.dropdown("year", "All", ["All", "2023", "2024", "2025", "2026"], "1. Year")
dbutils.widgets.dropdown("category", "All", CATEGORIES, "2. Category")
dbutils.widgets.text("product_keyword", "echo dot", "3. Product Search")
dbutils.widgets.dropdown("verdict", "All", ["All", "BUY", "DON'T BUY", "INSUFFICIENT DATA"], "4. Verdict Filter")

sel_year = dbutils.widgets.get("year")
sel_cat = dbutils.widgets.get("category")
sel_kw = dbutils.widgets.get("product_keyword").strip().lower()
sel_verdict = dbutils.widgets.get("verdict")

print(f"Active Filters -> Year: {sel_year} | Category: {sel_cat} | Product: '{sel_kw}' | Verdict: {sel_verdict}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 2: Executive KPIs & 4-Year Sentiment Trajectory

# COMMAND ----------

yr_summary = spark.table(f"{DB}.agg_year_summary").toPandas()
yr_only = yr_summary[yr_summary["year"] != "All"].sort_values("year")

if sel_cat == "All":
    kpi = yr_summary[yr_summary["year"] == sel_year].iloc[0]
    total_rev, pos_pct, neu_pct, neg_pct, avg_rat = (
        int(kpi["reviews"]), float(kpi["positive_pct"]), float(kpi["neutral_pct"]),
        float(kpi["negative_pct"]), float(kpi["avg_rating"])
    )
else:
    yc_df = spark.table(f"{DB}.agg_year_category").filter(F.col("category") == sel_cat).toPandas()
    if sel_year != "All":
        yc_df = yc_df[yc_df["year"] == int(sel_year)]
    total_rev = int(yc_df["reviews"].sum())
    pos_pct = round(yc_df["positive"].sum() * 100 / total_rev, 2)
    neu_pct = round(yc_df["neutral"].sum() * 100 / total_rev, 2)
    neg_pct = round(yc_df["negative"].sum() * 100 / total_rev, 2)
    avg_rat = round((yc_df["avg_rating"] * yc_df["reviews"]).sum() / total_rev, 2)

print("=" * 68)
print(f"  SCOPE: Year = {sel_year}  |  Category = {sel_cat}")
print(f"  Total Reviews : {total_rev:,}")
print(f"  Positive      : {pos_pct:.1f}%   |   Neutral : {neu_pct:.1f}%   |   Negative : {neg_pct:.1f}%")
print(f"  Average Rating: {avg_rat:.2f} / 5.0")
print("=" * 68)

fig_yr = go.Figure()
for name, col, clr in [
    ("Positive", "positive_pct", "#10b981"),
    ("Neutral", "neutral_pct", "#f59e0b"),
    ("Negative", "negative_pct", "#f43f5e"),
]:
    fig_yr.add_trace(go.Bar(
        name=name, x=yr_only["year"], y=yr_only[col],
        marker_color=clr, text=yr_only[col].map(lambda v: f"{v:.1f}%"), textposition="inside",
    ))
fig_yr.update_layout(
    barmode="stack", template="plotly_dark", height=380,
    title="Four-Year Sentiment Distribution (2023–2026)",
    yaxis_title="Share of Reviews (%)", xaxis_title="Year",
)
fig_yr.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 3: Category Performance & Monthly Trend

# COMMAND ----------

yc_all = spark.table(f"{DB}.agg_year_category").toPandas()
yc_all["year_str"] = yc_all["year"].astype(str)
if sel_cat != "All":
    yc_all = yc_all[yc_all["category"] == sel_cat]

fig_cat = px.bar(
    yc_all.sort_values(["category", "year_str"]),
    x="category", y="positive_pct", color="year_str", barmode="group",
    text_auto=".1f", template="plotly_dark", height=380,
    title="Category-wise Positive Sentiment Across 4 Years",
    labels={"positive_pct": "Positive (%)", "category": "Category", "year_str": "Year"},
)
fig_cat.update_layout(yaxis_range=[65, 92])
fig_cat.show()

m_df = spark.table(f"{DB}.agg_monthly")
if sel_cat != "All":
    m_df = m_df.filter(F.col("category") == sel_cat)
if sel_year != "All":
    m_df = m_df.filter(F.col("year") == int(sel_year))

m_pd = (
    m_df.groupBy("review_month")
    .agg(F.sum("reviews").alias("reviews"),
         F.round(F.sum("positive") * 100.0 / F.sum("reviews"), 2).alias("positive_pct"))
    .orderBy("review_month")
    .toPandas()
)
fig_m = px.line(
    m_pd, x="review_month", y="positive_pct", markers=True, template="plotly_dark", height=340,
    title=f"Monthly Positive Sentiment Trend ({sel_cat} · {sel_year})",
    labels={"review_month": "Month", "positive_pct": "Positive (%)"},
)
fig_m.update_traces(line_color="#2dd4bf")
fig_m.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 4: 1–5 Star Distribution & Customer Complaint Themes

# COMMAND ----------

rev_df = spark.table(f"{DB}.reviews_scored")
if sel_cat != "All":
    rev_df = rev_df.filter(F.col("category") == sel_cat)
if sel_year != "All":
    rev_df = rev_df.filter(F.col("year") == int(sel_year))

stars_pd = (
    rev_df.withColumn("stars", F.col("rating").cast("int"))
    .groupBy("stars").count().orderBy("stars").toPandas()
)
stars_pd["star_label"] = stars_pd["stars"].map(lambda s: f"{s} Star" if s == 1 else f"{s} Stars")
fig_stars = px.bar(
    stars_pd, x="star_label", y="count", text_auto=",", template="plotly_dark", height=340,
    color="star_label",
    color_discrete_map={"1 Star": "#f43f5e", "2 Stars": "#fb923c", "3 Stars": "#f59e0b", "4 Stars": "#60a5fa", "5 Stars": "#10b981"},
    title=f"Star Rating Distribution ({sel_cat} · {sel_year})",
)
fig_stars.update_layout(showlegend=False, xaxis_title="", yaxis_title="Reviews")
fig_stars.show()

cmp_df = spark.table(f"{DB}.agg_complaints")
if sel_cat != "All":
    cmp_df = cmp_df.filter(F.col("category") == sel_cat)
if sel_year != "All":
    cmp_df = cmp_df.filter(F.col("year") == int(sel_year))

cmp_pd = (
    cmp_df.groupBy("theme")
    .agg(F.round(F.sum("mentions") * 100.0 / F.sum("negative_reviews"), 2).alias("pct_of_negative"))
    .orderBy("pct_of_negative")
    .toPandas()
)
fig_cmp = px.bar(
    cmp_pd, x="pct_of_negative", y="theme", orientation="h", text_auto=".1f",
    color="pct_of_negative", color_continuous_scale="Reds", template="plotly_dark", height=360,
    title=f"Customer Complaint Themes (% of Negative Reviews · {sel_cat} · {sel_year})",
    labels={"pct_of_negative": "% of Negative Reviews", "theme": ""},
)
fig_cmp.update_layout(coloraxis_showscale=False)
fig_cmp.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 5: Top Positive & Negative Words (Lift > 1.3)

# COMMAND ----------

words_df = spark.table(f"{DB}.agg_top_words")
if sel_cat != "All":
    words_df = words_df.filter(F.col("category") == sel_cat)
if sel_year != "All":
    words_df = words_df.filter(F.col("year") == int(sel_year))

top_w_pd = (
    words_df.groupBy("sentiment", "word")
    .agg(F.sum("n").alias("mentions"), F.round(F.avg("lift"), 2).alias("avg_lift"))
    .orderBy(F.desc("mentions"))
    .limit(30)
    .toPandas()
)
fig_w = px.bar(
    top_w_pd.sort_values("mentions"),
    x="mentions", y="word", color="sentiment", orientation="h",
    color_discrete_map={"Positive": "#10b981", "Negative": "#f43f5e"},
    hover_data=["avg_lift"], template="plotly_dark", height=480,
    title=f"Distinctive Sentiment Words (Lift > 1.3 · {sel_cat} · {sel_year})",
)
fig_w.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 6: ML Model Benchmark (Naive Bayes vs Logistic Regression vs Random Forest)

# COMMAND ----------

metrics_pd = spark.table(f"{DB}.model_metrics").orderBy(F.desc("macro_f1")).toPandas()
display(metrics_pd[["model", "accuracy", "macro_f1", "weighted_f1", "recall_negative", "recall_neutral", "recall_positive", "train_rows", "train_seconds", "is_best"]])

melted_m = metrics_pd.melt(
    id_vars="model",
    value_vars=["accuracy", "macro_f1", "weighted_f1", "recall_negative", "recall_neutral"],
    var_name="Metric", value_name="Score",
)
fig_ml = px.bar(
    melted_m, x="Metric", y="Score", color="model", barmode="group",
    text_auto=".3f", template="plotly_dark", height=360,
    title="PySpark MLlib Classifier Benchmark (80/20 Held-Out Test Set)",
)
fig_ml.update_layout(yaxis_range=[0, 1.05])
fig_ml.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 7: Product Search & BUY / DON'T BUY Advisor (from Phase 10–11 Table)

# COMMAND ----------

recs_df = spark.table(f"{DB}.product_recommendations")
if sel_cat != "All":
    recs_df = recs_df.filter(F.col("category") == sel_cat)
if sel_verdict != "All":
    recs_df = recs_df.filter(F.col("recommendation") == sel_verdict)
if sel_kw:
    recs_df = recs_df.filter(F.lower(F.coalesce(F.col("product_title"), F.lit(""))).contains(sel_kw))

top_matches = recs_df.orderBy(F.desc("reviews")).limit(20)
display(top_matches.select(
    "parent_asin", "product_title", "brand", "category", "reviews",
    "avg_rating", "positive_pct", "buy_score", "recommendation", "confidence", "reason_summary"
))

# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 8: Live Stream Monitor Summary (from Phase 12 Table)

# COMMAND ----------

try:
    stream_df = spark.table(f"{DB}.stream_scored")
    display(
        stream_df.groupBy("category", "predicted_sentiment")
        .count()
        .orderBy("category", "predicted_sentiment")
    )
    display(
        stream_df.filter(F.col("is_negative_alert") == True)
        .select("ingested_at", "category", "brand", "product_title", "rating", "predicted_sentiment", "title")
        .orderBy(F.desc("ingested_at"))
        .limit(15)
    )
except Exception as e:
    print("Stream table not created yet. Run 06_streaming_reviews.py first:", e)
