# Databricks notebook source
# MAGIC %md
# MAGIC # 07 - Interactive Review Analytics Dashboard (Databricks Native)
# MAGIC Reads directly from the Unity Catalog Delta tables in `workspace.review_analytics`.
# MAGIC
# MAGIC **How to use during live demo:**
# MAGIC 1. Use the interactive filter widgets at the top (`1. Year`, `2. Category`, `3. Product Search`, `4. Verdict Filter`).
# MAGIC 2. Click **Run all** to refresh every KPI, Spark chart, ML benchmark, Product Advisor table, and Stream monitor.
# MAGIC 3. Open **View -> Dashboard** -> **Review Sentiment Analytics Dashboard** for a clean executive presentation layout.

# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 1: Interactive Dashboard Filter Widgets & Palette Configuration

# COMMAND ----------

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from pyspark.sql import functions as F

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

BG_FIG = "#0b1020"
BG_AX = "#111827"
GRID_CLR = "#1f2937"
TEXT_CLR = "#f8fafc"
MUTED_CLR = "#94a3b8"
CLR_POS = "#10b981"
CLR_NEU = "#f59e0b"
CLR_NEG = "#f43f5e"
CLR_INDIGO = "#6366f1"
CLR_TEAL = "#2dd4bf"
CLR_BLUE = "#60a5fa"

def style_ax(ax, title, xlabel="", ylabel=""):
    ax.set_facecolor(BG_AX)
    ax.tick_params(colors=MUTED_CLR, labelsize=10)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#334155")
    ax.grid(axis="y", color=GRID_CLR, linestyle="--", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, color=TEXT_CLR, fontsize=13, fontweight="bold", pad=14, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED_CLR, fontsize=10, labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED_CLR, fontsize=10, labelpad=8)

print(f"Active Dashboard Filters -> Year: {sel_year} | Category: {sel_cat} | Product Search: '{sel_kw}' | Verdict: {sel_verdict}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 2: Executive KPI Scorecard (Spark Delta Gold)

# COMMAND ----------

yr_summary = spark.table(f"{DB}.agg_year_summary").toPandas()

if sel_cat == "All":
    kpi = yr_summary[yr_summary["year"] == sel_year].iloc[0]
    total_rev = int(kpi["reviews"])
    pos_pct = float(kpi["positive_pct"])
    neu_pct = float(kpi["neutral_pct"])
    neg_pct = float(kpi["negative_pct"])
    avg_rat = float(kpi["avg_rating"])
else:
    yc_df = spark.table(f"{DB}.agg_year_category").filter(F.col("category") == sel_cat).toPandas()
    if sel_year != "All":
        yc_df = yc_df[yc_df["year"] == int(sel_year)]
    total_rev = int(yc_df["reviews"].sum())
    pos_pct = round(yc_df["positive"].sum() * 100.0 / max(total_rev, 1), 2)
    neu_pct = round(yc_df["neutral"].sum() * 100.0 / max(total_rev, 1), 2)
    neg_pct = round(yc_df["negative"].sum() * 100.0 / max(total_rev, 1), 2)
    avg_rat = round((yc_df["avg_rating"] * yc_df["reviews"]).sum() / max(total_rev, 1), 2)

fig, ax = plt.subplots(figsize=(13, 2.5), dpi=130)
fig.patch.set_facecolor(BG_FIG)
ax.set_facecolor(BG_FIG)
ax.axis("off")

ax.text(
    0.0, 0.94,
    f"ReviewSense Executive KPIs   |   Scope: Year = {sel_year}   ·   Category = {sel_cat}",
    color=TEXT_CLR, fontsize=13, fontweight="bold", va="top",
)

cards = [
    ("TOTAL REVIEWS", f"{total_rev:,}", "Gold Delta Lake", CLR_INDIGO),
    ("POSITIVE SHARE", f"{pos_pct:.1f}%", "4-5 Star Verified", CLR_POS),
    ("NEUTRAL / NEGATIVE", f"{neu_pct:.1f}% / {neg_pct:.1f}%", "Risk & Watch Segment", CLR_NEG),
    ("AVERAGE RATING", f"{avg_rat:.2f} / 5.0", "Weighted Customer Score", CLR_TEAL),
]

for idx, (label, val, sub, accent) in enumerate(cards):
    x0 = idx * 0.255
    rect = patches.FancyBboxPatch(
        (x0, 0.06), 0.235, 0.72,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor=BG_AX, edgecolor=accent, linewidth=1.6,
    )
    ax.add_patch(rect)
    ax.text(x0 + 0.02, 0.68, label, color=MUTED_CLR, fontsize=9, fontweight="bold", va="top")
    ax.text(x0 + 0.02, 0.44, val, color=TEXT_CLR, fontsize=18, fontweight="bold", va="center")
    ax.text(x0 + 0.02, 0.16, sub, color=accent, fontsize=9, va="bottom")

ax.set_xlim(-0.01, 1.02)
ax.set_ylim(0, 1.02)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 3: Four-Year Sentiment Distribution (2023–2026)

# COMMAND ----------

yr_only = yr_summary[yr_summary["year"] != "All"].sort_values("year")
x = np.arange(len(yr_only))
pos_vals = yr_only["positive_pct"].values
neu_vals = yr_only["neutral_pct"].values
neg_vals = yr_only["negative_pct"].values

fig, ax = plt.subplots(figsize=(11, 4.4), dpi=130)
fig.patch.set_facecolor(BG_FIG)

b1 = ax.bar(x, pos_vals, width=0.52, color=CLR_POS, label="Positive")
b2 = ax.bar(x, neu_vals, width=0.52, bottom=pos_vals, color=CLR_NEU, label="Neutral")
b3 = ax.bar(x, neg_vals, width=0.52, bottom=pos_vals + neu_vals, color=CLR_NEG, label="Negative")

for i in range(len(x)):
    ax.text(x[i], pos_vals[i] / 2, f"{pos_vals[i]:.1f}%", ha="center", va="center", color="#052e16", fontweight="bold", fontsize=10)
    ax.text(x[i], pos_vals[i] + neu_vals[i] / 2, f"{neu_vals[i]:.1f}%", ha="center", va="center", color="#1e293b", fontweight="bold", fontsize=9)
    ax.text(x[i], pos_vals[i] + neu_vals[i] + neg_vals[i] / 2, f"{neg_vals[i]:.1f}%", ha="center", va="center", color=TEXT_CLR, fontweight="bold", fontsize=10)

ax.set_xticks(x)
ax.set_xticklabels(yr_only["year"].tolist(), color=TEXT_CLR, fontsize=11, fontweight="bold")
ax.set_ylim(0, 108)
style_ax(ax, "Four-Year Sentiment Distribution (2023–2026)", xlabel="Year", ylabel="Share of Reviews (%)")
ax.legend(loc="upper right", frameon=True, facecolor=BG_AX, edgecolor="#334155", labelcolor=TEXT_CLR, ncol=3)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 4: Category-wise Positive Sentiment Across 4 Years

# COMMAND ----------

yc_all = spark.table(f"{DB}.agg_year_category").toPandas()
if sel_cat != "All":
    yc_all = yc_all[yc_all["category"] == sel_cat]

piv = yc_all.pivot(index="category", columns="year", values="positive_pct").sort_index()
years_cols = sorted(piv.columns.tolist())
cats = piv.index.tolist()
y_pos = np.arange(len(cats))
bar_w = 0.18
yr_colors = [CLR_INDIGO, CLR_BLUE, CLR_TEAL, CLR_POS]

fig, ax = plt.subplots(figsize=(12, 4.8), dpi=130)
fig.patch.set_facecolor(BG_FIG)

for i, yr_col in enumerate(years_cols):
    offset = (i - (len(years_cols) - 1) / 2) * bar_w
    vals = piv[yr_col].values
    bars = ax.barh(y_pos + offset, vals, height=bar_w * 0.9, color=yr_colors[i % len(yr_colors)], label=str(yr_col))
    for bar, v in zip(bars, vals):
        ax.text(v + 0.35, bar.get_y() + bar.get_height() / 2, f"{v:.1f}%", va="center", color=TEXT_CLR, fontsize=8.5)

ax.set_yticks(y_pos)
ax.set_yticklabels(cats, color=TEXT_CLR, fontsize=10, fontweight="bold")
ax.set_xlim(62, 95)
style_ax(ax, "Category-wise Positive Sentiment Across 4 Years (%)", xlabel="Positive Reviews (%)")
ax.grid(axis="x", color=GRID_CLR, linestyle="--", linewidth=0.7, alpha=0.8)
ax.grid(axis="y", visible=False)
ax.legend(loc="lower right", frameon=True, facecolor=BG_AX, edgecolor="#334155", labelcolor=TEXT_CLR, ncol=4)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 5: Monthly Positive Sentiment Trajectory

# COMMAND ----------

m_df = spark.table(f"{DB}.agg_monthly")
if sel_cat != "All":
    m_df = m_df.filter(F.col("category") == sel_cat)
if sel_year != "All":
    m_df = m_df.filter(F.col("year") == int(sel_year))

m_pd = (
    m_df.groupBy("review_month")
    .agg(
        F.sum("reviews").alias("reviews"),
        F.round(F.sum("positive") * 100.0 / F.sum("reviews"), 2).alias("positive_pct"),
    )
    .orderBy("review_month")
    .toPandas()
)

fig, ax = plt.subplots(figsize=(12, 4.2), dpi=130)
fig.patch.set_facecolor(BG_FIG)

xm = np.arange(len(m_pd))
ym = m_pd["positive_pct"].values
ax.plot(xm, ym, color=CLR_TEAL, linewidth=2.3, marker="o", markersize=4.5)
ax.fill_between(xm, ym, max(float(ym.min()) - 4, 40), color=CLR_TEAL, alpha=0.14)

step = max(1, len(xm) // 10)
tick_idx = xm[::step]
ax.set_xticks(tick_idx)
ax.set_xticklabels(m_pd["review_month"].iloc[tick_idx].tolist(), rotation=25, ha="right", color=MUTED_CLR, fontsize=9)
ax.set_ylim(max(float(ym.min()) - 4, 40), min(float(ym.max()) + 4, 96))
style_ax(ax, f"Monthly Positive Sentiment Trend ({sel_cat} · {sel_year})", xlabel="Month", ylabel="Positive Share (%)")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 6: 1–5 Star Rating Distribution

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
stars_pd["label"] = stars_pd["stars"].map(lambda s: f"{s} Star" if s == 1 else f"{s} Stars")
star_clrs = {1: CLR_NEG, 2: "#fb923c", 3: CLR_NEU, 4: CLR_BLUE, 5: CLR_POS}

fig, ax = plt.subplots(figsize=(10.5, 4.2), dpi=130)
fig.patch.set_facecolor(BG_FIG)

bars = ax.bar(
    stars_pd["label"],
    stars_pd["count"],
    width=0.52,
    color=[star_clrs.get(int(s), CLR_INDIGO) for s in stars_pd["stars"]],
)
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h * 1.015, f"{int(h):,}", ha="center", va="bottom", color=TEXT_CLR, fontsize=9.5, fontweight="bold")

ax.set_ylim(0, float(stars_pd["count"].max()) * 1.14)
style_ax(ax, f"Star Rating Distribution ({sel_cat} · {sel_year})", ylabel="Number of Reviews")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 7: Customer Complaint Themes (% of Negative Reviews)

# COMMAND ----------

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

fig, ax = plt.subplots(figsize=(11, 4.4), dpi=130)
fig.patch.set_facecolor(BG_FIG)

bars = ax.barh(cmp_pd["theme"], cmp_pd["pct_of_negative"], height=0.58, color=CLR_NEG)
for bar in bars:
    w = bar.get_width()
    ax.text(w + 0.4, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", color=TEXT_CLR, fontsize=9.5, fontweight="bold")

ax.set_xlim(0, float(cmp_pd["pct_of_negative"].max()) * 1.18)
style_ax(ax, f"Customer Complaint Themes (% of Negative Reviews · {sel_cat} · {sel_year})", xlabel="% of Negative Reviews")
ax.grid(axis="x", color=GRID_CLR, linestyle="--", linewidth=0.7, alpha=0.8)
ax.grid(axis="y", visible=False)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 8: Top Positive & Negative Distinctive Words (Lift > 1.3)

# COMMAND ----------

words_df = spark.table(f"{DB}.agg_top_words")
if sel_cat != "All":
    words_df = words_df.filter(F.col("category") == sel_cat)
if sel_year != "All":
    words_df = words_df.filter(F.col("year") == int(sel_year))

pos_w = (
    words_df.filter(F.col("sentiment") == "Positive")
    .groupBy("word")
    .agg(F.sum("n").alias("mentions"), F.round(F.avg("lift"), 2).alias("avg_lift"))
    .orderBy(F.desc("mentions"))
    .limit(12)
    .toPandas()
    .sort_values("mentions")
)
neg_w = (
    words_df.filter(F.col("sentiment") == "Negative")
    .groupBy("word")
    .agg(F.sum("n").alias("mentions"), F.round(F.avg("lift"), 2).alias("avg_lift"))
    .orderBy(F.desc("mentions"))
    .limit(12)
    .toPandas()
    .sort_values("mentions")
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8), dpi=130)
fig.patch.set_facecolor(BG_FIG)

b_pos = ax1.barh(pos_w["word"], pos_w["mentions"], color=CLR_POS, height=0.62)
for bar, lft in zip(b_pos, pos_w["avg_lift"]):
    w = bar.get_width()
    ax1.text(w * 1.02, bar.get_y() + bar.get_height() / 2, f"{int(w):,} ({lft}x)", va="center", color=TEXT_CLR, fontsize=8.5)
ax1.set_xlim(0, float(pos_w["mentions"].max()) * 1.28)
style_ax(ax1, f"Top Positive Words ({sel_cat} · {sel_year})", xlabel="Mentions (Avg Lift)")
ax1.grid(axis="x", color=GRID_CLR, linestyle="--", linewidth=0.7, alpha=0.8)
ax1.grid(axis="y", visible=False)

b_neg = ax2.barh(neg_w["word"], neg_w["mentions"], color=CLR_NEG, height=0.62)
for bar, lft in zip(b_neg, neg_w["avg_lift"]):
    w = bar.get_width()
    ax2.text(w * 1.02, bar.get_y() + bar.get_height() / 2, f"{int(w):,} ({lft}x)", va="center", color=TEXT_CLR, fontsize=8.5)
ax2.set_xlim(0, float(neg_w["mentions"].max()) * 1.28)
style_ax(ax2, f"Top Negative Words ({sel_cat} · {sel_year})", xlabel="Mentions (Avg Lift)")
ax2.grid(axis="x", color=GRID_CLR, linestyle="--", linewidth=0.7, alpha=0.8)
ax2.grid(axis="y", visible=False)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 9: PySpark MLlib Classifier Benchmark Chart (80/20 Test Split)

# COMMAND ----------

metrics_pd = spark.table(f"{DB}.model_metrics").orderBy(F.desc("macro_f1")).toPandas()
metric_cols = ["accuracy", "macro_f1", "weighted_f1", "recall_negative", "recall_neutral"]
metric_labels = ["Accuracy", "Macro F1", "Weighted F1", "Recall (Neg)", "Recall (Neu)"]

fig, ax = plt.subplots(figsize=(12, 4.4), dpi=130)
fig.patch.set_facecolor(BG_FIG)

xm = np.arange(len(metric_cols))
bw = 0.24
model_clrs = [CLR_INDIGO, CLR_TEAL, CLR_NEU]

for idx, (_, row) in enumerate(metrics_pd.iterrows()):
    offset = (idx - (len(metrics_pd) - 1) / 2) * bw
    vals = [float(row[c]) for c in metric_cols]
    bars = ax.bar(xm + offset, vals, width=bw * 0.9, color=model_clrs[idx % len(model_clrs)], label=row["model"])
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", va="bottom", color=TEXT_CLR, fontsize=8.5)

ax.set_xticks(xm)
ax.set_xticklabels(metric_labels, color=TEXT_CLR, fontsize=10, fontweight="bold")
ax.set_ylim(0, 1.06)
style_ax(ax, "PySpark MLlib Classifier Benchmark (80/20 Held-Out Test Set)", ylabel="Score (0.0 – 1.0)")
ax.legend(loc="upper right", frameon=True, facecolor=BG_AX, edgecolor="#334155", labelcolor=TEXT_CLR, ncol=3)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 10: MLlib Model Metrics & Training Runtime Table

# COMMAND ----------

display(metrics_pd[[
    "model", "accuracy", "macro_f1", "weighted_f1",
    "recall_negative", "recall_neutral", "recall_positive",
    "train_rows", "train_seconds", "is_best"
]])

# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 11: Product Search & BUY / DON'T BUY Advisor (Live Widget Filter)

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
# MAGIC ### CELL 12: Structured Streaming Category x Sentiment Summary (Phase 12 Delta Table)

# COMMAND ----------

try:
    stream_df = spark.table(f"{DB}.stream_scored")
    display(
        stream_df.groupBy("category", "predicted_sentiment")
        .count()
        .orderBy("category", "predicted_sentiment")
    )
except Exception as e:
    print("Stream table not created yet. Run 06_streaming_reviews.py first:", e)

# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 13: Structured Streaming Negative Review Alerts

# COMMAND ----------

try:
    stream_df = spark.table(f"{DB}.stream_scored")
    display(
        stream_df.filter(F.col("is_negative_alert") == True)
        .select("ingested_at", "category", "brand", "product_title", "rating", "predicted_sentiment", "title")
        .orderBy(F.desc("ingested_at"))
        .limit(15)
    )
except Exception as e:
    print("Stream table not created yet. Run 06_streaming_reviews.py first:", e)
