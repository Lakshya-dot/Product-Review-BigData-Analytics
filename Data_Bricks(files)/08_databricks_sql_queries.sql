-- ============================================================================
-- 08_databricks_sql_queries.sql
-- Ready-to-use SQL datasets for a Native Databricks AI/BI Dashboard (Lakeview)
-- Catalog / Schema: workspace.review_analytics
-- ============================================================================

-- 1. KPI Summary Dataset (Use for Counter Widgets + 4-Year Stacked Bar)
SELECT
    year,
    reviews,
    positive,
    neutral,
    negative,
    positive_pct,
    neutral_pct,
    negative_pct,
    avg_rating,
    products,
    brands
FROM workspace.review_analytics.agg_year_summary
ORDER BY year;

-- 2. Category Performance Across 4 Years (Use for Grouped Bar & Heatmap)
SELECT
    CAST(year AS STRING) AS year,
    category,
    reviews,
    avg_rating,
    positive_pct,
    neutral_pct,
    negative_pct
FROM workspace.review_analytics.agg_year_category
ORDER BY category, year;

-- 3. Monthly Sentiment & Volume Trend (Use for Area & Line Charts)
SELECT
    review_month,
    category,
    SUM(reviews) AS total_reviews,
    ROUND(SUM(positive) * 100.0 / SUM(reviews), 2) AS positive_pct,
    ROUND(SUM(negative) * 100.0 / SUM(reviews), 2) AS negative_pct
FROM workspace.review_analytics.agg_monthly
GROUP BY review_month, category
ORDER BY review_month, category;

-- 4. Star Rating Distribution (1-5 Stars)
SELECT
    CAST(year AS STRING) AS year,
    category,
    CAST(rating AS INT) AS stars,
    COUNT(*) AS reviews
FROM workspace.review_analytics.reviews_scored
GROUP BY year, category, CAST(rating AS INT)
ORDER BY year, category, stars;

-- 5. Customer Complaint Themes (% of Negative Reviews)
SELECT
    CAST(year AS STRING) AS year,
    category,
    theme,
    SUM(mentions) AS mentions,
    SUM(negative_reviews) AS negative_reviews,
    ROUND(SUM(mentions) * 100.0 / SUM(negative_reviews), 2) AS pct_of_negative
FROM workspace.review_analytics.agg_complaints
GROUP BY year, category, theme
ORDER BY pct_of_negative DESC;

-- 6. Top Brands Leaderboard (Min 30 Reviews)
SELECT
    CAST(year AS STRING) AS year,
    category,
    brand,
    reviews,
    avg_rating,
    positive_pct,
    negative_pct
FROM workspace.review_analytics.agg_brand
ORDER BY reviews DESC
LIMIT 200;

-- 7. ML Model Comparison Benchmark
SELECT
    model,
    CASE WHEN is_best THEN 'Selected Model' ELSE 'Baseline' END AS status,
    ROUND(accuracy * 100, 2) AS accuracy_pct,
    ROUND(macro_f1, 4) AS macro_f1,
    ROUND(weighted_f1, 4) AS weighted_f1,
    ROUND(recall_negative * 100, 2) AS negative_recall_pct,
    ROUND(recall_neutral * 100, 2) AS neutral_recall_pct,
    train_rows,
    ROUND(train_seconds, 1) AS train_seconds
FROM workspace.review_analytics.model_metrics
ORDER BY macro_f1 DESC;

-- 8. Product Buying Advisor (Top 500 Reviewed Products + BUY/DON'T BUY)
SELECT
    parent_asin,
    product_title,
    brand,
    category,
    price,
    reviews,
    avg_rating,
    positive_pct,
    model_positive_pct,
    buy_score,
    recommendation,
    confidence,
    top_aspect,
    top_complaint,
    reason_summary
FROM workspace.review_analytics.product_recommendations
WHERE reviews >= 20
ORDER BY reviews DESC
LIMIT 500;

-- 9. Live Structured Streaming Alerts (Phase 12)
SELECT
    ingested_at,
    category,
    brand,
    product_title,
    rating,
    actual_sentiment,
    predicted_sentiment,
    is_negative_alert,
    title
FROM workspace.review_analytics.stream_scored
ORDER BY ingested_at DESC
LIMIT 200;
