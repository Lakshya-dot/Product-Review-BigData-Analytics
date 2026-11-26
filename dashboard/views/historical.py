"""Page 1: Executive Overview, 1-5 Star Distribution, Word Lift, Complaints & Brand Intelligence."""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from utils import (
    ACCENT_AMBER,
    ACCENT_CORAL,
    ACCENT_EMERALD,
    ACCENT_INDIGO,
    ACCENT_TEAL,
    CATEGORIES,
    HEATMAP_CORAL,
    HEATMAP_TEAL,
    PLOTLY_CONFIG,
    SENTIMENT_COLORS,
    YEAR_PALETTE,
    download_csv_button,
    load_gold_table,
    render_kpi_card,
    render_page_header,
    style_fig,
)

render_page_header(
    badge="23RAI351 · Stages A–D · Market Intelligence",
    title="Amazon Product Review Sentiment Intelligence (2023–2026)",
    subtitle=(
        "Four-year longitudinal analytics across 1,139,264 cleaned genuine Amazon reviews and 6 categories "
        "(2020–2023 real timestamps shifted +3 years for prototype forecasting)."
    ),
)

year_summary = load_gold_table("agg_year_summary")
year_cat = load_gold_table("agg_year_category")
monthly = load_gold_table("agg_monthly")
brand_df = load_gold_table("agg_brand")
top_words = load_gold_table("agg_top_words")
complaints = load_gold_table("agg_complaints")
rating_dist = load_gold_table("agg_rating_dist")

# Inline Website Filter Bar
with st.expander("Market Scope & Category Filters", expanded=False):
    flt1, flt2 = st.columns([1.0, 3.0])
    selected_year = flt1.selectbox("Reporting Year", ["All", "2023", "2024", "2025", "2026"], index=0)
    selected_cats = flt2.multiselect("Product Categories", CATEGORIES, default=CATEGORIES)
    if not selected_cats:
        selected_cats = CATEGORIES

if set(selected_cats) == set(CATEGORIES):
    kpi_row = year_summary[year_summary["year"] == selected_year].iloc[0]
    total_reviews = int(kpi_row["reviews"])
    pos_pct = float(kpi_row["positive_pct"])
    neu_pct = float(kpi_row["neutral_pct"])
    neg_pct = float(kpi_row["negative_pct"])
    avg_rating = float(kpi_row["avg_rating"])
    n_products = int(kpi_row["products"])
    n_brands = int(kpi_row["brands"])
else:
    sub_yc = year_cat[year_cat["category"].isin(selected_cats)]
    if selected_year != "All":
        sub_yc = sub_yc[sub_yc["year"] == int(selected_year)]
    total_reviews = int(sub_yc["reviews"].sum())
    pos_sum = int(sub_yc["positive"].sum())
    neu_sum = int(sub_yc["neutral"].sum())
    neg_sum = int(sub_yc["negative"].sum())
    pos_pct = round(pos_sum * 100 / total_reviews, 2) if total_reviews else 0.0
    neu_pct = round(neu_sum * 100 / total_reviews, 2) if total_reviews else 0.0
    neg_pct = round(neg_sum * 100 / total_reviews, 2) if total_reviews else 0.0
    avg_rating = round((sub_yc["avg_rating"] * sub_yc["reviews"]).sum() / total_reviews, 3) if total_reviews else 0.0
    n_products = None
    n_brands = None

# Top KPI Cards
k1, k2, k3, k4, k5 = st.columns(5)
render_kpi_card(k1, "Total Reviews", f"{total_reviews:,}", f"Scope: {selected_year} · {len(selected_cats)} categories", ACCENT_INDIGO)
render_kpi_card(k2, "Positive Share", f"{pos_pct:.1f}%", "4–5 star proxy sentiment", ACCENT_EMERALD)
render_kpi_card(k3, "Neutral Share", f"{neu_pct:.1f}%", "3 star mixed reviews", ACCENT_AMBER)
render_kpi_card(k4, "Negative Share", f"{neg_pct:.1f}%", "1–2 star complaints", ACCENT_CORAL)
render_kpi_card(
    k5,
    "Average Rating",
    f"{avg_rating:.2f} / 5.0",
    f"{n_products:,} products · {n_brands:,} brands" if n_products else f"{len(selected_cats)} categories selected",
    ACCENT_TEAL,
)

# Executive Findings Strip
st.markdown("<div style='height:0.65rem'></div>", unsafe_allow_html=True)
i1, i2, i3 = st.columns(3)
i1.markdown(
    "<div class='insight-pill' style='border-left-color:#818cf8;'><strong>4-Year Trajectory:</strong> Positive share dipped from "
    "<strong>79.2% (2023)</strong> to <strong>74.3% (2025)</strong> across categories before recovering to <strong>75.6% (2026)</strong>.</div>",
    unsafe_allow_html=True,
)
i2.markdown(
    "<div class='insight-pill' style='border-left-color:#2dd4bf;'><strong>Category Benchmark:</strong> "
    "<strong>Books</strong> consistently leads (84–86% positive); <strong>Beauty & Personal Care</strong> recorded the steepest 2025 decline.</div>",
    unsafe_allow_html=True,
)
i3.markdown(
    "<div class='insight-pill' style='border-left-color:#f43f5e;'><strong>Primary Pain Points:</strong> "
    "<strong>Quality / durability</strong> (~24%) and <strong>Does not work</strong> (~21%) account for the majority of negative reviews.</div>",
    unsafe_allow_html=True,
)

st.markdown("<div style='height:0.65rem'></div>", unsafe_allow_html=True)

tab_trends, tab_voc, tab_brands, tab_arch = st.tabs([
    "Trends, Ratings & Categories",
    "Voice of Customer",
    "Brand Intelligence",
    "System Architecture (15 Phases)",
])

with tab_trends:
    col_left, col_right = st.columns([1.05, 1.25])

    with col_left:
        yr_only = year_summary[year_summary["year"] != "All"].sort_values("year")
        fig_yr = make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.28, 0.72],
            shared_xaxes=True,
            vertical_spacing=0.07,
        )
        fig_yr.add_trace(
            go.Scatter(
                name="Avg Rating",
                x=yr_only["year"],
                y=yr_only["avg_rating"],
                mode="lines+markers+text",
                text=yr_only["avg_rating"].map(lambda v: f"{v:.2f}"),
                textposition="top center",
                line=dict(color="#38bdf8", width=2.4),
                marker=dict(size=7, color="#f8fafc"),
            ),
            row=1,
            col=1,
        )
        for sent in ["Positive", "Neutral", "Negative"]:
            fig_yr.add_trace(
                go.Bar(
                    name=sent,
                    x=yr_only["year"],
                    y=yr_only[f"{sent.lower()}_pct"],
                    marker_color=SENTIMENT_COLORS[sent],
                    text=yr_only[f"{sent.lower()}_pct"].map(lambda v: f"{v:.1f}%"),
                    textposition="inside",
                    insidetextanchor="middle",
                ),
                row=2,
                col=1,
            )
        style_fig(fig_yr, height=405, title="Four-Year Average Rating & Sentiment Mix", has_top_legend=True)
        fig_yr.update_layout(barmode="stack")
        fig_yr.update_yaxes(title_text="Avg ★", range=[3.8, 4.55], nticks=3, row=1, col=1)
        fig_yr.update_yaxes(title_text="Share (%)", range=[0, 102], row=2, col=1)
        st.plotly_chart(fig_yr, width="stretch", config=PLOTLY_CONFIG)

    with col_right:
        view_mode = st.radio(
            "Category View",
            ["Grouped Bars", "Year × Category Heatmap"],
            horizontal=True,
            label_visibility="collapsed",
        )
        yc_filtered = year_cat[year_cat["category"].isin(selected_cats)].copy()
        yc_filtered["year_str"] = yc_filtered["year"].astype(str)

        if view_mode == "Grouped Bars":
            fig_cat = px.bar(
                yc_filtered.sort_values(["category", "year_str"]),
                x="category",
                y="positive_pct",
                color="year_str",
                barmode="group",
                text_auto=".1f",
                color_discrete_sequence=YEAR_PALETTE,
                labels={"positive_pct": "Positive (%)", "category": "", "year_str": ""},
            )
            style_fig(fig_cat, height=365, title="Positive Sentiment (%) by Category & Year", has_top_legend=True)
            fig_cat.update_layout(yaxis_range=[65, 91])
            fig_cat.update_xaxes(tickangle=-15)
        else:
            piv = yc_filtered.pivot(index="category", columns="year_str", values="positive_pct")
            fig_cat = px.imshow(
                piv,
                text_auto=".1f",
                color_continuous_scale=HEATMAP_TEAL,
                labels=dict(x="Year", y="Category", color="Positive %"),
            )
            style_fig(fig_cat, height=365, title="Positive Sentiment (%) Heatmap — Category × Year")
        st.plotly_chart(fig_cat, width="stretch", config=PLOTLY_CONFIG)

    r_col, m_col1, m_col2 = st.columns([0.95, 1.05, 1.05])
    with r_col:
        rd_sub = rating_dist[rating_dist["category"].isin(selected_cats)]
        if selected_year != "All":
            rd_sub = rd_sub[rd_sub["year"] == int(selected_year)]
        rd_agg = rd_sub.groupby("stars", as_index=False)["reviews"].sum().sort_values("stars")
        rd_agg["pct"] = (rd_agg["reviews"] * 100 / rd_agg["reviews"].sum()).round(1)
        rd_agg["star_label"] = rd_agg["stars"].map(lambda s: f"{s} Star" if s == 1 else f"{s} Stars")
        rd_agg["color"] = rd_agg["stars"].map({1: "#f43f5e", 2: "#fb923c", 3: "#f59e0b", 4: "#60a5fa", 5: "#10b981"})

        fig_rd = go.Figure(
            go.Bar(
                x=rd_agg["star_label"],
                y=rd_agg["reviews"],
                marker_color=rd_agg["color"],
                text=rd_agg["pct"].map(lambda p: f"{p:.1f}%"),
                textposition="outside",
                hovertemplate="%{x}: %{y:,} reviews (%{text})<extra></extra>",
            )
        )
        style_fig(fig_rd, height=330, title=f"Star Rating Distribution (1–5 · {selected_year})")
        fig_rd.update_layout(yaxis_title="Reviews", yaxis_range=[0, rd_agg["reviews"].max() * 1.16])
        st.plotly_chart(fig_rd, width="stretch", config=PLOTLY_CONFIG)

    m_filtered = monthly[monthly["category"].isin(selected_cats)]
    if selected_year != "All":
        m_filtered = m_filtered[m_filtered["year"] == int(selected_year)]
    m_agg = (
        m_filtered.groupby("review_month", as_index=False)
        .agg(reviews=("reviews", "sum"), positive=("positive", "sum"), negative=("negative", "sum"))
        .sort_values("review_month")
    )
    m_agg["positive_pct"] = (m_agg["positive"] * 100 / m_agg["reviews"]).round(2)
    m_agg["negative_pct"] = (m_agg["negative"] * 100 / m_agg["reviews"]).round(2)

    with m_col1:
        fig_vol = px.area(
            m_agg, x="review_month", y="reviews",
            labels={"review_month": "Month", "reviews": "Reviews"},
            color_discrete_sequence=[ACCENT_INDIGO],
        )
        style_fig(fig_vol, height=330, title="Monthly Review Volume (2026 ends ~Sep)")
        st.plotly_chart(fig_vol, width="stretch", config=PLOTLY_CONFIG)

    with m_col2:
        fig_pos_m = go.Figure()
        fig_pos_m.add_trace(
            go.Scatter(
                x=m_agg["review_month"], y=m_agg["positive_pct"],
                name="Positive %", mode="lines+markers",
                line=dict(color=ACCENT_TEAL, width=2.2), marker=dict(size=4.5),
            )
        )
        fig_pos_m.add_trace(
            go.Scatter(
                x=m_agg["review_month"], y=m_agg["negative_pct"],
                name="Negative %", mode="lines+markers",
                line=dict(color=ACCENT_CORAL, width=1.9), marker=dict(size=4.5),
            )
        )
        style_fig(fig_pos_m, height=330, title="Monthly Positive vs Negative Share (%)", has_top_legend=True)
        fig_pos_m.update_layout(yaxis_title="Share (%)")
        st.plotly_chart(fig_pos_m, width="stretch", config=PLOTLY_CONFIG)


with tab_voc:
    wc_col, cmp_col = st.columns([1.08, 1.02])

    with wc_col:
        ctrl1, ctrl2 = st.columns([1.5, 1.0])
        word_cat = ctrl1.selectbox("Category for Distinctive Words", selected_cats, index=0)
        top_n_words = ctrl2.slider("Top Words per Polarity", 5, 20, 12, step=1)

        w_sub = top_words[(top_words["year"] == selected_year) & (top_words["category"] == word_cat)]
        pos_w = w_sub[w_sub["sentiment"] == "Positive"].sort_values("n", ascending=False).head(top_n_words).sort_values("n", ascending=True)
        neg_w = w_sub[w_sub["sentiment"] == "Negative"].sort_values("n", ascending=False).head(top_n_words).sort_values("n", ascending=True)

        if pos_w.empty and neg_w.empty:
            st.info("No words met lift > 1.3 and count >= 20 for this filter.")
        else:
            fig_bf = go.Figure()
            fig_bf.add_trace(
                go.Bar(
                    name="Negative (Lift > 1.3)",
                    y=neg_w["word"],
                    x=-neg_w["n"],
                    orientation="h",
                    marker_color=ACCENT_CORAL,
                    customdata=neg_w[["n", "lift"]],
                    hovertemplate="<b>%{y}</b><br>Count: %{customdata[0]:,}<br>Lift: %{customdata[1]:.2f}x<extra></extra>",
                )
            )
            fig_bf.add_trace(
                go.Bar(
                    name="Positive (Lift > 1.3)",
                    y=pos_w["word"],
                    x=pos_w["n"],
                    orientation="h",
                    marker_color=ACCENT_TEAL,
                    customdata=pos_w[["n", "lift"]],
                    hovertemplate="<b>%{y}</b><br>Count: %{customdata[0]:,}<br>Lift: %{customdata[1]:.2f}x<extra></extra>",
                )
            )
            style_fig(fig_bf, height=455, title=f"Polarity Word Lift Butterfly ({word_cat} · {selected_year})", has_top_legend=True)
            fig_bf.update_layout(
                barmode="relative",
                xaxis_title="Negative Occurrences  |  Positive Occurrences",
            )
            st.plotly_chart(fig_bf, width="stretch", config=PLOTLY_CONFIG)

    with cmp_col:
        cmp_mode = st.radio(
            "Complaint View",
            ["Overall Themes Bar", "Category × Theme Heatmap"],
            horizontal=True,
            label_visibility="collapsed",
        )
        comp_sub = complaints[complaints["category"].isin(selected_cats)]
        if selected_year != "All":
            comp_sub = comp_sub[comp_sub["year"] == int(selected_year)]

        if cmp_mode == "Overall Themes Bar":
            comp_agg = (
                comp_sub.groupby("theme", as_index=False)
                .agg(mentions=("mentions", "sum"), neg_total=("negative_reviews", "sum"))
            )
            comp_agg["pct_of_negative"] = (comp_agg["mentions"] * 100 / comp_agg["neg_total"]).round(2)
            comp_agg = comp_agg.sort_values("pct_of_negative", ascending=True)
            fig_cmp = px.bar(
                comp_agg, x="pct_of_negative", y="theme", orientation="h",
                text_auto=".1f", color="pct_of_negative", color_continuous_scale=HEATMAP_CORAL,
                labels={"pct_of_negative": "% of Negative Reviews", "theme": "Complaint Theme"},
            )
            style_fig(fig_cmp, height=420, title=f"Customer Complaint Themes ({selected_year})")
            fig_cmp.update_layout(coloraxis_showscale=False)
        else:
            comp_cat = (
                comp_sub.groupby(["category", "theme"], as_index=False)
                .agg(mentions=("mentions", "sum"), neg_total=("negative_reviews", "sum"))
            )
            comp_cat["pct"] = (comp_cat["mentions"] * 100 / comp_cat["neg_total"]).round(1)
            piv_c = comp_cat.pivot(index="theme", columns="category", values="pct")
            fig_cmp = px.imshow(
                piv_c, text_auto=".1f", color_continuous_scale=HEATMAP_CORAL,
                labels=dict(x="Category", y="Complaint Theme", color="% of Neg"),
            )
            style_fig(fig_cmp, height=420, title=f"Complaint Intensity by Category ({selected_year})")
        st.plotly_chart(fig_cmp, width="stretch", config=PLOTLY_CONFIG)


with tab_brands:
    b_sub = brand_df[(brand_df["year"] == selected_year) & (brand_df["category"].isin(selected_cats))].copy()
    bc1, bc2, bc3 = st.columns([1.4, 1.2, 1.0])
    brand_search = bc1.text_input("Filter Brand Name", value="", placeholder="e.g. Anker, Amazon, Sony")
    brand_sort = bc2.selectbox("Rank By", ["Most Reviewed", "Highest Positive %", "Lowest Positive %"])
    min_brand_rev = bc3.slider("Minimum Brand Reviews", 30, 1000, 100, step=10)

    b_sub = b_sub[b_sub["reviews"] >= min_brand_rev]
    if brand_search.strip():
        b_sub = b_sub[b_sub["brand"].str.contains(brand_search.strip(), case=False, regex=False)]

    if brand_sort == "Most Reviewed":
        b_top = b_sub.sort_values("reviews", ascending=False).head(30)
    elif brand_sort == "Highest Positive %":
        b_top = b_sub.sort_values(["positive_pct", "reviews"], ascending=[False, False]).head(30)
    else:
        b_top = b_sub.sort_values(["positive_pct", "reviews"], ascending=[True, False]).head(30)

    bl, br = st.columns([1.15, 1.0])
    with bl:
        if b_sub.empty:
            st.warning("No brands match the current filters.")
        else:
            scatter_df = b_sub.sort_values("reviews", ascending=False).head(150)
            fig_bs = px.scatter(
                scatter_df,
                x="reviews",
                y="positive_pct",
                size="reviews",
                color="category",
                hover_name="brand",
                hover_data={"avg_rating": ":.2f", "negative_pct": ":.1f"},
                labels={"reviews": "Review Volume", "positive_pct": "Positive %", "category": "Category"},
            )
            style_fig(fig_bs, height=420, title=f"Brand Volume vs Positive Sentiment (Top 150 · {selected_year})", has_top_legend=True)
            st.plotly_chart(fig_bs, width="stretch", config=PLOTLY_CONFIG)

    with br:
        st.dataframe(
            b_top[["brand", "category", "reviews", "avg_rating", "positive_pct", "negative_pct"]]
            .rename(columns={
                "brand": "Brand",
                "category": "Category",
                "reviews": "Reviews",
                "avg_rating": "Avg Rating",
                "positive_pct": "Pos %",
                "negative_pct": "Neg %",
            }),
            width="stretch",
            hide_index=True,
            height=360,
        )
        if not b_top.empty:
            download_csv_button(b_top, f"top_brands_{selected_year}.csv", "Export Brand Leaderboard CSV")


with tab_arch:
    st.subheader("End-to-End 5-Stage Big Data Architecture (Phases 1–15)")
    p1, p2, p3, p4, p5 = st.columns(5)
    stages = [
        (p1, "Stage A · Phases 1–4", "Data Engineering", "1.14M genuine Amazon 2023 reviews sampled across 6 categories, shifted +3 yrs to 2023–2026, joined with 543K meta_* product records.", "#3b82f6"),
        (p2, "Stage B · Phases 5–6", "Databricks Delta", "Ingested to Unity Catalog Volumes & Delta tables (reviews_all, products); regex cleaning & rating-derived proxy labels (reviews_clean).", "#06b6d4"),
        (p3, "Stage C · Phases 7–8", "PySpark MLlib", "Deterministic 80/20 review_id hash split, TF-IDF vectorization, and class-weighted Naive Bayes, Logistic Regression & Random Forest.", "#8b5cf6"),
        (p4, "Stage D · Phases 9–12", "Gold & Streaming", "Gold summary tables, lift word analysis, complaint themes, 7-factor BUY / DON'T BUY engine, and Structured Streaming micro-batch pipeline.", "#10b981"),
        (p5, "Stage E · Phases 13–15", "Analytics UI", "5-page interactive analytics, side-by-side product comparison, live ML predictor with token explanations, stream monitor & 16-run benchmark.", "#f59e0b"),
    ]
    for col, stg, nm, desc, clr in stages:
        col.markdown(
            f"""
            <div class="pipe-step" style="border-top-color:{clr};">
                <div class="pipe-stage" style="color:{clr};">{stg}</div>
                <div class="pipe-name">{nm}</div>
                <div class="pipe-desc">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:0.9rem'></div>", unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    with d1:
        st.info(
            "**Dataset Integrity & Methodology**\n"
            "- **100% Genuine Review Text:** Zero synthetic rows (`is_synthetic = False`), `date_shifted = True` (+3 yrs).\n"
            "- **Proxy Label Mapping:** `1–2 Stars -> Negative`, `3 Stars -> Neutral`, `4–5 Stars -> Positive`.\n"
            "- **Variant Integrity:** Products are grouped strictly by Amazon `parent_asin` so color/size options never split or mix models."
        )
    with d2:
        st.success(
            "**Explainable BUY / DON'T BUY Formula (0–100)**\n"
            "- `20%` Rating Positive % + `15%` ML Model Positive % + `20%` Avg Rating Scaled\n"
            "- `+ 15%` Recent Year % + `10%` YoY Trend + `10%` Consistency + `10%` Low Complaints\n"
            "- `< 20 reviews -> INSUFFICIENT DATA` | `>= 65 -> BUY` | `< 65 -> DON'T BUY`"
        )
