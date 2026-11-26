"""Page 2: Customer-First Product Buying Advisor & Side-by-Side Radar Comparison."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    RECOMMENDATION_COLORS,
    SENTIMENT_COLORS,
    download_csv_button,
    load_gold_table,
    render_kpi_card,
    render_page_header,
    style_fig,
)

render_page_header(
    badge="Phases 10–11 & 14 · Consumer Decision Engine",
    title="Product Buying Advisor & Recommendation Scorecard",
    subtitle=(
        "Search 543,496 unique products before you buy. Each product is evaluated across rating sentiment, "
        "ML text predictions, multi-year consistency, aspect strengths, and recurring customer complaints."
    ),
)

recs = load_gold_table("product_recommendations")
aspects = load_gold_table("product_aspect_sentiment")
prod_complaints = load_gold_table("product_complaints")
sample_reviews = load_gold_table("product_sample_reviews")

# Catalog Summary KPIs
counts = recs["recommendation"].value_counts().to_dict()
buy_cnt = counts.get("BUY", 0)
dont_buy_cnt = counts.get("DON'T BUY", 0)
insuff_cnt = counts.get("INSUFFICIENT DATA", 0)

s1, s2, s3, s4 = st.columns(4)
render_kpi_card(s1, "Catalog Coverage", f"{len(recs):,}", "Unique parent_asin products", ACCENT_INDIGO)
render_kpi_card(s2, "Recommended (BUY)", f"{buy_cnt:,}", "Score >= 65 & >= 20 reviews", ACCENT_EMERALD)
render_kpi_card(s3, "Avoid (DON'T BUY)", f"{dont_buy_cnt:,}", "Score < 65 & >= 20 reviews", ACCENT_CORAL)
render_kpi_card(s4, "Insufficient Data", f"{insuff_cnt:,}", "Fewer than 20 reviews", ACCENT_AMBER)

st.markdown("<div style='height:0.65rem'></div>", unsafe_allow_html=True)

# Clean Quick Search Presets
st.caption("Popular Consumer Searches:")
qp1, qp2, qp3, qp4, qp5 = st.columns(5)
if "search_kw" not in st.session_state:
    st.session_state.search_kw = "echo dot"
if qp1.button("Echo Dot", width="stretch"):
    st.session_state.search_kw = "echo dot"
if qp2.button("Fire TV Stick", width="stretch"):
    st.session_state.search_kw = "fire tv stick"
if qp3.button("Anker Charger", width="stretch"):
    st.session_state.search_kw = "anker"
if qp4.button("Yoga Mat", width="stretch"):
    st.session_state.search_kw = "yoga mat"
if qp5.button("Browse All Top Products", width="stretch"):
    st.session_state.search_kw = ""

with st.expander("Search & Filter Controls", expanded=True):
    f1, f2, f3, f4 = st.columns([2.2, 1.3, 1.3, 1.2])
    q_name = f1.text_input("Product Name Keyword", key="search_kw", placeholder="e.g. echo dot, charger, yoga mat")
    q_brand = f2.text_input("Brand (optional)", value="", placeholder="e.g. Amazon, Anker")
    q_cat = f3.selectbox("Category", ["All"] + CATEGORIES)
    q_verdict = f4.selectbox("Verdict", ["All", "BUY", "DON'T BUY", "INSUFFICIENT DATA"])

    g1, g2, g3, g4 = st.columns([1.2, 1.2, 1.2, 1.4])
    q_conf = g1.selectbox("Confidence", ["All", "High", "Medium", "Low"])
    min_rev = g2.number_input("Minimum Reviews", min_value=1, max_value=5000, value=20, step=10)
    sort_by = g3.selectbox("Sort Results By", ["Most Reviewed", "Highest BUY Score", "Lowest BUY Score", "Highest Rating"])
    max_price = g4.slider("Max Price ($)", 10, 1000, 1000, step=10, help="Includes unpriced products when set to 1000.")

# Apply filters
filtered = recs[recs["reviews"] >= min_rev]
if q_cat != "All":
    filtered = filtered[filtered["category"] == q_cat]
if q_verdict != "All":
    filtered = filtered[filtered["recommendation"] == q_verdict]
if q_conf != "All":
    filtered = filtered[filtered["confidence"] == q_conf]
if max_price < 1000:
    filtered = filtered[filtered["price"].notna() & (filtered["price"] <= max_price)]
if q_name.strip():
    filtered = filtered[filtered["product_title"].fillna("").str.contains(q_name.strip(), case=False, regex=False)]
if q_brand.strip():
    filtered = filtered[filtered["brand"].fillna("").str.contains(q_brand.strip(), case=False, regex=False)]

if sort_by == "Most Reviewed":
    filtered = filtered.sort_values("reviews", ascending=False)
elif sort_by == "Highest BUY Score":
    filtered = filtered.sort_values(["buy_score", "reviews"], ascending=[False, False])
elif sort_by == "Lowest BUY Score":
    filtered = filtered.sort_values(["buy_score", "reviews"], ascending=[True, False])
else:
    filtered = filtered.sort_values(["avg_rating", "reviews"], ascending=[False, False])

if filtered.empty:
    st.warning("No matching products found. Try clearing the keyword, lowering Minimum Reviews, or selecting Category 'All'.")
    st.stop()

top_matches = filtered.head(40).copy()

hdr_res, hdr_exp = st.columns([3, 1])
hdr_res.subheader(f"Matched Products ({len(filtered):,} total · displaying top {len(top_matches)})")
with hdr_exp:
    download_csv_button(top_matches, "product_search_results.csv", "Export Matches CSV")

display_cols = [
    "parent_asin", "product_title", "brand", "category", "price", "reviews",
    "avg_rating", "positive_pct", "buy_score", "recommendation", "confidence",
]
st.dataframe(
    top_matches[display_cols].rename(columns={
        "parent_asin": "Parent ASIN",
        "product_title": "Product Title",
        "brand": "Brand",
        "category": "Category",
        "price": "Price ($)",
        "reviews": "Reviews",
        "avg_rating": "Avg Rating",
        "positive_pct": "Pos %",
        "buy_score": "BUY Score",
        "recommendation": "Verdict",
        "confidence": "Confidence",
    }),
    width="stretch",
    hide_index=True,
    height=240,
)

tab_single, tab_compare = st.tabs(["Product Intelligence Report", "Side-by-Side Comparison"])

options = [
    f"{r['parent_asin']}  |  {(r['product_title'] or 'Untitled')[:68]}  ({int(r['reviews'])} rev · {r['recommendation']})"
    for _, r in top_matches.iterrows()
]

with tab_single:
    selected_label = st.selectbox("Select Product to Inspect", options, index=0)
    selected_asin = selected_label.split("|")[0].strip()
    p = top_matches[top_matches["parent_asin"] == selected_asin].iloc[0]

    verdict = p["recommendation"]
    badge_cls = "badge-buy" if verdict == "BUY" else ("badge-dont-buy" if verdict == "DON'T BUY" else "badge-insufficient")
    price_str = f"${p['price']:.2f}" if pd.notna(p["price"]) else "N/A"
    trend_val = float(p["sentiment_trend"]) if pd.notna(p["sentiment_trend"]) else 0.0
    trend_str = f"{trend_val:+.1f}%"

    hdr_l, hdr_r = st.columns([2.2, 1.0])
    with hdr_l:
        st.markdown(f"### {p['product_title'] or p['parent_asin']}")
        st.markdown(
            f"<span class='meta-tag'>ASIN: {p['parent_asin']}</span>"
            f"<span class='meta-tag'>Brand: {p['brand']}</span>"
            f"<span class='meta-tag'>{p['category']}</span>"
            f"<span class='meta-tag'>Price: {price_str}</span>"
            f"<span class='meta-tag'>Verified Purchases: {p['verified_pct']:.1f}%</span>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:0.55rem'></div>", unsafe_allow_html=True)
        st.markdown(
            f"<span class='verdict-badge {badge_cls}'>{verdict}</span> &nbsp;&nbsp; "
            f"<strong>Confidence:</strong> {p['confidence']} &nbsp;|&nbsp; "
            f"<strong>Volume:</strong> {int(p['reviews']):,} reviews ({int(p['years_active'])} yrs) &nbsp;|&nbsp; "
            f"<strong>YoY Sentiment Shift:</strong> {trend_str}",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:0.65rem'></div>", unsafe_allow_html=True)

        # Customer-friendly Pros & Watch-outs summary cards
        p_col1, p_col2 = st.columns(2)
        top_asp_txt = f"{p['top_aspect']} ({p['top_aspect_pos_pct']}% positive)" if pd.notna(p["top_aspect"]) else "Consistent overall rating"
        top_cmp_txt = f"{p['top_complaint']} ({p['top_complaint_pct']}% of negatives)" if pd.notna(p["top_complaint"]) else "No recurring defect pattern"
        p_col1.markdown(
            f"<div class='insight-pill' style='border-left-color:#2dd4bf;'><strong>Key Strength:</strong><br/>{top_asp_txt} · Avg Rating {p['avg_rating']:.2f} / 5.0</div>",
            unsafe_allow_html=True,
        )
        p_col2.markdown(
            f"<div class='insight-pill' style='border-left-color:#f43f5e;'><strong>Watch Out For:</strong><br/>{top_cmp_txt} · {p['negative_pct']:.1f}% negative rate</div>",
            unsafe_allow_html=True,
        )

        if pd.notna(p.get("product_features")) and str(p["product_features"]).strip():
            with st.expander("Product Specifications & Key Features"):
                for feat in str(p["product_features"]).split(" | "):
                    if feat.strip():
                        st.markdown(f"- {feat.strip()}")

    with hdr_r:
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=float(p["buy_score"]),
                title={"text": "BUY Score (0–100)", "font": {"size": 15, "color": "#f1f5f9"}},
                number={"font": {"color": "#f8fafc"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
                    "bar": {"color": RECOMMENDATION_COLORS.get(verdict, "#3b82f6")},
                    "bgcolor": "rgba(17,24,39,0.5)",
                    "steps": [
                        {"range": [0, 45], "color": "rgba(244,63,94,0.16)"},
                        {"range": [45, 65], "color": "rgba(245,158,11,0.16)"},
                        {"range": [65, 100], "color": "rgba(45,212,191,0.18)"},
                    ],
                    "threshold": {"line": {"color": "#f8fafc", "width": 3}, "thickness": 0.78, "value": 65},
                },
            )
        )
        style_fig(fig_gauge, height=255)
        fig_gauge.update_layout(margin=dict(t=56, b=10, l=24, r=24))
        st.plotly_chart(fig_gauge, width="stretch", config=PLOTLY_CONFIG)

    b1, b2, b3 = st.columns(3)
    with b1:
        factor_df = pd.DataFrame([
            {"Factor": "Positive % (20%)", "Score": float(p["positive_pct"])},
            {"Factor": "ML Model Pos % (15%)", "Score": float(p["model_positive_pct"])},
            {"Factor": "Avg Rating Scaled (20%)", "Score": float(p["rating_score"])},
            {"Factor": "Recent Year Pos % (15%)", "Score": float(p["recent_year_pos_pct"]) if pd.notna(p["recent_year_pos_pct"]) else float(p["positive_pct"])},
            {"Factor": "Trend Score (10%)", "Score": float(p["trend_score"])},
            {"Factor": "Consistency (10%)", "Score": float(p["consistency_score"])},
            {"Factor": "Low Complaints (10%)", "Score": float(p["complaint_inv_score"])},
        ])
        fig_fac = px.bar(
            factor_df, x="Score", y="Factor", orientation="h", text_auto=".1f",
            color="Score", color_continuous_scale=HEATMAP_TEAL, range_color=[0, 100],
        )
        style_fig(fig_fac, height=340, title="7-Factor Score Inputs")
        fig_fac.update_layout(coloraxis_showscale=False, xaxis_range=[0, 108])
        st.plotly_chart(fig_fac, width="stretch", config=PLOTLY_CONFIG)

    with b2:
        p_asp = aspects[(aspects["parent_asin"] == selected_asin) & (aspects["category"] == p["category"])].sort_values("mentions", ascending=True)
        if p_asp.empty:
            st.info("Insufficient aspect mentions (requires >= 5 product reviews).")
        else:
            fig_asp = px.bar(
                p_asp, x="positive_pct", y="aspect", orientation="h",
                hover_data=["mentions", "negative_pct"], text_auto=".0f",
                color="positive_pct", color_continuous_scale=HEATMAP_TEAL, range_color=[0, 100],
                labels={"positive_pct": "Positive %", "aspect": "Aspect"},
            )
            style_fig(fig_asp, height=340, title="Aspect-Level Sentiment")
            fig_asp.update_layout(coloraxis_showscale=False, xaxis_range=[0, 108])
            st.plotly_chart(fig_asp, width="stretch", config=PLOTLY_CONFIG)

    with b3:
        p_cmp = prod_complaints[(prod_complaints["parent_asin"] == selected_asin) & (prod_complaints["category"] == p["category"])].sort_values("mentions", ascending=True)
        if p_cmp.empty:
            st.success("No recurring complaint themes flagged for this product.")
        else:
            fig_pc = px.bar(
                p_cmp, x="pct_of_negative", y="theme", orientation="h",
                hover_data=["mentions"], text_auto=".1f",
                color_discrete_sequence=[ACCENT_CORAL],
                labels={"pct_of_negative": "% of Negative Reviews", "theme": "Theme"},
            )
            style_fig(fig_pc, height=340, title="Complaint Breakdown (% of Negatives)")
            st.plotly_chart(fig_pc, width="stretch", config=PLOTLY_CONFIG)

    # Customer Review Quotes
    st.subheader("Verified Customer Feedback Samples")
    p_revs = sample_reviews[(sample_reviews["parent_asin"] == selected_asin) & (sample_reviews["category"] == p["category"])]
    if p_revs.empty:
        st.info("Sample customer review quotes are pre-cached for the top 5,000 most-reviewed products.")
    else:
        rev_filter = st.radio("Filter Customer Quotes", ["All", "Positive", "Neutral", "Negative"], horizontal=True)
        if rev_filter != "All":
            p_revs = p_revs[p_revs["sentiment"] == rev_filter]
        if p_revs.empty:
            st.info(f"No {rev_filter} quotes in the cached sample for this product.")
        else:
            for _, rv in p_revs.iterrows():
                color = SENTIMENT_COLORS.get(rv["sentiment"], "#3b82f6")
                ver_tag = " · Verified Purchase" if rv["verified_purchase"] else ""
                st.markdown(
                    f"""
                    <div class="review-card" style="border-left-color: {color};">
                        <strong>{rv['rating']:.1f} / 5.0 · {rv['sentiment']}</strong>
                        <span style="color:#94a3b8; font-size:0.83rem;"> · ML Pred: {rv['predicted_sentiment']} · {rv['review_date']}{ver_tag} · Helpful votes: {int(rv['helpful_vote'])}</span><br/>
                        <div style="font-weight:600; margin:0.2rem 0; color:#f1f5f9;">{rv['title'] or ''}</div>
                        <span style="font-size:0.90rem; color:#cbd5e1;">{(rv['text'] or '')[:520]}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

with tab_compare:
    if len(options) < 2:
        st.info("Search results need at least 2 matching products for side-by-side comparison.")
    else:
        c_sel1, c_sel2 = st.columns(2)
        sel_a = c_sel1.selectbox("Product A", options, index=0, key="cmp_a")
        sel_b = c_sel2.selectbox("Product B", options, index=1, key="cmp_b")
        asin_a = sel_a.split("|")[0].strip()
        asin_b = sel_b.split("|")[0].strip()
        pa = top_matches[top_matches["parent_asin"] == asin_a].iloc[0]
        pb = top_matches[top_matches["parent_asin"] == asin_b].iloc[0]

        diff = round(float(pa["buy_score"]) - float(pb["buy_score"]), 1)
        if abs(diff) >= 0.5:
            winner_txt = f"Product A ({asin_a}) leads by +{abs(diff):.1f} BUY Score points" if diff > 0 else f"Product B ({asin_b}) leads by +{abs(diff):.1f} BUY Score points"
            st.success(f"**Comparison Summary:** {winner_txt}")

        ca, cb = st.columns(2)
        for col, item, label in [(ca, pa, "Product A"), (cb, pb, "Product B")]:
            v = item["recommendation"]
            bcls = "badge-buy" if v == "BUY" else ("badge-dont-buy" if v == "DON'T BUY" else "badge-insufficient")
            col.markdown(
                f"""
                <div class="kpi-card" style="border-top-color:{RECOMMENDATION_COLORS.get(v, '#3b82f6')};">
                    <div class="kpi-label">{label} · {item['parent_asin']}</div>
                    <div style="font-weight:700; font-size:1.02rem; margin-bottom:0.45rem;">{(item['product_title'] or '')[:80]}</div>
                    <span class='verdict-badge {bcls}'>{v} · {item['buy_score']}/100</span>
                    <div class="kpi-sub" style="margin-top:0.55rem;">
                        Brand: <strong>{item['brand']}</strong> · Reviews: <strong>{int(item['reviews']):,}</strong> ·
                        Avg Rating: <strong>{item['avg_rating']:.2f}/5</strong> · Positive: <strong>{item['positive_pct']}%</strong><br/>
                        Strength: <strong>{item['top_aspect'] or 'N/A'}</strong> · Risk: <strong>{item['top_complaint'] or 'None'}</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        rc1, rc2 = st.columns(2)
        radar_cats = ["Positive %", "ML Model %", "Rating Scaled", "Trend", "Consistency", "Low Complaints"]
        vals_a = [float(pa["positive_pct"]), float(pa["model_positive_pct"]), float(pa["rating_score"]), float(pa["trend_score"]), float(pa["consistency_score"]), float(pa["complaint_inv_score"])]
        vals_b = [float(pb["positive_pct"]), float(pb["model_positive_pct"]), float(pb["rating_score"]), float(pb["trend_score"]), float(pb["consistency_score"]), float(pb["complaint_inv_score"])]

        with rc1:
            fig_rad = go.Figure()
            fig_rad.add_trace(go.Scatterpolar(r=vals_a + [vals_a[0]], theta=radar_cats + [radar_cats[0]], fill="toself", name=f"A: {asin_a}", line_color=ACCENT_INDIGO))
            fig_rad.add_trace(go.Scatterpolar(r=vals_b + [vals_b[0]], theta=radar_cats + [radar_cats[0]], fill="toself", name=f"B: {asin_b}", line_color=ACCENT_TEAL))
            style_fig(fig_rad, height=380, title="6-Axis Product Strength Radar", has_top_legend=True)
            fig_rad.update_layout(
                polar=dict(bgcolor="rgba(17,20,29,0.65)", radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255,255,255,0.10)")),
            )
            st.plotly_chart(fig_rad, width="stretch", config=PLOTLY_CONFIG)

        with rc2:
            comp_metrics = pd.DataFrame([
                {"Metric": "BUY Score", "Product A": float(pa["buy_score"]), "Product B": float(pb["buy_score"])},
                {"Metric": "Positive %", "Product A": float(pa["positive_pct"]), "Product B": float(pb["positive_pct"])},
                {"Metric": "ML Positive %", "Product A": float(pa["model_positive_pct"]), "Product B": float(pb["model_positive_pct"])},
                {"Metric": "Rating Scaled", "Product A": float(pa["rating_score"]), "Product B": float(pb["rating_score"])},
                {"Metric": "Low Complaints", "Product A": float(pa["complaint_inv_score"]), "Product B": float(pb["complaint_inv_score"])},
            ]).melt(id_vars="Metric", var_name="Product", value_name="Score")
            fig_side = px.bar(
                comp_metrics, x="Metric", y="Score", color="Product", barmode="group",
                text_auto=".1f", color_discrete_sequence=[ACCENT_INDIGO, ACCENT_TEAL],
            )
            style_fig(fig_side, height=380, title="Head-to-Head Metric Breakdown", has_top_legend=True)
            fig_side.update_layout(yaxis_range=[0, 108])
            st.plotly_chart(fig_side, width="stretch", config=PLOTLY_CONFIG)
