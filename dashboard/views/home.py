"""Website Landing Page: ReviewSense Product Intelligence Platform."""

import pandas as pd
import streamlit as st

from utils import (
    ACCENT_AMBER,
    ACCENT_CORAL,
    ACCENT_EMERALD,
    ACCENT_INDIGO,
    ACCENT_TEAL,
    load_gold_table,
    render_kpi_card,
)

# Website Hero Banner
st.markdown(
    """
    <div class="web-hero">
        <div class="hero-badge">ReviewSense Analytics · 23RAI351 Big Data Platform</div>
        <div class="web-hero-title">Intelligent Product Review Analytics &amp; Buying Advisor</div>
        <p class="hero-sub">
            Built for shoppers, product analysts, and big data evaluation. Explore sentiment across
            <strong>1,139,264 genuine Amazon reviews</strong> and <strong>543,496 unique products</strong>
            using Apache Spark, Databricks Delta Lakehouse, TF-IDF NLP, and real-time stream scoring.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

recs = load_gold_table("product_recommendations")
metrics = load_gold_table("model_metrics")
scal = load_gold_table("scalability_results")
best_m = metrics[metrics["is_best"]].iloc[0]
max_tput = float(scal["throughput_rps"].max())

# Platform KPI Strip
k1, k2, k3, k4 = st.columns(4)
render_kpi_card(k1, "Corpus Scale", "1,139,264", "Cleaned reviews · 6 categories · 2023–2026", ACCENT_INDIGO)
render_kpi_card(k2, "Tracked Products", f"{len(recs):,}", "3,018 BUY · 1,276 DON'T BUY verdicts", ACCENT_EMERALD)
render_kpi_card(k3, "ML Accuracy", f"{best_m['accuracy']*100:.2f}%", "Logistic Regression · 227,992 test reviews", ACCENT_TEAL)
render_kpi_card(k4, "Peak Throughput", f"{max_tput:,.0f} rev/s", "Measured across 16-run benchmark grid", ACCENT_AMBER)

st.markdown("<div style='height:1.0rem'></div>", unsafe_allow_html=True)

# Instant Hero Product Search
st.subheader("Instant Product Buying Check")
sc1, sc2 = st.columns([3.5, 1.0])
hero_q = sc1.text_input(
    "Search any product before you buy",
    value="",
    placeholder="Try: echo dot, fire tv stick, anker charger, yoga mat, air fryer...",
    label_visibility="collapsed",
)
if sc2.button("Open Full Buying Advisor", type="primary", width="stretch"):
    if hero_q.strip():
        st.session_state.search_kw = hero_q.strip()
    st.switch_page("views/search.py")

if hero_q.strip():
    matches = recs[recs["product_title"].fillna("").str.contains(hero_q.strip(), case=False, regex=False)].sort_values("reviews", ascending=False).head(3)
    if matches.empty:
        st.info("No matching products found. Try a broader keyword or open the full Product Buying Advisor.")
    else:
        cols = st.columns(len(matches))
        for col, (_, item) in zip(cols, matches.iterrows()):
            v = item["recommendation"]
            bcls = "badge-buy" if v == "BUY" else ("badge-dont-buy" if v == "DON'T BUY" else "badge-insufficient")
            card_color = ACCENT_EMERALD if v == "BUY" else (ACCENT_CORAL if v == "DON'T BUY" else ACCENT_AMBER)
            price_str = f"${item['price']:.2f}" if pd.notna(item["price"]) else "N/A"
            col.markdown(
                f"""
                <div class="kpi-card" style="border-top-color:{card_color};">
                    <div class="kpi-label">{item['category']} · {item['brand']}</div>
                    <div style="font-weight:700; font-size:0.96rem; margin-bottom:0.45rem; color:#f8fafc;">{(item['product_title'] or '')[:72]}</div>
                    <span class='verdict-badge {bcls}'>{v} · {item['buy_score']}/100</span>
                    <div class="kpi-sub" style="margin-top:0.5rem;">
                        Reviews: <strong>{int(item['reviews']):,}</strong> · Avg Rating: <strong>{item['avg_rating']:.2f}/5</strong> · Price: <strong>{price_str}</strong><br/>
                        {item['reason_summary'][:130]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.markdown("<div style='height:1.0rem'></div>", unsafe_allow_html=True)

# Featured Showcase: Top Recommended vs Products to Avoid
st.subheader("Live Catalog Highlights")
high_vol = recs[recs["reviews"] >= 100].copy()
top_buy = high_vol[high_vol["recommendation"] == "BUY"].sort_values(["buy_score", "reviews"], ascending=[False, False]).head(3)
top_avoid = high_vol[high_vol["recommendation"] == "DON'T BUY"].sort_values(["buy_score", "reviews"], ascending=[True, False]).head(3)

col_buy, col_avoid = st.columns(2)
with col_buy:
    st.markdown("**Top Rated Products to Buy (100+ Reviews)**")
    for _, item in top_buy.iterrows():
        st.markdown(
            f"""
            <div class="review-card" style="border-left-color:{ACCENT_EMERALD};">
                <span class="verdict-badge badge-buy" style="font-size:0.72rem; padding:0.18rem 0.55rem;">BUY · {item['buy_score']}/100</span>
                <strong style="margin-left:0.45rem; color:#f8fafc;">{(item['product_title'] or item['parent_asin'])[:65]}</strong><br/>
                <span style="font-size:0.81rem; color:#94a3b8;">
                    {item['category']} · Brand: {item['brand']} · {int(item['reviews']):,} reviews · Avg {item['avg_rating']:.2f}/5 · {item['positive_pct']}% positive
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

with col_avoid:
    st.markdown("**High-Volume Products Flagged as DON'T BUY (100+ Reviews)**")
    for _, item in top_avoid.iterrows():
        st.markdown(
            f"""
            <div class="review-card" style="border-left-color:{ACCENT_CORAL};">
                <span class="verdict-badge badge-dont-buy" style="font-size:0.72rem; padding:0.18rem 0.55rem;">DON'T BUY · {item['buy_score']}/100</span>
                <strong style="margin-left:0.45rem; color:#f8fafc;">{(item['product_title'] or item['parent_asin'])[:65]}</strong><br/>
                <span style="font-size:0.81rem; color:#94a3b8;">
                    {item['category']} · Brand: {item['brand']} · {int(item['reviews']):,} reviews · Avg {item['avg_rating']:.2f}/5 · Main risk: {item['top_complaint'] or 'Low rating'}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<div style='height:0.9rem'></div>", unsafe_allow_html=True)

# Explore Platform Modules
st.subheader("Platform Modules")
m1, m2, m3, m4, m5 = st.columns(5)
modules = [
    (m1, "Stage D · Phase 9", "Market Analytics", "4-year sentiment trends, 1–5 star ratings, lift words, complaint heatmaps, and brand scatter.", "views/historical.py", "Open Analytics", ACCENT_INDIGO),
    (m2, "Stage D · Phases 10–11", "Buying Advisor", "Search 543K+ products, inspect 7-factor BUY scores, aspect sentiment, and side-by-side radar.", "views/search.py", "Open Advisor", ACCENT_EMERALD),
    (m3, "Stage C · Phases 7–8", "ML & Review Checker", "Compare Naive Bayes, Logistic Regression & Random Forest; analyze single or batch reviews live.", "views/ml_models.py", "Open ML Lab", ACCENT_TEAL),
    (m4, "Stage D · Phase 12", "Live Stream Monitor", "Real-time micro-batch sentiment inference on 2026 reviews with automated negative alerts.", "views/streaming.py", "Open Stream", ACCENT_AMBER),
    (m5, "Module 3 · Phase 15", "Scalability & Scope", "16-run execution benchmark across 10K–500K rows and 2–16 partitions, plus project scope.", "views/scalability.py", "Open Benchmark", "#a855f7"),
]
for col, stg, nm, desc, page_path, btn_txt, clr in modules:
    with col:
        st.markdown(
            f"""
            <div class="pipe-step" style="border-top-color:{clr}; margin-bottom:0.5rem;">
                <div class="pipe-stage" style="color:{clr};">{stg}</div>
                <div class="pipe-name">{nm}</div>
                <div class="pipe-desc">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(btn_txt, key=f"nav_{page_path}", width="stretch"):
            st.switch_page(page_path)
