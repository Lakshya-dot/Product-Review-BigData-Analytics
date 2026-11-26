"""Page 4: Live Review Stream Telemetry & Alert Monitor (Phase 12 - Spark Structured Streaming)."""

import time
import numpy as np
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
    PLOTLY_CONFIG,
    SENTIMENT_COLORS,
    download_csv_button,
    load_gold_table,
    load_nlp_models,
    render_kpi_card,
    render_page_header,
    style_fig,
)

render_page_header(
    badge="Phase 12 · Spark Structured Streaming",
    title="Real-Time Review Stream & Alert Telemetry",
    subtitle=(
        "Simulates continuous micro-batch ingestion over 2026 Amazon reviews, running live TF-IDF + Logistic Regression "
        "inference, rolling sentiment momentum tracking, and automated negative-spike alerting."
    ),
)

pool = load_gold_table("stream_sample_pool")
vec, lr, _ = load_nlp_models()
LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}

if "stream_cursor" not in st.session_state:
    st.session_state.stream_cursor = 0
if "stream_events" not in st.session_state:
    st.session_state.stream_events = pd.DataFrame()
if "stream_batches" not in st.session_state:
    st.session_state.stream_batches = []
if "stream_running" not in st.session_state:
    st.session_state.stream_running = False

# Live Stream Status Indicator
status_html = (
    "<span class='status-pill'><span class='dot-live'></span>STREAM ACTIVE · AUTO-INGESTING</span>"
    if st.session_state.stream_running
    else "<span class='status-pill'><span class='dot-idle'></span>STREAM STANDBY · MANUAL / STEP MODE</span>"
)
st.markdown(status_html, unsafe_allow_html=True)

# Stream Control Toolbar
c1, c2, c3, c4, c5, c6 = st.columns([1.5, 1.0, 1.0, 1.0, 1.0, 0.9])
stream_cat = c1.selectbox("Category Stream Filter", ["All"] + CATEGORIES)
batch_size = c2.slider("Micro-Batch Size", 10, 100, 25, step=5)
tick_sec = c3.slider("Interval (sec)", 0.5, 3.0, 1.0, step=0.5)

active_pool = pool if stream_cat == "All" else pool[pool["category"] == stream_cat]
active_pool = active_pool.reset_index(drop=True)


def process_one_batch():
    cur = st.session_state.stream_cursor
    if cur >= len(active_pool):
        cur = 0
    end = min(cur + batch_size, len(active_pool))
    batch = active_pool.iloc[cur:end].copy()
    st.session_state.stream_cursor = end if end < len(active_pool) else 0

    X = vec.transform(batch["clean_text"].fillna("").tolist())
    preds = lr.predict(X)
    probs = lr.predict_proba(X)

    batch["predicted_sentiment"] = [LABEL_MAP[int(p)] for p in preds]
    batch["confidence_pct"] = (np.max(probs, axis=1) * 100).round(1)
    batch["batch_id"] = len(st.session_state.stream_batches) + 1
    batch["ingested_at"] = pd.Timestamp.now().strftime("%H:%M:%S")
    batch["is_alert"] = batch["predicted_sentiment"] == "Negative"

    st.session_state.stream_events = pd.concat([batch, st.session_state.stream_events], ignore_index=True).head(1000)

    st.session_state.stream_batches.append({
        "batch_id": batch["batch_id"].iloc[0],
        "time": batch["ingested_at"].iloc[0],
        "events": len(batch),
        "Positive %": round((batch["predicted_sentiment"] == "Positive").mean() * 100, 1),
        "Neutral %": round((batch["predicted_sentiment"] == "Neutral").mean() * 100, 1),
        "Negative %": round((batch["predicted_sentiment"] == "Negative").mean() * 100, 1),
    })


c4.markdown("<div style='height:1.65rem'></div>", unsafe_allow_html=True)
b_step = c4.button("+1 Micro-Batch", width="stretch")

c5.markdown("<div style='height:1.65rem'></div>", unsafe_allow_html=True)
if st.session_state.stream_running:
    if c5.button("Pause Stream", width="stretch"):
        st.session_state.stream_running = False
        st.rerun()
else:
    if c5.button("Start Auto-Stream", type="primary", width="stretch"):
        st.session_state.stream_running = True
        st.rerun()

c6.markdown("<div style='height:1.65rem'></div>", unsafe_allow_html=True)
if c6.button("Reset", width="stretch"):
    st.session_state.stream_cursor = 0
    st.session_state.stream_events = pd.DataFrame()
    st.session_state.stream_batches = []
    st.session_state.stream_running = False
    st.rerun()

if b_step or st.session_state.stream_running or st.session_state.stream_events.empty:
    process_one_batch()

events = st.session_state.stream_events
batches_df = pd.DataFrame(st.session_state.stream_batches)

# Telemetry KPI Cards
total_ev = len(events)
pos_live = (events["predicted_sentiment"] == "Positive").mean() * 100
neg_live = (events["predicted_sentiment"] == "Negative").mean() * 100
alerts_n = int(events["is_alert"].sum())
agree_pct = (events["predicted_sentiment"] == events["sentiment"]).mean() * 100

k1, k2, k3, k4, k5 = st.columns(5)
render_kpi_card(k1, "Events Processed", f"{total_ev:,}", f"{len(batches_df)} micro-batches", ACCENT_INDIGO)
render_kpi_card(k2, "Live Positive Share", f"{pos_live:.1f}%", "Predicted Positive", ACCENT_EMERALD)
render_kpi_card(k3, "Live Negative Share", f"{neg_live:.1f}%", "Predicted Negative", ACCENT_CORAL)
render_kpi_card(k4, "Negative Alerts", f"{alerts_n:,}", "Flagged for triage", ACCENT_AMBER)
render_kpi_card(k5, "Model Agreement", f"{agree_pct:.1f}%", "ML prediction vs rating", ACCENT_TEAL)

st.markdown("<div style='height:0.65rem'></div>", unsafe_allow_html=True)

g1, g2, g3 = st.columns([1.25, 0.85, 1.0])
with g1:
    melted_b = batches_df.tail(30).melt(
        id_vars=["batch_id", "time"],
        value_vars=["Positive %", "Neutral %", "Negative %"],
        var_name="Sentiment",
        value_name="Share (%)",
    )
    melted_b["Sentiment"] = melted_b["Sentiment"].str.replace(" %", "", regex=False)
    fig_b = px.line(
        melted_b, x="batch_id", y="Share (%)", color="Sentiment",
        color_discrete_map=SENTIMENT_COLORS, markers=True,
        labels={"batch_id": "Micro-Batch #"},
    )
    style_fig(fig_b, height=345, title="Rolling Micro-Batch Sentiment Trend", has_top_legend=True)
    fig_b.update_layout(yaxis_range=[0, 105])
    st.plotly_chart(fig_b, width="stretch", config=PLOTLY_CONFIG)

with g2:
    fig_g = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(pos_live, 1),
            number={"suffix": "%", "font": {"color": "#f8fafc"}},
            title={"text": "Live Positive Pulse", "font": {"size": 14, "color": "#f1f5f9"}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": ACCENT_TEAL if pos_live >= 65 else (ACCENT_AMBER if pos_live >= 50 else ACCENT_CORAL)},
                "steps": [
                    {"range": [0, 50], "color": "rgba(244,63,94,0.16)"},
                    {"range": [50, 70], "color": "rgba(245,158,11,0.16)"},
                    {"range": [70, 100], "color": "rgba(45,212,191,0.18)"},
                ],
            },
        )
    )
    style_fig(fig_g, height=345)
    fig_g.update_layout(margin=dict(t=58, b=15, l=22, r=22))
    st.plotly_chart(fig_g, width="stretch", config=PLOTLY_CONFIG)

with g3:
    cat_cnt = events.groupby(["category", "predicted_sentiment"], as_index=False).size()
    fig_c = px.bar(
        cat_cnt, x="category", y="size", color="predicted_sentiment",
        color_discrete_map=SENTIMENT_COLORS, barmode="stack",
        labels={"size": "Reviews", "category": "", "predicted_sentiment": ""},
    )
    style_fig(fig_c, height=345, title="Streamed Category Mix", has_top_legend=True)
    fig_c.update_xaxes(tickangle=-18)
    st.plotly_chart(fig_c, width="stretch", config=PLOTLY_CONFIG)

a_col, e_col = st.columns([1.0, 1.3])
with a_col:
    st.subheader("Negative Sentiment Alerts")
    alert_rows = events[events["is_alert"]].head(8)
    if alert_rows.empty:
        st.success("No negative reviews in the current stream window.")
    else:
        for _, rv in alert_rows.iterrows():
            st.markdown(
                f"""
                <div class="review-card" style="border-left-color: #f43f5e;">
                    <span class="meta-tag" style="background:rgba(244,63,94,0.16); border-color:rgba(244,63,94,0.32); color:#fda4af;">NEGATIVE ALERT</span>
                    <strong>{rv['ingested_at']} · {rv['category']}</strong><br/>
                    <div style="font-weight:600; margin-top:0.25rem; color:#f1f5f9;">{(rv['product_title'] or rv['parent_asin'])[:65]}</div>
                    <span style="font-size:0.82rem; color:#94a3b8;">Brand: {rv['brand']} · Rating: {rv['rating']:.1f} / 5.0 · ML Confidence: {rv['confidence_pct']}%</span><br/>
                    <span style="font-size:0.88rem; color:#cbd5e1;">{(rv['title'] or '')} — {(rv['text'] or '')[:175]}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

with e_col:
    eh1, eh2 = st.columns([2, 1])
    eh1.subheader("Live Event Log")
    with eh2:
        download_csv_button(events.drop(columns=["clean_text"], errors="ignore"), "live_stream_log.csv", "Export Log CSV")

    st.dataframe(
        events[["ingested_at", "batch_id", "category", "brand", "rating", "sentiment", "predicted_sentiment", "confidence_pct", "title"]]
        .head(30)
        .rename(columns={
            "ingested_at": "Time",
            "batch_id": "Batch",
            "category": "Category",
            "brand": "Brand",
            "rating": "Rating",
            "sentiment": "Actual",
            "predicted_sentiment": "ML Pred",
            "confidence_pct": "Conf %",
            "title": "Review Title",
        }),
        width="stretch",
        hide_index=True,
        height=400,
    )

if st.session_state.stream_running:
    time.sleep(tick_sec)
    st.rerun()
