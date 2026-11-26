"""Page 5: Big Data Scalability Analysis, Speedup & Academic Limitations (Module 3 & Slide 14)."""

import plotly.express as px
import streamlit as st

from utils import (
    ACCENT_AMBER,
    ACCENT_INDIGO,
    ACCENT_TEAL,
    HEATMAP_INDIGO,
    HEATMAP_TEAL,
    PLOTLY_CONFIG,
    STAGE_PALETTE,
    YEAR_PALETTE,
    download_csv_button,
    load_gold_table,
    render_kpi_card,
    render_page_header,
    style_fig,
)

render_page_header(
    badge="Module 3 & Phase 15 · Scalability & Evaluation",
    title="Big Data Scalability Benchmark & System Scope",
    subtitle=(
        "Empirical 16-run execution benchmark across 4 dataset scales (10K, 100K, 250K, 500K reviews) × "
        "4 partition levels (2, 4, 8, 16), plus methodological limitations and future enhancements."
    ),
)

scal = load_gold_table("scalability_results").copy()
scal["partitions_str"] = scal["partitions"].astype(str) + " partitions"

base_2 = scal[scal["partitions"] == 2][["reviews", "total_sec"]].rename(columns={"total_sec": "base_sec"})
scal = scal.merge(base_2, on="reviews", how="left")
scal["speedup"] = (scal["base_sec"] / scal["total_sec"]).round(2)

fastest_500k = scal[scal["reviews"] == 500_000].sort_values("total_sec").iloc[0]
slowest_500k = scal[scal["reviews"] == 500_000].sort_values("total_sec", ascending=False).iloc[0]
speedup_500k = slowest_500k["total_sec"] / fastest_500k["total_sec"]
max_tput = scal.sort_values("throughput_rps", ascending=False).iloc[0]

k1, k2, k3, k4 = st.columns(4)
render_kpi_card(k1, "Experiment Grid", f"{len(scal)} runs", "4 sizes × 4 partition counts", ACCENT_INDIGO)
render_kpi_card(k2, "Peak Throughput", f"{max_tput['throughput_rps']:,.0f} rev/s", f"{int(max_tput['reviews']):,} rows @ {int(max_tput['partitions'])} parts", ACCENT_TEAL)
render_kpi_card(k3, "Best 500K Runtime", f"{fastest_500k['total_sec']:.2f}s", f"{speedup_500k:.2f}x faster vs slowest 500K run", "#a855f7")
render_kpi_card(k4, "Full Corpus Scale", "1,139,264", "Cleaned reviews · 6 categories", ACCENT_AMBER)

st.markdown("<div style='height:0.65rem'></div>", unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    scale_log = st.checkbox("Logarithmic Y-Axis (Execution Time)", value=False)
    fig_time = px.line(
        scal.sort_values(["partitions", "reviews"]),
        x="reviews",
        y="total_sec",
        color="partitions_str",
        markers=True,
        log_y=scale_log,
        color_discrete_sequence=YEAR_PALETTE,
        labels={"reviews": "Dataset Size (Reviews)", "total_sec": "Total Time (seconds)", "partitions_str": "Parallelism"},
    )
    style_fig(fig_time, height=385, title="Total Execution Time vs Dataset Size", has_top_legend=True)
    st.plotly_chart(fig_time, width="stretch", config=PLOTLY_CONFIG)

with c2:
    heat_metric = st.radio("Heatmap Metric", ["Throughput (rev/s)", "Speedup vs 2-Partition"], horizontal=True, label_visibility="collapsed")
    if heat_metric == "Throughput (rev/s)":
        heat = scal.pivot(index="reviews", columns="partitions", values="throughput_rps")
        fig_heat = px.imshow(
            heat, text_auto=",.0f", color_continuous_scale=HEATMAP_INDIGO,
            labels=dict(x="Partitions / Workers", y="Dataset Size (Reviews)", color="Reviews / sec"),
        )
        style_fig(fig_heat, height=370, title="Throughput Heatmap (Reviews / Second)")
    else:
        heat = scal.pivot(index="reviews", columns="partitions", values="speedup")
        fig_heat = px.imshow(
            heat, text_auto=".2f", color_continuous_scale=HEATMAP_TEAL,
            labels=dict(x="Partitions / Workers", y="Dataset Size (Reviews)", color="Speedup (x)"),
        )
        style_fig(fig_heat, height=370, title="Relative Speedup Heatmap (vs 2-Partition Baseline)")
    st.plotly_chart(fig_heat, width="stretch", config=PLOTLY_CONFIG)

st.markdown("---")
ctrl1, ctrl2 = st.columns([1.6, 1.0])
sel_parts = ctrl1.radio("Partition Level for Stage Breakdown", sorted(scal["partitions"].unique().tolist()), horizontal=True, index=3)
brk_mode = ctrl2.radio("Stage Metric", ["Absolute Seconds", "Share (%)"], horizontal=True)

sub = scal[scal["partitions"] == sel_parts].sort_values("reviews").copy()
stage_cols = ["ingestion_sec", "cleaning_sec", "tfidf_sec", "training_sec"]
if brk_mode == "Share (%)":
    for col in stage_cols:
        sub[col] = (sub[col] * 100.0 / sub["total_sec"]).round(1)
    y_label = "Share of Total Time (%)"
    fmt = ".1f"
else:
    y_label = "Execution Time (seconds)"
    fmt = ".2f"

melted = sub.melt(id_vars="reviews", value_vars=stage_cols, var_name="Stage", value_name="Value")
melted["Stage"] = melted["Stage"].map({
    "ingestion_sec": "1. Ingestion",
    "cleaning_sec": "2. Text Cleaning",
    "tfidf_sec": "3. NLP / TF-IDF",
    "training_sec": "4. Model Training",
})
melted["reviews_label"] = melted["reviews"].map(lambda n: f"{n:,}")

fig_brk = px.bar(
    melted, x="reviews_label", y="Value", color="Stage", barmode="stack", text_auto=fmt,
    color_discrete_sequence=STAGE_PALETTE,
    labels={"reviews_label": "Dataset Size", "Value": y_label},
)
style_fig(fig_brk, height=370, title=f"Pipeline Stage Breakdown at {sel_parts} Partitions", has_top_legend=True)
st.plotly_chart(fig_brk, width="stretch", config=PLOTLY_CONFIG)

th1, th2 = st.columns([3, 1])
th1.subheader("Full 16-Run Benchmark Measurements")
with th2:
    download_csv_button(scal.drop(columns=["partitions_str", "base_sec"]), "scalability_benchmark_16_runs.csv", "Export Benchmark CSV")

st.dataframe(
    scal[[
        "reviews", "partitions", "ingestion_sec", "cleaning_sec",
        "tfidf_sec", "training_sec", "total_sec", "throughput_rps", "speedup",
    ]].rename(columns={
        "reviews": "Reviews",
        "partitions": "Partitions",
        "ingestion_sec": "Ingest (s)",
        "cleaning_sec": "Clean (s)",
        "tfidf_sec": "TF-IDF (s)",
        "training_sec": "Train (s)",
        "total_sec": "Total (s)",
        "throughput_rps": "Throughput (rev/s)",
        "speedup": "Speedup (x)",
    }),
    width="stretch",
    hide_index=True,
)

st.markdown("---")
st.subheader("Academic Scope: Known Limitations & Future Enhancements")
l_col, f_col = st.columns(2)
with l_col:
    st.warning(
        "**Current Limitations**\n"
        "- **Proxy Sentiment Labels:** Derived from 1–5 star ratings (`1–2 Neg`, `3 Neu`, `4–5 Pos`), not manually annotated ground truth.\n"
        "- **Date-Shifted Prototype:** Real 2020–2023 Amazon reviews shifted +3 years to 2023–2026 (`2026` ends ~September).\n"
        "- **Bag-of-Words / TF-IDF Context:** Sarcasm and complex multi-sentence negation remain challenging.\n"
        "- **Compute Constraints:** Executed on Databricks Free Edition and local workstation rather than a multi-node cluster."
    )
with f_col:
    st.info(
        "**Future Enhancements**\n"
        "- **Transformer Fine-Tuning:** Upgrade from TF-IDF + Logistic Regression to DistilBERT / RoBERTa for sarcasm & nuance.\n"
        "- **Multi-Node Spark Cluster + Kafka:** Connect live e-commerce webhooks to distributed Kafka + Spark Streaming.\n"
        "- **Fake / Bot Review Filtering:** Incorporate verified-purchase anomaly scoring and reviewer burst detection.\n"
        "- **Cross-Lingual Support:** Multilingual sentiment models across global marketplaces."
    )
