"""Page 3: ML Model Benchmark, Feature Explainability & Live/Batch Review Checker."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import (
    ACCENT_CORAL,
    ACCENT_EMERALD,
    ACCENT_INDIGO,
    ACCENT_TEAL,
    HEATMAP_INDIGO,
    MODEL_PALETTE,
    PLOTLY_CONFIG,
    SENTIMENT_COLORS,
    clean_review_text,
    download_csv_button,
    load_gold_table,
    load_nlp_models,
    predict_single_review,
    render_kpi_card,
    render_page_header,
    style_fig,
)

render_page_header(
    badge="Phases 7–8 & Module 2 · NLP & Classification",
    title="ML Model Benchmark, Explainability & Review Analyzer",
    subtitle=(
        "Three TF-IDF classifiers evaluated on a deterministic 80/20 review_id hash split (911,272 train / 227,992 held-out test) "
        "with inverse-frequency balanced class weights to preserve Negative and Neutral complaint recall."
    ),
)

metrics = load_gold_table("model_metrics").sort_values("macro_f1", ascending=False)
cms = load_gold_table("confusion_matrices")
lr_feats = load_gold_table("top_lr_features")
best_row = metrics[metrics["is_best"]].iloc[0]

# Top KPI Cards
k1, k2, k3, k4 = st.columns(4)
render_kpi_card(k1, "Selected Production Model", "Logistic Regression", f"Macro F1: {best_row['macro_f1']:.3f} (highest)", ACCENT_TEAL)
render_kpi_card(k2, "Overall Accuracy", f"{best_row['accuracy']*100:.2f}%", f"Weighted F1: {best_row['weighted_f1']:.3f}", ACCENT_INDIGO)
render_kpi_card(k3, "Negative Recall", f"{best_row['recall_negative']*100:.2f}%", "Complaint detection sensitivity", ACCENT_CORAL)
render_kpi_card(k4, "Held-Out Test Volume", "227,992", f"Inference latency: {best_row['test_seconds']:.2f}s", ACCENT_EMERALD)

st.markdown("<div style='height:0.65rem'></div>", unsafe_allow_html=True)

tab_eval, tab_feat, tab_pred, tab_batch = st.tabs([
    "Model Benchmark & Confusion Matrix",
    "Learned TF-IDF Features",
    "Interactive Review Checker",
    "Batch Review Scorer",
])

with tab_eval:
    display_m = metrics.copy()
    display_m["Status"] = display_m["is_best"].map(lambda x: "Selected" if x else "Baseline")
    h1, h2 = st.columns([3, 1])
    h1.subheader("Classifier Comparison Table")
    with h2:
        download_csv_button(display_m, "model_metrics.csv", "Export Metrics CSV")

    st.dataframe(
        display_m[[
            "model", "Status", "accuracy", "macro_f1", "weighted_f1",
            "recall_negative", "recall_neutral", "recall_positive",
            "train_rows", "train_seconds", "test_seconds",
        ]].rename(columns={
            "model": "Model",
            "accuracy": "Accuracy",
            "macro_f1": "Macro F1",
            "weighted_f1": "Weighted F1",
            "recall_negative": "Neg Recall",
            "recall_neutral": "Neu Recall",
            "recall_positive": "Pos Recall",
            "train_rows": "Train Rows",
            "train_seconds": "Train (s)",
            "test_seconds": "Test (s)",
        }),
        width="stretch",
        hide_index=True,
    )

    c_left, c_right = st.columns([1.15, 1.0])
    with c_left:
        chart_type = st.radio("Metric Chart", ["Grouped Bars", "Multi-Metric Radar"], horizontal=True, label_visibility="collapsed")
        if chart_type == "Grouped Bars":
            melted = display_m.melt(
                id_vars="model",
                value_vars=["accuracy", "macro_f1", "weighted_f1", "recall_negative", "recall_neutral", "recall_positive"],
                var_name="Metric",
                value_name="Score",
            )
            fig_m = px.bar(
                melted, x="Metric", y="Score", color="model", barmode="group",
                text_auto=".3f", color_discrete_sequence=MODEL_PALETTE,
            )
            style_fig(fig_m, height=385, title="Accuracy, F1 & Per-Class Recall Comparison", has_top_legend=True)
            fig_m.update_layout(yaxis_range=[0, 1.05])
        else:
            axes = ["Accuracy", "Macro F1", "Weighted F1", "Neg Recall", "Neu Recall", "Pos Recall"]
            cols_r = ["accuracy", "macro_f1", "weighted_f1", "recall_negative", "recall_neutral", "recall_positive"]
            colors_r = MODEL_PALETTE
            fig_m = go.Figure()
            for idx, (_, row) in enumerate(display_m.iterrows()):
                vals = [float(row[c]) for c in cols_r]
                fig_m.add_trace(
                    go.Scatterpolar(
                        r=vals + [vals[0]], theta=axes + [axes[0]],
                        fill="toself", name=row["model"], line_color=colors_r[idx % len(colors_r)],
                    )
                )
            style_fig(fig_m, height=385, title="Classifier Profile Radar", has_top_legend=True)
            fig_m.update_layout(
                polar=dict(bgcolor="rgba(17,20,29,0.55)", radialaxis=dict(visible=True, range=[0, 1.0], gridcolor="rgba(255,255,255,0.10)")),
            )
        st.plotly_chart(fig_m, width="stretch", config=PLOTLY_CONFIG)

    with c_right:
        ctrl_cm1, ctrl_cm2 = st.columns(2)
        cm_model = ctrl_cm1.selectbox("Model", display_m["model"].tolist(), index=0)
        cm_norm = ctrl_cm2.radio("Display", ["Row % (Recall)", "Raw Counts"], horizontal=True)

        cm_sub = cms[cms["model"] == cm_model].pivot(index="actual", columns="predicted", values="count")
        order = ["Negative", "Neutral", "Positive"]
        cm_sub = cm_sub.reindex(index=order, columns=order, fill_value=0)
        if cm_norm == "Row % (Recall)":
            cm_plot = cm_sub.div(cm_sub.sum(axis=1), axis=0) * 100.0
            fmt = ".1f"
            cbar = "Row %"
        else:
            cm_plot = cm_sub
            fmt = ","
            cbar = "Reviews"

        fig_cm = px.imshow(
            cm_plot, text_auto=fmt, color_continuous_scale=HEATMAP_INDIGO,
            labels=dict(x="Predicted Sentiment", y="Actual Rating Label", color=cbar),
        )
        style_fig(fig_cm, height=340, title=f"Confusion Matrix — {cm_model}")
        st.plotly_chart(fig_cm, width="stretch", config=PLOTLY_CONFIG)

with tab_feat:
    n_feats = st.slider("Top Features per Polarity", 10, 30, 15, step=5)
    pos_f = lr_feats[lr_feats["sentiment"] == "Positive"].head(n_feats).sort_values("weight", ascending=True)
    neg_f = lr_feats[lr_feats["sentiment"] == "Negative"].head(n_feats).sort_values("weight", ascending=True)

    fig_bf = go.Figure()
    fig_bf.add_trace(
        go.Bar(
            name="Negative Weight",
            y=neg_f["feature"],
            x=-neg_f["weight"],
            orientation="h",
            marker_color=ACCENT_CORAL,
            text=neg_f["weight"].map(lambda w: f"-{w:.2f}"),
            textposition="outside",
        )
    )
    fig_bf.add_trace(
        go.Bar(
            name="Positive Weight",
            y=pos_f["feature"],
            x=pos_f["weight"],
            orientation="h",
            marker_color=ACCENT_TEAL,
            text=pos_f["weight"].map(lambda w: f"+{w:.2f}"),
            textposition="outside",
        )
    )
    max_w = max(float(pos_f["weight"].max()), float(neg_f["weight"].max())) * 1.25
    style_fig(fig_bf, height=540, title="Logistic Regression Learned Feature Polarity (Unigrams & Bigrams)", has_top_legend=True)
    fig_bf.update_layout(
        barmode="relative",
        xaxis_range=[-max_w, max_w],
        xaxis_title="Negative Coefficient  |  Positive Coefficient",
    )
    st.plotly_chart(fig_bf, width="stretch", config=PLOTLY_CONFIG)

with tab_pred:
    st.caption("Uses the exact regex cleaning + fitted 80K-feature TF-IDF + classifier pipeline (no re-fitting).")
    p1, p2, p3, p4 = st.columns(4)
    if "checker_text" not in st.session_state:
        st.session_state.checker_text = "The battery lasts all day and the sound quality is excellent!"

    if p1.button("Sample: Positive Review", width="stretch"):
        st.session_state.checker_text = "The battery lasts all day, setup took two minutes, and the build quality feels sturdy. Worth every penny!"
    if p2.button("Sample: Negative Review", width="stretch"):
        st.session_state.checker_text = "Stopped working after three days. Cheap plastic, terrible battery life, and I am returning it for a refund."
    if p3.button("Sample: Mixed / 3-Star", width="stretch"):
        st.session_state.checker_text = "The sound is decent for the price, not bad overall, but the cable is a bit short and packaging arrived damaged."
    if p4.button("Clear Input", width="stretch"):
        st.session_state.checker_text = ""

    col_in, col_cfg = st.columns([3, 1.1])
    user_text = col_in.text_area("Customer Review Text", key="checker_text", height=115, placeholder="Paste or type a product review to evaluate...")
    model_mode = col_cfg.radio("Evaluation Mode", ["Compare Both", "Logistic Regression", "Naive Bayes"], index=0)

    if st.button("Analyze Sentiment", type="primary"):
        res_lr = None
        res_main = None
        if model_mode == "Compare Both":
            res_lr = predict_single_review(user_text, "Logistic Regression")
            res_nb = predict_single_review(user_text, "Naive Bayes")
            if res_lr is None:
                st.warning("Please enter at least one valid alphabetic word.")
            else:
                mcol1, mcol2 = st.columns(2)
                for col, mname, res in [(mcol1, "Logistic Regression (Selected)", res_lr), (mcol2, "Naive Bayes (Baseline)", res_nb)]:
                    pred = res["prediction"]
                    probs = res["probabilities"]
                    color = SENTIMENT_COLORS[pred]
                    col.markdown(
                        f"""
                        <div class="kpi-card" style="border-top-color:{color};">
                            <div class="kpi-label">{mname}</div>
                            <div class="kpi-value" style="color:{color};">{pred} ({probs[pred]*100:.1f}%)</div>
                            <div class="kpi-sub">Neg: {probs['Negative']*100:.1f}% · Neu: {probs['Neutral']*100:.1f}% · Pos: {probs['Positive']*100:.1f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                res_main = res_lr
        else:
            res_main = predict_single_review(user_text, model_mode)
            if res_main is None:
                st.warning("Please enter at least one valid alphabetic word.")
            else:
                pred = res_main["prediction"]
                probs = res_main["probabilities"]
                color = SENTIMENT_COLORS[pred]
                st.markdown(
                    f"""
                    <div class="kpi-card" style="border-top-color:{color}; margin-bottom:0.8rem;">
                        <div class="kpi-label">Predicted Sentiment ({model_mode})</div>
                        <div class="kpi-value" style="color:{color};">{pred} · {probs[pred]*100:.1f}% Confidence</div>
                        <div class="kpi-sub">Preprocessed text: <code>{res_main['cleaned_text'][:140]}</code></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        if res_main is not None:
            r1, r2 = st.columns([1.0, 1.3])
            with r1:
                probs = res_main["probabilities"]
                prob_df = pd.DataFrame([
                    {"Class": k, "Probability (%)": round(v * 100, 1)}
                    for k, v in probs.items()
                ])
                fig_pr = px.bar(
                    prob_df, x="Class", y="Probability (%)", color="Class",
                    color_discrete_map=SENTIMENT_COLORS, text_auto=".1f",
                )
                style_fig(fig_pr, height=320, title="Class Probability Distribution (LR)")
                fig_pr.update_layout(showlegend=False, yaxis_range=[0, 108])
                st.plotly_chart(fig_pr, width="stretch", config=PLOTLY_CONFIG)

            with r2:
                contrib_df = pd.DataFrame(res_main["top_contributions"])
                if contrib_df.empty:
                    st.info("No vocabulary words matched the fitted TF-IDF dictionary.")
                else:
                    fig_exp = px.bar(
                        contrib_df.sort_values("impact", ascending=True),
                        x="impact", y="term", orientation="h", color="direction",
                        color_discrete_map={"Positive signal": ACCENT_TEAL, "Negative signal": ACCENT_CORAL},
                        labels={"impact": "Log-Odds Signal (+ Pos / - Neg)", "term": "Matched Term"},
                    )
                    style_fig(fig_exp, height=320, title="Token-Level Feature Attribution")
                    st.plotly_chart(fig_exp, width="stretch", config=PLOTLY_CONFIG)

with tab_batch:
    st.caption("Score multiple reviews simultaneously: upload a CSV (with a `text` or `review` column) or evaluate the built-in sample batch.")
    uploaded = st.file_uploader("Upload Review CSV (optional)", type=["csv"])
    sample_batch = pd.DataFrame({
        "review": [
            "Absolutely love this blender, crushes ice effortlessly and easy to clean!",
            "Arrived cracked and the motor smelled like burning plastic on first use.",
            "Works fine for basic tasks, nothing special, instructions could be clearer.",
            "Best wireless mouse I have owned, battery lasted 4 months already.",
            "Completely stopped charging after 2 weeks. Waste of money, returning.",
            "Good fabric quality and fits nicely, though color is slightly darker than pictured.",
        ]
    })

    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        text_col = next((c for c in batch_df.columns if c.lower() in {"text", "review", "review_text", "comment"}), batch_df.columns[0])
    else:
        batch_df = sample_batch
        text_col = "review"

    if st.button("Score Batch with Logistic Regression", type="primary"):
        vec, lr, _ = load_nlp_models()
        cleaned = [clean_review_text(str(x)) for x in batch_df[text_col].fillna("")]
        X_b = vec.transform(cleaned)
        preds = lr.predict(X_b)
        probs = lr.predict_proba(X_b)
        label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        out_df = batch_df.copy()
        out_df["Predicted Sentiment"] = [label_map[int(p)] for p in preds]
        out_df["Confidence (%)"] = (np.max(probs, axis=1) * 100).round(1)
        out_df["Pos Prob (%)"] = (probs[:, 2] * 100).round(1)
        out_df["Neg Prob (%)"] = (probs[:, 0] * 100).round(1)
        st.dataframe(out_df, width="stretch", hide_index=True)
        download_csv_button(out_df, "batch_scored_reviews.csv", "Export Scored Batch CSV")
