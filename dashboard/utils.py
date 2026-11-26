"""Shared enterprise UI/UX styling, signature Midnight Carbon & Aurora palette, loaders, and ML helpers."""

import os
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD_DIR = os.path.join(BASE_DIR, "data", "gold")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Signature Midnight Carbon + Aurora Teal/Indigo + Rose Coral Palette
ACCENT_INDIGO = "#818cf8"
ACCENT_TEAL = "#2dd4bf"
ACCENT_EMERALD = "#10b981"
ACCENT_AMBER = "#f59e0b"
ACCENT_CORAL = "#f43f5e"
ACCENT_SKY = "#38bdf8"

SENTIMENT_COLORS = {
    "Positive": ACCENT_EMERALD,
    "Neutral": ACCENT_AMBER,
    "Negative": ACCENT_CORAL,
}

RECOMMENDATION_COLORS = {
    "BUY": ACCENT_EMERALD,
    "DON'T BUY": ACCENT_CORAL,
    "INSUFFICIENT DATA": ACCENT_AMBER,
}

YEAR_PALETTE = ["#64748b", "#38bdf8", "#818cf8", "#2dd4bf"]
MODEL_PALETTE = ["#2dd4bf", "#818cf8", "#f59e0b"]
STAGE_PALETTE = ["#64748b", "#38bdf8", "#818cf8", "#2dd4bf"]

HEATMAP_TEAL = [[0.0, "#12151f"], [0.45, "#1e3a5f"], [0.75, "#0d9488"], [1.0, "#2dd4bf"]]
HEATMAP_INDIGO = [[0.0, "#12151f"], [0.5, "#3730a3"], [1.0, "#818cf8"]]
HEATMAP_CORAL = [[0.0, "#12151f"], [0.5, "#881337"], [1.0, "#fb7185"]]

CATEGORIES = [
    "Electronics",
    "Home and Kitchen",
    "Health and Household",
    "Beauty and Personal Care",
    "Sports and Outdoors",
    "Books",
]

PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
}


def inject_custom_css():
    st.markdown(
        """
        <style>
        /* Global Canvas & Typography */
        html, body, [class*="css"] {
            font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
            max-width: 1380px;
        }
        h1, h2, h3, h4 {
            letter-spacing: -0.02em;
            color: #f8fafc;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        [data-testid="stAppDeployButton"] {display: none;}

        /* Top Website Navigation Styling */
        [data-testid="stHeader"] {
            background: rgba(9, 11, 16, 0.90);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        }

        /* Executive Page Header */
        .hero-wrap {
            background: radial-gradient(circle at top right, rgba(45, 212, 191, 0.14), transparent 42%),
                        radial-gradient(circle at bottom left, rgba(129, 140, 248, 0.14), transparent 45%),
                        linear-gradient(135deg, #11141d 0%, #0d1017 100%);
            border: 1px solid rgba(255, 255, 255, 0.09);
            border-left: 3.5px solid #818cf8;
            border-radius: 14px;
            padding: 1.35rem 1.65rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.35);
        }
        .hero-badge {
            display: inline-block;
            font-size: 0.70rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.10em;
            color: #a5b4fc;
            background: rgba(129, 140, 248, 0.13);
            border: 1px solid rgba(129, 140, 248, 0.30);
            border-radius: 6px;
            padding: 0.20rem 0.65rem;
            margin-bottom: 0.55rem;
        }
        .hero-title {
            font-size: 1.72rem;
            font-weight: 800;
            color: #f8fafc;
            margin: 0 0 0.30rem 0;
            line-height: 1.22;
        }
        .hero-sub {
            font-size: 0.93rem;
            color: #94a3b8;
            margin: 0;
            line-height: 1.5;
            max-width: 1050px;
        }

        /* Website Landing Hero */
        .web-hero {
            background: radial-gradient(circle at 85% 20%, rgba(45, 212, 191, 0.18), transparent 40%),
                        radial-gradient(circle at 15% 80%, rgba(129, 140, 248, 0.18), transparent 45%),
                        linear-gradient(135deg, #121622 0%, #0b0e15 100%);
            border: 1px solid rgba(255, 255, 255, 0.10);
            border-radius: 16px;
            padding: 2.0rem 2.2rem;
            margin-bottom: 1.35rem;
            box-shadow: 0 16px 40px rgba(0, 0, 0, 0.40);
        }
        .web-hero-title {
            font-size: 2.15rem;
            font-weight: 800;
            line-height: 1.18;
            letter-spacing: -0.025em;
            background: linear-gradient(95deg, #f8fafc 0%, #c7d2fe 55%, #5eead4 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0.35rem 0 0.65rem 0;
        }

        /* Executive KPI Cards */
        .kpi-card {
            background: linear-gradient(180deg, rgba(22, 26, 38, 0.92) 0%, rgba(15, 18, 26, 0.96) 100%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-top: 2.5px solid #818cf8;
            border-radius: 10px;
            padding: 0.95rem 1.08rem;
            height: 100%;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.24);
            transition: border-color 0.15s ease, transform 0.15s ease;
        }
        .kpi-card:hover {
            transform: translateY(-2px);
            border-color: rgba(255, 255, 255, 0.18);
        }
        .kpi-label {
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: #94a3b8;
            margin-bottom: 0.32rem;
        }
        .kpi-value {
            font-size: 1.56rem;
            font-weight: 750;
            color: #f8fafc;
            line-height: 1.15;
            font-variant-numeric: tabular-nums;
        }
        .kpi-sub {
            font-size: 0.78rem;
            color: #94a3b8;
            margin-top: 0.34rem;
            line-height: 1.35;
        }

        /* Key Insight Callouts */
        .insight-pill {
            background: rgba(17, 20, 29, 0.85);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-left: 3px solid #818cf8;
            border-radius: 8px;
            padding: 0.75rem 0.95rem;
            font-size: 0.84rem;
            color: #cbd5e1;
            line-height: 1.42;
            height: 100%;
        }

        /* Pipeline & Module Cards */
        .pipe-step {
            background: rgba(17, 20, 29, 0.92);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-top: 3px solid #818cf8;
            border-radius: 10px;
            padding: 0.95rem 1.0rem;
            height: 100%;
        }
        .pipe-stage {
            font-size: 0.69rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #a5b4fc;
            margin-bottom: 0.24rem;
        }
        .pipe-name {
            font-size: 0.96rem;
            font-weight: 700;
            color: #f8fafc;
            margin-bottom: 0.28rem;
        }
        .pipe-desc {
            font-size: 0.80rem;
            color: #94a3b8;
            line-height: 1.42;
        }

        /* Product Advisor Verdict Badges */
        .verdict-badge {
            display: inline-block;
            padding: 0.34rem 0.90rem;
            border-radius: 6px;
            font-weight: 700;
            font-size: 0.83rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: #ffffff;
        }
        .badge-buy { background-color: #059669; border: 1px solid #10b981; }
        .badge-dont-buy { background-color: #e11d48; border: 1px solid #f43f5e; }
        .badge-insufficient { background-color: #d97706; border: 1px solid #f59e0b; color: #ffffff; }

        /* Status Dot for Stream Console */
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            font-size: 0.74rem;
            font-weight: 700;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            padding: 0.26rem 0.75rem;
            border-radius: 999px;
            border: 1px solid rgba(255, 255, 255, 0.12);
            background: rgba(17, 20, 29, 0.92);
            color: #e2e8f0;
            margin-bottom: 0.6rem;
        }
        .dot-live {
            width: 8px; height: 8px; border-radius: 50%;
            background-color: #2dd4bf;
            box-shadow: 0 0 8px #2dd4bf;
            display: inline-block;
        }
        .dot-idle {
            width: 8px; height: 8px; border-radius: 50%;
            background-color: #f59e0b;
            display: inline-block;
        }

        /* Review & Alert Feed Cards */
        .review-card {
            border-left: 3.5px solid #818cf8;
            background: rgba(17, 20, 29, 0.85);
            border-top: 1px solid rgba(255, 255, 255, 0.07);
            border-right: 1px solid rgba(255, 255, 255, 0.07);
            border-bottom: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 0 8px 8px 0;
            padding: 0.85rem 1.05rem;
            margin-bottom: 0.65rem;
        }
        .meta-tag {
            display: inline-block;
            font-size: 0.72rem;
            font-weight: 600;
            padding: 0.12rem 0.48rem;
            border-radius: 4px;
            background: rgba(129, 140, 248, 0.12);
            border: 1px solid rgba(129, 140, 248, 0.22);
            color: #c7d2fe;
            margin-right: 0.35rem;
        }

        /* Website Footer */
        .web-footer {
            margin-top: 2.5rem;
            padding: 1.1rem 1.4rem;
            border-top: 1px solid rgba(255, 255, 255, 0.08);
            background: rgba(13, 16, 23, 0.85);
            border-radius: 10px;
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
            gap: 0.75rem;
            font-size: 0.80rem;
            color: #94a3b8;
        }

        /* Clean Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.4rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.09);
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 6px 6px 0 0;
            padding: 0.55rem 1.05rem;
            font-weight: 600;
            font-size: 0.88rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(badge: str, title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="hero-wrap">
            <div class="hero-badge">{badge}</div>
            <div class="hero-title">{title}</div>
            <p class="hero-sub">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(col, label: str, value: str, sub: str = "", accent: str = ACCENT_INDIGO):
    col.markdown(
        f"""
        <div class="kpi-card" style="border-top-color: {accent};">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_fig(fig, height: int = 370, title: str | None = None, has_top_legend: bool = False):
    top_margin = 84 if (title and has_top_legend) else (52 if title else 24)
    layout_kwargs = dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,20,29,0.60)",
        font=dict(family="Inter, Segoe UI, sans-serif", size=12, color="#cbd5e1"),
        height=height,
        margin=dict(t=top_margin, b=32, l=28, r=24),
        hoverlabel=dict(bgcolor="#11141d", bordercolor="#334155", font=dict(color="#f8fafc")),
        xaxis=dict(automargin=True, gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.12)"),
        yaxis=dict(automargin=True, gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.12)"),
    )
    if title:
        layout_kwargs["title"] = dict(
            text=title,
            font=dict(size=14, color="#f1f5f9"),
            x=0.01,
            xanchor="left",
            y=0.96,
            yanchor="top",
        )
    if has_top_legend:
        layout_kwargs["legend"] = dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            title_text="",
            font=dict(size=11.5),
        )
    fig.update_layout(**layout_kwargs)
    return fig


def databricks_configured() -> bool:
    return all(
        os.environ.get(k)
        for k in ("DATABRICKS_HOST", "DATABRICKS_HTTP_PATH", "DATABRICKS_TOKEN")
    )


def render_website_footer():
    src_label = (
        "Databricks Unity Catalog (workspace.review_analytics)"
        if databricks_configured()
        else "Local Delta-Aligned Gold Parquet (Databricks-Compatible)"
    )
    st.markdown(
        f"""
        <div class="web-footer">
            <div>
                <strong style="color:#f8fafc;">ReviewSense Analytics</strong> · 23RAI351 Big Data Analytics ·
                Team No. 1: Sreenidhi and Lakshya Yarlagadda
            </div>
            <div>
                Faculty Guide: <strong style="color:#cbd5e1;">Dr. Ravi Kumar V.</strong> ·
                Source: <strong style="color:#a5b4fc;">{src_label}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def download_csv_button(df: pd.DataFrame, filename: str, label: str = "Export CSV"):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
    )


@st.cache_data(show_spinner=False)
def load_gold_table(name: str) -> pd.DataFrame:
    path = os.path.join(GOLD_DIR, f"{name}.parquet")
    read_cols = None
    if name == "product_recommendations":
        # Defer heavy 368 MB product_features text until a single product is inspected
        read_cols = [
            "parent_asin", "category", "product_title", "brand", "price",
            "reviews", "avg_rating", "positive_pct", "neutral_pct", "negative_pct",
            "model_positive_pct", "verified_pct", "years_active",
            "recent_year_pos_pct", "sentiment_trend",
            "rating_score", "trend_score", "consistency_score", "complaint_inv_score",
            "buy_score", "recommendation", "confidence",
            "top_aspect", "top_aspect_pos_pct", "top_complaint", "top_complaint_pct",
            "reason_summary",
        ]

    # Optional live Databricks Unity Catalog SQL Warehouse mode
    databricks_tables = {
        "agg_year_summary", "agg_year_category", "agg_monthly", "agg_brand",
        "agg_top_words", "agg_complaints", "model_metrics",
        "product_recommendations", "product_aspect_sentiment", "product_complaints",
    }
    if databricks_configured() and name in databricks_tables:
        try:
            from databricks import sql as dbsql

            col_sql = ", ".join(read_cols) if read_cols else "*"
            with dbsql.connect(
                server_hostname=os.environ["DATABRICKS_HOST"],
                http_path=os.environ["DATABRICKS_HTTP_PATH"],
                access_token=os.environ["DATABRICKS_TOKEN"],
            ) as conn:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT {col_sql} FROM workspace.review_analytics.{name}")
                    df = cur.fetchall_arrow().to_pandas()
            if name == "product_recommendations":
                for cat_col in ["category", "recommendation", "confidence", "top_aspect", "top_complaint"]:
                    if cat_col in df.columns:
                        df[cat_col] = df[cat_col].astype("category")
            return df
        except Exception:
            pass

    if os.path.exists(path):
        df = pd.read_parquet(path, columns=read_cols)
    else:
        shard_dir = os.path.join(GOLD_DIR, f"{name}_shards")
        if os.path.isdir(shard_dir):
            parts = sorted(
                os.path.join(shard_dir, f)
                for f in os.listdir(shard_dir)
                if f.endswith(".parquet")
            )
            if parts:
                df = pd.concat([pd.read_parquet(p, columns=read_cols) for p in parts], ignore_index=True)
            else:
                raise FileNotFoundError(f"Missing Gold table: {path}")
        else:
            raise FileNotFoundError(f"Missing Gold table: {path}. Run scripts/build_local_gold.py first.")

    if name == "product_recommendations":
        for cat_col in ["category", "recommendation", "confidence", "top_aspect", "top_complaint"]:
            if cat_col in df.columns:
                df[cat_col] = df[cat_col].astype("category")
    return df


@st.cache_data(show_spinner=False)
def load_product_features(parent_asin: str, category: str) -> str | None:
    """Lazily read product_features for a single selected product without holding 368 MB in RAM."""
    import polars as pl

    single_path = os.path.join(GOLD_DIR, "product_recommendations.parquet")
    shard_glob = os.path.join(GOLD_DIR, "product_recommendations_shards", "part_*.parquet")
    src = single_path if os.path.exists(single_path) else shard_glob
    res = (
        pl.scan_parquet(src)
        .filter((pl.col("parent_asin") == parent_asin) & (pl.col("category") == category))
        .select("product_features")
        .collect()
    )
    if res.height == 0:
        return None
    val = res["product_features"][0]
    return str(val) if val is not None else None


@st.cache_resource(show_spinner=False)
def load_nlp_models():
    vec = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
    lr = joblib.load(os.path.join(MODEL_DIR, "sentiment_lr.joblib"))
    nb = joblib.load(os.path.join(MODEL_DIR, "sentiment_nb.joblib"))
    return vec, lr, nb


def clean_review_text(text: str) -> str:
    s = (text or "").lower()
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"(https?://\S+|www\.\S+)", " ", s)
    s = re.sub(r"n't\b", " not", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def predict_single_review(raw_text: str, model_choice: str = "Logistic Regression"):
    vec, lr, nb = load_nlp_models()
    cleaned = clean_review_text(raw_text)
    if not cleaned:
        return None
    X = vec.transform([cleaned])
    clf = lr if model_choice == "Logistic Regression" else nb
    probs = clf.predict_proba(X)[0]
    pred_idx = int(np.argmax(probs))
    labels = ["Negative", "Neutral", "Positive"]

    feature_names = vec.get_feature_names_out()
    nz_indices = X.nonzero()[1]
    contributions = []
    if hasattr(lr, "coef_"):
        pos_coef = lr.coef_[2]
        neg_coef = lr.coef_[0]
        for idx in nz_indices:
            tfidf_val = float(X[0, idx])
            net_pos_impact = tfidf_val * float(pos_coef[idx] - neg_coef[idx])
            contributions.append({
                "term": str(feature_names[idx]),
                "impact": round(net_pos_impact, 4),
                "direction": "Positive signal" if net_pos_impact >= 0 else "Negative signal",
            })
        contributions.sort(key=lambda d: abs(d["impact"]), reverse=True)

    return {
        "cleaned_text": cleaned,
        "prediction": labels[pred_idx],
        "probabilities": {
            "Negative": float(probs[0]),
            "Neutral": float(probs[1]),
            "Positive": float(probs[2]),
        },
        "top_contributions": contributions[:12],
    }
