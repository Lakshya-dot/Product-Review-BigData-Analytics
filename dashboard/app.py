r"""
Main Website Router (st.navigation with top horizontal navbar + website footer).
Launch from E:\Projects\Product_Review_BigData:
    streamlit run dashboard/app.py
"""

import streamlit as st
from utils import inject_custom_css, render_website_footer

st.set_page_config(
    page_title="ReviewSense · Product Review Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_custom_css()

pages = [
    st.Page("views/home.py", title="Home", default=True),
    st.Page("views/historical.py", title="Market Analytics"),
    st.Page("views/search.py", title="Buying Advisor"),
    st.Page("views/ml_models.py", title="ML & Review Checker"),
    st.Page("views/streaming.py", title="Live Stream"),
    st.Page("views/scalability.py", title="Scalability & Scope"),
]

nav = st.navigation(pages, position="top")
nav.run()
render_website_footer()
