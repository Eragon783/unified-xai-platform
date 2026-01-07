"""
Unified Explainable AI Platform
Main Streamlit Application Entry Point
"""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Unified XAI Platform",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import views
from views.home import render_home_page
from views.comparison import render_comparison_page


def main():
    """Main application controller."""

    # Sidebar navigation
    st.sidebar.title("🔍 Unified XAI Platform")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Home", "📊 Comparison"]
    )

    # Page routing
    if page == "🏠 Home":
        render_home_page()
    elif page == "📊 Comparison":
        render_comparison_page()


if __name__ == "__main__":
    main()
