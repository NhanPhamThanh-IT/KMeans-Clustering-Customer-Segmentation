"""
ui.py

Defines the UI for the Streamlit app.
"""

import streamlit as st
from model import CustomerSegmentModel


class CustomerSegmentationApp:
    """Streamlit UI for Customer Segmentation."""

    def __init__(self, model: CustomerSegmentModel):
        """Initializes the app with the provided model."""
        self.model = model

    def run(self):
        """Renders the app UI."""
        st.set_page_config(
            page_title="K-Means Clustering Customer Segmentation",
            layout="centered"
        )

        st.markdown(
            """
            <h3 style="text-align: center; color: green;">
            K-Means Clustering Customer Segmentation</h3>
            <p style="text-align: center;">
            This app predicts customer segments based on annual income and spending score.</p>
            """,
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:
            annual_income = st.number_input(
                "Annual Income (k$)", min_value=0, max_value=200, value=50
            )

        with col2:
            spending_score = st.number_input(
                "Spending Score (1-100)", min_value=1, max_value=100, value=50
            )

        if st.button("Predict", use_container_width=True):
            features = [[annual_income, spending_score]]
            cluster = self.model.predict(features)

            st.markdown(
                f"""
                <h4 style="text-align: center; color: blue;">
                Predicted Customer Segment: {cluster}</h4>
                """,
                unsafe_allow_html=True
            )
