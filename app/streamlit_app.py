"""A simple dashboard for churn data.

Has overview, customer explorer and model insights pages.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import PROCESSED_DIR, MODELS_DIR
from src.models.predict import score_all_customers
from src.models.train import prepare_data

# page configuration
st.set_page_config(
    page_title="Churn Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 8px;
        padding: 20px;
        border-left: 4px solid #4C72B0;
    }
    .high-risk { border-left-color: #DD8452 !important; }
    .stMetric label { font-size: 0.85rem; color: #888; }
    .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# Applied to every chart for consistent, smooth, borderless look
PLOT_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='sans-serif', size=11),
    margin=dict(l=4, r=4, t=16, b=4),
    xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=10)),
    yaxis=dict(gridcolor='rgba(255,255,255,0.06)', gridwidth=1,
               zeroline=False, tickfont=dict(size=10)),
    legend=dict(orientation='h', y=-0.2, x=0, font=dict(size=10),
                bgcolor='rgba(0,0,0,0)'),
    hoverlabel=dict(font=dict(size=11)),
)

@st.cache_data
def load_data():
    df = score_all_customers(model_name="xgboost")
    return df

@st.cache_data
def load_metrics():
    path = os.path.join(MODELS_DIR, "metrics_summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return pd.DataFrame(json.load(f))

@st.cache_data
def load_shap_importance():
    path = os.path.join(MODELS_DIR, "shap_importance.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


with st.sidebar:
    st.title("Churn Intelligence")
    st.caption("Telco Customer Analytics")
    st.divider()

    view = st.radio(
        "Navigate",
        ["Overview", "Customer Explorer", "Model Insights"],
        label_visibility="collapsed"
    )

    st.divider()
    st.caption("Dataset: Telco Churn\nModel: XGBoost")


try:
    df = load_data()
    data_loaded = True
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.info("Run the ETL pipeline and model training first:\n\n"
            "```\npython src/etl/ingest.py\n"
            "python src/etl/clean.py\n"
            "python src/etl/features.py\n"
            "python src/models/train.py\n```")
    data_loaded = False
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# VIEW 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if view == "Overview":
    st.title("Churn Overview")
    st.caption("High-level KPIs and risk breakdown")
    st.divider()

    # KPI row
    total = len(df)
    actual_churn = df['Churn'].sum()
    predicted_churn = df['churn_prediction'].sum()
    high_risk = (df['risk_label'] == 'High').sum()
    avg_prob = df['churn_probability'].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Customers", f"{total:,}")
    c2.metric("Actual Churn", f"{actual_churn:,}", f"{actual_churn/total:.1%}")
    c3.metric("Predicted to Churn", f"{predicted_churn:,}", f"{predicted_churn/total:.1%}")
    c4.metric("High Risk Customers", f"{high_risk:,}", f"{high_risk/total:.1%}")
    c5.metric("Avg Churn Probability", f"{avg_prob:.2%}")

    st.divider()

    # charts row 1
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Segment Distribution")
        seg_counts = df['risk_segment'].value_counts().reset_index()
        seg_counts.columns = ['Segment', 'Count']
        color_map = {'High Risk': '#DD8452', 'Medium Risk': '#8DA0CB', 'Low Risk': '#4C72B0'}
        fig = px.bar(seg_counts, x='Segment', y='Count',
                     color='Segment', color_discrete_map=color_map,
                     text='Count')
        fig.update_traces(textposition='outside', textfont_size=11,
                          marker_line_width=0)
        fig.update_layout(**PLOT_LAYOUT, height=320, showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with col2:
        st.subheader("Churn Probability Distribution")
        fig = px.histogram(df, x='churn_probability', nbins=40,
                           color_discrete_sequence=['#4C72B0'])
        fig.add_vline(x=0.5, line_dash='dash', line_color='#DD8452', line_width=1.5,
                      annotation_text='Decision threshold (0.5)',
                      annotation_font_size=10)
        fig.update_traces(marker_line_width=0, opacity=0.85)
        fig.update_layout(**PLOT_LAYOUT, height=320,
                          xaxis_title='Predicted Churn Probability',
                          yaxis_title='Number of Customers')
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # charts row 2
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Churn Rate by Contract Type")
        churn_by_contract = df.groupby('Contract')['Churn'].mean().reset_index()
        churn_by_contract.columns = ['Contract', 'Churn Rate']
        churn_by_contract = churn_by_contract.sort_values('Churn Rate', ascending=False)
        fig = px.bar(churn_by_contract, x='Contract', y='Churn Rate',
                     color='Churn Rate', color_continuous_scale='RdYlGn_r',
                     text=churn_by_contract['Churn Rate'].apply(lambda x: f'{x:.1%}'))
        fig.update_traces(textposition='outside', textfont_size=11,
                          marker_line_width=0)
        fig.update_layout(**PLOT_LAYOUT, height=320, coloraxis_showscale=False,
                          yaxis_tickformat='.0%')
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with col4:
        st.subheader("Avg Churn Probability by Tenure Band")
        tb = df.groupby('tenure_band', observed=True)['churn_probability'].mean().reset_index()
        tb.columns = ['Tenure Band', 'Avg Churn Probability']
        fig = px.bar(tb, x='Tenure Band', y='Avg Churn Probability',
                     color_discrete_sequence=['#DD8452'],
                     text=tb['Avg Churn Probability'].apply(lambda x: f'{x:.2%}'))
        fig.update_traces(textposition='outside', textfont_size=11,
                          marker_line_width=0)
        fig.update_layout(**PLOT_LAYOUT, height=320,
                          yaxis_tickformat='.0%')
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


# ══════════════════════════════════════════════════════════════════════════════
# VIEW 2: CUSTOMER EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif view == "Customer Explorer":
    st.title("Customer Explorer")
    st.caption("Filter and explore customer groups")
    st.divider()

    # filters
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        risk_filter = st.multiselect(
            "Risk Label", options=['High', 'Medium', 'Low'],
            default=['High', 'Medium', 'Low']
        )
    with col2:
        contract_filter = st.multiselect(
            "Contract Type", options=df['Contract'].unique().tolist(),
            default=df['Contract'].unique().tolist()
        )
    with col3:
        tenure_filter = st.multiselect(
            "Tenure Band", options=['New', 'Developing', 'Established'],
            default=['New', 'Developing', 'Established']
        )
    with col4:
        min_prob = st.slider("Min Churn Probability", 0.0, 1.0, 0.0, 0.05)

    # Apply filters
    filtered = df[
        (df['risk_label'].isin(risk_filter)) &
        (df['Contract'].isin(contract_filter)) &
        (df['tenure_band'].isin(tenure_filter)) &
        (df['churn_probability'] >= min_prob)
    ]

    st.caption(f"Showing **{len(filtered):,}** of {len(df):,} customers")
    st.divider()

    # summary metrics for filtered set
    c1, c2, c3 = st.columns(3)
    c1.metric("Customers in Filter", f"{len(filtered):,}")
    c2.metric("Avg Churn Probability", f"{filtered['churn_probability'].mean():.2%}" if len(filtered) > 0 else "—")
    c3.metric("Predicted Churners", f"{filtered['churn_prediction'].sum():,}" if len(filtered) > 0 else "—")

    # scatter plot
    if len(filtered) > 0:
        st.subheader("Tenure vs Monthly Charges — Churn Risk")
        fig = px.scatter(
            filtered, x='tenure', y='MonthlyCharges',
            color='churn_probability',
            color_continuous_scale='RdYlGn_r',
            hover_data=['customerID', 'Contract', 'PaymentMethod', 'risk_label'],
            opacity=0.65,
        )
        fig.update_traces(marker=dict(size=6, line=dict(width=0)))
        fig.update_layout(**PLOT_LAYOUT, height=380,
                          coloraxis_colorbar=dict(title='Churn Prob',
                                                  thickness=12, len=0.75,
                                                  tickfont=dict(size=9)))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # customer table
        st.subheader("Customer Records")
        display_cols = [
            'customerID', 'tenure', 'Contract', 'PaymentMethod',
            'MonthlyCharges', 'engagement_score', 'risk_segment',
            'churn_probability', 'churn_prediction', 'risk_label'
        ]
        st.dataframe(
            filtered[display_cols].sort_values('churn_probability', ascending=False),
            use_container_width=True,
            height=380
        )

        # download
        csv = filtered[display_cols].to_csv(index=False)
        st.download_button(
            "Download filtered customers as CSV",
            data=csv, file_name="filtered_customers.csv", mime="text/csv"
        )


# ══════════════════════════════════════════════════════════════════════════════
# VIEW 3: MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif view == "Model Insights":
    st.title("Model Insights")
    st.caption("Performance, importance and SHAP notes")
    st.divider()

    # ── Model comparison table ────────────────────────────────────────────────
    metrics = load_metrics()
    if metrics is not None:
        st.subheader("Model Comparison")
        st.dataframe(
            metrics.set_index('model').style.highlight_max(
                subset=['roc_auc', 'f1_churn', 'precision_churn', 'recall_churn'],
                color='#2d5f3f'
            ),
            use_container_width=True
        )
        st.divider()

    # ── Distribution charts ───────────────────────────────────────────────────
    st.subheader("Churn Probability Distribution by Actual Churn Status")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            df, x='churn_probability',
            color=df['Churn'].map({1: 'Churned', 0: 'Retained'}),
            nbins=40, barmode='overlay', opacity=0.72,
            color_discrete_map={'Churned': '#DD8452', 'Retained': '#4C72B0'},
            labels={'color': 'Actual Status'}
        )
        fig.update_traces(marker_line_width=0)
        fig.update_layout(**PLOT_LAYOUT, height=320,
                          xaxis_title='Predicted Churn Probability')
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with col2:
        fig = px.scatter(
            df.sample(min(1000, len(df)), random_state=42),
            x='engagement_score', y='churn_probability',
            color=df['Churn'].sample(min(1000, len(df)), random_state=42).map(
                {1: 'Churned', 0: 'Retained'}),
            color_discrete_map={'Churned': '#DD8452', 'Retained': '#4C72B0'},
            opacity=0.5,
            labels={'color': 'Actual Status',
                    'engagement_score': 'Engagement Score',
                    'churn_probability': 'Predicted Churn Probability'}
        )
        fig.update_traces(marker=dict(size=5, line=dict(width=0)))
        fig.update_layout(**PLOT_LAYOUT, height=320)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # ── SHAP key drivers ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("Key Model Drivers")
    st.markdown("""
    Based on SHAP analysis from the modeling notebook, the top drivers of churn prediction are:

    | Rank | Feature | Direction | Business Meaning |
    |------|---------|-----------|-----------------|
    | 1 | `contract_risk` | ↑ Higher risk = more churn | Month-to-month customers churn most |
    | 2 | `tenure` / `rfm_recency` | ↑ Newer = more churn | New customers haven't committed |
    | 3 | `payment_risk` | ↑ Electronic check = more churn | Manual payers are more likely to cancel |
    | 4 | `engagement_score` | ↓ Lower engagement = more churn | Few services = easy to leave |
    | 5 | `MonthlyCharges` | ↑ Higher charges = more churn | Price sensitivity is real |

    > **Marketing action:** Target high-risk customers (month-to-month, electronic check, low engagement)
    > with contract upgrade incentives and service bundle promotions.
    """)

    st.info("For full SHAP waterfall plots and individual customer explanations, "
            "see notebooks/03_modeling.ipynb")
