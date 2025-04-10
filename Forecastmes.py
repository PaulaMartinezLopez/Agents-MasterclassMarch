import streamlit as st
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import plotly.graph_objects as go
from groq import Groq

# Load API Key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("❌ API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# Streamlit UI Config
st.set_page_config(page_title="Forecasting Agent - Scenario Planner", page_icon=":bar_chart:", layout="wide")
st.title(":bar_chart: CFO-Friendly Revenue Forecast + Scenario Planner")

# Upload Excel File
uploaded_file = st.file_uploader("📂 Upload your Excel file with 'Date', 'Revenue' and '% GPM' columns", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.subheader(":page_facing_up: Preview of Uploaded Data")
    st.dataframe(df.head())

    # Preprocessing
    df.columns = df.columns.str.strip().str.lower()
    if 'date' not in df.columns or 'revenue' not in df.columns or '% gpm' not in df.columns:
        st.error("❌ The file must contain 'Date', 'Revenue' and '% GPM' columns.")
        st.stop()

    df['date'] = pd.to_datetime(df['date'])
    df = df.rename(columns={'date': 'ds', 'revenue': 'y', '% gpm': 'gpm'})
    df = df[['ds', 'y', 'gpm']].dropna()

    df['month'] = df['ds'].dt.month
    df['year'] = df['ds'].dt.year

    # Seasonal index for revenue
    seasonal_index = df.groupby('month')['y'].mean() / df['y'].mean()
    last_12m_avg = df.sort_values('ds').tail(12)['y'].mean()

    # Forecast horizon
    months = st.slider("📅 Forecast Horizon (in months)", 1, 12, 6)
    future_months = pd.date_range(df['ds'].max() + pd.offsets.MonthBegin(), periods=months, freq='MS')
    forecast_df = pd.DataFrame({'ds': future_months})
    forecast_df['month'] = forecast_df['ds'].dt.month
    forecast_df['y'] = forecast_df['month'].map(seasonal_index) * last_12m_avg

    # Sliders for revenue and margin adjustments
    st.sidebar.header("🔧 Scenario Planner")
    col1, col2 = st.sidebar.columns(2)

    q1_rev = col1.slider("Q1 Rev (%)", -50, 50, 0)
    q2_rev = col1.slider("Q2 Rev (%)", -50, 50, 0)
    q3_rev = col1.slider("Q3 Rev (%)", -50, 50, 0)
    q4_rev = col1.slider("Q4 Rev (%)", -50, 50, 0)

    q1_gpm = col2.slider("Q1 GPM (p.p.)", -10, 10, 0)
    q2_gpm = col2.slider("Q2 GPM (p.p.)", -10, 10, 0)
    q3_gpm = col2.slider("Q3 GPM (p.p.)", -10, 10, 0)
    q4_gpm = col2.slider("Q4 GPM (p.p.)", -10, 10, 0)

    def apply_adjustments(row):
        month = row['ds'].month
        revenue = row['y']
        base_margin = df['gpm'].mean()

        if month in [1,2,3]:
            adj_y = revenue * (1 + q1_rev / 100)
            adj_gpm = base_margin + (q1_gpm / 100)
        elif month in [4,5,6]:
            adj_y = revenue * (1 + q2_rev / 100)
            adj_gpm = base_margin + (q2_gpm / 100)
        elif month in [7,8,9]:
            adj_y = revenue * (1 + q3_rev / 100)
            adj_gpm = base_margin + (q3_gpm / 100)
        else:
            adj_y = revenue * (1 + q4_rev / 100)
            adj_gpm = base_margin + (q4_gpm / 100)

        return pd.Series({'adjusted_y': adj_y, 'adjusted_margin': adj_y * adj_gpm})

    forecast_df[['adjusted_y', 'adjusted_margin']] = forecast_df.apply(apply_adjustments, axis=1)

    # Plot
    st.subheader(":bar_chart: Forecast with Scenario")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='markers+lines', name='Actuals', line=dict(color='gray')))
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['y'], mode='lines', name='Baseline Forecast', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['adjusted_y'], mode='lines+markers', name='Adjusted Forecast', line=dict(color='green')))
    fig.update_layout(title="Monthly Revenue Forecast (Scenario Model)", xaxis_title="Date", yaxis_title="Revenue", hovermode="x")
    st.plotly_chart(fig, use_container_width=True)

    # Summary Table
    st.subheader(":clipboard: Forecast Summary Table")
    real_2023_df = df[df['year'] == 2023]
    real_2024_df = df[df['year'] == 2024]

    real_2023 = real_2023_df['y'].sum() if not real_2023_df.empty else 0
    real_2024 = real_2024_df['y'].sum() if not real_2024_df.empty else 0
    gpm_2023 = (real_2023_df['y'] * real_2023_df['gpm']).sum() if not real_2023_df.empty else 0
    gpm_2024 = (real_2024_df['y'] * real_2024_df['gpm']).sum() if not real_2024_df.empty else 0
    gpm_pct_2023 = gpm_2023 / real_2023 if real_2023 else 0
    gpm_pct_2024 = gpm_2024 / real_2024 if real_2024 else 0

    fcst_2025_base = forecast_df['y'].sum()
    margin_2025_base = forecast_df['y'].mean() * df['gpm'].mean() * len(forecast_df)
    gpm_pct_2025_base = margin_2025_base / fcst_2025_base if fcst_2025_base else 0

    fcst_2025_adjusted = forecast_df['adjusted_y'].sum()
    margin_2025_adjusted = forecast_df['adjusted_margin'].sum()
    gpm_pct_2025_adjusted = margin_2025_adjusted / fcst_2025_adjusted if fcst_2025_adjusted else 0

    summary_df = pd.DataFrame({
        'Metric': ['2023', '2024', 'Forecast 2025 (base)', 'Forecast 2025 (adjusted)'],
        'Revenue (€)': [real_2023, real_2024, fcst_2025_base, fcst_2025_adjusted],
        'Gross Margin (€)': [gpm_2023, gpm_2024, margin_2025_base, margin_2025_adjusted],
        'Gross Margin (%)': [gpm_pct_2023, gpm_pct_2024, gpm_pct_2025_base, gpm_pct_2025_adjusted]
    })

    summary_df['Revenue (€)'] = summary_df['Revenue (€)'].apply(lambda x: f"€{x/1_000_000:,.2f}M")
    summary_df['Gross Margin (€)'] = summary_df['Gross Margin (€)'].apply(lambda x: f"€{x/1_000_000:,.2f}M")
    summary_df['Gross Margin (%)'] = summary_df['Gross Margin (%)'].apply(lambda x: f"{x*100:.1f}%")

    st.dataframe(summary_df.style.applymap(
        lambda v: 'color: green; font-weight: bold' if 'adjusted' in str(v) else '',
        subset=['Metric']
    ), use_container_width=True)

    # AI Commentary using Groq
    st.subheader(":robot_face: Executive Commentary")
    delta = fcst_2025_adjusted - fcst_2025_base

    prompt = f"""
    You are an FP&A analyst. Based on the user's adjustments:
    - Revenue adjustments: Q1={q1_rev}%, Q2={q2_rev}%, Q3={q3_rev}%, Q4={q4_rev}%
    - Margin adjustments: Q1={q1_gpm}p.p., Q2={q2_gpm}p.p., Q3={q3_gpm}p.p., Q4={q4_gpm}p.p.

    Forecast base: €{fcst_2025_base:,.0f}, adjusted: €{fcst_2025_adjusted:,.0f}.
    Gross margin base: €{margin_2025_base:,.0f}, adjusted: €{margin_2025_adjusted:,.0f}

    Summarize the impact and provide a CFO-level executive summary.
    """

    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a senior FP&A expert specialized in business insight."},
            {"role": "user", "content": prompt}
        ],
        model="llama3-8b-8192",
    )

    commentary = response.choices[0].message.content
    st.markdown("### :blue_book: AI-Generated Commentary")
    st.write(commentary)






