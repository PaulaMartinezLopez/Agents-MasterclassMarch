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
uploaded_file = st.file_uploader("📂 Upload your Excel file with 'Date' and 'Revenue' columns", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.subheader(":page_facing_up: Preview of Uploaded Data")
    st.dataframe(df.head())

    # Preprocessing
    df.columns = df.columns.str.strip().str.lower()
    if 'date' not in df.columns or 'revenue' not in df.columns:
        st.error("❌ The file must contain 'Date' and 'Revenue' columns.")
        st.stop()

    df['date'] = pd.to_datetime(df['date'])
    df = df.rename(columns={'date': 'ds', 'revenue': 'y'})
    df = df[['ds', 'y']].dropna()

    df['month'] = df['ds'].dt.month
    df['year'] = df['ds'].dt.year

    # Seasonal index
    seasonal_index = df.groupby('month')['y'].mean() / df['y'].mean()

    # Forecast horizon
    months = st.slider("📅 Forecast Horizon (in months)", 1, 12, 6)
    last_12m_avg = df.sort_values('ds').tail(12)['y'].mean()
    future_months = pd.date_range(df['ds'].max() + pd.offsets.MonthBegin(), periods=months, freq='MS')
    forecast_df = pd.DataFrame({'ds': future_months})
    forecast_df['month'] = forecast_df['ds'].dt.month
    forecast_df['y'] = forecast_df['month'].map(seasonal_index) * last_12m_avg

    # Scenario adjustments
    st.sidebar.header("🔧 Scenario Planner")
    q1_adj = st.sidebar.slider("Q1 adjustment (%)", -50, 50, 0)
    q2_adj = st.sidebar.slider("Q2 adjustment (%)", -50, 50, 0)
    q3_adj = st.sidebar.slider("Q3 adjustment (%)", -50, 50, 0)
    q4_adj = st.sidebar.slider("Q4 adjustment (%)", -50, 50, 0)

    def apply_adjustment(row):
        if row['ds'].month in [1, 2, 3]:
            return row['y'] * (1 + q1_adj / 100)
        elif row['ds'].month in [4, 5, 6]:
            return row['y'] * (1 + q2_adj / 100)
        elif row['ds'].month in [7, 8, 9]:
            return row['y'] * (1 + q3_adj / 100)
        else:
            return row['y'] * (1 + q4_adj / 100)

    forecast_df['adjusted_y'] = forecast_df.apply(apply_adjustment, axis=1)

    # Plot
    st.subheader(":bar_chart: Forecast with Scenario")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='markers+lines', name='Actuals'))
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['y'], mode='lines', name='Baseline Forecast'))
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['adjusted_y'], mode='lines+markers', name='Adjusted Forecast', line=dict(color='green')))
    fig.update_layout(title="Monthly Revenue Forecast (Scenario Model)", xaxis_title="Date", yaxis_title="Revenue", hovermode="x")
    st.plotly_chart(fig, use_container_width=True)

    # Summary Table
    st.subheader(":clipboard: Forecast Summary Table")
    real_2023 = df[df['year'] == 2023]['y'].sum() if 2023 in df['year'].values else 0
    real_2024 = df[df['year'] == 2024]['y'].sum() if 2024 in df['year'].values else 0
    fcst_2025_base = forecast_df['y'].sum()
    fcst_2025_adjusted = forecast_df['adjusted_y'].sum()

    summary_df = pd.DataFrame({
        'Metric': ['Real 2023', 'Real 2024', 'Forecast 2025 (base)', 'Forecast 2025 (adjusted)'],
        'Value (€)': [real_2023, real_2024, fcst_2025_base, fcst_2025_adjusted]
    })

    def format_millions(euros):
        return f"€{euros/1_000_000:,.2f}M"

    summary_df['Value (€)'] = summary_df['Value (€)'].apply(format_millions)
    st.dataframe(summary_df.style.applymap(
        lambda v: 'color: green; font-weight: bold' if 'adjusted' in v else '',
        subset=['Metric']
    ), use_container_width=True)

    # AI Commentary using Groq
    st.subheader(":robot_face: Executive Commentary")
    delta = fcst_2025_adjusted - fcst_2025_base

    prompt = f"""
    You are an FP&A analyst. Based on the user's adjustments:
    - Q1 adj: {q1_adj}%
    - Q2 adj: {q2_adj}%
    - Q3 adj: {q3_adj}%
    - Q4 adj: {q4_adj}%

    The baseline forecast for the next {months} months is €{fcst_2025_base:,.0f}.
    The adjusted forecast is €{fcst_2025_adjusted:,.0f}, a change of €{delta:,.0f}.

    Please summarize the key business implications and provide a CFO-ready narrative based on this short-term outlook.
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


