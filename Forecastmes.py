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
st.set_page_config(page_title="Forecasting Agent - Moving Average", page_icon=":bar_chart:", layout="wide")
st.title(":bar_chart: CFO-Friendly Revenue Forecast (Simple Model)")

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

    # Create seasonal index: average revenue per month over all years
    seasonal_index = df.groupby('month')['y'].mean() / df['y'].mean()

    # Forecast next 6 months using average of last 12 months adjusted by seasonality
    last_12m_avg = df.sort_values('ds').tail(12)['y'].mean()
    future_months = pd.date_range(df['ds'].max() + pd.offsets.MonthBegin(), periods=6, freq='MS')
    forecast_df = pd.DataFrame({'ds': future_months})
    forecast_df['month'] = forecast_df['ds'].dt.month
    forecast_df['y'] = forecast_df['month'].map(seasonal_index) * last_12m_avg

    # Plot
    st.subheader(":bar_chart: Forecast Plot")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='markers+lines', name='Actuals'))
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['y'], mode='lines+markers', name='Forecast'))
    fig.update_layout(title="Monthly Revenue Forecast (Simple Model)", xaxis_title="Date", yaxis_title="Revenue", hovermode="x")
    st.plotly_chart(fig, use_container_width=True)

    # AI Commentary using Groq
    st.subheader(":robot_face: Executive Commentary")
    historical_total = df[df['year'] == 2024]['y'].sum() if 2024 in df['year'].values else 0
    future_total = forecast_df['y'].sum()

    prompt = f"""
    You are an FP&A analyst. Revenue from Jan-Dec 2024 was €{historical_total:,.0f}.
    Based on this, we forecast the next 6 months with a seasonal model adjusted from the last 12 months' average.
    The forecasted total for the next 6 months is €{future_total:,.0f}.

    Write a brief, data-based executive summary for a CFO, including:
    - Trends observed in recent data.
    - The seasonal impact expected.
    - Whether the next 6 months are expected to be above/below average.
    - Recommendations based on this short-term outlook.
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

