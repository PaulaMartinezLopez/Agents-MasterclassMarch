import streamlit as st
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from prophet import Prophet
import plotly.graph_objects as go
from groq import Groq

# Load API Key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("❌ API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# Streamlit UI Config
st.set_page_config(page_title="AI Forecasting Agent", page_icon=":crystal_ball:", layout="wide")
st.title(":crystal_ball: AI Agent - Revenue Forecasting with Prophet")

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

    # Filter data only for 2023 and 2024 (real data)
    df_train = df[df['ds'].dt.year <= 2024]

    # Calculate CAGR between 2023 and 2024
    start = df_train[df_train['ds'].dt.year == 2023]['y'].sum()
    end = df_train[df_train['ds'].dt.year == 2024]['y'].sum()
    cagr = ((end / start) ** (1 / 1)) - 1 if start > 0 else 0

    # Prophet Forecast
    periods = st.slider("📅 Forecast Horizon (in months)", 1, 12, 6)
    
    m = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    seasonality_mode='multiplicative'  # o 'additive' si prefieres
)
m.add_seasonality(name='monthly', period=30.5, fourier_order=5)

    m.fit(df_train)

    future = m.make_future_dataframe(periods=periods, freq='MS')  # Monthly start
    forecast = m.predict(future)

    # Plot Forecast and actuals
    st.subheader(":bar_chart: Forecast Plot")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_train['ds'], y=df_train['y'], mode='markers+lines', name='Actuals'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
    fig.update_layout(title="Monthly Revenue Forecast", xaxis_title="Date", yaxis_title="Revenue", hovermode="x")
    st.plotly_chart(fig, use_container_width=True)

    # Revenue by Year
    st.subheader(":calendar: Revenue by Year")
    df_train['year'] = df_train['ds'].dt.year
    revenue_by_year = df_train.groupby('year')['y'].sum().reset_index()
    st.table(revenue_by_year)

    # AI Commentary using Groq
    st.subheader(":robot_face: AI Analysis of Forecast")

    data_for_ai = df_train.tail(12).to_json(orient="records", date_format="iso")

    prompt = f"""
    You are an expert FP&A analyst.
    Revenue in 2023: €{start:,.0f}
    Revenue in 2024: €{end:,.0f}
    CAGR from 2023 to 2024: {cagr:.2%}

    Only analyze the trends based on this real historical data. Do not assume growth or volatility unless explicitly shown by the data.
    Here's the monthly breakdown for the last 12 months: {data_for_ai}

    Please provide:
    - Key revenue trends (backed by data)
    - Business implications
    - A realistic executive summary for the CFO (avoid assumptions)
    """

    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a senior FP&A expert specialized in forecasting and data-driven decision-making."},
            {"role": "user", "content": prompt}
        ],
        model="llama3-8b-8192",
    )

    commentary = response.choices[0].message.content
    st.markdown("### :blue_book: AI-Generated Commentary")
    st.write(commentary)

