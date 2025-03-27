import streamlit as st
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go
from groq import Groq

# Load API Key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# Streamlit UI Config
st.set_page_config(page_title="AI Forecasting Agent", page_icon="🔮", layout="wide")
st.title("🔮 AI Agent - Revenue Forecasting with Prophet")

# Upload Excel File
uploaded_file = st.file_uploader("📂 Upload your Excel file with 'Date' and 'Revenue' columns", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.subheader("🧾 Preview of Uploaded Data")
    st.dataframe(df.head())

    # Preprocessing
    df.columns = df.columns.str.strip().str.lower()
    if 'date' not in df.columns or 'revenue' not in df.columns:
        st.error("❌ The file must contain 'Date' and 'Revenue' columns.")
        st.stop()

    df['date'] = pd.to_datetime(df['date'])
    df = df.rename(columns={'date': 'ds', 'revenue': 'y'})
    df = df[['ds', 'y']].dropna()

    # Prophet Forecast
    periods = st.slider("📅 Forecast Horizon (in months)", 1, 12, 6)
    freq = st.selectbox("📈 Frequency of Data", options=["D", "W", "M"], index=2)

    m = Prophet()
    m.fit(df)

    future = m.make_future_dataframe(periods=periods, freq=freq)
    forecast = m.predict(future)

    # Plot Forecast
    st.subheader("📊 Forecast Plot")
    fig = plot_plotly(m, forecast)
    st.plotly_chart(fig, use_container_width=True)

    # AI Commentary using Groq
    st.subheader("🤖 AI Analysis of Forecast")

    data_for_ai = df.tail(12).to_json(orient="records", date_format="iso")

    prompt = f"""
    You are an expert financial analyst.
    Based on the following historical revenue data, explain:
    - Key revenue trends and patterns.
    - Potential business implications of the forecast.
    - A CFO-level executive summary and suggestions for the next 6 months.
    Data: {data_for_ai}
    """

    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a senior FP&A expert specialized in time series forecasting and business insights."},
            {"role": "user", "content": prompt}
        ],
        model="llama3-8b-8192",
    )

    commentary = response.choices[0].message.content
    st.markdown("### 📖 AI-Generated Commentary")
    st.write(commentary)
