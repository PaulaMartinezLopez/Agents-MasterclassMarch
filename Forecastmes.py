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
    st.error("\ud83d\udea8 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# Streamlit UI Config
st.set_page_config(page_title="AI Forecasting Agent", page_icon="\ud83d\udd2e", layout="wide")
st.title("\ud83d\udd2e AI Agent - Revenue Forecasting with Prophet")

# Upload Excel File
uploaded_file = st.file_uploader("\ud83d\udcc2 Upload your Excel file with 'Date' and 'Revenue' columns", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.subheader("\ud83d\udcdc Preview of Uploaded Data")
    st.dataframe(df.head())

    # Preprocessing
    df.columns = df.columns.str.strip().str.lower()
    if 'date' not in df.columns or 'revenue' not in df.columns:
        st.error("\u274c The file must contain 'Date' and 'Revenue' columns.")
        st.stop()

    df['date'] = pd.to_datetime(df['date'])
    df = df.rename(columns={'date': 'ds', 'revenue': 'y'})
    df = df[['ds', 'y']].dropna()

    # Filter only real data (2023-2024)
    df_real = df[df['ds'].dt.year <= 2024]

    # Prophet Forecast
    periods = st.slider("\ud83d\uddd3\ufe0f Forecast Horizon (in months)", 1, 12, 6)
    freq = st.selectbox("\ud83d\udcc8 Frequency of Data", options=["D", "W", "M"], index=2)

    m = Prophet(growth='flat')
    m.fit(df_real)

    future = m.make_future_dataframe(periods=periods, freq=freq)
    forecast = m.predict(future)

    # Plot Forecast
    st.subheader("\ud83d\udcca Forecast Plot")
    fig = plot_plotly(m, forecast)
    fig.add_vline(
        x=pd.to_datetime("2024-12-31"),
        line=dict(color="red", dash="dot"),
        annotation_text="Forecast Start",
        annotation_position="top right"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Revenue by Year
    st.subheader("\ud83d\udcc5 Revenue by Year")
    df_real['year'] = df_real['ds'].dt.year
    revenue_by_year = df_real.groupby('year')['y'].sum().reset_index()
    st.table(revenue_by_year)

    # AI Commentary using Groq
    st.subheader("\ud83e\udd16 AI Analysis of Forecast")

    data_for_ai = df_real.tail(12).to_json(orient="records", date_format="iso")

    prompt = f"""
    You are an expert financial analyst.
    Based only on the historical revenue data (excluding forecasted data), analyze:
    - Key revenue trends and patterns between 2023 and 2024.
    - Business implications assuming no automatic growth.
    - Executive summary for CFO with suggestions for the next 6 months based on real trends.
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
    st.markdown("### \ud83d\udcd6 AI-Generated Commentary")
    st.write(commentary)
