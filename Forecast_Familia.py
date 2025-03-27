import streamlit as st
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from prophet import Prophet
import plotly.graph_objects as go
from groq import Groq

# 🎯 Configuración inicial
st.set_page_config(page_title="Forecast per Categoria", page_icon="🦑", layout="wide")
st.title("🧠 Forecast 2025 per Categoria")

# 🔐 Cargar clave API Groq
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# 📂 Subida de archivo
uploaded_file = st.file_uploader("📁 Sube un archivo Excel con columnas: Years, Months, Categoria, Eur", type=["xlsx"])
if uploaded_file:
    df_raw = pd.read_excel(uploaded_file)

    # 🧹 Preprocesamiento
    df_raw['date_str'] = df_raw['Years'].astype(str) + "-" + df_raw['Months'].astype(str)
    df_raw['ds'] = pd.to_datetime(df_raw['date_str'], format='%Y-%b')  # Ej: 2023-Jan
    df_raw = df_raw[['ds', 'Categoria', 'Eur']].rename(columns={'Eur': 'y'})

    # 🎛️ Selector de categoría
    categoria_sel = st.selectbox("🗂️ Elige la categoría", sorted(df_raw['Categoria'].unique()))
    df_categoria = df_raw[df_raw['Categoria'] == categoria_sel].copy()

    # 🔮 Forecast con Prophet
    m = Prophet()
    m.fit(df_categoria[['ds', 'y']])

    future = m.make_future_dataframe(periods=12, freq='M')  # 12 meses = todo 2025
    forecast = m.predict(future)

    # 📊 Gráfico interactivo y visual
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_categoria['ds'], y=df_categoria['y'],
                             mode='lines+markers', name='Actual',
                             line=dict(color='#0C769E')))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'],
                             mode='lines', name='Forecast', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'],
                             mode='lines', name='Upper', line=dict(width=0)))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'],
                             fill='tonexty', fillcolor='rgba(173,216,230,0.3)',
                             mode='lines', line=dict(width=0),
                             name='Confidence Interval'))
    fig.update_layout(title=f"📈 Forecast 2025 - {categoria_sel}",
                      xaxis_title="Fecha", yaxis_title="EUR",
                      height=500)

    st.plotly_chart(fig, use_container_width=True)

    # 🤖 Comentario automático con IA
    st.subheader("📖 Análisis AI para esta categoría")
    data_json = df_categoria.to_json(orient="records", date_format="iso")

    prompt = f"""
    You are a sales analyst working for a company specialized in frozen seafood (fish and shellfish). 
    Based on the following historical sales data for the product category '{categoria_sel}':
    - Describe the overall revenue evolution.
    - Identify relevant seasonal patterns (e.g. Easter, summer, Christmas).
    - Detect any trends or irregularities.
    - Comment on the forecast for 2025.
    - Provide actionable recommendations to increase sales performance and prepare for seasonal demand.
    
    Sales data (JSON format):
    {data_json}
    """


    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "Eres un analista de negocio especializado en predicción de ventas por categoría."},
            {"role": "user", "content": prompt}
        ],
        model="llama3-8b-8192",
    )

    commentary = response.choices[0].message.content
    st.write(commentary)

