# ==== DATOS ====
df = pd.read_excel(uploaded_file)
df.columns = [col.strip().lower() for col in df.columns]
df = df.rename(columns={'date': 'ds', 'revenue': 'y', '% gpm': 'gpm'})
df['ds'] = pd.to_datetime(df['ds'])
df = df[['ds', 'y', 'gpm']].dropna()
df['month'] = df['ds'].dt.month
df['year'] = df['ds'].dt.year

# ==== FORECAST ====
last_date = df['ds'].max()
last_month = last_date.month

# Últimos 12 meses reales
last_12_months_df = df[df['ds'] >= (last_date - pd.DateOffset(months=12))]
seasonal_index = last_12_months_df.groupby(last_12_months_df['ds'].dt.month)['y'].mean() / last_12_months_df['y'].mean()
last_12m_avg = last_12_months_df['y'].mean()

# Horizonte de forecast
months = st.slider("📅 Forecast Horizon (in months from next real month)", 1, 12, 6)
future_months = pd.date_range(last_date + pd.offsets.MonthBegin(), periods=months, freq='MS')
forecast_df = pd.DataFrame({'ds': future_months})
forecast_df['month'] = forecast_df['ds'].dt.month
forecast_df['y'] = forecast_df['month'].map(seasonal_index) * last_12m_avg

# ==== SCENARIO SLIDERS ====
trimesters = {
    "Q1": [1, 2, 3],
    "Q2": [4, 5, 6],
    "Q3": [7, 8, 9],
    "Q4": [10, 11, 12],
}

# Mostrar solo si el último mes real es menor que el último mes del trimestre
active_trimesters = {q: months for q, months in trimesters.items() if max(months) > last_month}

st.sidebar.header("🔧 Scenario Planner")
col1, col2 = st.sidebar.columns(2)
scenario_inputs = {}

for q, months_list in active_trimesters.items():
    scenario_inputs[q] = {
        'rev': col1.slider(f"{q} Rev (%)", -50, 50, 0),
        'gpm': col2.slider(f"{q} GPM (p.p.)", -10, 10, 0)
    }

# ==== AJUSTES ====
def apply_adjustments(row):
    month = row['ds'].month
    revenue = row['y']
    base_margin = df['gpm'].mean()

    for q, months_list in trimesters.items():
        if month in months_list and q in scenario_inputs:
            rev_adj = scenario_inputs[q]['rev']
            gpm_adj = scenario_inputs[q]['gpm']
            adj_y = revenue * (1 + rev_adj / 100)
            adj_gpm = base_margin + (gpm_adj / 100)
            break
    else:
        adj_y = revenue
        adj_gpm = base_margin

    return pd.Series({'adjusted_y': adj_y, 'adjusted_margin': adj_y * adj_gpm})

forecast_df[['adjusted_y', 'adjusted_margin']] = forecast_df.apply(apply_adjustments, axis=1)






