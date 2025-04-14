if uploaded_file:
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

    # Últimos 12 meses reales para estacionalidad
    last_12_months_df = df[df['ds'] >= (last_date - pd.DateOffset(months=12))]
    seasonal_index = last_12_months_df.groupby(last_12_months_df['ds'].dt.month)['y'].mean() / last_12_months_df['y'].mean()
    last_12m_avg = last_12_months_df['y'].mean()

    # Horizonte desde el mes posterior al último real
    months = st.slider("📅 Forecast Horizon (in months from next real month)", 1, 12, 6)
    future_months = pd.date_range(last_date + pd.offsets.MonthBegin(), periods=months, freq='MS')

    forecast_df = pd.DataFrame({'ds': future_months})
    forecast_df['month'] = forecast_df['ds'].dt.month
    forecast_df['y'] = forecast_df['month'].map(seasonal_index) * last_12m_avg

    # ==== SLIDERS POR TRIMESTRE ====
    trimesters = {
        "Q1": [1, 2, 3],
        "Q2": [4, 5, 6],
        "Q3": [7, 8, 9],
        "Q4": [10, 11, 12],
    }

    active_trimesters = {q: months_list for q, months_list in trimesters.items() if max(months_list) > last_month}

    st.sidebar.header("🔧 Scenario Planner")
    col1, col2 = st.sidebar.columns(2)
    scenario_inputs = {}

    for q, months_list in active_trimesters.items():
        scenario_inputs[q] = {
            'rev': col1.slider(f"{q} Rev (%)", -50, 50, 0),
            'gpm': col2.slider(f"{q} GPM (p.p.)", -10, 10, 0)
        }

    # ==== AJUSTES AL FORECAST ====
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

    # ==== CALCULAR TOTALES 2025 CON Q1 REAL ====
    real_2025_df = df[(df['ds'].dt.year == 2025) & (df['ds'].dt.month <= 3)]
    forecast_df['year'] = forecast_df['ds'].dt.year
    forecast_2025_df = forecast_df[forecast_df['ds'].dt.year == 2025]

    combined_base = pd.concat([
        real_2025_df[['ds', 'y', 'gpm']],
        forecast_2025_df[['ds', 'y']].assign(gpm=df['gpm'].mean())
    ])

    combined_adjusted = pd.concat([
        real_2025_df[['ds', 'y', 'gpm']],
        forecast_2025_df[['ds', 'adjusted_y']].rename(columns={'adjusted_y': 'y'}).assign(gpm=df['gpm'].mean())
    ])

    fcst_2025_base = combined_base['y'].sum()
    margin_2025_base = (combined_base['y'] * combined_base['gpm']).sum()
    gpm_pct_2025_base = margin_2025_base / fcst_2025_base

    fcst_2025_adjusted = combined_adjusted['y'].sum()
    margin_2025_adjusted = (combined_adjusted['y'] * combined_adjusted['gpm']).sum()
    gpm_pct_2025_adjusted = margin_2025_adjusted / fcst_2025_adjusted

    # ==== COMENTARIO DINÁMICO PARA GROQ ====
    adjustment_summary = ""
    for q in scenario_inputs:
        rev = scenario_inputs[q]['rev']
        gpm = scenario_inputs[q]['gpm']
        adjustment_summary += f"- {q}: Revenue adjustment = {rev}%, GPM adjustment = {gpm}p.p.\n"

    prompt = f"""
    You are an FP&A analyst. Based on the user's adjustments:

    {adjustment_summary}

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

    st.subheader(":robot_face: Executive Commentary")
    st.markdown("### :blue_book: AI-Generated Commentary")
    st.write(commentary)







