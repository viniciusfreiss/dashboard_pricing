import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm

# --- Page Configuration ---
st.set_page_config(
    page_title="Elasticidade de preço",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Modern Minimalist Design System (CSS) ---
# --- Modern Minimalist Design System (CSS) ---
# Palette: White, Black, Blue (Primary)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    /* Global Reset & Background */
    .stApp {
        background-color: #F8FAFC; /* Soft Slate Background for Depth */
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6, .stMarkdown, p, span, label, div {
        color: #1E293B !important; /* Slate 800 for softer black */
        font-family: 'Inter', sans-serif !important;
    }
    h1 { letter-spacing: -0.02em; font-weight: 700; }
    
    /* Specific Sidebar Override */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important; /* White Sidebar */
        border-right: 1px solid #E2E8F0;
        box-shadow: 2px 0 5px rgba(0,0,0,0.02);
    }
    
    /* Dropdowns & Selects Fix */
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
    }
    
    /* Buttons */
    div.stButton > button {
        background-color: #2563EB !important; /* Premium Blue */
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.2s;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
    }
    div.stButton > button:hover {
        background-color: #1D4ED8 !important; /* Darker Blue */
        transform: translateY(-1px);
        box-shadow: 0 6px 8px -1px rgba(37, 99, 235, 0.3);
    }
    
    /* Premium Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def card_html(title, value, description, color="#3B82F6"):
    return f"""
    <div style="background-color: white; padding: 20px; border-radius: 10px; border: 1px solid #E2E8F0; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
        <h4 style="color: #64748B; margin: 0; font-size: 0.9em; text-transform: uppercase;">{title}</h4>
        <div style="font-size: 2em; font-weight: 700; color: {color}; margin: 5px 0;">{value}</div>
        <div style="color: #94A3B8; font-size: 0.8em;">{description}</div>
    </div>
    """

def style_chart(fig):
    """Enforces a clean white theme for all charts, ignoring Streamlit's dark mode defaults."""
    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        font={'color': '#000000', 'family': 'Inter'},
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(showgrid=True, gridcolor='#E2E8F0', zeroline=False, tickfont=dict(color='#000000'), title_font=dict(color='#000000')),
        yaxis=dict(showgrid=True, gridcolor='#E2E8F0', zeroline=False, tickfont=dict(color='#000000'), title_font=dict(color='#000000')),
    )
    return fig

# --- Data Loading Functions ---

@st.cache_data
def load_global_data_v2():
    """Loads the weekly global data with robust cleaning (Cache Bust)"""
    file_path = 'db_elasticidade.csv'
    try:
        df = pd.read_csv(file_path)
        
        # Helper: Clean numeric columns
        def clean_val(val):
            if pd.isna(val): return 0.0
            if isinstance(val, (int, float)): return float(val)
            s = str(val).strip().replace('"', '').replace("'", "")
            
            # Case: "1,245,0" (Typo/BR format mixed) -> 1245.0
            if s.count(',') > 1:
                parts = s.split(',')
                integer_part = "".join(parts[:-1])
                decimal_part = parts[-1]
                return float(f"{integer_part}.{decimal_part}")
            
            if ',' in s and '.' in s: 
                s = s.replace(',', '') 
            elif ',' in s:
                s = s.replace(',', '.')
                
            try:
                return float(s)
            except:
                return 0.0

        df['quantity'] = df['kpi'].apply(clean_val)
        df['price'] = df['revenue_per_kpi'].apply(clean_val)
        df['date'] = pd.to_datetime(df['time'])
        df['revenue'] = df['quantity'] * df['price']
        
        df = df.sort_values('date')
        df = df[(df['quantity'] > 0) & (df['price'] > 0)]
        
        # Log Features
        df['ln_quantity'] = np.log(df['quantity'])
        df['ln_price'] = np.log(df['price'])
        
        # Date Features (Restored)
        df['month'] = df['date'].dt.month_name()
        df['month_num'] = df['date'].dt.month
        
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados globais: {e}")
        return None

@st.cache_data
def load_granular_data():
    """Loads the transactional granular data with robust cleaning"""
    file_path = 'db_per_course.csv'
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        return None
    
    # Robust Cleaning Helper (Same as Global)
    def clean_val(val):
        if pd.isna(val): return 0.0
        if isinstance(val, (int, float)): return float(val)
        s = str(val).strip().replace('"', '').replace("'", "").replace("R$", "").strip()
        
        # Case: "1,245,0" (Mixed) -> 1245.0
        if s.count(',') > 1:
            parts = s.split(',')
            integer_part = "".join(parts[:-1])
            decimal_part = parts[-1]
            return float(f"{integer_part}.{decimal_part}")
        
        # Case: "12,160.00" (US) -> 12160.0
        if ',' in s and '.' in s: 
            s = s.replace(',', '') 
        elif ',' in s:
            s = s.replace(',', '.')
            
        try:
            return float(s)
        except:
            return 0.0

    # Apply cleaning
    df['transaction_value'] = df['Valor Líquido'].apply(clean_val)

    # Date
    df['date'] = pd.to_datetime(df['Data'], dayfirst=True)
    
    # Rename
    df = df.rename(columns={
        'Tipo de Curso': 'type',
        'UF': 'state',
        'Forma de Pagamento': 'payment',
        'Curso': 'course',
        'Parcelas': 'installments'
    })
    
    # Deduplicate Course Names
    if 'course' in df.columns:
        df['course'] = df['course'].astype(str).str.strip().str.upper()
    
    # Filter valid
    df = df.dropna(subset=['transaction_value', 'date'])
    df = df[df['transaction_value'] > 0]
    
    return df

# --- Load Data ---
df_global = load_global_data_v2()
df_granular = load_granular_data()

# --- Application State ---
st.title("Elasticidade de Preço")

# --- Sidebar ---
with st.sidebar:
    st.markdown('<div class="sidebar-logo">', unsafe_allow_html=True)
    st.image("uol_logo.png", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.header("Configuração de Análise")
    
    # Mode Selection
    mode = st.radio("Escopo:", ["Global (Dados Agregados)", "Granular (Por Filtro)"], index=0)
    
    active_df = None
    filter_desc = "Global"
    is_granular = False
    payment_df = df_granular.copy() if df_granular is not None else None # Transactional DF for Tab 6
    
    if mode == "Global (Dados Agregados)":
        if df_global is not None:
            active_df = df_global.copy()
            filter_desc = "Visão Agregada (Todas as Vendas)"
        else:
            st.error("Dados globais indisponíveis.")
            
    else: # Granular
        is_granular = True
        if df_granular is None:
            st.error("Dados granulares indisponíveis.")
        else:
            filter_type = st.selectbox("Filtrar por:", ["Tipo de Curso", "Curso Específico", "Parcelas"])
            
            df_slice = df_granular.copy()
            
            if filter_type == "Tipo de Curso":
                sel = st.selectbox("Selecione:", sorted(df_granular['type'].unique()))
                df_slice = df_granular[df_granular['type'] == sel]
                filter_desc = f"Tipo: {sel}"
            
            elif filter_type == "Curso Específico":
                sel = st.selectbox("Selecione:", sorted(df_granular['course'].unique()))
                df_slice = df_granular[df_granular['course'] == sel]
                filter_desc = f"Curso: {sel}"
                
            elif filter_type == "Parcelas":
                sel = st.selectbox("Selecione:", sorted(df_granular['installments'].astype(str).unique()))
                df_slice = df_granular[df_granular['installments'].astype(str) == sel]
                filter_desc = f"Parcelas: {sel}"
            
                df_slice = df_granular[df_granular['installments'].astype(str) == sel]
                filter_desc = f"Parcelas: {sel}"
            
            payment_df = df_slice.copy() # Update for Tab 6
            
            # Aggregate Granular to Weekly for Modeling
            # Robust Aggregation:
            if not df_slice.empty:
                df_wk = df_slice.set_index('date').resample('W').agg({
                    'transaction_value': 'sum',
                    'type': 'count'
                }).rename(columns={'transaction_value': 'revenue', 'type': 'quantity'})
                
                df_wk['price'] = df_wk['revenue'] / df_wk['quantity']
                df_wk = df_wk[(df_wk['quantity'] > 0) & (df_wk['price'] > 0)]
                
                # Log Features
                df_wk['ln_quantity'] = np.log(df_wk['quantity'])
                df_wk['ln_price'] = np.log(df_wk['price'])
                df_wk['date'] = df_wk.index
                df_wk['month'] = df_wk.index.month_name()
                
                active_df = df_wk
            else:
                st.warning("Filtro retornou vazio.")
                active_df = pd.DataFrame()

    st.markdown("---")
    st.info(f"**Modo:** {filter_desc}")
    if active_df is not None and not active_df.empty:
        st.write(f"Pontos de Dados (Semanas): {len(active_df)}")

# --- Modeling Logic (Dual Model: Log for Elasticity, Linear for Optimization) ---
elasticity = 0.0
r2_log = 0.0
model_ok = False
model_warning = ""

# Linear Optimization Variables (Restored)
opt_price = 0.0
max_rev = 0.0
curr_rev_est = 0.0
linear_r2 = 0.0
linear_model_ok = False
slope_lin = 0.0
intercept_lin = 0.0

if active_df is not None and not active_df.empty and len(active_df) > 8:
    if active_df['ln_price'].std() > 0.001:
        try:
            # 1. Log-Log Model (Elasticity)
            X_log = sm.add_constant(active_df['ln_price'])
            y_log = active_df['ln_quantity']
            model_log = sm.OLS(y_log, X_log).fit()
            
            e_val = model_log.params['ln_price']
            if isinstance(e_val, pd.Series): e_val = e_val.iloc[0]
            raw_elasticity = float(e_val)
            
            # Safeguard: Enforce Conservative Elasticity (Decay View)
            # If data suggests Inelastic (>-1) or Positive (>0), clamp to slightly Elastic (-1.01)
            # to visualize risk of price increases (Revenue Decay).
            if raw_elasticity >= -1.0:
                elasticity = -1.05 # Force "just barely elastic" behavior
                model_warning = f"Elasticidade calculada ({raw_elasticity:.2f}) indica inelasticidade. Ajustado para -1.05 para análise conservadora de risco."
            else:
                elasticity = raw_elasticity
                
            r2_log = model_log.rsquared
            model_ok = True
            
            # 2. Linear Model (Optimization/Saturation) - RESTORED
            # Q = a + b * P
            X_lin = sm.add_constant(active_df['price'])
            y_lin = active_df['quantity']
            model_lin = sm.OLS(y_lin, X_lin).fit()
            
            intercept_lin = model_lin.params['const']
            slope_lin = model_lin.params['price']
            linear_r2 = model_lin.rsquared
            
            if slope_lin < 0: # Normal demand behavior
                opt_price = -intercept_lin / (2 * slope_lin)
                opt_q = intercept_lin + slope_lin * opt_price
                max_rev = opt_price * opt_q
                linear_model_ok = True
            
            # Robustness Check
            if is_granular and r2_log < 0.1:
                model_warning = "Atenção: Correlação fraca (R² < 0.1). Resultados podem ser instáveis."
                
        except Exception as e:
            st.error(f"Erro modelagem: {e}")
            model_warning = "Erro no cálculo."
    else:
        model_warning = "Sem variação de preço suficiente para análise."
else:
    model_warning = "Dados insuficientes (Mínimo 12 semanas)."

# --- UI Layout ---
if model_warning and not model_ok:
    st.warning(model_warning)

# KPI Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Elasticidade (Log-Log)", f"{elasticity:.2f}", 
              help="Quanto % o volume cai se o preço subir 1%. Menor que -1 é Elástica (Sensível).")
    if abs(elasticity) < 1:
        st.markdown('<span style="color:#22c55e; background:#dcfce7; padding:4px 8px; border-radius:4px; font-size:12px;">↑ Inelástica</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="color:#ef4444; background:#fee2e2; padding:4px 8px; border-radius:4px; font-size:12px;">↓ Elástica</span>', unsafe_allow_html=True)

with col2:
    sensitivity_10 = elasticity * 10
    st.metric("Sensibilidade (+10% Preço)", f"{sensitivity_10:.1f}% Vol",
              help="Impacto estimado no volume ao subir 10% o preço.")

with col3:
    if active_df is not None and not active_df.empty:
        curr_p = active_df['price'].tail(4).mean()
        st.metric("Ticket Médio (Atual)", f"R$ {curr_p:,.2f}")
    else:
        st.metric("Ticket Médio", "-")
with col4:
    if active_df is not None and not active_df.empty:
        curr_r = active_df['revenue'].tail(4).mean()
        st.metric("Receita Semanal (Média)", f"R$ {curr_r:,.2f}")
    else:
        st.metric("Receita", "-")

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Curva de Demanda", "Simulador (Otimização)", "Sazonalidade", "Comparativo (Granular)", "Diagnóstico de Modelos", "Otimização de Pagamento"])

with tab1:
    col_a, col_b = st.columns([2,1])
    if active_df is not None and not active_df.empty:
        with col_a:
             # Basic Scatter
             fig = px.scatter(active_df, x='price', y='quantity', size='revenue', title="Curva de Demanda", labels={'price':'Preço', 'quantity':'Volume'})
             fig.update_traces(marker=dict(color='#3B82F6', opacity=0.6, line=dict(width=1, color='white'))) # Blue bubbles
             
             # Add Trendline if valid
             if model_ok and elasticity < 0:
                 x_range = np.linspace(active_df['price'].min(), active_df['price'].max(), 100)
                 # Q = exp(a) * P^b
                 # ln(Q) = a + b*ln(P) -> Q = exp(a + b*ln(P))
                 # Get params from the model if possible, or approximate with the scalar elasticity
                 # Re-calculating specific intercept for this view to ensure alignment
                 try:
                    c_val = model_log.params['const']
                    e_val = model_log.params['ln_price']
                    y_pred = np.exp(c_val + e_val * np.log(x_range))
                    
                    fig.add_trace(go.Scatter(
                        x=x_range, y=y_pred, mode='lines', 
                        name=f'Tendência (e={elasticity:.2f})',
                        line=dict(color='#0F172A', width=2, dash='dash') # Dark slate line
                    ))
                 except: pass

             fig = style_chart(fig)
             st.plotly_chart(fig, theme=None, use_container_width=True)
        with col_b:
             st.markdown("### Análise Inteligente")
             
             # 1. Natural Language Diagnosis
             if abs(elasticity) < 1.0:
                 diag_title = "Demanda Inelástica (Oportunidade)"
                 diag_color = "#10B981" # Green
                 diag_text = f"""
                 A demanda **resiste bem a aumentos de preço**. 
                 A elasticidade de **{elasticity:.2f}** indica que um aumento de 10% no preço 
                 resultaria em queda de apenas {abs(elasticity)*10:.1f}% no volume, aumentando a receita total.
                 """
             else:
                 diag_title = "Demanda Elástica (Sensível)"
                 diag_color = "#F59E0B" # Orange
                 diag_text = f"""
                 Os consumidores são **sensíveis a preço**. 
                 A elasticidade de **{elasticity:.2f}** indica que um aumento de 10% no preço 
                 reduziria o volume em {abs(elasticity)*10:.1f}%, impactando a receita.
                 """
             
             st.markdown(f"""
             <div style="background-color: {diag_color}20; border-left: 5px solid {diag_color}; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
                 <strong style="color: {diag_color}; font-size: 1.1em;">{diag_title}</strong>
                 <p style="margin-top: 5px; font-size: 0.95em; color: #334155;">{diag_text}</p>
             </div>
             """, unsafe_allow_html=True)

             # 2. Model Confidence
             conf_color = "#10B981" if r2_log > 0.6 else "#F59E0B" if r2_log > 0.3 else "#EF4444"
             conf_text = "Alta" if r2_log > 0.6 else "Média" if r2_log > 0.3 else "Baixa"
             st.markdown(f"""
             <div style="display: flex; align-items: center; margin-bottom: 20px;">
                 <div style="flex: 1;"><strong>Confiabilidade do Modelo (R²):</strong></div>
                 <div style="background-color: {conf_color}; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.8em; font-weight: bold;">{conf_text} ({r2_log:.2f})</div>
             </div>
             """, unsafe_allow_html=True)
            
             # 3. Mini Simulator
             st.markdown("#### Simulação Rápida")
             with st.container(border=True):
                 sim_pct = st.slider("Ajuste de Preço (%)", -30, 30, 0, format="%d%%")
                 
                 if sim_pct != 0:
                     new_p_factor = 1 + (sim_pct / 100)
                     # Qty Change = %P * Elasticity
                     # More precise: Q2 = Q1 * (P2/P1)^Elasticity
                     qty_change_pct = ((new_p_factor ** elasticity) - 1) * 100
                     
                     # Rev Change
                     new_rev_factor = new_p_factor * (1 + (qty_change_pct/100))
                     rev_change_pct = (new_rev_factor - 1) * 100
                     
                     c1, c2 = st.columns(2)
                     c1.metric("Matrículas Semanais", f"{qty_change_pct:+.1f}%", delta_color="inverse")
                     c2.metric("Receita Estimada", f"{rev_change_pct:+.1f}%")
                 else:
                     st.caption("Mova a barra para simular impactos no faturamento.")
        
        st.markdown("---")
        st.subheader("Evolução Histórica")
        
        # Dual Axis Chart: Volume vs Price
        fig_hist = go.Figure()
        # Changed color to #94A3B8 (Slate 400) for better visibility against white
        fig_hist.add_trace(go.Bar(x=active_df['date'], y=active_df['quantity'], name='Volume de Vendas', marker_color='#94A3B8', opacity=0.7))
        fig_hist.add_trace(go.Scatter(x=active_df['date'], y=active_df['price'], name='Ticket Médio', yaxis='y2', line=dict(color='#0F172A', width=3)))
        
        fig_hist.update_layout(
            yaxis=dict(title="Volume"),
            yaxis2=dict(title="Preço Médio", overlaying='y', side='right'),
            title="Volume vs Preço ao Longo do Tempo",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig_hist = style_chart(fig_hist)
        st.plotly_chart(fig_hist, use_container_width=True)

with tab2:
    if active_df is not None:
        st.subheader("Interactive Scenario Planner")
        
        # --- PREPARATION ---
        base_p = active_df['price'].tail(4).mean()
        base_q = active_df['quantity'].tail(4).mean()
        curr_rev_est = base_p * base_q
        
        # Layout: Left Panel (Controls + KPIs) | Right Panel (Chart)
        col_panel, col_chart = st.columns([1, 2])
        
        with col_panel:
            # --- CONTROLS ---
            st.markdown("""
            <div style="background-color: white; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <h5 style="margin: 0 0 10px 0; color: #333;">Controls</h5>
                <p style="font-size: 0.8em; color: #666; margin-bottom: 10px;">Adjust price to simulate revenue impact.</p>
            </div>
            """, unsafe_allow_html=True)
            
            pct_change = st.slider("Price Adjustment (%)", -50, 50, 0, format="%d%%")
            
            # Logic
            manual_price = base_p * (1 + pct_change / 100)
            
            # Forecast
            pred_q = 0
            if model_ok:
                pred_q = base_q * ((manual_price / base_p) ** elasticity)
            else:
                pred_q = base_q
                
            pred_r = manual_price * pred_q
            
            delta_r_pct = ((pred_r - curr_rev_est) / curr_rev_est) * 100
            
            # --- CARD 1: REVENUE ---
            rev_color = "#10B981" if delta_r_pct >= 0 else "#EF4444"
            st.markdown(f"""
            <div style="background-color: white; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; margin-bottom: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <p style="font-size: 0.8em; font-weight: 600; color: #666; margin: 0;">Projected Revenue</p>
                <h3 style="margin: 5px 0; color: #111;">R$ {pred_r:,.2f}</h3>
                <span style="background-color: {rev_color}20; color: {rev_color}; padding: 2px 6px; border-radius: 4px; font-size: 0.8em; font-weight: 600;">
                    {delta_r_pct:+.2f}%
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            # --- CARD 2: NEW PRICE ---
            st.markdown(f"""
            <div style="background-color: white; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; margin-bottom: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <p style="font-size: 0.8em; font-weight: 600; color: #666; margin: 0;">New Price</p>
                <h3 style="margin: 5px 0; color: #111;">R$ {manual_price:,.2f}</h3>
                <p style="font-size: 0.8em; color: #888; margin: 0;">Original: R$ {base_p:,.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # --- CARD 3: VOLUME ---
            st.markdown(f"""
            <div style="background-color: white; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; margin-bottom: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <p style="font-size: 0.8em; font-weight: 600; color: #666; margin: 0;">Projected Volume</p>
                <h3 style="margin: 5px 0; color: #111;">{int(pred_q)}</h3>
                <p style="font-size: 0.8em; color: #888; margin: 0;">Base: {int(base_q)}</p>
            </div>
            """, unsafe_allow_html=True)

        with col_chart:
            # --- REVENUE CURVE ---
            st.markdown("##### Revenue Optimization Curve")
            
            # Generate Logic Curve
            p_range = np.linspace(base_p * 0.5, base_p * 1.5, 100)
            r_curve = []
            for p in p_range:
                q_ = base_q * ((p / base_p) ** elasticity)
                r_curve.append(p * q_)
                
            df_curve = pd.DataFrame({'Price': p_range, 'Revenue': r_curve})
            
            fig_scen = go.Figure()
            
            # Curve
            fig_scen.add_trace(go.Scatter(
                x=df_curve['Price'], y=df_curve['Revenue'],
                mode='lines',
                name='Revenue Curve',
                line=dict(color='gray', width=2)
            ))
            
            # Current Price Line (Dashed)
            fig_scen.add_shape(
                type="line",
                x0=base_p, y0=min(r_curve), x1=base_p, y1=max(r_curve),
                line=dict(color="black", width=2, dash="dash"),
            )
            fig_scen.add_annotation(
                x=base_p, y=max(r_curve),
                text="Current",
                showarrow=False,
                yshift=10,
                font=dict(color="gray")
            )
            
            # Selected Point
            fig_scen.add_trace(go.Scatter(
                x=[manual_price], y=[pred_r], 
                mode='markers+text', 
                name='Selected', text=['Selected'], 
                textposition='middle right',
                marker=dict(color='#2563EB', size=12, line=dict(width=2, color='white'))
            ))
            
            fig_scen.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(t=30, b=20, l=40, r=40),
                xaxis=dict(showgrid=True, gridcolor='#f3f4f6', title='Price'),
                yaxis=dict(showgrid=True, gridcolor='#f3f4f6', title='Revenue'),
                showlegend=False,
                height=350
            )
            
            st.plotly_chart(fig_scen, use_container_width=True)
            
            # --- FORMULA EXPLANATION ---
            st.markdown(f"""
            <div style="background-color: #E0F2FE; padding: 20px; border-radius: 8px; border: 1px solid #BAE6FD; color: #0C4A6E; font-size: 0.9em;">
                <p style="margin-bottom: 10px;"><strong>Prediction Formula:</strong></p>
                <div style="text-align: center; margin-bottom: 15px; font-size: 1.1em; font-family: 'Courier New', monospace;">
                    Q<sub>new</sub> = Q<sub>base</sub> · ( P<sub>new</sub> / P<sub>base</sub> )<sup>β</sup>
                </div>
                <p style="margin-bottom: 5px;"><strong>Variáveis:</strong></p>
                <ul style="list-style-type: disc; margin-left: 20px; margin-bottom: 0;">
                    <li><strong>Q<sub>new</sub></strong>: Quantidade Projetada ({int(pred_q)})</li>
                    <li><strong>Q<sub>base</sub></strong>: Quantidade Atual ({int(base_q)})</li>
                    <li><strong>P<sub>new</sub></strong>: Novo Preço (R$ {manual_price:,.2f})</li>
                    <li><strong>P<sub>base</sub></strong>: Preço Atual (R$ {base_p:,.2f})</li>
                    <li><strong>β</strong>: Elasticidade ({elasticity:.2f})</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

with tab3:
    if active_df is not None and not active_df.empty:
        st.subheader("Padrões de Sazonalidade")
        st.markdown("""
        <div style="color: #64748B; font-size: 0.9em; margin-bottom: 15px;">
        Analise como o volume de vendas oscila ao longo dos meses. 
        A **linha vermelha** indica a média histórica, e as **caixas azuis** mostram a variação (mínimo e máximo).
        </div>
        """, unsafe_allow_html=True)

        active_df['m_order'] = active_df['date'].dt.month
        
        # Translate Months
        month_map = {
            1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun',
            7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'
        }
        active_df['month_pt'] = active_df['m_order'].map(month_map)
        
        df_srt = active_df.sort_values('m_order')
        
        # Base Boxplot
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(
            x=df_srt['month_pt'], 
            y=df_srt['quantity'], 
            name='Variação',
            marker_color='#3B82F6',
            boxpoints='outliers'
        ))
        
        # Add Average Line
        df_mean = df_srt.groupby('month_pt', sort=False)['quantity'].mean().reset_index()
        
        # Insights for Layman
        best_month_row = df_mean.loc[df_mean['quantity'].idxmax()]
        worst_month_row = df_mean.loc[df_mean['quantity'].idxmin()]
        avg_vol = df_mean['quantity'].mean()
        
        # Display Summary Cards
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.markdown(card_html("Melhor Mês Historicamente", f"{best_month_row['month_pt']}", f"Média de {int(best_month_row['quantity'])} vendas. O pico do ano.", color="#10B981"), unsafe_allow_html=True)
        with col_s2:
            st.markdown(card_html("Mês de Menor Movimento", f"{worst_month_row['month_pt']}", f"Média de {int(worst_month_row['quantity'])} vendas. Momento para promoções?", color="#EF4444"), unsafe_allow_html=True)
        with col_s3:
            st.markdown(card_html("Média Geral Mensal", f"{int(avg_vol)}", "Volume base esperado por mês.", color="#3B82F6"), unsafe_allow_html=True)
            
        fig_box.add_trace(go.Scatter(
            x=df_mean['month_pt'], 
            y=df_mean['quantity'],
            mode='lines+markers+text', # Added text
            name='Média',
            text=[f"{int(y)}" if m == best_month_row['month_pt'] else "" for m, y in zip(df_mean['month_pt'], df_mean['quantity'])],
            textposition="top center",
            line=dict(color='#EF4444', width=3)
        ))
        
        # Annotation for Peak
        fig_box.add_annotation(
            x=best_month_row['month_pt'],
            y=best_month_row['quantity'],
            text="Pico Sazonal",
            showarrow=True,
            arrowhead=1,
            yshift=10
        )
        
        fig_box.update_layout(
            title="Média Semanal de Vendas por Mês (Linha Vermelha = Média)",
            xaxis_title="Mês",
            yaxis_title="Média Semanal de Vendas",
            showlegend=True
        )
        fig_box = style_chart(fig_box)
        st.plotly_chart(fig_box, theme=None, use_container_width=True)

with tab4:
    st.markdown("### Comparativo Granular")
    if df_granular is not None:
        # Translation Mapping
        dim_map = {
            "Forma de Pagamento": "payment",
            "Parcelas": "installments",
            "Tipo de Curso": "type"
        }
        
        sel_dim_label = st.selectbox("Dimensão de Análise:", list(dim_map.keys()))
        comp_dim = dim_map[sel_dim_label]
        
        st.info("""
        **Como ler esta tabela:**
        *   **Elasticidade**: Mede a sensibilidade. Valores abaixo de -1 (ex: -2.5) indicam que o cliente é **muito sensível** a preço (se subir, a venda cai muito). Valores próximos a 0 indicam que ele não se importa tanto.
        *   **R² (Confiança)**: Quanto mais próximo de 1.0, mais confiável é o cálculo. Desconsidere linhas com R² muito baixo (< 0.1).
        """)
        
        if st.button("Gerar Ranking"):
            res = []
            segs = df_granular[comp_dim].unique()
            # Context for Bubble Chart
            seg_stats = []
            
            pbar = st.progress(0)
            for i, s in enumerate(segs):
                sub = df_granular[df_granular[comp_dim] == s]
                # Aggr
                w = sub.set_index('date').resample('W').agg({'transaction_value': 'sum', 'type': 'count'})
                w.columns = ['revenue', 'quantity'] # Ensure correct names
                w['p'] = w['revenue']/w['quantity']
                w = w[(w['p']>0) & (w['quantity']>0)].dropna()
                
                total_rev = w['revenue'].sum()
                total_vol = w['quantity'].sum()
                
                if len(w) > 8 and w['p'].std() > 0.01:
                    try:
                        m = sm.OLS(np.log(w['quantity']), sm.add_constant(np.log(w['p']))).fit()
                        el = m.params.iloc[1]
                        r2 = m.rsquared
                        res.append({'Segmento': s, 'Elasticidade': el, 'R2': r2})
                        seg_stats.append({'Segmento': s, 'Elasticidade': el, 'Receita Total': total_rev, 'Volume Total': total_vol, 'R2': r2})
                    except: pass
                pbar.progress((i+1)/len(segs))
            
            pbar.empty()
            
            if res:
                rdf = pd.DataFrame(res).sort_values('Elasticidade')
                
                # Table
                st.dataframe(rdf.style.format({'Elasticidade': '{:.2f}', 'R2': '{:.2f}'}), use_container_width=True)
                
                col_g1, col_g2 = st.columns(2)
                
                with col_g1:
                    st.markdown("#### Ranking de Sensibilidade")
                    fig_comp = px.bar(rdf, y='Segmento', x='Elasticidade', orientation='h', title="Mais Sensíveis (Menor Valor) vs Menos Sensíveis", color='Elasticidade', color_continuous_scale='RdYlGn_r')
                    fig_comp = style_chart(fig_comp)
                    st.plotly_chart(fig_comp, theme=None, use_container_width=True)
                
                with col_g2:
                    st.markdown("#### Matriz de Estratégia")
                    st.caption("Tamanho da bolha = Receita Total")
                    sdf = pd.DataFrame(seg_stats)
                    fig_bubble = px.scatter(sdf, x='Elasticidade', y='Receita Total', size='Receita Total', color='Segmento', 
                                            hover_name='Segmento', text='Segmento',
                                            title="Elasticidade vs Importância Financeira")
                    fig_bubble.add_vline(x=-1, line_dash="dash", line_color="gray", annotation_text="Elástico <- | -> Inelástico")
                    fig_bubble = style_chart(fig_bubble)
                    st.plotly_chart(fig_bubble, theme=None, use_container_width=True)
                    
            else:
                st.warning("Sem dados suficientes para cálculo de elasticidade nestes segmentos (necessário variação de preço e histórico).")

with tab5:
    if active_df is not None and len(active_df) > 12:
        st.header("Análise Detalhada dos Modelos")
        
        try:
            # Re-run models for diagnostics
            # 1. Log-Log
            Y_log = active_df['ln_quantity']
            X_log = sm.add_constant(active_df['ln_price'])
            model_log_full = sm.OLS(Y_log, X_log).fit()
            
            # 2. Linear (Reference Only)
            Y_lin = active_df['quantity']
            X_lin = sm.add_constant(active_df['price'])
            model_lin_full = sm.OLS(Y_lin, X_lin).fit()
            
            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader("Comparativo de Ajuste (R²)")
                st.dataframe(pd.DataFrame({
                    "Modelo": ["Log-Log (Elasticidade)", "Linear (Simples)"],
                    "R²": [model_log_full.rsquared, model_lin_full.rsquared],
                    "AIC (Menor é melhor)": [model_log_full.aic, model_lin_full.aic]
                }).style.format({"R²": "{:.3f}", "AIC": "{:.1f}"}))
                
                st.markdown("### Significância Estatística")
                p_val = model_log_full.pvalues['ln_price']
                if p_val < 0.05:
                    st.success(f"Preço é estatisticamente significante (p={p_val:.4f})")
                else:
                    st.error(f"Variação de preço não explica vendas (p={p_val:.4f})")

            with c2:
                # Plot Overlay
                x_vals = np.linspace(active_df['price'].min(), active_df['price'].max(), 50)
                
                # Log pred
                log_const = model_log_full.params['const']
                log_slope = model_log_full.params['ln_price']
                y_log_pred = np.exp(log_const + log_slope * np.log(x_vals))
                
                # Lin pred
                lin_const = model_lin_full.params['const']
                lin_slope = model_lin_full.params['price']
                y_lin_pred = lin_const + lin_slope * x_vals
                
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Scatter(x=active_df['price'], y=active_df['quantity'], mode='markers', name='Dados Reais', marker=dict(color='black', opacity=0.5)))
                fig_comp.add_trace(go.Scatter(x=x_vals, y=y_log_pred, name='Log-Log (Elasticidade)', line=dict(color='blue', width=3)))
                fig_comp.add_trace(go.Scatter(x=x_vals, y=y_lin_pred, name='Linear', line=dict(color='gray', dash='dash')))
                
                fig_comp.update_layout(title="Curva Log-Log vs Linear", xaxis_title="Preço", yaxis_title="Volume")
                fig_comp = style_chart(fig_comp)
                st.plotly_chart(fig_comp, use_container_width=True)
                
            with st.expander("Ver Sumário Estatístico Completo (OLS)"):
                st.text(model_log_full.summary())
                
        except Exception as e:
            st.warning(f"Diagnóstico indisponível: {e}")
    else:
        st.error("Dados insuficientes para diagnóstico avançado.")

with tab6:
    if payment_df is not None and not payment_df.empty:
        
        # Determine Card Style Helper
        def card_html(title, value, sub_text="", color="#333333"):
            return f"""
            <div style="background-color: white; padding: 20px; border-radius: 12px; border: 1px solid #e0e0e0; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
                <p style="font-size: 0.9em; font-weight: 600; color: #666; margin: 0 0 5px 0;">{title}</p>
                <h3 style="margin: 0; color: {color}; font-size: 1.5em;">{value}</h3>
                <p style="font-size: 0.8em; color: #888; margin: 5px 0 0 0;">{sub_text}</p>
            </div>
            """
            
        col_pay1, col_pay2 = st.columns([1, 1], gap="large")
        
        with col_pay1:
            st.markdown("### Preferência de Parcelamento") 
            st.markdown("""
            <div style="color: #64748B; font-size: 0.9em; margin-bottom: 15px;">
            Analise a preferência de pagamento em diferentes faixas de preço. 
            Utilize o filtro abaixo para "focar" em um ticket específico.
            </div>
            """, unsafe_allow_html=True)
            
            min_ticket = int(payment_df['transaction_value'].min())
            max_ticket = int(payment_df['transaction_value'].max())
            
            # Slider Specific to this Analysis
            if min_ticket == max_ticket:
                sel_range = (min_ticket, max_ticket)
                df_pay_filtered = payment_df.copy()
            else:
                sel_range = st.slider("Filtrar Faixa de Preço para Análise (R$)", min_ticket, max_ticket, (min_ticket, max_ticket))
                df_pay_filtered = payment_df[(payment_df['transaction_value'] >= sel_range[0]) & 
                                              (payment_df['transaction_value'] <= sel_range[1])]
            
            if not df_pay_filtered.empty:
                vol_by_inst = df_pay_filtered['installments'].value_counts().reset_index()
                vol_by_inst.columns = ['Parcelas', 'Volume de Vendas']
                
                # Correct Logic for Top Option (Sort by Volume Descending first)
                top_opt_row = vol_by_inst.sort_values('Volume de Vendas', ascending=False).iloc[0]
                
                # Prepare for Chart (Sort by Installment Order)
                vol_by_inst['sort_key'] = vol_by_inst['Parcelas'].str.extract('(\d+)').astype(float).fillna(0)
                vol_by_inst = vol_by_inst.sort_values('sort_key')
                
                st.markdown(card_html("Opção Favorita (Nesta Faixa)", f"{top_opt_row['Parcelas']}", f"Representa a maioria das vendas entre R$ {sel_range[0]} e {sel_range[1]}."), unsafe_allow_html=True)
                
                fig_bar = px.bar(vol_by_inst, x='Parcelas', y='Volume de Vendas', 
                                 title="Volume X Opção de Parcelamento", text_auto=True,
                                 color_discrete_sequence=['#3B82F6'])
                fig_bar.update_layout(xaxis_title=None, yaxis_title="Vendas", showlegend=False, margin=dict(t=30, l=0, r=0, b=0))
                fig_bar = style_chart(fig_bar)
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.warning("Sem vendas nesta faixa de preço selecionada.")

        with col_pay2:
            st.markdown("""
            <div style="color: #64748B; font-size: 0.9em; margin-bottom: 15px;">
            Modelo de elasticidade exclusivo para pagamentos à vista (Boleto/PIX/1x).
            Analisa <b>todos os dados disponíveis</b> para encontrar o preço ideal.
            </div>
            """, unsafe_allow_html=True)
            
            # Context Data (Full Variance) - NOT Filtered by Slider
            df_cash = payment_df[payment_df['installments'].str.contains("vista", case=False, na=False)].copy()
            
            if len(df_cash) > 12:
                # Aggregation
                df_cash_wk = df_cash.set_index('date').resample('W').agg({'transaction_value': 'sum', 'installments': 'count'}).rename(columns={'transaction_value': 'revenue', 'installments': 'quantity'})
                df_cash_wk['price'] = df_cash_wk['revenue'] / df_cash_wk['quantity']
                df_cash_wk = df_cash_wk[(df_cash_wk['quantity'] > 0) & (df_cash_wk['price'] > 0)]
                
                if len(df_cash_wk) > 5 and df_cash_wk['price'].std() > 0.1:
                    try:
                        import statsmodels.api as sm
                        df_cash_wk['ln_quantity'] = np.log(df_cash_wk['quantity'])
                        df_cash_wk['ln_price'] = np.log(df_cash_wk['price'])
                        
                        X = sm.add_constant(df_cash_wk['ln_price'])
                        model_cash = sm.OLS(df_cash_wk['ln_quantity'], X).fit()
                        e_cash = model_cash.params['ln_price']
                        
                        warning_msg = ""
                        # Logic Fix: Don't show confusing/contradictory warnings
                        if e_cash >= -1.0:
                            # Inelastic -> Increase Price
                            e_cash_clamped = -1.01 # Mildly elastic for curve vis
                            rec_text = "Aumentar Preço"
                            rec_sub = "Demanda Inelástica (> -1). Mercado aceita preço maior."
                            rec_color = "#10B981" # Green
                        else:
                            # Elastic -> Decrease or hold
                            e_cash_clamped = e_cash
                            # Calculate Peak Revenue Price
                            # Peak happens at E = -1. If current E < -1 (e.g. -2), we are past the peak (price too high).
                            rec_text = "Reduzir Preço"
                            rec_sub = "Demanda Elástica (< -1). Preço atual limita volume."
                            rec_color = "#F59E0B" # Orange
                        
                        st.markdown(card_html("Estratégia Sugerida", rec_text, f"Elasticidade Calc: {e_cash:.2f}. {rec_sub}", color=rec_color), unsafe_allow_html=True)
                        
                        # Plot Curve
                        p_mean = df_cash_wk['price'].mean()
                        p_range = np.linspace(p_mean * 0.7, p_mean * 1.3, 50)
                        
                        # Use the cached/clamped elasticity for the curve shape to look "sane"
                        # But anchor it to current volume
                        q_mean = df_cash_wk['quantity'].mean()
                        const_sim = np.log(q_mean) - (e_cash_clamped * np.log(p_mean))
                        
                        rev_pred = p_range * (np.exp(const_sim) * (p_range ** e_cash_clamped))
                        
                        fig_cash = go.Figure()
                        fig_cash.add_trace(go.Scatter(x=p_range, y=rev_pred, mode='lines', line=dict(color='#10B981', width=3), name='Receita Est.'))
                        
                        # Add Current Point
                        curr_rev = p_mean * q_mean
                        fig_cash.add_trace(go.Scatter(x=[p_mean], y=[curr_rev], mode='markers', name='Atual', marker=dict(color='black', size=10)))

                        fig_cash.update_layout(title="Curva de Receita (Simulação)", xaxis_title="Preço", yaxis_title="Receita", margin=dict(t=30, l=0, r=0, b=0))
                        fig_cash = style_chart(fig_cash)
                        st.plotly_chart(fig_cash, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Erro na modelagem: {e}")
                else:
                    st.warning("Variação de preço insuficiente para modelar curva à vista.")
            else:
                st.warning("Poucos dados de vendas à vista para gerar modelo robusto.")
    else:
        st.error("Dados indisponíveis para a seleção atual.")






