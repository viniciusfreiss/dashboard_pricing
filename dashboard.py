import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm

# --- Page Configuration ---
st.set_page_config(
    page_title="Price Elasticity Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Modern Minimalist Design System (CSS) ---
# Palette: White, Black, Blue (Primary)
st.markdown("""
<style>
    /* Global Reset & Background */
    .stApp {
        background-color: #FFFFFF; /* Pure White */
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important; /* Pure Black */
        font-weight: 700 !important;
        letter-spacing: -0.025em;
    }
    p, label, .stMarkdown, li {
        color: #000000 !important; /* Pure Black */
        line-height: 1.6;
    }
    
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    div[data-testid="stMetricLabel"] {
        color: #000000 !important; /* Pure Black */
        font-size: 0.875rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetricValue"] {
        color: #000000 !important; /* Pure Black */
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    div[data-testid="stMetricDelta"] {
        font-weight: 600;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        border-radius: 8px;
        color: #000000;
        font-weight: 600;
        border: none;
        background-color: transparent;
        padding: 0 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #EFF6FF; /* Blue 50 */
        color: #2563EB !important; /* Blue 600 */
    }
    
    /* Custom Insight Box */
    .insight-card {
        background-color: #FFFFFF;
        border-left: 4px solid #2563EB;
        padding: 20px;
        border-radius: 0 12px 12px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 32px;
        border: 1px solid #E2E8F0;
    }
    .insight-header {
        color: #000000; /* Black */
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .insight-body {
        color: #000000; /* Black */
        font-size: 1rem;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }
    
    /* Sidebar Logo Container */
    .sidebar-logo {
        text-align: center;
        padding-bottom: 20px;
        border-bottom: 1px solid #E2E8F0;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- Data Loading & Processing ---
@st.cache_data
def load_data():
    file_path = r'c:\Users\vfelisberto\Desktop\elasticidade\db_elasticidade.csv'
    df = pd.read_csv(file_path)
    
    # Clean KPI (Quantity)
    def clean_kpi(val):
        if isinstance(val, (int, float)):
            return float(val)
        val = str(val).strip()
        if ',' in val:
            if val.endswith(',0'):
                val = val[:-2]
            val = val.replace(',', '')
        return float(val)
    
    df['quantity'] = df['kpi'].apply(clean_kpi)
    df['price'] = df['revenue_per_kpi'] # Assuming revenue_per_kpi is Price
    df['date'] = pd.to_datetime(df['time'])
    df = df.sort_values('date')
    
    # Filter valid data
    df = df[(df['quantity'] > 0) & (df['price'] > 0)].copy()
    
    # Log transformation for elasticity
    df['ln_quantity'] = np.log(df['quantity'])
    df['ln_price'] = np.log(df['price'])
    
    # Derived features
    df['month'] = df['date'].dt.month_name()
    df['month_num'] = df['date'].dt.month
    df['revenue'] = df['quantity'] * df['price']
    
    return df

df = load_data()

# --- Analysis (Backend) ---
X = sm.add_constant(df['ln_price'])
y = df['ln_quantity']
model = sm.OLS(y, X).fit()
elasticity = model.params['ln_price']
r2 = model.rsquared

# --- Sidebar ---
with st.sidebar:
    st.markdown('<div class="sidebar-logo">', unsafe_allow_html=True)
    st.image(r"c:\Users\vfelisberto\Desktop\elasticidade\uol_logo.png", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("### Settings")
    rolling_window = st.slider("Rolling Window (Weeks)", 12, 52, 26)
    show_raw_data = st.checkbox("Show Raw Data", False)
    
    st.markdown("---")
    st.info(f"**Data Range**\n{df['date'].min().strftime('%b %Y')} - {df['date'].max().strftime('%b %Y')}")
    
    st.markdown("---")
    st.markdown("""
    <small style="color: #64748B;">
    **About**<br>
    Strategic tool for price scenario simulation and revenue forecasting based on historical elasticity.
    </small>
    """, unsafe_allow_html=True)

# --- Main Content ---

# Header
st.title("Price Elasticity & Revenue Strategy")
st.markdown("Strategic analysis of demand sensitivity to optimize pricing and maximize revenue.")

# Key Insight Section
elasticity_label = "Elastic" if abs(elasticity) > 1 else "Inelastic"

with st.container(border=True):
    st.subheader(f"Key Insight: Demand is Highly {elasticity_label}")
    st.markdown(f"""
    The model calculates a **Price Elasticity of {elasticity:.2f}**. 
    This indicates high sensitivity to price changes. A **10% decrease in price** correlates with a **{abs(elasticity)*10:.1f}% increase in enrollment volume**.
    """)
    
    st.markdown("---")
    st.markdown("**Methodology (Log-Log Regression):**")
    st.latex(r"\ln(Q) = \alpha + \beta \cdot \ln(P) + \epsilon")
    st.markdown(f"""
    **Onde (Variáveis):**
    *   $Q$: Quantidade (Número de Matrículas)
    *   $P$: Preço (Ticket Médio)
    *   $\\beta$: Coeficiente de Elasticidade ({elasticity:.2f})
    *   $\\alpha$: Intercepto (Volume Basal)
    """)

# KPI Cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Price Elasticity", f"{elasticity:.2f}", delta="High Sensitivity" if abs(elasticity) > 1 else "Low Sensitivity", delta_color="inverse")
with col2:
    st.metric("Model Confidence (R²)", f"{r2:.2f}")
with col3:
    avg_price = df['price'].tail(4).mean()
    st.metric("Current Avg Price", f"R$ {avg_price:,.2f}")
with col4:
    avg_rev = df['revenue'].tail(4).mean()
    st.metric("Avg Weekly Revenue", f"R$ {avg_rev:,.2f}")

# Tabs
tab_demand, tab_sim, tab_deep = st.tabs(["Demand Analysis", "Revenue Simulator", "Deep Dive"])

# --- Tab 1: Demand Analysis ---
with tab_demand:
    st.markdown("### Market Demand Behavior")
    
    col_d1, col_d2 = st.columns([2, 1])
    
    with col_d1:
        # Scatter Plot
        fig_scatter = px.scatter(
            df, x='price', y='quantity',
            title="Demand Curve (Price vs. Quantity)",
            labels={'price': 'Price', 'quantity': 'Quantity'},
            trendline="ols",
            trendline_color_override="#2563EB", # Blue 600
            opacity=0.6
        )
        fig_scatter.update_traces(marker=dict(size=10, color='#64748B', line=dict(width=1, color='white')))
        fig_scatter.update_layout(
            template="plotly_white",
            height=450,
            font=dict(family="Inter", color="#000000"),
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor='#F1F5F9', title_font=dict(color='#000000'), tickfont=dict(color='#000000')),
            yaxis=dict(showgrid=True, gridcolor='#F1F5F9', title_font=dict(color='#000000'), tickfont=dict(color='#000000')),
            title_font=dict(color='#000000')
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        st.info(f"""
        **Methodology (Power Law):**
        The demand curve follows a Power Law relationship:
        $$ Q = A \\cdot P^{{\\beta}} $$
        
        **Variáveis:**
        *   $Q$: Quantidade Demandada
        *   $P$: Preço
        *   $\\beta$: Elasticidade ({elasticity:.2f})
        *   $A$: Constante de Escala
        """)
        
    with col_d2:
        st.markdown("#### Trend Overview")
        # Price Trend
        fig_price = px.line(df, x='date', y='price', title="Price Evolution")
        fig_price.update_traces(line_color='#000000', line_width=2)
        fig_price.update_layout(
            template="plotly_white", height=200,
            font=dict(family="Inter", size=10, color="#000000"),
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis=dict(showgrid=False, tickfont=dict(color='#000000')), 
            yaxis=dict(showgrid=True, gridcolor='#F1F5F9', tickfont=dict(color='#000000')),
            plot_bgcolor="white", paper_bgcolor="white", title_font=dict(color='#000000')
        )
        st.plotly_chart(fig_price, use_container_width=True)
        
        # Quantity Trend
        fig_qty = px.line(df, x='date', y='quantity', title="Quantity Evolution")
        fig_qty.update_traces(line_color='#2563EB', line_width=2)
        fig_qty.update_layout(
            template="plotly_white", height=200,
            font=dict(family="Inter", size=10, color="#000000"),
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis=dict(showgrid=False, tickfont=dict(color='#000000')), 
            yaxis=dict(showgrid=True, gridcolor='#F1F5F9', tickfont=dict(color='#000000')),
            plot_bgcolor="white", paper_bgcolor="white", title_font=dict(color='#000000')
        )
        st.plotly_chart(fig_qty, use_container_width=True)
        
        st.markdown("""
        <small><strong>Observation:</strong> Inverse correlation visible: $P \\uparrow \\Rightarrow Q \\downarrow$</small>
        """, unsafe_allow_html=True)

# --- Tab 2: Revenue Simulator ---
with tab_sim:
    st.markdown("### Interactive Scenario Planner")
    
    col_s1, col_s2 = st.columns([1, 2])
    
    with col_s1:
        st.markdown("""
        <div style="background-color: #FFFFFF; padding: 20px; border-radius: 12px; border: 1px solid #E2E8F0;">
            <h4 style="margin-top:0; color: #000000;">Controls</h4>
            <p style="font-size: 0.9rem; color: #000000;">Adjust price to simulate revenue impact.</p>
        </div>
        """, unsafe_allow_html=True)
        
        price_change_pct = st.slider("Price Adjustment (%)", -30, 30, 0, 1)
        
        # Calculations
        base_price = df['price'].tail(4).mean()
        base_qty = df['quantity'].tail(4).mean()
        base_rev = base_price * base_qty
        
        new_price = base_price * (1 + price_change_pct/100)
        new_qty = base_qty * ((new_price / base_price) ** elasticity)
        new_rev = new_price * new_qty
        rev_change_pct = (new_rev - base_rev) / base_rev * 100
        rev_delta_abs = new_rev - base_rev
        
        st.markdown("---")
        st.metric("Projected Revenue", f"R$ {new_rev:,.2f}", f"{rev_change_pct:+.2f}%")
        st.metric("New Price", f"R$ {new_price:,.2f}")
        st.metric("Projected Volume", f"{new_qty:,.0f}")
        
        st.info(f"""
        **Prediction Formula:**
        $$ Q_{{new}} = Q_{{base}} \\cdot \\left( \\frac{{P_{{new}}}}{{P_{{base}}}} \\right)^{{\\beta}} $$
        
        **Variáveis:**
        *   $Q_{{new}}$: Quantidade Projetada
        *   $Q_{{base}}$: Quantidade Atual ({base_qty:,.0f})
        *   $P_{{new}}$: Novo Preço (R$ {new_price:,.2f})
        *   $P_{{base}}$: Preço Atual (R$ {base_price:,.2f})
        *   $\\beta$: Elasticidade ({elasticity:.2f})
        """)

    with col_s2:
        # Revenue Curve
        changes = np.arange(-0.30, 0.35, 0.05)
        scenarios = []
        for chg in changes:
            p = base_price * (1 + chg)
            q = base_qty * ((p / base_price) ** elasticity)
            r = p * q
            scenarios.append({'Price': p, 'Revenue': r})
        
        df_sens = pd.DataFrame(scenarios)
        
        fig_curve = px.line(df_sens, x='Price', y='Revenue', title="Revenue Optimization Curve")
        fig_curve.update_traces(line_color='#000000', line_width=3)
        
        # Current vs Selected
        fig_curve.add_vline(x=base_price, line_dash="dash", line_color="#94A3B8", annotation_text="Current")
        fig_curve.add_trace(go.Scatter(
            x=[new_price], y=[new_rev], mode='markers',
            marker=dict(color='#2563EB', size=15, symbol='circle', line=dict(width=2, color='white')),
            name='Selected'
        ))
        
        fig_curve.update_layout(
            template="plotly_white",
            height=400,
            font=dict(family="Inter", color="#000000"),
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor='#F1F5F9', title_font=dict(color='#000000'), tickfont=dict(color='#000000')),
            yaxis=dict(showgrid=True, gridcolor='#F1F5F9', title_font=dict(color='#000000'), tickfont=dict(color='#000000')),
            title_font=dict(color='#000000'),
            legend=dict(font=dict(color='#000000'))
        )
        st.plotly_chart(fig_curve, use_container_width=True)
        
        st.markdown("""
        **Interpretation:**
        The curve illustrates the trade-off between Price ($P$) and Quantity ($Q$) on Total Revenue ($R = P \\cdot Q$). The peak represents the revenue-maximizing price point given the calculated elasticity.
        """)

# --- Tab 3: Deep Dive ---
with tab_deep:
    col_dd1, col_dd2 = st.columns(2)
    
    with col_dd1:
        st.markdown("#### Seasonality Analysis")
        df_sorted = df.sort_values('month_num')
        fig_box = px.box(
            df_sorted, x='month', y='quantity',
            color_discrete_sequence=['#000000']
        )
        fig_box.update_layout(
            template="plotly_white", height=350,
            xaxis_title=None, yaxis_title="Enrollment Volume",
            font=dict(family="Inter", color="#000000"),
            plot_bgcolor="white",
            paper_bgcolor="white",
            yaxis=dict(gridcolor='#F1F5F9', tickfont=dict(color='#000000'), title_font=dict(color='#000000')),
            xaxis=dict(tickfont=dict(color='#000000'))
        )
        st.plotly_chart(fig_box, use_container_width=True)
        
        st.markdown("""
        **Interpretation:**
        Displays the distribution of enrollments per month. The box represents the Interquartile Range (IQR), and the central line indicates the median volume.
        """)
        
    with col_dd2:
        st.markdown(f"#### Rolling Elasticity ({rolling_window}-Week Window)")
        
        rolling_elast = []
        dates = []
        for i in range(len(df) - rolling_window):
            w = df.iloc[i : i + rolling_window]
            if len(w) < 10: continue
            try:
                m = sm.OLS(w['ln_quantity'], sm.add_constant(w['ln_price'])).fit()
                rolling_elast.append(m.params['ln_price'])
                dates.append(w['date'].iloc[-1])
            except: pass
            
        df_roll = pd.DataFrame({'Date': dates, 'Elasticity': rolling_elast})
        
        fig_roll = px.line(df_roll, x='Date', y='Elasticity')
        fig_roll.add_hline(y=-1, line_dash="dash", line_color="#EF4444", annotation_text="Unit Elasticity")
        fig_roll.update_traces(line_color='#2563EB', line_width=2)
        fig_roll.update_layout(
            template="plotly_white", height=350,
            xaxis_title=None,
            font=dict(family="Inter", color="#000000"),
            plot_bgcolor="white",
            paper_bgcolor="white",
            yaxis=dict(gridcolor='#F1F5F9', tickfont=dict(color='#000000'), title_font=dict(color='#000000')),
            xaxis=dict(tickfont=dict(color='#000000'))
        )
        st.plotly_chart(fig_roll, use_container_width=True)
        
        st.markdown("""
        **Methodology:**
        Elasticity $\\beta_t$ is recalculated for each moving window $t$:
        $$ \\ln(Q_t) = \\alpha_t + \\beta_t \\ln(P_t) $$
        
        **Variáveis:**
        *   $Q_t$: Quantidade na janela $t$
        *   $P_t$: Preço na janela $t$
        *   $\\beta_t$: Elasticidade Móvel
        """)

if show_raw_data:
    st.markdown("### Dados Brutos")
    st.dataframe(df)
