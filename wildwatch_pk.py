import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WildWatch PK — Pakistan Wildlife Intelligence System",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500&display=swap');

:root {
    --bg-deep:    #020b08;
    --bg-card:    #041410;
    --bg-glass:   rgba(4,20,16,0.85);
    --accent-g:   #00ff88;
    --accent-g2:  #00d4aa;
    --accent-warn:#ffaa00;
    --accent-red: #ff4444;
    --accent-safe:#00cc66;
    --text-main:  #e8f5f0;
    --text-muted: #5a8a78;
    --border:     rgba(0,255,136,0.12);
    --glow:       0 0 30px rgba(0,255,136,0.15);
}

html, body, [class*="css"] {
    background-color: var(--bg-deep) !important;
    color: var(--text-main) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020f0a 0%, #041410 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-main) !important; }

/* Main header */
.hero-banner {
    background: linear-gradient(135deg, #020f0a 0%, #041a12 50%, #020f0a 100%);
    border: 1px solid rgba(0,255,136,0.2);
    border-radius: 16px;
    padding: 40px 48px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 20% 50%, rgba(0,255,136,0.06) 0%, transparent 60%),
                radial-gradient(ellipse at 80% 20%, rgba(0,212,170,0.04) 0%, transparent 50%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Orbitron', monospace !important;
    font-size: 3rem !important;
    font-weight: 900 !important;
    letter-spacing: 0.08em !important;
    background: linear-gradient(90deg, #00ff88 0%, #00d4aa 50%, #00ff88 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    margin: 0 !important;
    line-height: 1.1 !important;
}
.hero-sub {
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
    color: var(--text-muted) !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    margin-top: 8px !important;
    font-weight: 300 !important;
}
.hero-badge {
    display: inline-block;
    background: rgba(0,255,136,0.1);
    border: 1px solid rgba(0,255,136,0.3);
    color: #00ff88 !important;
    font-family: 'Orbitron', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    padding: 4px 12px;
    border-radius: 2px;
    margin-top: 16px;
}

/* Metric cards */
.metric-grid { display: flex; gap: 16px; margin-bottom: 32px; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 160px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
}
.metric-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
}
.metric-card.green::after  { background: linear-gradient(90deg, transparent, #00ff88, transparent); }
.metric-card.red::after    { background: linear-gradient(90deg, transparent, #ff4444, transparent); }
.metric-card.warn::after   { background: linear-gradient(90deg, transparent, #ffaa00, transparent); }
.metric-card.blue::after   { background: linear-gradient(90deg, transparent, #44aaff, transparent); }
.metric-label { font-size: 0.7rem; letter-spacing: 0.15em; text-transform: uppercase; color: var(--text-muted); }
.metric-value { font-family: 'Orbitron', monospace; font-size: 2rem; font-weight: 700; margin-top: 6px; line-height: 1; }
.metric-card.green .metric-value { color: #00ff88; }
.metric-card.red .metric-value   { color: #ff4444; }
.metric-card.warn .metric-value  { color: #ffaa00; }
.metric-card.blue .metric-value  { color: #44aaff; }

/* Risk pill */
.risk-high   { background: rgba(255,68,68,0.15);  color: #ff6666; border: 1px solid rgba(255,68,68,0.3);  padding: 2px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.08em; display: inline-block; }
.risk-medium { background: rgba(255,170,0,0.15);  color: #ffaa00; border: 1px solid rgba(255,170,0,0.3);  padding: 2px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.08em; display: inline-block; }
.risk-low    { background: rgba(0,255,136,0.12);  color: #00cc88; border: 1px solid rgba(0,255,136,0.3);  padding: 2px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.08em; display: inline-block; }

/* Section headers */
.section-tag {
    font-family: 'Orbitron', monospace;
    font-size: 0.65rem;
    color: var(--accent-g);
    letter-spacing: 0.3em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--text-main);
    margin-bottom: 20px;
}

/* Forest officer alert box */
.alert-box {
    border-radius: 10px;
    padding: 18px 24px;
    margin: 12px 0;
    border-left: 3px solid;
    font-size: 0.9rem;
    line-height: 1.6;
}
.alert-danger { background: rgba(255,68,68,0.08); border-color: #ff4444; }
.alert-warn   { background: rgba(255,170,0,0.08); border-color: #ffaa00; }
.alert-safe   { background: rgba(0,255,136,0.07); border-color: #00cc88; }
.alert-title  { font-weight: 600; font-size: 0.8rem; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 6px; }
.alert-danger .alert-title { color: #ff6666; }
.alert-warn   .alert-title { color: #ffaa00; }
.alert-safe   .alert-title { color: #00cc88; }

/* Divider */
.cyber-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,255,136,0.2), transparent);
    margin: 32px 0;
}

/* Streamlit overrides */
.stSelectbox > div > div { background: var(--bg-card) !important; border-color: var(--border) !important; color: var(--text-main) !important; }
.stButton > button { background: rgba(0,255,136,0.1) !important; border: 1px solid rgba(0,255,136,0.4) !important; color: #00ff88 !important; font-family: 'Orbitron', monospace !important; font-size: 0.75rem !important; letter-spacing: 0.1em !important; border-radius: 6px !important; padding: 8px 20px !important; }
.stButton > button:hover { background: rgba(0,255,136,0.2) !important; }
[data-testid="stMetric"] { background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px; padding: 16px !important; }
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; }
[data-testid="stMetricValue"] { color: #00ff88 !important; font-family: 'Orbitron', monospace !important; }
h1,h2,h3,h4 { font-family: 'Space Grotesk', sans-serif !important; color: var(--text-main) !important; }
.stTabs [data-baseweb="tab"] { font-family: 'Space Grotesk', sans-serif !important; color: var(--text-muted) !important; }
.stTabs [aria-selected="true"] { color: var(--accent-g) !important; border-bottom-color: var(--accent-g) !important; }
[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

# ─── DATA & MODEL ─────────────────────────────────────────────────────────────
@st.cache_data
def load_and_train():
    df_raw = pd.read_csv("animal_dataset.csv")
    df = df_raw.copy()

    # --- Preprocessing (mirrors notebook exactly) ---
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    for col in df.select_dtypes(include='float64').columns:
        df[col] = df[col].fillna(df[col].mean())

    df.drop(columns=['Animal ID', 'Kingdom', 'Phylum', 'Subphylum', 'Class'], inplace=True, errors='ignore')

    df['Min-Population size'] = df['Min-Population size'].astype(str).str.replace(',', '').str.replace(' ', '85000')
    df['Max-Population size'] = df['Max-Population size'].astype(str).str.replace(',', '').str.replace(' ', '85000')
    df['Min-Population size'] = pd.to_numeric(df['Min-Population size'], errors='coerce').fillna(85000).astype(int)
    df['Max-Population size'] = pd.to_numeric(df['Max-Population size'], errors='coerce').fillna(85000).astype(int)

    animal_names = df['Name'].copy()

    # Extinction risk target
    df['Extinction_Risk'] = 0
    df.loc[df['Current_population'] < 10000, 'Extinction_Risk'] = 2
    df.loc[(df['Current_population'] >= 10000) & (df['Current_population'] < 100000), 'Extinction_Risk'] = 1

    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    features = ['Min-Population size', 'Max-Population size', 'Min-Life Span',
                'Max-Life Span', 'Min-Length (M)', 'Min-Wingspan (cm)', 'Current_population']
    X = df[features]
    y = df['Extinction_Risk']

    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)

    preds = rf.predict(X_scaled)
    acc = accuracy_score(y_test, rf.predict(X_test))
    df['prediction'] = preds
    df['Animal_Name'] = animal_names.values
    return df, rf, acc, X_test, y_test, features

df, rf, acc, X_test, y_test, features = load_and_train()
df_raw = pd.read_csv("animal_dataset.csv")

RISK_MAP = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
COLOR_MAP = {0: "#00cc88", 1: "#ffaa00", 2: "#ff4444"}

high_risk  = df[df['prediction'] == 2]
med_risk   = df[df['prediction'] == 1]
low_risk   = df[df['prediction'] == 0]

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 16px 0 24px'>
        <div style='font-family:Orbitron,monospace;font-size:0.9rem;color:#00ff88;letter-spacing:0.2em;'>WILDWATCH</div>
        <div style='font-family:Orbitron,monospace;font-size:0.9rem;color:#00ff88;letter-spacing:0.2em;'>PK</div>
        <div style='color:#5a8a78;font-size:0.7rem;letter-spacing:0.15em;margin-top:6px;'>WILDLIFE INTELLIGENCE SYSTEM</div>
    </div>
    <hr style='border:none;border-top:1px solid rgba(0,255,136,0.15);margin-bottom:24px;'>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "🌿  Mission Control",
        "⚠️  Threat Assessment",
        "🌲  Forest Officer Briefing",
        "🔬  ML Model Analysis",
        "🔍  Species Lookup",
    ], label_visibility="collapsed")

    st.markdown("""<hr style='border:none;border-top:1px solid rgba(0,255,136,0.1);margin:24px 0;'>""", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size:0.7rem;color:#5a8a78;line-height:2;'>
        <div>🤖 Model: Random Forest</div>
        <div>📊 Dataset: 42 Species</div>
        <div>✅ Accuracy: <span style='color:#00ff88;font-family:Orbitron,monospace;'>{acc*100:.1f}%</span></div>
        <div>🇵🇰 Region: Pakistan</div>
        <div>📅 System v2030.1</div>
    </div>
    """, unsafe_allow_html=True)

# ─── PAGE: MISSION CONTROL ────────────────────────────────────────────────────
if "Mission Control" in page:
    st.markdown("""
    <div class="hero-banner">
        <p class="hero-sub">Pakistan Wildlife Intelligence System</p>
        <p class="hero-title">WildWatch PK</p>
        <div class="hero-badge">■ LIVE MONITORING ACTIVE — ML-POWERED EXTINCTION RISK ENGINE</div>
    </div>
    """, unsafe_allow_html=True)

    # Metric cards
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card red">
            <div class="metric-label">High Risk Species</div>
            <div class="metric-value">{len(high_risk)}</div>
        </div>
        <div class="metric-card warn">
            <div class="metric-label">Medium Risk Species</div>
            <div class="metric-value">{len(med_risk)}</div>
        </div>
        <div class="metric-card green">
            <div class="metric-label">Stable Species</div>
            <div class="metric-value">{len(low_risk)}</div>
        </div>
        <div class="metric-card blue">
            <div class="metric-label">Total Species Tracked</div>
            <div class="metric-value">{len(df)}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown('<div class="section-tag">// RISK DISTRIBUTION</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Extinction Risk Breakdown</div>', unsafe_allow_html=True)

        risk_counts = df['prediction'].value_counts().reset_index()
        risk_counts.columns = ['Risk', 'Count']
        risk_counts['Label'] = risk_counts['Risk'].map(RISK_MAP)
        risk_counts['Color'] = risk_counts['Risk'].map(COLOR_MAP)

        fig_donut = go.Figure(go.Pie(
            labels=risk_counts['Label'],
            values=risk_counts['Count'],
            hole=0.65,
            marker=dict(colors=list(risk_counts['Color']), line=dict(color='#020b08', width=3)),
            textinfo='label+percent',
            textfont=dict(family='Space Grotesk', size=12, color='#e8f5f0'),
            hovertemplate="<b>%{label}</b><br>Species: %{value}<extra></extra>"
        ))
        fig_donut.add_annotation(text=f"<b>{len(df)}</b><br><span style='font-size:10'>Species</span>",
                                  x=0.5, y=0.5, showarrow=False,
                                  font=dict(size=20, color='#00ff88', family='Orbitron'))
        fig_donut.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e8f5f0'), showlegend=True,
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#e8f5f0')),
            margin=dict(t=10, b=10, l=10, r=10), height=320
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col2:
        st.markdown('<div class="section-tag">// POPULATION STATUS</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Species with Known Population</div>', unsafe_allow_html=True)

        known_pop = df[df['Current_population'] > 0].sort_values('Current_population')
        known_pop['Risk_Label'] = known_pop['prediction'].map(RISK_MAP)
        known_pop['Color'] = known_pop['prediction'].map(COLOR_MAP)

        fig_bar = go.Figure()
        for risk_level in [2, 1, 0]:
            sub = known_pop[known_pop['prediction'] == risk_level]
            fig_bar.add_trace(go.Bar(
                x=sub['Current_population'],
                y=sub['Animal_Name'],
                name=RISK_MAP[risk_level],
                orientation='h',
                marker_color=COLOR_MAP[risk_level],
                hovertemplate="<b>%{y}</b><br>Population: %{x:,.0f}<extra></extra>"
            ))
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(4,20,16,0.5)',
            font=dict(color='#e8f5f0', family='Space Grotesk'), barmode='stack',
            xaxis=dict(gridcolor='rgba(0,255,136,0.06)', tickformat=',', title='Population'),
            yaxis=dict(gridcolor='rgba(0,255,136,0.06)'),
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#e8f5f0')),
            margin=dict(t=5, b=5, l=5, r=5), height=320
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown('<hr class="cyber-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-tag">// SPECIES CLASS ANALYSIS</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Risk by Animal Class</div>', unsafe_allow_html=True)

    df_raw2 = df_raw.copy()
    df_raw2['prediction'] = df['prediction'].values
    class_risk = df_raw2.groupby('Class')['prediction'].mean().reset_index()
    class_risk.columns = ['Class', 'Avg Risk Score']

    fig_class = px.bar(class_risk.sort_values('Avg Risk Score', ascending=False),
                        x='Class', y='Avg Risk Score',
                        color='Avg Risk Score',
                        color_continuous_scale=[[0,'#00cc88'],[0.5,'#ffaa00'],[1,'#ff4444']],
                        title='')
    fig_class.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(4,20,16,0.5)',
        font=dict(color='#e8f5f0', family='Space Grotesk'),
        xaxis=dict(gridcolor='rgba(0,255,136,0.06)'),
        yaxis=dict(gridcolor='rgba(0,255,136,0.06)', title='Average Risk (0=Low, 2=High)'),
        coloraxis_showscale=False, margin=dict(t=5, b=5), height=300
    )
    st.plotly_chart(fig_class, use_container_width=True)

# ─── PAGE: THREAT ASSESSMENT ──────────────────────────────────────────────────
elif "Threat Assessment" in page:
    st.markdown("""
    <p class="section-tag">// THREAT ASSESSMENT MODULE</p>
    <h1 style='font-family:Orbitron,monospace;font-size:2rem;color:#ff4444;margin-bottom:32px;'>
    ⚠️ Danger vs Safe Species
    </h1>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🔴  HIGH RISK", "🟡  MEDIUM RISK", "🟢  STABLE"])

    with tab1:
        st.markdown('<p style="color:#ff6666;font-size:0.9rem;margin-bottom:16px;">These species require IMMEDIATE intervention. Population critically low — extinction risk is real.</p>', unsafe_allow_html=True)
        for _, row in high_risk.sort_values('Current_population').iterrows():
            pop_text = f"{int(row['Current_population']):,}" if row['Current_population'] > 0 else "Unknown"
            st.markdown(f"""
            <div style='background:rgba(255,68,68,0.07);border:1px solid rgba(255,68,68,0.25);border-radius:10px;padding:16px 20px;margin-bottom:10px;display:flex;align-items:center;gap:20px;'>
                <div style='font-size:1.8rem;'>🚨</div>
                <div style='flex:1;'>
                    <div style='font-weight:600;font-size:1rem;color:#ff8888;'>{row['Animal_Name']}</div>
                    <div style='color:#5a8a78;font-size:0.8rem;margin-top:2px;'>Population: <span style='color:#ff4444;font-family:Orbitron,monospace;'>{pop_text}</span></div>
                </div>
                <span class="risk-high">HIGH RISK</span>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        st.markdown('<p style="color:#ffaa00;font-size:0.9rem;margin-bottom:16px;">These species are under stress. Monitoring and habitat protection needed urgently.</p>', unsafe_allow_html=True)
        for _, row in med_risk.sort_values('Current_population').iterrows():
            pop_text = f"{int(row['Current_population']):,}" if row['Current_population'] > 0 else "Unknown"
            st.markdown(f"""
            <div style='background:rgba(255,170,0,0.07);border:1px solid rgba(255,170,0,0.25);border-radius:10px;padding:16px 20px;margin-bottom:10px;display:flex;align-items:center;gap:20px;'>
                <div style='font-size:1.8rem;'>⚠️</div>
                <div style='flex:1;'>
                    <div style='font-weight:600;font-size:1rem;color:#ffcc66;'>{row['Animal_Name']}</div>
                    <div style='color:#5a8a78;font-size:0.8rem;margin-top:2px;'>Population: <span style='color:#ffaa00;font-family:Orbitron,monospace;'>{pop_text}</span></div>
                </div>
                <span class="risk-medium">MED RISK</span>
            </div>
            """, unsafe_allow_html=True)

    with tab3:
        st.markdown('<p style="color:#00cc88;font-size:0.9rem;margin-bottom:16px;">These species are currently stable. Routine monitoring recommended.</p>', unsafe_allow_html=True)
        for _, row in low_risk.sort_values('Current_population', ascending=False).iterrows():
            pop_text = f"{int(row['Current_population']):,}" if row['Current_population'] > 0 else "Unknown"
            st.markdown(f"""
            <div style='background:rgba(0,255,136,0.05);border:1px solid rgba(0,255,136,0.15);border-radius:10px;padding:16px 20px;margin-bottom:10px;display:flex;align-items:center;gap:20px;'>
                <div style='font-size:1.8rem;'>✅</div>
                <div style='flex:1;'>
                    <div style='font-weight:600;font-size:1rem;color:#88ffcc;'>{row['Animal_Name']}</div>
                    <div style='color:#5a8a78;font-size:0.8rem;margin-top:2px;'>Population: <span style='color:#00cc88;font-family:Orbitron,monospace;'>{pop_text}</span></div>
                </div>
                <span class="risk-low">STABLE</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="cyber-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-tag">// COMPARATIVE VISUALIZATION</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Population vs Risk Level</div>', unsafe_allow_html=True)

    known = df[df['Current_population'] > 0].copy()
    known['Risk_Label'] = known['prediction'].map(RISK_MAP)
    fig_scatter = px.scatter(
        known, x='Animal_Name', y='Current_population',
        color='Risk_Label',
        color_discrete_map={v: COLOR_MAP[k] for k, v in RISK_MAP.items()},
        size='Current_population',
        size_max=50,
        title='',
        labels={'Animal_Name': 'Species', 'Current_population': 'Population'}
    )
    fig_scatter.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(4,20,16,0.5)',
        font=dict(color='#e8f5f0', family='Space Grotesk'),
        xaxis=dict(gridcolor='rgba(0,255,136,0.06)', tickangle=-45),
        yaxis=dict(gridcolor='rgba(0,255,136,0.06)', type='log', title='Population (log scale)'),
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        margin=dict(t=10, b=80), height=400
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ─── PAGE: FOREST OFFICER BRIEFING ────────────────────────────────────────────
elif "Forest Officer" in page:
    st.markdown("""
    <p class="section-tag">// FIELD OPERATIONS MODULE</p>
    <h1 style='font-family:Orbitron,monospace;font-size:2rem;color:#00ff88;margin-bottom:8px;'>
    🌲 Forest Officer Briefing
    </h1>
    <p style='color:#5a8a78;margin-bottom:32px;'>Pakistan Wildlife Department — AI-Generated Field Intelligence Report</p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="alert-box alert-danger">
        <div class="alert-title">🚨 CRITICAL ALERT — IMMEDIATE ACTION REQUIRED</div>
        <p>Our ML model has detected <strong style='color:#ff6666;'>critically endangered species</strong> in Pakistan with populations below 10,000.
        These animals face <strong>real extinction risk</strong> and require emergency protection protocols.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="section-tag">// DANGER ZONE — THESE ANIMALS KO BACHAO</div>
        <div class="section-title" style='color:#ff6666;'>⛔ Khatra Mein Hain (Danger)</div>
        """, unsafe_allow_html=True)

        danger_info = {
            "Snow Leopard": {"pop": "~200", "habitat": "Karakoram, Hindu Kush, Himalayas", "threat": "Poaching, habitat loss, prey decline", "action": "Anti-poaching patrols, corridor protection"},
            "Bengal Tiger": {"pop": "~5,300 (global)", "habitat": "Mangrove forests", "threat": "Habitat destruction, hunting", "action": "Sanctuary expansion, camera traps"},
            "Markhor": {"pop": "~6,000", "habitat": "Chitral, Gilgit-Baltistan rocky terrain", "threat": "Trophy hunting, grazing competition", "action": "Community hunting bans, breeding programs"},
            "Bearded Vulture": {"pop": "~2,500", "habitat": "Northern mountainous regions", "threat": "Poison, lead bullets in carcasses", "action": "Ban lead ammunition, carcass monitoring"},
            "Blue Whale": {"pop": "~25,000", "habitat": "Arabian Sea coast", "threat": "Ship strikes, fishing nets", "action": "Shipping lane restrictions near Karachi"},
            "Mugger Crocodile": {"pop": "~10,000", "habitat": "Indus River, Manchar Lake", "threat": "Habitat loss, hunting for skin", "action": "River protection zones, nest monitoring"},
        }

        for animal, info in danger_info.items():
            with st.expander(f"🔴 {animal} — Pop: {info['pop']}"):
                st.markdown(f"""
                <div style='font-size:0.85rem;line-height:1.8;'>
                    <div><span style='color:#5a8a78;'>📍 Habitat:</span> <span style='color:#e8f5f0;'>{info['habitat']}</span></div>
                    <div><span style='color:#5a8a78;'>⚡ Threats:</span> <span style='color:#ff8888;'>{info['threat']}</span></div>
                    <div><span style='color:#5a8a78;'>✅ Action:</span> <span style='color:#00cc88;'>{info['action']}</span></div>
                </div>
                """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="section-tag">// SAFE ZONE — MONITORING MEIN RAKHEIN</div>
        <div class="section-title" style='color:#00cc88;'>✅ Mehfooz Hain (Safe)</div>
        """, unsafe_allow_html=True)

        safe_info = {
            "Grey Wolf": {"pop": "~3,000,000", "status": "Stable", "note": "Maintain prey base (deer, ibex)"},
            "Golden Eagle": {"pop": "~250,000", "status": "Stable", "note": "Protect nesting cliffs from disturbance"},
            "Brown Bear": {"pop": "~200,000", "status": "Stable", "note": "Seasonal movement corridors needed"},
            "Osprey": {"pop": "~200,000", "status": "Stable", "note": "Waterway pollution monitoring"},
            "Nilgai": {"pop": "~100,000", "status": "Stable", "note": "Grassland conservation"},
            "Eurasian Lynx": {"pop": "~50,000", "status": "Stable", "note": "Forest fragmentation watch"},
        }

        for animal, info in safe_info.items():
            with st.expander(f"🟢 {animal} — Pop: {info['pop']}"):
                st.markdown(f"""
                <div style='font-size:0.85rem;line-height:1.8;'>
                    <div><span style='color:#5a8a78;'>📊 Status:</span> <span style='color:#00cc88;'>{info['status']}</span></div>
                    <div><span style='color:#5a8a78;'>📋 Note:</span> <span style='color:#e8f5f0;'>{info['note']}</span></div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown('<hr class="cyber-divider">', unsafe_allow_html=True)

    st.markdown("""
    <div class="section-tag">// IMMEDIATE PRIORITIES</div>
    <div class="section-title">Field Officer Action Plan</div>
    """, unsafe_allow_html=True)

    priorities = [
        ("🔴", "PRIORITY 1", "Snow Leopard Patrol", "Deploy ranger units in Karakoram. Set up camera traps. Report all sightings within 24 hours.", "danger"),
        ("🔴", "PRIORITY 2", "Markhor Protection", "Coordinate with local communities. Implement hunting bans. Monitor population in Chitral.", "danger"),
        ("🟡", "PRIORITY 3", "Bearded Vulture Survey", "Conduct aerial surveys in northern Pakistan. Monitor feeding grounds for lead poisoning signs.", "warn"),
        ("🟡", "PRIORITY 4", "Mugger Crocodile Census", "Conduct Indus River survey. Mark and monitor nesting sites. Report illegal fishing activity.", "warn"),
        ("🟢", "PRIORITY 5", "Routine Monitoring", "Monthly population checks for all stable species. Update database with field observations.", "safe"),
    ]

    for icon, label, title, desc, style in priorities:
        st.markdown(f"""
        <div class="alert-box alert-{style}">
            <div class="alert-title">{icon} {label} — {title}</div>
            <div style='color:#b0c8bf;font-size:0.85rem;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# ─── PAGE: ML MODEL ANALYSIS ──────────────────────────────────────────────────
elif "ML Model" in page:
    st.markdown("""
    <p class="section-tag">// MACHINE LEARNING ENGINE</p>
    <h1 style='font-family:Orbitron,monospace;font-size:2rem;color:#00ff88;margin-bottom:32px;'>
    🔬 Model Performance Analysis
    </h1>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Model Accuracy", f"{acc*100:.1f}%")
    col2.metric("Algorithm", "Random Forest")
    col3.metric("Trees", "100")

    st.markdown('<hr class="cyber-divider">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-tag">// FEATURE IMPORTANCE</div><div class="section-title">What the Model Uses</div>', unsafe_allow_html=True)
        fi = pd.DataFrame({'Feature': features, 'Importance': rf.feature_importances_}).sort_values('Importance', ascending=True)
        fig_fi = go.Figure(go.Bar(
            x=fi['Importance'], y=fi['Feature'], orientation='h',
            marker=dict(
                color=fi['Importance'],
                colorscale=[[0,'#00cc88'],[1,'#00ff88']],
                line=dict(color='rgba(0,255,136,0.3)', width=1)
            ),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>"
        ))
        fig_fi.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(4,20,16,0.5)',
            font=dict(color='#e8f5f0', family='Space Grotesk'),
            xaxis=dict(gridcolor='rgba(0,255,136,0.06)', title='Importance Score'),
            yaxis=dict(gridcolor='rgba(0,255,136,0.06)'),
            margin=dict(t=5, b=5), height=320
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    with col2:
        st.markdown('<div class="section-tag">// CONFUSION MATRIX</div><div class="section-title">Prediction Accuracy</div>', unsafe_allow_html=True)
        y_pred_test = rf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred_test)
        fig_cm = go.Figure(go.Heatmap(
            z=cm,
            x=['Predicted Low', 'Predicted Med', 'Predicted High'],
            y=['Actual Low', 'Actual Med', 'Actual High'],
            colorscale=[[0,'#020b08'],[0.5,'rgba(0,212,170,0.4)'],[1,'#00ff88']],
            text=cm, texttemplate="%{text}",
            textfont=dict(size=18, family='Orbitron', color='#e8f5f0'),
            hovertemplate="<b>%{y} → %{x}</b><br>Count: %{z}<extra></extra>"
        ))
        fig_cm.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(4,20,16,0.5)',
            font=dict(color='#e8f5f0', family='Space Grotesk'),
            margin=dict(t=5, b=5), height=320
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown('<hr class="cyber-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-tag">// CLASSIFICATION REPORT</div><div class="section-title">Detailed Metrics</div>', unsafe_allow_html=True)

    report = classification_report(y_test, rf.predict(X_test), target_names=['Low Risk', 'Medium Risk', 'High Risk'], output_dict=True)
    report_df = pd.DataFrame(report).transpose().iloc[:3]
    report_df = report_df.round(2)

    col1, col2, col3 = st.columns(3)
    for i, (risk_name, col) in enumerate(zip(['Low Risk', 'Medium Risk', 'High Risk'], [col1, col2, col3])):
        color = ['#00cc88', '#ffaa00', '#ff4444'][i]
        row = report_df.loc[risk_name]
        col.markdown(f"""
        <div style='background:var(--bg-card);border:1px solid rgba({[0,255,136][0] if i==0 else 255},{[255,170,0][0] if i==1 else [68,170,0][0]},{[136,0,0][0]},0.2);
             border-radius:10px;padding:20px;text-align:center;'>
            <div style='color:{color};font-family:Orbitron,monospace;font-size:0.7rem;letter-spacing:0.15em;margin-bottom:10px;'>{risk_name.upper()}</div>
            <div style='display:flex;justify-content:space-around;'>
                <div><div style='color:#5a8a78;font-size:0.65rem;'>PRECISION</div><div style='font-family:Orbitron,monospace;color:{color};font-size:1.2rem;'>{row['precision']:.2f}</div></div>
                <div><div style='color:#5a8a78;font-size:0.65rem;'>RECALL</div><div style='font-family:Orbitron,monospace;color:{color};font-size:1.2rem;'>{row['recall']:.2f}</div></div>
                <div><div style='color:#5a8a78;font-size:0.65rem;'>F1</div><div style='font-family:Orbitron,monospace;color:{color};font-size:1.2rem;'>{row['f1-score']:.2f}</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─── PAGE: SPECIES LOOKUP ─────────────────────────────────────────────────────
elif "Species Lookup" in page:
    st.markdown("""
    <p class="section-tag">// SPECIES INTELLIGENCE LOOKUP</p>
    <h1 style='font-family:Orbitron,monospace;font-size:2rem;color:#00ff88;margin-bottom:32px;'>
    🔍 Individual Species Analysis
    </h1>
    """, unsafe_allow_html=True)

    animal_list = sorted(df['Animal_Name'].unique())
    selected = st.selectbox("Select a Species", animal_list)

    if selected:
        row = df[df['Animal_Name'] == selected].iloc[0]
        raw_row = df_raw[df_raw['Name'] == selected].iloc[0]
        risk_level = int(row['prediction'])
        risk_label = RISK_MAP[risk_level]
        risk_color = COLOR_MAP[risk_level]
        risk_class = ['low', 'medium', 'high'][risk_level]

        st.markdown(f"""
        <div style='background:var(--bg-card);border:1px solid {risk_color}33;border-radius:16px;padding:32px;margin-bottom:24px;'>
            <div style='display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:16px;'>
                <div>
                    <div style='font-family:Orbitron,monospace;font-size:2rem;color:{risk_color};font-weight:700;'>{selected}</div>
                    <div style='color:#5a8a78;margin-top:6px;'>{raw_row.get('Class','—')} · {raw_row.get('Order','—')}</div>
                </div>
                <span class="risk-{risk_class}">{risk_label.upper()}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            pop_val = int(row['Current_population']) if row['Current_population'] > 0 else None
            st.markdown(f"""
            <div style='background:var(--bg-card);border:1px solid var(--border);border-radius:12px;padding:24px;margin-bottom:16px;'>
                <div class='section-tag'>// POPULATION DATA</div>
                <div style='margin-top:12px;display:grid;grid-template-columns:1fr 1fr;gap:12px;'>
                    <div>
                        <div style='color:#5a8a78;font-size:0.7rem;letter-spacing:0.1em;'>CURRENT POP</div>
                        <div style='font-family:Orbitron,monospace;font-size:1.3rem;color:{risk_color};margin-top:4px;'>{f'{pop_val:,}' if pop_val else 'Unknown'}</div>
                    </div>
                    <div>
                        <div style='color:#5a8a78;font-size:0.7rem;letter-spacing:0.1em;'>RISK SCORE</div>
                        <div style='font-family:Orbitron,monospace;font-size:1.3rem;color:{risk_color};margin-top:4px;'>{risk_level}/2</div>
                    </div>
                    <div>
                        <div style='color:#5a8a78;font-size:0.7rem;letter-spacing:0.1em;'>MIN POP SIZE</div>
                        <div style='color:#e8f5f0;margin-top:4px;'>{raw_row.get('Min-Population size','—')}</div>
                    </div>
                    <div>
                        <div style='color:#5a8a78;font-size:0.7rem;letter-spacing:0.1em;'>MAX POP SIZE</div>
                        <div style='color:#e8f5f0;margin-top:4px;'>{raw_row.get('Max-Population size','—')}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div style='background:var(--bg-card);border:1px solid var(--border);border-radius:12px;padding:24px;margin-bottom:16px;'>
                <div class='section-tag'>// BIOLOGICAL DATA</div>
                <div style='margin-top:12px;display:grid;grid-template-columns:1fr 1fr;gap:12px;'>
                    <div>
                        <div style='color:#5a8a78;font-size:0.7rem;letter-spacing:0.1em;'>LIFE SPAN (MIN)</div>
                        <div style='color:#e8f5f0;margin-top:4px;'>{raw_row.get('Min-Life Span','—')} yrs</div>
                    </div>
                    <div>
                        <div style='color:#5a8a78;font-size:0.7rem;letter-spacing:0.1em;'>LIFE SPAN (MAX)</div>
                        <div style='color:#e8f5f0;margin-top:4px;'>{raw_row.get('Max-Life Span','—')} yrs</div>
                    </div>
                    <div>
                        <div style='color:#5a8a78;font-size:0.7rem;letter-spacing:0.1em;'>TOP SPEED</div>
                        <div style='color:#e8f5f0;margin-top:4px;'>{raw_row.get('Top speed km/h','—')} km/h</div>
                    </div>
                    <div>
                        <div style='color:#5a8a78;font-size:0.7rem;letter-spacing:0.1em;'>WEIGHT RANGE</div>
                        <div style='color:#e8f5f0;margin-top:4px;'>{raw_row.get('Min-Weight (kg)','—')}–{raw_row.get('Max-Weight (kg)','—')} kg</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Risk gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_level,
            title={'text': "Extinction Risk Score", 'font': {'family': 'Orbitron', 'color': '#e8f5f0'}},
            delta={'reference': 1, 'font': {'family': 'Orbitron'}},
            gauge={
                'axis': {'range': [0, 2], 'tickfont': {'color': '#5a8a78', 'family': 'Space Grotesk'}},
                'bar': {'color': risk_color},
                'bgcolor': '#041410',
                'bordercolor': '#1a3a30',
                'steps': [
                    {'range': [0, 0.67], 'color': 'rgba(0,204,136,0.1)'},
                    {'range': [0.67, 1.33], 'color': 'rgba(255,170,0,0.1)'},
                    {'range': [1.33, 2], 'color': 'rgba(255,68,68,0.1)'},
                ],
                'threshold': {'line': {'color': risk_color, 'width': 3}, 'thickness': 0.75, 'value': risk_level}
            },
            number={'font': {'family': 'Orbitron', 'color': risk_color}, 'suffix': '/2'}
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e8f5f0'),
            margin=dict(t=20, b=20, l=30, r=30), height=260
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<hr class="cyber-divider">
<div style='text-align:center;color:#2a5a48;font-size:0.7rem;font-family:Orbitron,monospace;letter-spacing:0.2em;padding-bottom:24px;'>
    WILDWATCH PK · ML-POWERED EXTINCTION RISK ENGINE · PAKISTAN WILDLIFE DEPT · v2030.1
</div>
""", unsafe_allow_html=True)
