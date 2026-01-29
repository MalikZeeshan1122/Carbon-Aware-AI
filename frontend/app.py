import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- Configuration ---
API_URL = "http://localhost:8000"
st.set_page_config(
    page_title="Carbon-Aware AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Professional Light Theme ---
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }
    
    /* Professional Cards */
    .glass-card {
        background: #ffffff;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        padding: 24px;
        margin: 10px 0;
        transition: all 0.2s ease;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
        border-color: #3b82f6;
    }
    
    /* Metric Display */
    .metric-value {
        font-size: 2.5em;
        font-weight: 700;
        color: #0f172a;
        text-shadow: none;
    }
    .metric-label {
        font-size: 0.85em;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
        font-weight: 500;
    }
    .metric-unit {
        font-size: 0.8em;
        color: #3b82f6;
    }
    
    /* Hero Section */
    .hero-title {
        font-size: 2.8em;
        font-weight: 700;
        color: #0f172a;
        text-align: center;
        margin-bottom: 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: #64748b;
        text-align: center;
        font-size: 1.1em;
        margin-top: 10px;
    }
    
    /* Recommendation Badge */
    .rec-badge {
        background: #3b82f6;
        padding: 12px 24px;
        border-radius: 8px;
        color: white;
        font-weight: 600;
        display: inline-block;
    }
    
    /* Savings Highlight */
    .savings-highlight {
        background: #0d9488;
        color: #ffffff;
        padding: 18px;
        border-radius: 10px;
        font-size: 1.4em;
        font-weight: 600;
        text-align: center;
        margin: 15px 0;
    }
    
    /* Status Indicators */
    .status-good { color: #16a34a; }
    .status-warning { color: #ca8a04; }
    .status-bad { color: #dc2626; }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: #f1f5f9;
        border-radius: 10px;
        padding: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #64748b;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: #3b82f6;
        color: #ffffff !important;
    }
    
    /* Button Styling */
    .stButton > button {
        background: #3b82f6;
        color: #ffffff;
        border: none;
        border-radius: 8px;
        padding: 12px 28px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: #2563eb;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    /* Info boxes */
    .info-box {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 15px;
        border-radius: 0 8px 8px 0;
        color: #1e40af;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        padding: 20px;
        border-top: 1px solid #e2e8f0;
        margin-top: 40px;
    }
    
    /* Override Streamlit defaults for light theme */
    .stMarkdown, .stText, p, span, label {
        color: #334155 !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #0f172a !important;
    }
    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #475569 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State for History ---
if 'history' not in st.session_state:
    st.session_state.history = []

# --- Model Info Database (2026 Models) ---
MODEL_INFO = {
    # OpenAI Models
    "gpt-5": {"params": "~3T", "provider": "OpenAI", "tier": "Frontier"},
    "gpt-4.5-turbo": {"params": "~2T", "provider": "OpenAI", "tier": "Frontier"},
    "gpt-4o": {"params": "~1.8T", "provider": "OpenAI", "tier": "Frontier"},
    "gpt-4o-mini": {"params": "~500B", "provider": "OpenAI", "tier": "Standard"},
    "gpt-4-turbo": {"params": "~1.7T", "provider": "OpenAI", "tier": "Standard"},
    "o1": {"params": "~2T", "provider": "OpenAI", "tier": "Reasoning"},
    "o1-mini": {"params": "~500B", "provider": "OpenAI", "tier": "Reasoning"},
    "o1-pro": {"params": "~3T", "provider": "OpenAI", "tier": "Reasoning"},
    "o3": {"params": "~4T", "provider": "OpenAI", "tier": "Frontier"},
    "o3-mini": {"params": "~1T", "provider": "OpenAI", "tier": "Reasoning"},
    
    # Anthropic Models
    "claude-4-opus": {"params": "~3T", "provider": "Anthropic", "tier": "Frontier"},
    "claude-4-sonnet": {"params": "~1T", "provider": "Anthropic", "tier": "Frontier"},
    "claude-3.5-opus": {"params": "~2.5T", "provider": "Anthropic", "tier": "Frontier"},
    "claude-3.5-sonnet": {"params": "~175B", "provider": "Anthropic", "tier": "Standard"},
    "claude-3.5-haiku": {"params": "~35B", "provider": "Anthropic", "tier": "Efficient"},
    "claude-3-opus": {"params": "~2T", "provider": "Anthropic", "tier": "Frontier"},
    "claude-3-sonnet": {"params": "~70B", "provider": "Anthropic", "tier": "Standard"},
    "claude-3-haiku": {"params": "~20B", "provider": "Anthropic", "tier": "Efficient"},
    
    # Google Models
    "gemini-2.0-ultra": {"params": "~2T", "provider": "Google", "tier": "Frontier"},
    "gemini-2.0-pro": {"params": "~1T", "provider": "Google", "tier": "Frontier"},
    "gemini-2.0-flash": {"params": "~200B", "provider": "Google", "tier": "Standard"},
    "gemini-1.5-ultra": {"params": "~1.5T", "provider": "Google", "tier": "Frontier"},
    "gemini-1.5-pro": {"params": "~500B", "provider": "Google", "tier": "Standard"},
    "gemini-1.5-flash": {"params": "~150B", "provider": "Google", "tier": "Efficient"},
    
    # Meta Models
    "llama-4-405b": {"params": "405B", "provider": "Meta", "tier": "Open Source"},
    "llama-4-70b": {"params": "70B", "provider": "Meta", "tier": "Open Source"},
    "llama-4-8b": {"params": "8B", "provider": "Meta", "tier": "Efficient"},
    "llama-3.3-70b": {"params": "70B", "provider": "Meta", "tier": "Open Source"},
    "llama-3.2-90b-vision": {"params": "90B", "provider": "Meta", "tier": "Multimodal"},
    "llama-3.1-405b": {"params": "405B", "provider": "Meta", "tier": "Open Source"},
    
    # Mistral Models
    "mistral-large-2": {"params": "~200B", "provider": "Mistral AI", "tier": "Frontier"},
    "mistral-medium": {"params": "~123B", "provider": "Mistral AI", "tier": "Standard"},
    "mistral-small": {"params": "~22B", "provider": "Mistral AI", "tier": "Efficient"},
    "mixtral-8x22b": {"params": "176B", "provider": "Mistral AI", "tier": "Standard"},
    "mixtral-8x7b": {"params": "56B", "provider": "Mistral AI", "tier": "Efficient"},
    "codestral": {"params": "~22B", "provider": "Mistral AI", "tier": "Code"},
    
    # Cohere Models
    "command-r-plus": {"params": "~104B", "provider": "Cohere", "tier": "Standard"},
    "command-r": {"params": "~35B", "provider": "Cohere", "tier": "Efficient"},
    
    # xAI Models
    "grok-3": {"params": "~2T", "provider": "xAI", "tier": "Frontier"},
    "grok-2": {"params": "~500B", "provider": "xAI", "tier": "Standard"},
    
    # Amazon Models
    "amazon-nova-pro": {"params": "~300B", "provider": "Amazon", "tier": "Standard"},
    "amazon-nova-lite": {"params": "~70B", "provider": "Amazon", "tier": "Efficient"},
    "amazon-titan-express": {"params": "~50B", "provider": "Amazon", "tier": "Efficient"},
    
    # DeepSeek Models
    "deepseek-v3": {"params": "~671B", "provider": "DeepSeek", "tier": "Open Source"},
    "deepseek-coder-v2": {"params": "~236B", "provider": "DeepSeek", "tier": "Code"},
    "deepseek-r1": {"params": "~671B", "provider": "DeepSeek", "tier": "Reasoning"},
    
    # Alibaba Models
    "qwen-2.5-max": {"params": "~500B", "provider": "Alibaba", "tier": "Standard"},
    "qwen-2.5-72b": {"params": "72B", "provider": "Alibaba", "tier": "Open Source"},
    "qwen-2.5-coder-32b": {"params": "32B", "provider": "Alibaba", "tier": "Code"},
    
    # Efficient/Micro Models
    "phi-4": {"params": "14B", "provider": "Microsoft", "tier": "Efficient"},
    "phi-3-medium": {"params": "14B", "provider": "Microsoft", "tier": "Efficient"},
    "phi-3-mini": {"params": "3.8B", "provider": "Microsoft", "tier": "Micro"},
    "gemma-2-27b": {"params": "27B", "provider": "Google", "tier": "Efficient"},
    "gemma-2-9b": {"params": "9B", "provider": "Google", "tier": "Efficient"},
    "distilbert": {"params": "66M", "provider": "Hugging Face", "tier": "Micro"},
    "flan-t5-xxl": {"params": "11B", "provider": "Google", "tier": "Efficient"}
}

# --- Sidebar ---
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")
    
    # Model Selection with Info
    model_name = st.selectbox(
        "🤖 AI Model",
        list(MODEL_INFO.keys()),
        format_func=lambda x: f"{x} ({MODEL_INFO[x]['tier']})"
    )
    
    # Show model info
    info = MODEL_INFO[model_name]
    st.markdown(f"""
    <div class="info-box">
        <strong>Provider:</strong> {info['provider']}<br>
        <strong>Parameters:</strong> {info['params']}<br>
        <strong>Tier:</strong> {info['tier']}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    task_type = st.selectbox(
        "📋 Task Type",
        ["Classification", "Summarization", "Reasoning", "Translation", "Extraction", "Creative Writing", "Code Generation", "Simple Q&A", "Complex Analysis"]
    )
    
    region = st.selectbox(
        "🌍 Cloud Region",
        ["Global", "US-East", "US-West", "EU-West", "AsiaPac"],
        help="Different regions have different carbon intensities based on their energy mix."
    )
    
    st.markdown("---")
    st.markdown("### 📊 Token Estimation")
    
    prompt_tokens = st.slider("Prompt Tokens", 10, 10000, 500, 50)
    completion_tokens = st.slider("Completion Tokens", 10, 5000, 200, 50)
    
    # Batch Mode
    st.markdown("---")
    st.markdown("### 🔄 Batch Mode")
    batch_queries = st.number_input("Number of Queries", min_value=1, max_value=10000, value=1, step=10)
    
    st.markdown("---")
    estimate_btn = st.button("🌿 Estimate Impact", type="primary", use_container_width=True)
    
    # Developer Info
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-top: 10px;">
        <p style="color: white; margin: 0; font-size: 0.9em;">Developed by</p>
        <p style="color: white; margin: 5px 0; font-weight: 700; font-size: 1.1em;">Muhammad Zeeshan</p>
        <a href="https://www.linkedin.com/in/muhammadzeeshan007/" target="_blank" style="text-decoration: none;">
            <div style="background: white; color: #0077B5; padding: 8px 16px; border-radius: 5px; display: inline-block; font-weight: 600; margin-top: 5px;">
                🔗 Connect on LinkedIn
            </div>
        </a>
    </div>
    """, unsafe_allow_html=True)

# --- Main Content ---
st.markdown('<h1 class="hero-title">🌱 Carbon-Aware AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Make the Climate Impact of AI Visible, Measurable, and Actionable</p>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📊 Text AI", "🖼️ Image AI", "🎬 Video AI", "🏢 Data Centers", 
    "⚡ Live Carbon", "🌳 Offsets", "🏆 Analytics", "📈 History", "📚 Learn"
])

with tab1:
    if estimate_btn:
        try:
            # 1. Get Estimation
            est_resp = requests.post(f"{API_URL}/estimate", json={
                "model_name": model_name,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "region": region
            })
            est_data = est_resp.json()
            if 'error' in est_data:
                st.error(f"Estimation error: {est_data['error']}")
                st.stop()
            # Apply batch multiplier
            est_data['energy_kwh'] *= batch_queries
            est_data['co2_grams'] *= batch_queries
            # 2. Get Recommendation
            rec_resp = requests.post(f"{API_URL}/recommend", json={
                "task_type": task_type,
                "current_model": model_name
            })
            rec_data = rec_resp.json()
            # Save to history
            st.session_state.history.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "model": model_name,
                "task": task_type,
                "tokens": prompt_tokens + completion_tokens,
                "co2": est_data['co2_grams'],
                "queries": batch_queries
            })
            # --- Display Results ---
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### 📊 Carbon Footprint Analysis")
                
                # Metrics Row
                m1, m2, m3, m4 = st.columns(4)
                
                with m1:
                    st.markdown(f"""
                    <div class="glass-card">
                        <div class="metric-label">Energy Consumed</div>
                        <div class="metric-value">{est_data['energy_kwh']:.6f}</div>
                        <div class="metric-unit">kWh</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with m2:
                    st.markdown(f"""
                    <div class="glass-card">
                        <div class="metric-label">CO₂ Emissions</div>
                        <div class="metric-value">{est_data['co2_grams']:.4f}</div>
                        <div class="metric-unit">grams</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with m3:
                    # Real-world equivalents
                    phones = est_data['energy_kwh'] / 0.005
                    st.markdown(f"""
                    <div class="glass-card">
                        <div class="metric-label">Equivalent To</div>
                        <div class="metric-value">{phones:.2f}</div>
                        <div class="metric-unit">📱 Phone Charges</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with m4:
                    # Tree offset (1 tree absorbs ~21kg CO2/year)
                    trees_needed = (est_data['co2_grams'] / 1000) / 21 * 365
                    st.markdown(f"""
                    <div class="glass-card">
                        <div class="metric-label">Tree-Days Needed</div>
                        <div class="metric-value">{trees_needed:.4f}</div>
                        <div class="metric-unit">🌳 to offset</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Methodology Info
                st.markdown(f"""
                <div class="info-box">
                    <strong>📐 Methodology:</strong> {est_data['methodology']}
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                st.markdown("### 💡 Smart Recommendation")
                
                if rec_data['recommended_model'].lower() != model_name.lower():
                    st.markdown(f"""
                    <div class="glass-card" style="border-color: #eab308;">
                        <div class="metric-label" style="color: #ca8a04;">⚠️ Optimization Available</div>
                        <p style="color: #334155; margin: 15px 0;">Switch to:</p>
                        <div class="rec-badge">{rec_data['recommended_model'].upper()}</div>
                        <p style="color: #64748b; margin-top: 15px; font-size: 0.9em;">{rec_data['reason']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="savings-highlight">
                        🎯 Potential Savings: {rec_data['savings_percentage']}%
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="glass-card" style="border-color: #16a34a;">
                        <div class="metric-label status-good">✅ Optimal Choice</div>
                        <p style="color: #334155;">You're using an efficient model for this task!</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # --- Visualizations ---
            st.markdown("---")
            st.markdown("### 📈 Visual Analysis")
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                # Model Comparison Chart
                models_to_compare = ["gpt-4", "gpt-3.5-turbo", "llama-3-70b", "llama-3-8b", "distilbert"]
                comparison_data = []
                
                for m in models_to_compare:
                    resp = requests.post(f"{API_URL}/estimate", json={
                        "model_name": m,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "region": region
                    })
                    data = resp.json()
                    if 'error' in data:
                        st.warning(f"Model {m}: {data['error']}")
                        continue
                    # Handle missing model info gracefully
                    tier = MODEL_INFO.get(m, {}).get('tier', 'Unknown')
                    comparison_data.append({
                        "Model": m,
                        "CO2 (g)": data['co2_grams'] * batch_queries,
                        "Energy (Wh)": data['energy_kwh'] * 1000 * batch_queries,
                        "Tier": tier
                    })
                
                df_compare = pd.DataFrame(comparison_data)
                
                fig = px.bar(
                    df_compare, 
                    x="Model", 
                    y="CO2 (g)", 
                    color="Tier",
                    title="🔍 Model Emissions Comparison",
                    color_discrete_map={
                        "Frontier": "#dc2626",
                        "Standard": "#f59e0b", 
                        "Open Source": "#3b82f6",
                        "Efficient": "#16a34a",
                        "Micro": "#0891b2"
                    }
                )
                fig.update_layout(
                    plot_bgcolor='#ffffff',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#334155',
                    title_font_color='#0f172a'
                )
                st.plotly_chart(fig, use_container_width=True)
                
            with viz_col2:
                # Regional Comparison
                regions = ["Global", "US-East", "US-West", "EU-West", "AsiaPac"]
                regional_data = []
                
                for r in regions:
                    resp = requests.post(f"{API_URL}/estimate", json={
                        "model_name": model_name,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "region": r
                    })
                    data = resp.json()
                    if 'error' in data:
                        st.warning(f"Region {r}: {data['error']}")
                        continue
                    regional_data.append({
                        "Region": r,
                        "CO2 (g)": data['co2_grams'] * batch_queries
                    })
                
                df_region = pd.DataFrame(regional_data)
                
                fig2 = px.bar(
                    df_region,
                    x="Region",
                    y="CO2 (g)",
                    title="🌍 Regional Carbon Intensity Impact",
                    color="CO2 (g)",
                    color_continuous_scale=["#16a34a", "#eab308", "#dc2626"]
                )
                fig2.update_layout(
                    plot_bgcolor='#ffffff',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#334155',
                    title_font_color='#0f172a'
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # --- Advanced Visualizations Row 2 ---
            st.markdown("### 🎨 Advanced Analytics")
            
            adv_col1, adv_col2, adv_col3 = st.columns(3)
            
            with adv_col1:
                # Gauge Chart for Current Emission Level
                max_co2 = max(d['CO2 (g)'] for d in comparison_data)
                current_co2 = est_data['co2_grams']
                
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=current_co2,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "CO₂ Emission Level", 'font': {'size': 16, 'color': '#0f172a'}},
                    delta={'reference': df_compare['CO2 (g)'].mean(), 'relative': True, 'valueformat': '.1%'},
                    gauge={
                        'axis': {'range': [0, max_co2], 'tickcolor': '#64748b'},
                        'bar': {'color': "#3b82f6"},
                        'bgcolor': "#f1f5f9",
                        'borderwidth': 2,
                        'bordercolor': "#e2e8f0",
                        'steps': [
                            {'range': [0, max_co2 * 0.3], 'color': '#dcfce7'},
                            {'range': [max_co2 * 0.3, max_co2 * 0.7], 'color': '#fef3c7'},
                            {'range': [max_co2 * 0.7, max_co2], 'color': '#fee2e2'}
                        ],
                        'threshold': {
                            'line': {'color': "#16a34a", 'width': 4},
                            'thickness': 0.75,
                            'value': df_compare['CO2 (g)'].min()
                        }
                    }
                ))
                fig_gauge.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    font={'color': '#334155'},
                    height=300
                )
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with adv_col2:
                # Radar Chart - Model Efficiency Profile
                categories = ['Energy Efficiency', 'Cost Efficiency', 'Carbon Score', 'Speed', 'Quality']
                
                # Simulate efficiency scores based on model tier
                tier_scores = {
                    "Frontier": [20, 30, 25, 70, 100],
                    "Standard": [50, 60, 55, 80, 85],
                    "Open Source": [65, 80, 70, 75, 80],
                    "Efficient": [85, 90, 88, 90, 70],
                    "Micro": [100, 100, 95, 100, 50]
                }
                current_tier = MODEL_INFO[model_name]['tier']
                scores = tier_scores.get(current_tier, [50, 50, 50, 50, 50])
                
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=scores,
                    theta=categories,
                    fill='toself',
                    name=model_name,
                    line_color='#3b82f6',
                    fillcolor='rgba(59,130,246,0.3)'
                ))
                
                # Add comparison with best efficient model
                fig_radar.add_trace(go.Scatterpolar(
                    r=tier_scores["Efficient"],
                    theta=categories,
                    fill='toself',
                    name='Efficient Baseline',
                    line_color='#22c55e',
                    fillcolor='rgba(34,197,94,0.1)'
                ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 100], tickfont={'color': '#64748b'}),
                        bgcolor='#f8fafc'
                    ),
                    showlegend=True,
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#334155',
                    title={'text': '🎯 Model Efficiency Profile', 'font': {'color': '#0f172a'}},
                    height=300,
                    legend=dict(font=dict(color='#334155'))
                )
                st.plotly_chart(fig_radar, use_container_width=True)
            
            with adv_col3:
                # Pie Chart - Emission Breakdown
                emission_breakdown = {
                    'GPU Compute': est_data['energy_kwh'] * 0.7 * 1000,
                    'Memory Access': est_data['energy_kwh'] * 0.15 * 1000,
                    'Network': est_data['energy_kwh'] * 0.05 * 1000,
                    'Cooling (PUE)': est_data['energy_kwh'] * 0.1 * 1000
                }
                
                fig_pie = px.pie(
                    values=list(emission_breakdown.values()),
                    names=list(emission_breakdown.keys()),
                    title='⚡ Energy Breakdown',
                    color_discrete_sequence=['#3b82f6', '#0d9488', '#eab308', '#64748b'],
                    hole=0.4
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_pie.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#334155',
                    title_font_color='#0f172a',
                    height=300,
                    showlegend=False
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # --- Advanced Visualizations Row 3 ---
            adv_row2_col1, adv_row2_col2 = st.columns(2)
            
            with adv_row2_col1:
                # Treemap - Model Categories
                treemap_data = []
                for m in models_to_compare:
                    tier = MODEL_INFO[m]['tier']
                    co2 = next((d['CO2 (g)'] for d in comparison_data if d['Model'] == m), 0)
                    treemap_data.append({
                        'Model': m,
                        'Tier': tier,
                        'CO2': co2,
                        'Provider': MODEL_INFO[m]['provider']
                    })
                
                df_tree = pd.DataFrame(treemap_data)
                
                fig_tree = px.treemap(
                    df_tree,
                    path=['Tier', 'Provider', 'Model'],
                    values='CO2',
                    color='CO2',
                    color_continuous_scale=['#22c55e', '#eab308', '#dc2626'],
                    title='🌳 Model Hierarchy by Emissions'
                )
                fig_tree.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#334155',
                    title_font_color='#0f172a',
                    height=350
                )
                st.plotly_chart(fig_tree, use_container_width=True)
            
            with adv_row2_col2:
                # Scatter Plot - Energy vs CO2 with bubble size
                scatter_data = []
                for m in models_to_compare:
                    co2 = next((d['CO2 (g)'] for d in comparison_data if d['Model'] == m), 0)
                    energy = next((d['Energy (Wh)'] for d in comparison_data if d['Model'] == m), 0)
                    tier = MODEL_INFO[m]['tier']
                    params = MODEL_INFO[m]['params']
                    scatter_data.append({
                        'Model': m,
                        'CO2 (g)': co2,
                        'Energy (Wh)': energy,
                        'Tier': tier,
                        'Size': co2 * 10 + 10  # Bubble size
                    })
                
                df_scatter = pd.DataFrame(scatter_data)
                
                fig_scatter = px.scatter(
                    df_scatter,
                    x='Energy (Wh)',
                    y='CO2 (g)',
                    size='Size',
                    color='Tier',
                    hover_name='Model',
                    title='🔬 Energy vs Emissions Analysis',
                    color_discrete_map={
                        "Frontier": "#dc2626",
                        "Standard": "#f59e0b", 
                        "Open Source": "#3b82f6",
                        "Efficient": "#16a34a",
                        "Micro": "#0891b2"
                    }
                )
                fig_scatter.update_layout(
                    plot_bgcolor='#ffffff',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#334155',
                    title_font_color='#0f172a',
                    height=350
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            # --- Waterfall & Funnel Charts ---
            st.markdown("### 📊 Detailed Breakdown")
            
            waterfall_col, funnel_col = st.columns(2)
            
            with waterfall_col:
                # Waterfall Chart - Emission Factors
                base_co2 = est_data['co2_grams'] / 1.5  # Remove PUE effect
                pue_addition = base_co2 * 0.2
                region_factor = base_co2 * 0.3
                
                fig_waterfall = go.Figure(go.Waterfall(
                    name="CO₂ Breakdown",
                    orientation="v",
                    measure=["absolute", "relative", "relative", "total"],
                    x=["Base GPU", "PUE Overhead", "Region Impact", "Total"],
                    textposition="outside",
                    text=[f"{base_co2:.4f}g", f"+{pue_addition:.4f}g", f"+{region_factor:.4f}g", f"{est_data['co2_grams']:.4f}g"],
                    y=[base_co2, pue_addition, region_factor, 0],
                    connector={"line": {"color": "#94a3b8"}},
                    increasing={"marker": {"color": "#dc2626"}},
                    decreasing={"marker": {"color": "#16a34a"}},
                    totals={"marker": {"color": "#3b82f6"}}
                ))
                fig_waterfall.update_layout(
                    title={'text': '💧 Emission Waterfall', 'font': {'color': '#0f172a'}},
                    plot_bgcolor='#ffffff',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#334155',
                    height=350,
                    showlegend=False
                )
                st.plotly_chart(fig_waterfall, use_container_width=True)
            
            with funnel_col:
                # Funnel Chart - Optimization Potential
                current = est_data['co2_grams']
                with_better_model = current * 0.4
                with_better_region = with_better_model * 0.7
                with_caching = with_better_region * 0.8
                optimized = with_caching * 0.9
                
                fig_funnel = go.Figure(go.Funnel(
                    y=['Current Usage', 'Switch to Efficient Model', 'Use Low-Carbon Region', 'Enable Caching', 'Fully Optimized'],
                    x=[current, with_better_model, with_better_region, with_caching, optimized],
                    textposition="inside",
                    textinfo="value+percent initial",
                    marker={"color": ["#dc2626", "#f59e0b", "#3b82f6", "#0891b2", "#16a34a"]},
                    connector={"line": {"color": "#94a3b8", "dash": "dot", "width": 2}}
                ))
                fig_funnel.update_layout(
                    title={'text': '🎯 Optimization Funnel', 'font': {'color': '#0f172a'}},
                    plot_bgcolor='#ffffff',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#334155',
                    height=350
                )
                st.plotly_chart(fig_funnel, use_container_width=True)
            
            # --- Heatmap for Model x Region ---
            st.markdown("### 🗺️ Model × Region Heatmap")
            
            heatmap_data = []
            heatmap_models = ["gpt-4", "gpt-3.5-turbo", "llama-3-70b", "llama-3-8b", "distilbert"]
            heatmap_regions = ["Global", "US-East", "US-West", "EU-West", "AsiaPac"]
            
            for m in heatmap_models:
                row = []
                for r in heatmap_regions:
                    resp = requests.post(f"{API_URL}/estimate", json={
                        "model_name": m,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "region": r
                    })
                    data = resp.json()
                    if 'error' in data:
                        row.append('N/A')
                    else:
                        row.append(data['co2_grams'] * batch_queries)
                heatmap_data.append(row)
            
            # Prepare text for heatmap, showing 'N/A' for missing/error values
            heatmap_text = [[f"{val:.4f}g" if isinstance(val, (int, float)) else "N/A" for val in row] for row in heatmap_data]
            # Replace 'N/A' with np.nan for z values so plotly can handle missing data
            import numpy as np
            heatmap_z = [[val if isinstance(val, (int, float)) else np.nan for val in row] for row in heatmap_data]
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=heatmap_z,
                x=heatmap_regions,
                y=heatmap_models,
                colorscale=[[0, '#16a34a'], [0.5, '#eab308'], [1, '#dc2626']],
                text=heatmap_text,
                texttemplate="%{text}",
                textfont={"size": 10, "color": "#ffffff"},
                hovertemplate="Model: %{y}<br>Region: %{x}<br>CO₂: %{text}<extra></extra>"
            ))
            fig_heatmap.update_layout(
                title={'text': '🔥 CO₂ Emissions Heatmap (grams)', 'font': {'color': '#0f172a'}},
                plot_bgcolor='#ffffff',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#334155',
                height=400,
                xaxis_title="Region",
                yaxis_title="Model"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
                
        except requests.exceptions.ConnectionError:
            st.error("🚨 Backend API is not running. Please start the backend server first.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        # Welcome State
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 60px;">
            <h2 style="color: #ccd6f6;">👈 Configure Your Query</h2>
            <p style="color: #8892b0; font-size: 1.1em;">
                Select a model, task type, and token count in the sidebar.<br>
                Then click <strong style="color: #00ff88;">Estimate Impact</strong> to see your carbon footprint.
        """, unsafe_allow_html=True)

# ============================================================================
# TAB 2: IMAGE AI GENERATION
# ============================================================================
with tab2:
    st.markdown("### 🖼️ Image Generation Carbon Footprint")
    
    img_col1, img_col2 = st.columns([1, 2])
    
    with img_col1:
        st.markdown("#### Configure Image Generation")
        
        img_model = st.selectbox(
            "🎨 Image Model",
            ["dall-e-3", "dall-e-2", "midjourney-v6", "midjourney-v5", 
             "stable-diffusion-xl", "stable-diffusion-3", "stable-diffusion-2.1",
             "sdxl-turbo", "sdxl-lightning", "lcm",
             "imagen-2", "flux-pro", "flux-schnell"],
            key="img_model"
        )
        
        img_resolution = st.selectbox(
            "📐 Resolution",
            ["512x512", "768x768", "1024x1024", "1536x1536", "2048x2048"],
            index=2,
            key="img_res"
        )
        
        img_steps = st.slider("🔄 Denoising Steps", 1, 100, 50, key="img_steps")
        img_count = st.number_input("🖼️ Number of Images", 1, 100, 1, key="img_count")
        img_region = st.selectbox("🌍 Region", ["global", "us-west", "us-east", "eu-west", "eu-north"], key="img_region")
        
        estimate_img_btn = st.button("🌿 Estimate Image Impact", type="primary", key="img_btn")
    
    with img_col2:
        if estimate_img_btn:
            try:
                resp = requests.get(f"{API_URL}/image/estimate", params={
                    "model_name": img_model,
                    "num_images": img_count,
                    "resolution": img_resolution,
                    "steps": img_steps,
                    "region": img_region
                })
                data = resp.json()
                
                # Display Results
                st.markdown("#### 📊 Results")
                
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(f"""
                    <div class="glass-card">
                        <div class="metric-label">Energy Used</div>
                        <div class="metric-value">{data['total_energy_kwh']*1000:.2f}</div>
                        <div class="metric-unit">Wh</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""
                    <div class="glass-card">
                        <div class="metric-label">CO₂ Emissions</div>
                        <div class="metric-value">{data['co2_grams']:.3f}</div>
                        <div class="metric-unit">grams</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m3:
                    st.markdown(f"""
                    <div class="glass-card">
                        <div class="metric-label">Equivalent To</div>
                        <div class="metric-value">{data['equivalent_google_searches']:.0f}</div>
                        <div class="metric-unit">🔍 Google Searches</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.info(f"📐 Methodology: {data['methodology']}")
                
                # Model Comparison Chart
                st.markdown("#### 📈 Model Comparison")
                compare_models = ["dall-e-3", "stable-diffusion-xl", "sdxl-turbo", "flux-pro", "lcm"]
                compare_data = []
                for m in compare_models:
                    r = requests.get(f"{API_URL}/image/estimate", params={
                        "model_name": m, "num_images": img_count, "resolution": img_resolution,
                        "steps": img_steps, "region": img_region
                    })
                    d = r.json()
                    compare_data.append({"Model": m, "CO2 (g)": d['co2_grams'], "Energy (Wh)": d['total_energy_kwh']*1000})
                
                df = pd.DataFrame(compare_data)
                fig = px.bar(df, x="Model", y="CO2 (g)", color="Model", title="CO₂ Emissions by Image Model",
                            color_discrete_sequence=["#ff6b6b", "#ffa502", "#00ff88", "#00d4ff", "#a855f7"])
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#8892b0')
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 40px;">
                <h3 style="color: #ccd6f6;">🎨 Image Generation Emissions</h3>
                <p style="color: #8892b0;">
                    Configure your image generation parameters and click <strong style="color: #00ff88;">Estimate Impact</strong>
                </p>
                <p style="color: #64ffda; font-size: 0.9em;">
                    💡 Tip: SDXL-Turbo uses 20x less energy than DALL-E 3!
                </p>
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# TAB 3: VIDEO AI GENERATION
# ============================================================================
with tab3:
    st.markdown("### 🎬 Video Generation Carbon Footprint")
    
    vid_col1, vid_col2 = st.columns([1, 2])
    
    with vid_col1:
        st.markdown("#### Configure Video Generation")
        
        vid_model = st.selectbox(
            "🎥 Video Model",
            ["sora", "runway-gen3", "runway-gen2", "veo", "lumiere",
             "pika-1.5", "pika-1.0", "stable-video-diffusion", "animatediff"],
            key="vid_model"
        )
        
        vid_duration = st.slider("⏱️ Duration (seconds)", 1, 60, 4, key="vid_duration")
        vid_resolution = st.selectbox("📐 Resolution", ["480p", "720p", "1080p", "4k"], index=2, key="vid_res")
        vid_fps = st.selectbox("🎞️ FPS", [12, 24, 30, 60], index=1, key="vid_fps")
        vid_region = st.selectbox("🌍 Region", ["global", "us-west", "us-east", "eu-west", "eu-north"], key="vid_region")
        
        estimate_vid_btn = st.button("🌿 Estimate Video Impact", type="primary", key="vid_btn")
    
    with vid_col2:
        if estimate_vid_btn:
            try:
                resp = requests.get(f"{API_URL}/video/estimate", params={
                    "model_name": vid_model,
                    "duration_seconds": vid_duration,
                    "resolution": vid_resolution,
                    "fps": vid_fps,
                    "region": vid_region
                })
                data = resp.json()
                
                # Display Results
                st.markdown("#### 📊 Results")
                
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.markdown(f"""
                    <div class="glass-card">
                        <div class="metric-label">Energy Used</div>
                        <div class="metric-value">{data['total_energy_kwh']*1000:.1f}</div>
                        <div class="metric-unit">Wh</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""
                    <div class="glass-card">
                        <div class="metric-label">CO₂ Emissions</div>
                        <div class="metric-value">{data['co2_grams']:.2f}</div>
                        <div class="metric-unit">grams</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m3:
                    st.markdown(f"""
                    <div class="glass-card">
                        <div class="metric-label">Phone Charges</div>
                        <div class="metric-value">{data['equivalent_phone_charges']:.1f}</div>
                        <div class="metric-unit">📱</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m4:
                    st.markdown(f"""
                    <div class="glass-card">
                        <div class="metric-label">Streaming Hours</div>
                        <div class="metric-value">{data['equivalent_streaming_hours']:.1f}</div>
                        <div class="metric-unit">📺</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.warning(f"⚠️ {vid_duration}s of {vid_model} video = {data['equivalent_car_meters']:.0f} meters of car driving emissions!")
                st.info(f"📐 Methodology: {data['methodology']}")
                
                # Model Comparison
                st.markdown("#### 📈 Video Model Comparison")
                compare_models = ["sora", "runway-gen3", "veo", "stable-video-diffusion", "animatediff"]
                compare_data = []
                for m in compare_models:
                    r = requests.get(f"{API_URL}/video/estimate", params={
                        "model_name": m, "duration_seconds": vid_duration, "resolution": vid_resolution,
                        "fps": vid_fps, "region": vid_region
                    })
                    d = r.json()
                    compare_data.append({"Model": m, "CO2 (g)": d['co2_grams'], "Energy (Wh)": d['total_energy_kwh']*1000})
                
                df = pd.DataFrame(compare_data)
                fig = px.bar(df, x="Model", y="CO2 (g)", color="Model", title=f"CO₂ for {vid_duration}s Video Generation",
                            color_discrete_sequence=["#ff6b6b", "#ffa502", "#00d4ff", "#00ff88", "#a855f7"])
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#8892b0')
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 40px;">
                <h3 style="color: #ccd6f6;">🎬 Video Generation Emissions</h3>
                <p style="color: #8892b0;">
                    Video generation is <strong style="color: #ff6b6b;">extremely energy intensive</strong>!<br>
                    A 10-second Sora video uses ~50x more energy than generating an image.
                </p>
                <p style="color: #ffa502; font-size: 0.9em;">
                    ⚡ Sora: ~50 Wh/second | Runway Gen3: ~25 Wh/second | AnimateDiff: ~5 Wh/second
                </p>
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# TAB 5: LIVE CARBON INTENSITY
# ============================================================================
with tab5:
    st.markdown("### ⚡ Real-Time Carbon Intensity")
    
    live_col1, live_col2 = st.columns([1, 2])
    
    with live_col1:
        st.markdown("#### Select Region")
        live_region = st.selectbox(
            "🌍 Region",
            ["global", "us-west", "us-east", "eu-west", "eu-north"],
            key="live_region"
        )
        
        refresh_btn = st.button("🔄 Refresh Data", type="primary", key="refresh_live")
        
        st.markdown("#### ⏰ Schedule Optimizer")
        schedule_duration = st.slider("Workload Duration (hours)", 1, 8, 2, key="schedule_duration")
        schedule_btn = st.button("🎯 Find Optimal Time", key="schedule_btn")
    
    with live_col2:
        try:
            # Get live carbon intensity
            live_resp = requests.get(f"{API_URL}/carbon/live", params={"region": live_region})
            live_data = live_resp.json()
            
            # Status display
            status_color = {
                "excellent": "#00ff88",
                "good": "#00d4ff",
                "moderate": "#ffa502",
                "poor": "#ff6b6b"
            }.get(live_data.get("status", "moderate"), "#ffa502")
            
            st.markdown(f"""
            <div class="glass-card" style="border-color: {status_color};">
                <div class="metric-label">Current Carbon Intensity</div>
                <div class="metric-value">{live_data.get('carbon_intensity', 'N/A')}</div>
                <div class="metric-unit">{live_data.get('unit', 'gCO2/kWh')}</div>
                <p style="color: {status_color}; margin-top: 15px; font-weight: bold;">
                    Status: {live_data.get('status', 'unknown').upper()}
                </p>
                <p style="color: #8892b0;">💡 {live_data.get('recommendation', '')}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # 24-hour forecast chart
            st.markdown("#### 📈 24-Hour Forecast")
            forecast = live_data.get("forecast_24h", [])
            if forecast:
                df_forecast = pd.DataFrame(forecast)
                fig = px.line(
                    df_forecast, x="hour", y="carbon_intensity",
                    title=f"Carbon Intensity Forecast - {live_region.upper()}",
                    markers=True
                )
                fig.add_hline(y=200, line_dash="dash", line_color="#00ff88", 
                             annotation_text="Low Carbon Threshold")
                fig.add_hline(y=400, line_dash="dash", line_color="#ff6b6b",
                             annotation_text="High Carbon Threshold")
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#8892b0'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Optimal scheduling
            if schedule_btn:
                schedule_resp = requests.get(f"{API_URL}/carbon/schedule", params={
                    "region": live_region,
                    "duration_hours": schedule_duration
                })
                schedule_data = schedule_resp.json()
                
                st.markdown(f"""
                <div class="glass-card" style="border-color: #00ff88;">
                    <h4 style="color: #00ff88;">🎯 Optimal Schedule Found</h4>
                    <p style="color: #ccd6f6;">
                        <strong>Best Time:</strong> {schedule_data.get('optimal_start_time', 'N/A')} - {schedule_data.get('optimal_end_time', 'N/A')}<br>
                        <strong>Avg Carbon Intensity:</strong> {schedule_data.get('optimal_avg_ci', 'N/A')} gCO2/kWh<br>
                        <strong>Worst Time:</strong> {schedule_data.get('worst_start_time', 'N/A')} ({schedule_data.get('worst_avg_ci', 'N/A')} gCO2/kWh)<br>
                    </p>
                    <div class="savings-highlight">
                        💚 Potential Savings: {schedule_data.get('potential_savings_percent', 0):.1f}% less carbon
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error fetching live data: {e}")

# ============================================================================
# TAB 6: CARBON OFFSETS
# ============================================================================
with tab6:
    st.markdown("### 🌳 Carbon Offset Calculator")
    st.markdown("*Calculate the environmental impact of your AI usage and discover offset options*")
    
    offset_col1, offset_col2 = st.columns([1, 2])
    
    with offset_col1:
        st.markdown("#### 📊 Enter Your Emissions")
        
        offset_co2_kg = st.number_input(
            "CO₂ to Offset (kg)",
            min_value=0.001, max_value=1000000.0, value=10.0,
            step=1.0, key="offset_co2",
            help="Enter the amount of CO₂ emissions you want to offset"
        )
        
        offset_type = st.selectbox(
            "🌿 Offset Type",
            ["tree_planting", "renewable_energy", "verified_carbon_standard", 
             "gold_standard", "methane_capture", "direct_air_capture"],
            format_func=lambda x: {
                "tree_planting": "🌳 Tree Planting ($15/tonne)",
                "renewable_energy": "⚡ Renewable Energy ($20/tonne)",
                "verified_carbon_standard": "✅ Verified Carbon Standard ($25/tonne)",
                "gold_standard": "🏆 Gold Standard ($35/tonne)",
                "methane_capture": "🔥 Methane Capture ($40/tonne)",
                "direct_air_capture": "🌬️ Direct Air Capture ($300/tonne)"
            }.get(x, x.replace("_", " ").title()),
            key="offset_type"
        )
        
        st.markdown("---")
        st.markdown("#### 💡 Quick Presets")
        preset_col1, preset_col2 = st.columns(2)
        with preset_col1:
            if st.button("🖥️ 1 Hour GPT-4", key="preset1"):
                st.session_state.offset_co2 = 0.05
        with preset_col2:
            if st.button("📊 1000 Queries", key="preset2"):
                st.session_state.offset_co2 = 5.0
        
        preset_col3, preset_col4 = st.columns(2)
        with preset_col3:
            if st.button("🎬 Video Gen", key="preset3"):
                st.session_state.offset_co2 = 25.0
        with preset_col4:
            if st.button("🤖 Train Model", key="preset4"):
                st.session_state.offset_co2 = 500.0
    
    with offset_col2:
        try:
            offset_resp = requests.get(f"{API_URL}/offset/calculate", params={
                "co2_kg": offset_co2_kg,
                "offset_type": offset_type
            })
            offset_data = offset_resp.json()
            
            # Main metrics row
            st.markdown("#### 🎯 Offset Results")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f"""
                <div class="glass-card" style="text-align: center;">
                    <div class="metric-label">CO₂ to Offset</div>
                    <div class="metric-value" style="font-size: 1.8em; color: #dc2626;">{offset_co2_kg:.2f}</div>
                    <div class="metric-unit">kg CO₂</div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="glass-card" style="text-align: center;">
                    <div class="metric-label">Trees (1 Year)</div>
                    <div class="metric-value" style="font-size: 1.8em; color: #16a34a;">{offset_data.get('trees_needed_annual', 0)}</div>
                    <div class="metric-unit">🌳 trees</div>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="glass-card" style="text-align: center;">
                    <div class="metric-label">Offset Cost</div>
                    <div class="metric-value" style="font-size: 1.8em; color: #3b82f6;">${offset_data.get('cost_usd', 0):.2f}</div>
                    <div class="metric-unit">USD</div>
                </div>
                """, unsafe_allow_html=True)
            with m4:
                st.markdown(f"""
                <div class="glass-card" style="text-align: center;">
                    <div class="metric-label">Permanent Trees</div>
                    <div class="metric-value" style="font-size: 1.8em; color: #0891b2;">{offset_data.get('trees_needed_permanent', 0)}</div>
                    <div class="metric-unit">🌲 40-year</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Real-world equivalents with visual
            st.markdown("#### 🔄 Real-World Impact Equivalents")
            equivalents = offset_data.get('real_world_equivalents', {})
            
            # Create visual comparison chart
            equiv_items = [
                {"item": "🚗 Car Distance", "value": equivalents.get('car_km_equivalent', 0), "unit": "km", "icon": "🚗"},
                {"item": "✈️ NYC-London Flights", "value": equivalents.get('flights_nyc_london', 0), "unit": "flights", "icon": "✈️"},
                {"item": "📱 Phone Charges", "value": equivalents.get('smartphones_charged', 0), "unit": "charges", "icon": "📱"},
                {"item": "🍔 Beef Meals", "value": equivalents.get('beef_meals', 0), "unit": "meals", "icon": "🍔"},
                {"item": "⛽ Gasoline", "value": equivalents.get('gallons_gasoline', 0), "unit": "gallons", "icon": "⛽"},
                {"item": "📺 Streaming", "value": equivalents.get('years_of_streaming', 0), "unit": "years", "icon": "📺"},
            ]
            
            eq_col1, eq_col2, eq_col3 = st.columns(3)
            for i, eq in enumerate(equiv_items[:3]):
                with [eq_col1, eq_col2, eq_col3][i]:
                    val_display = f"{eq['value']:.0f}" if eq['value'] >= 1 else f"{eq['value']:.2f}"
                    st.markdown(f"""
                    <div class="glass-card" style="text-align: center; padding: 15px;">
                        <div style="font-size: 2em;">{eq['icon']}</div>
                        <div style="font-size: 1.5em; font-weight: 700; color: #0f172a;">{val_display}</div>
                        <div style="color: #64748b; font-size: 0.85em;">{eq['unit']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            eq_col4, eq_col5, eq_col6 = st.columns(3)
            for i, eq in enumerate(equiv_items[3:]):
                with [eq_col4, eq_col5, eq_col6][i]:
                    val_display = f"{eq['value']:.0f}" if eq['value'] >= 1 else f"{eq['value']:.2f}"
                    st.markdown(f"""
                    <div class="glass-card" style="text-align: center; padding: 15px;">
                        <div style="font-size: 2em;">{eq['icon']}</div>
                        <div style="font-size: 1.5em; font-weight: 700; color: #0f172a;">{val_display}</div>
                        <div style="color: #64748b; font-size: 0.85em;">{eq['unit']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Offset options comparison chart
            st.markdown("#### 💰 Offset Options Comparison")
            options = offset_data.get('offset_options', {})
            if options:
                options_list = []
                for k, v in options.items():
                    options_list.append({
                        "Option": k.replace("_", " ").title(),
                        "Price/Tonne": v['price_per_tonne'],
                        "Total Cost": v['total_cost']
                    })
                df_options = pd.DataFrame(options_list)
                
                # Bar chart comparing costs
                fig_offset = px.bar(
                    df_options,
                    x='Option',
                    y='Total Cost',
                    color='Total Cost',
                    color_continuous_scale=['#16a34a', '#eab308', '#dc2626'],
                    title=f'💵 Cost to Offset {offset_co2_kg:.2f} kg CO₂'
                )
                fig_offset.update_layout(
                    plot_bgcolor='#ffffff',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#334155',
                    title_font_color='#0f172a',
                    xaxis_tickangle=-30,
                    height=350
                )
                st.plotly_chart(fig_offset, use_container_width=True)
                
                # Table view
                st.dataframe(
                    df_options.style.format({
                        "Price/Tonne": "${:.0f}",
                        "Total Cost": "${:.2f}"
                    }),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Recommendations
                cheapest = min(options.items(), key=lambda x: x[1]['total_cost'])
                most_effective = "direct_air_capture"  # Most permanent
                
                st.markdown(f"""
                <div class="glass-card" style="border-left: 4px solid #16a34a;">
                    <h4 style="color: #0f172a; margin-bottom: 10px;">💡 Recommendations</h4>
                    <p style="color: #334155;"><strong>🏷️ Most Affordable:</strong> {cheapest[0].replace('_', ' ').title()} - ${cheapest[1]['total_cost']:.2f}</p>
                    <p style="color: #334155;"><strong>🎯 Most Permanent:</strong> Direct Air Capture - Removes CO₂ permanently from atmosphere</p>
                    <p style="color: #334155;"><strong>🌳 Best for Nature:</strong> Tree Planting - Also supports biodiversity & ecosystems</p>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error calculating offset: {e}")
            st.info("Make sure the backend API is running on http://localhost:8000")

# ============================================================================
# TAB 7: ANALYTICS & SUSTAINABILITY
# ============================================================================
with tab7:
    st.markdown("### 🏆 Sustainability Analytics")
    
    analytics_subtab1, analytics_subtab2, analytics_subtab3 = st.tabs([
        "🎯 Sustainability Score", "🏭 Training Emissions", "📊 API Analytics"
    ])
    
    with analytics_subtab1:
        st.markdown("#### Calculate Your Sustainability Score")
        
        try:
            score_resp = requests.get(f"{API_URL}/analytics/sustainability")
            score_data = score_resp.json()
            
            score = score_data.get('score', 50)
            grade = score_data.get('grade', 'C')
            
            # Score display
            score_color = "#00ff88" if score >= 70 else "#ffa502" if score >= 50 else "#ff6b6b"
            
            score_col1, score_col2 = st.columns([1, 2])
            
            with score_col1:
                st.markdown(f"""
                <div class="glass-card" style="text-align: center; border-color: {score_color};">
                    <div class="metric-label">Sustainability Score</div>
                    <div class="metric-value" style="font-size: 4em;">{score:.0f}</div>
                    <div style="font-size: 3em; color: {score_color};">{grade}</div>
                    <p style="color: #8892b0; margin-top: 15px;">{score_data.get('message', '')}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with score_col2:
                # Breakdown chart
                breakdown = score_data.get('breakdown', {})
                if breakdown:
                    df_breakdown = pd.DataFrame([
                        {"Factor": k.replace("_", " ").title(), "Score": v}
                        for k, v in breakdown.items()
                    ])
                    fig = px.bar(df_breakdown, x="Factor", y="Score", 
                                color="Score", color_continuous_scale=["#ff6b6b", "#ffa502", "#00ff88"],
                                title="Score Breakdown")
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#8892b0'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Improvement tips
                tips = score_data.get('improvement_tips', [])
                if tips:
                    st.markdown("#### 💡 Improvement Tips")
                    for tip in tips:
                        st.info(tip)
                        
        except Exception as e:
            st.error(f"Error loading analytics: {e}")
    
    with analytics_subtab2:
        st.markdown("#### 🏭 AI Training Emissions Calculator")
        st.markdown("Estimate the carbon footprint of training AI models")
        
        train_col1, train_col2 = st.columns([1, 2])
        
        with train_col1:
            train_method = st.radio(
                "Estimation Method",
                ["By Model Type", "By GPU Hours", "By Parameters"],
                key="train_method"
            )
            
            if train_method == "By Model Type":
                train_model = st.selectbox(
                    "Model Type",
                    ["custom-small", "custom-medium", "custom-large", "bert-base", "bert-large",
                     "llama-3-8b", "llama-3-70b", "gpt-3", "gpt-3.5", "gpt-4", "stable-diffusion-xl"],
                    key="train_model"
                )
                gpu_hours = None
                params_b = None
            elif train_method == "By GPU Hours":
                train_model = "custom-medium"
                gpu_hours = st.number_input("GPU Hours", min_value=1, max_value=1000000, value=1000, key="gpu_hours")
                params_b = None
            else:
                train_model = "custom-medium"
                gpu_hours = None
                params_b = st.number_input("Parameters (Billions)", min_value=0.1, max_value=1000.0, value=7.0, key="params_b")
            
            train_gpu = st.selectbox("GPU Type", ["a100", "h100", "v100", "a10", "t4", "4090"], key="train_gpu")
            train_gpus = st.number_input("Number of GPUs", min_value=1, max_value=10000, value=8, key="train_gpus")
            train_region = st.selectbox("Training Region", ["global", "us-west", "us-east", "eu-west", "eu-north"], key="train_region")
            
            calc_train_btn = st.button("🔥 Calculate Training Emissions", type="primary", key="calc_train")
        
        with train_col2:
            if calc_train_btn:
                try:
                    train_resp = requests.post(f"{API_URL}/training/estimate", json={
                        "model_type": train_model,
                        "gpu_hours": gpu_hours,
                        "gpu_type": train_gpu,
                        "num_gpus": train_gpus,
                        "region": train_region,
                        "parameters_billions": params_b
                    })
                    train_data = train_resp.json()
                    
                    # Results
                    t1, t2, t3 = st.columns(3)
                    with t1:
                        st.markdown(f"""
                        <div class="glass-card">
                            <div class="metric-label">Energy Used</div>
                            <div class="metric-value">{train_data.get('total_energy_mwh', 0):.1f}</div>
                            <div class="metric-unit">MWh</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with t2:
                        st.markdown(f"""
                        <div class="glass-card">
                            <div class="metric-label">CO₂ Emissions</div>
                            <div class="metric-value">{train_data.get('co2_tonnes', 0):.1f}</div>
                            <div class="metric-unit">tonnes</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with t3:
                        st.markdown(f"""
                        <div class="glass-card">
                            <div class="metric-label">Offset Cost</div>
                            <div class="metric-value">${train_data.get('offset_cost_usd', 0):.0f}</div>
                            <div class="metric-unit">USD</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Comparisons
                    st.markdown("#### 🔄 Training Emissions Equivalents")
                    comps = train_data.get('comparisons', {})
                    comp_col1, comp_col2 = st.columns(2)
                    with comp_col1:
                        st.metric("🚗 Car Kilometers", f"{comps.get('equivalent_car_km', 0):,.0f}")
                        st.metric("✈️ NYC-SF Flights", f"{comps.get('equivalent_flights_nyc_sf', 0):.1f}")
                    with comp_col2:
                        st.metric("🏠 Home Years", f"{comps.get('equivalent_home_years', 0):.2f}")
                        st.metric("🌳 Tree Lifetimes", f"{comps.get('equivalent_trees_lifetime', 0):.0f}")
                    
                    st.info(f"📐 {train_data.get('methodology', '')}")
                    st.warning(f"⚖️ Break-even: This training cost is equivalent to {train_data.get('break_even_queries', 0):,} inference queries")
                    
                except Exception as e:
                    st.error(f"Error calculating training emissions: {e}")
    
    with analytics_subtab3:
        st.markdown("#### 📊 API Usage Analytics")
        
        try:
            analytics_resp = requests.get(f"{API_URL}/analytics/api", params={"hours": 24})
            api_data = analytics_resp.json()
            
            # Summary
            a1, a2, a3, a4 = st.columns(4)
            with a1:
                st.metric("Total API Calls", api_data.get('total_calls', 0))
            with a2:
                st.metric("Total CO₂ (g)", f"{api_data.get('total_co2_grams', 0):.4f}")
            with a3:
                st.metric("Avg Response (ms)", f"{api_data.get('avg_response_time_ms', 0):.1f}")
            with a4:
                st.metric("Calls/Hour", f"{api_data.get('calls_per_hour', 0):.1f}")
            
            # By endpoint chart
            by_endpoint = api_data.get('by_endpoint', {})
            if by_endpoint:
                df_ep = pd.DataFrame([
                    {"Endpoint": k, "Calls": v['calls'], "CO2 (g)": v['co2']}
                    for k, v in by_endpoint.items()
                ])
                fig = px.bar(df_ep, x="Endpoint", y="Calls", color="CO2 (g)",
                            title="API Calls by Endpoint",
                            color_continuous_scale=["#00ff88", "#ffa502", "#ff6b6b"])
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#8892b0',
                    xaxis_tickangle=-45
                )
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.info("API analytics will appear as you use the system")

# ============================================================================
# TAB 8: HISTORY (was tab5)
# ============================================================================
with tab4:
    st.markdown("### 🏢 Global Data Center Carbon Emissions")
    
    try:
        # Fetch data center info from API
        dc_resp = requests.get(f"{API_URL}/datacenters/comparison")
        dc_data = dc_resp.json()["comparison"]
        df_dc = pd.DataFrame(dc_data)
        
        # === WORLD MAP SECTION ===
        st.markdown("### 🗺️ World Map - Data Center Locations & Emissions")
        
        # Data center coordinates and details
        dc_locations = [
            {"name": "Google Cloud - Iowa", "lat": 41.2619, "lon": -95.8608, "provider": "Google", "ci": 410, "emissions": 0, "location": "Council Bluffs, Iowa, USA"},
            {"name": "Google Cloud - Oregon", "lat": 45.5946, "lon": -121.1787, "provider": "Google", "ci": 80, "emissions": 0, "location": "The Dalles, Oregon, USA"},
            {"name": "Google Cloud - Belgium", "lat": 50.4542, "lon": 3.9426, "provider": "Google", "ci": 150, "emissions": 0, "location": "St. Ghislain, Belgium"},
            {"name": "AWS US East (N. Virginia)", "lat": 39.0438, "lon": -77.4874, "provider": "AWS", "ci": 340, "emissions": 450000, "location": "Ashburn, Virginia, USA"},
            {"name": "AWS US West (Oregon)", "lat": 45.8399, "lon": -119.6884, "provider": "AWS", "ci": 100, "emissions": 50000, "location": "Boardman, Oregon, USA"},
            {"name": "AWS Europe (Ireland)", "lat": 53.3498, "lon": -6.2603, "provider": "AWS", "ci": 280, "emissions": 120000, "location": "Dublin, Ireland"},
            {"name": "AWS Asia Pacific (Singapore)", "lat": 1.3521, "lon": 103.8198, "provider": "AWS", "ci": 420, "emissions": 180000, "location": "Singapore"},
            {"name": "Azure East US", "lat": 37.4316, "lon": -78.6569, "provider": "Microsoft", "ci": 350, "emissions": 350000, "location": "Virginia, USA"},
            {"name": "Azure West Europe", "lat": 52.3676, "lon": 4.9041, "provider": "Microsoft", "ci": 200, "emissions": 80000, "location": "Amsterdam, Netherlands"},
            {"name": "Azure North Europe", "lat": 53.3498, "lon": -6.2603, "provider": "Microsoft", "ci": 280, "emissions": 100000, "location": "Dublin, Ireland"},
            {"name": "Azure Sweden Central", "lat": 60.6749, "lon": 17.1413, "provider": "Microsoft", "ci": 20, "emissions": 5000, "location": "Gävle, Sweden"},
            {"name": "Equinix SV5", "lat": 37.3382, "lon": -121.8863, "provider": "Equinix", "ci": 180, "emissions": 8000, "location": "San Jose, California, USA"},
            {"name": "CoreSite LA1", "lat": 34.0522, "lon": -118.2437, "provider": "CoreSite", "ci": 200, "emissions": 15000, "location": "Los Angeles, California, USA"},
        ]
        
        df_map = pd.DataFrame(dc_locations)
        df_map['size'] = df_map['emissions'].apply(lambda x: max(15, min(50, x / 10000 + 10)))
        df_map['emissions_display'] = df_map['emissions'].apply(lambda x: f"{x:,}" if x > 0 else "Carbon Neutral")
        
        # Color by carbon intensity
        provider_colors = {
            "Google": "#16a34a",
            "AWS": "#f59e0b", 
            "Microsoft": "#3b82f6",
            "Equinix": "#dc2626",
            "CoreSite": "#8b5cf6"
        }
        
        # Create world map
        fig_map = go.Figure()
        
        for provider in df_map['provider'].unique():
            df_provider = df_map[df_map['provider'] == provider]
            fig_map.add_trace(go.Scattergeo(
                lon=df_provider['lon'],
                lat=df_provider['lat'],
                text=df_provider.apply(lambda r: f"<b>{r['name']}</b><br>{r['location']}<br>Carbon Intensity: {r['ci']} gCO₂/kWh<br>Annual Emissions: {r['emissions_display']} tonnes", axis=1),
                hoverinfo='text',
                marker=dict(
                    size=df_provider['size'],
                    color=provider_colors.get(provider, '#64748b'),
                    line=dict(width=2, color='white'),
                    symbol='circle'
                ),
                name=provider
            ))
        
        fig_map.update_layout(
            title={'text': '🌍 Global Data Centers - CO₂ Emissions Map', 'font': {'size': 20, 'color': '#0f172a'}},
            geo=dict(
                showland=True,
                landcolor='#e2e8f0',
                showocean=True,
                oceancolor='#bfdbfe',
                showlakes=True,
                lakecolor='#93c5fd',
                showcountries=True,
                countrycolor='#94a3b8',
                countrywidth=0.5,
                showcoastlines=True,
                coastlinecolor='#64748b',
                projection_type='natural earth',
                bgcolor='rgba(0,0,0,0)'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#334155',
            height=500,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(255,255,255,0.9)',
                font=dict(color='#334155')
            ),
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        st.plotly_chart(fig_map, use_container_width=True)
        
        # Region summary cards
        st.markdown("### 🌐 Regional Emissions Overview")
        
        region_data = [
            {"region": "North America", "icon": "🇺🇸", "dcs": 7, "total_emissions": 873000, "avg_ci": 237, "cleanest": "Google Oregon (80 gCO₂/kWh)"},
            {"region": "Europe", "icon": "🇪🇺", "dcs": 4, "total_emissions": 305000, "avg_ci": 178, "cleanest": "Azure Sweden (20 gCO₂/kWh)"},
            {"region": "Asia Pacific", "icon": "🌏", "dcs": 1, "total_emissions": 180000, "avg_ci": 420, "cleanest": "AWS Singapore (420 gCO₂/kWh)"},
        ]
        
        rcol1, rcol2, rcol3 = st.columns(3)
        for col, region in zip([rcol1, rcol2, rcol3], region_data):
            with col:
                st.markdown(f"""
                <div class="glass-card">
                    <h3 style="color: #0f172a; margin-bottom: 10px;">{region['icon']} {region['region']}</h3>
                    <p style="color: #64748b; margin: 5px 0;"><strong>Data Centers:</strong> {region['dcs']}</p>
                    <p style="color: #64748b; margin: 5px 0;"><strong>Total Emissions:</strong> {region['total_emissions']:,} tonnes/year</p>
                    <p style="color: #64748b; margin: 5px 0;"><strong>Avg Carbon Intensity:</strong> {region['avg_ci']} gCO₂/kWh</p>
                    <p style="color: #16a34a; margin: 5px 0; font-size: 0.85em;">🌿 Cleanest: {region['cleanest']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Summary metrics
        dc1, dc2, dc3, dc4 = st.columns(4)
        with dc1:
            st.metric("Total Data Centers", len(df_dc))
        with dc2:
            avg_ci = df_dc['carbon_intensity'].mean()
            st.metric("Avg Carbon Intensity", f"{avg_ci:.0f} g/kWh")
        with dc3:
            avg_pue = df_dc['pue'].mean()
            st.metric("Avg PUE", f"{avg_pue:.2f}")
        with dc4:
            total_emissions = df_dc['annual_emissions_tonnes'].sum()
            st.metric("Total Annual Emissions", f"{total_emissions:,.0f} t")
        
        st.markdown("---")
        
        # Visualization columns
        viz1, viz2 = st.columns(2)
        
        with viz1:
            # Carbon Intensity by DC
            fig_ci = px.bar(
                df_dc.sort_values('carbon_intensity'),
                x='name',
                y='carbon_intensity',
                color='provider',
                title='🌍 Carbon Intensity by Data Center (gCO₂/kWh)',
                color_discrete_map={
                    "Google": "#4285f4",
                    "AWS": "#ff9900", 
                    "Microsoft": "#00a4ef",
                    "Equinix": "#ed1c24",
                    "CoreSite": "#6b2c91"
                }
            )
            fig_ci.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#8892b0',
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig_ci, use_container_width=True)
        
        with viz2:
            # PUE Comparison
            fig_pue = px.bar(
                df_dc.sort_values('pue'),
                x='name',
                y='pue',
                color='provider',
                title='⚡ Power Usage Effectiveness (Lower is Better)',
                color_discrete_map={
                    "Google": "#4285f4",
                    "AWS": "#ff9900", 
                    "Microsoft": "#00a4ef",
                    "Equinix": "#ed1c24",
                    "CoreSite": "#6b2c91"
                }
            )
            fig_pue.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#8892b0',
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig_pue, use_container_width=True)
        
        # Renewable percentage
        st.markdown("### 🌿 Renewable Energy Usage")
        fig_renew = px.bar(
            df_dc.sort_values('renewable_percent', ascending=False),
            x='name',
            y='renewable_percent',
            color='renewable_percent',
            color_continuous_scale=['#ff6b6b', '#ffa502', '#00ff88'],
            title='Renewable Energy Percentage by Data Center'
        )
        fig_renew.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#8892b0',
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig_renew, use_container_width=True)
        
        # Annual emissions by provider
        st.markdown("### 📊 Annual CO₂ Emissions by Provider")
        provider_emissions = df_dc.groupby('provider')['annual_emissions_tonnes'].sum().reset_index()
        fig_provider = px.pie(
            provider_emissions,
            values='annual_emissions_tonnes',
            names='provider',
            title='Total Annual Emissions Distribution',
            color_discrete_map={
                "Google": "#4285f4",
                "AWS": "#ff9900", 
                "Microsoft": "#00a4ef",
                "Equinix": "#ed1c24",
                "CoreSite": "#6b2c91"
            }
        )
        fig_provider.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#8892b0'
        )
        st.plotly_chart(fig_provider, use_container_width=True)
        
        # Data table
        st.markdown("### 📋 Full Data Center Details")
        st.dataframe(
            df_dc[['name', 'provider', 'carbon_intensity', 'pue', 'renewable_percent', 'annual_emissions_tonnes']].rename(
                columns={
                    'name': 'Data Center',
                    'provider': 'Provider',
                    'carbon_intensity': 'CI (g/kWh)',
                    'pue': 'PUE',
                    'renewable_percent': 'Renewable %',
                    'annual_emissions_tonnes': 'Annual CO₂ (tonnes)'
                }
            ),
            use_container_width=True
        )
        
    except Exception as e:
        st.error(f"Could not load data center info: {e}")

# ============================================================================
# TAB 8: HISTORY (was tab5)
# ============================================================================
with tab8:
    st.markdown("### 📈 Query History")
    
    if st.session_state.history:
        df_history = pd.DataFrame(st.session_state.history)
        
        # Summary Stats
        h1, h2, h3 = st.columns(3)
        with h1:
            total_co2 = df_history['co2'].sum()
            st.metric("Total CO₂", f"{total_co2:.4f} g")
        with h2:
            total_queries = df_history['queries'].sum()
            st.metric("Total Queries", f"{total_queries}")
        with h3:
            avg_co2 = df_history['co2'].mean()
            st.metric("Avg CO₂/Query", f"{avg_co2:.4f} g")
        
        # History Chart
        fig_hist = px.line(
            df_history,
            x="timestamp",
            y="co2",
            color="model",
            markers=True,
            title="CO₂ Emissions Over Time"
        )
        fig_hist.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#8892b0'
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Table
        st.dataframe(df_history, use_container_width=True)
        
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No queries yet. Make your first estimation in the Text AI tab to start tracking!")

# ============================================================================
# TAB 9: LEARN (was tab6)
# ============================================================================
with tab9:
    st.markdown("### 📚 Understanding AI Carbon Footprint")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: #00ff88;">🔬 How We Calculate</h4>
            <p style="color: #8892b0;">
                <strong>Energy (kWh)</strong> = (Tokens / 1000) × Energy_per_1k_tokens × PUE<br><br>
                <strong>CO₂ (grams)</strong> = Energy × Carbon_Intensity<br><br>
                Where PUE (Power Usage Effectiveness) accounts for datacenter overhead.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: #00d4ff;">📊 Key Insights</h4>
            <ul style="color: #8892b0;">
                <li>GPT-4 uses ~10x more energy than GPT-3.5</li>
                <li>Region choice can reduce emissions by 60%</li>
                <li>Right-sizing can save up to 95% on simple tasks</li>
                <li>Video generation uses 50-100x more energy than images</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <h4 style="color: #a855f7;">🌍 Why It Matters</h4>
        <p style="color: #8892b0;">
            AI inference is projected to consume as much energy as a small country by 2030. 
            By making informed choices about model selection and query design, 
            we can reduce this impact while maintaining quality outputs.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Additional educational content
    st.markdown("### 📊 Energy Comparison Table")
    energy_comparison = pd.DataFrame({
        "AI Task": ["1 GPT-4 Query", "1 DALL-E 3 Image", "10s Sora Video", "1 Google Search", "1 Netflix Hour"],
        "Energy (Wh)": [0.4, 2.9, 600, 0.3, 100],
        "CO₂ (grams)": [0.19, 1.38, 285, 0.14, 36]
    })
    st.dataframe(energy_comparison, use_container_width=True)

# --- Footer ---
st.markdown("""
<div class="footer">
    <p>🌱 Carbon-Aware AI | MIT Capstone Project</p>
    <p style="font-size: 0.8em;">Making sustainable AI accessible</p>
</div>
""", unsafe_allow_html=True)
