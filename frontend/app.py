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

# --- Premium Dark Theme Styles ---
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 25px;
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0, 255, 136, 0.1);
        border-color: rgba(0, 255, 136, 0.3);
    }
    
    /* Metric Display */
    .metric-value {
        font-size: 2.5em;
        font-weight: 700;
        background: linear-gradient(90deg, #00ff88, #00d4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(0, 255, 136, 0.3);
    }
    .metric-label {
        font-size: 0.9em;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 8px;
    }
    .metric-unit {
        font-size: 0.8em;
        color: #64ffda;
    }
    
    /* Hero Section */
    .hero-title {
        font-size: 3em;
        font-weight: 800;
        background: linear-gradient(135deg, #00ff88 0%, #00d4ff 50%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
        animation: glow 2s ease-in-out infinite alternate;
    }
    @keyframes glow {
        from { filter: drop-shadow(0 0 10px rgba(0, 255, 136, 0.3)); }
        to { filter: drop-shadow(0 0 20px rgba(0, 212, 255, 0.5)); }
    }
    .hero-subtitle {
        color: #8892b0;
        text-align: center;
        font-size: 1.2em;
        margin-top: 10px;
    }
    
    /* Recommendation Badge */
    .rec-badge {
        background: linear-gradient(135deg, #ff6b6b, #ffa502);
        padding: 15px 25px;
        border-radius: 50px;
        color: white;
        font-weight: 600;
        display: inline-block;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    /* Savings Highlight */
    .savings-highlight {
        background: linear-gradient(135deg, #00ff88, #00d4ff);
        color: #0f0f1a;
        padding: 20px;
        border-radius: 15px;
        font-size: 1.5em;
        font-weight: 700;
        text-align: center;
        margin: 15px 0;
    }
    
    /* Status Indicators */
    .status-good { color: #00ff88; }
    .status-warning { color: #ffa502; }
    .status-bad { color: #ff6b6b; }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #8892b0;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00ff88, #00d4ff);
        color: #0f0f1a !important;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #00ff88, #00d4ff);
        color: #0f0f1a;
        border: none;
        border-radius: 10px;
        padding: 15px 30px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(0, 255, 136, 0.3);
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(0, 212, 255, 0.1);
        border-left: 4px solid #00d4ff;
        padding: 15px;
        border-radius: 0 10px 10px 0;
        color: #ccd6f6;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #4a5568;
        padding: 20px;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        margin-top: 40px;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State for History ---
if 'history' not in st.session_state:
    st.session_state.history = []

# --- Model Info Database ---
MODEL_INFO = {
    "gpt-4": {"params": "~1.7T", "provider": "OpenAI", "tier": "Frontier"},
    "gpt-3.5-turbo": {"params": "~175B", "provider": "OpenAI", "tier": "Standard"},
    "claude-3-opus": {"params": "~2T", "provider": "Anthropic", "tier": "Frontier"},
    "claude-3-sonnet": {"params": "~70B", "provider": "Anthropic", "tier": "Standard"},
    "mistral-large": {"params": "~123B", "provider": "Mistral AI", "tier": "Standard"},
    "llama-3-70b": {"params": "70B", "provider": "Meta", "tier": "Open Source"},
    "llama-3-8b": {"params": "8B", "provider": "Meta", "tier": "Efficient"},
    "distilbert": {"params": "66M", "provider": "Hugging Face", "tier": "Micro"},
    "flan-t5-base": {"params": "250M", "provider": "Google", "tier": "Efficient"}
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

# --- Main Content ---
st.markdown('<h1 class="hero-title">🌱 Carbon-Aware AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Make the Climate Impact of AI Visible, Measurable, and Actionable</p>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Text AI", "🖼️ Image AI", "🎬 Video AI", "🏢 Data Centers", "📈 History", "📚 Learn"])

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
                    <div class="glass-card" style="border-color: #ffa502;">
                        <div class="metric-label" style="color: #ffa502;">⚠️ Optimization Available</div>
                        <p style="color: #ccd6f6; margin: 15px 0;">Switch to:</p>
                        <div class="rec-badge">{rec_data['recommended_model'].upper()}</div>
                        <p style="color: #8892b0; margin-top: 15px; font-size: 0.9em;">{rec_data['reason']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="savings-highlight">
                        🎯 Potential Savings: {rec_data['savings_percentage']}%
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="glass-card" style="border-color: #00ff88;">
                        <div class="metric-label status-good">✅ Optimal Choice</div>
                        <p style="color: #ccd6f6;">You're using an efficient model for this task!</p>
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
                    comparison_data.append({
                        "Model": m,
                        "CO2 (g)": data['co2_grams'] * batch_queries,
                        "Tier": MODEL_INFO[m]['tier']
                    })
                
                df_compare = pd.DataFrame(comparison_data)
                
                fig = px.bar(
                    df_compare, 
                    x="Model", 
                    y="CO2 (g)", 
                    color="Tier",
                    title="🔍 Model Emissions Comparison",
                    color_discrete_map={
                        "Frontier": "#ff6b6b",
                        "Standard": "#ffa502", 
                        "Open Source": "#00d4ff",
                        "Efficient": "#00ff88",
                        "Micro": "#a855f7"
                    }
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#8892b0',
                    title_font_color='#ccd6f6'
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
                    color_continuous_scale=["#00ff88", "#ffa502", "#ff6b6b"]
                )
                fig2.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#8892b0',
                    title_font_color='#ccd6f6'
                )
                st.plotly_chart(fig2, use_container_width=True)
                
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
# TAB 4: DATA CENTERS (was tab3)
# ============================================================================
with tab4:
    st.markdown("### 🏢 Global Data Center Carbon Emissions")
    
    try:
        # Fetch data center info from API
        dc_resp = requests.get(f"{API_URL}/datacenters/comparison")
        dc_data = dc_resp.json()["comparison"]
        df_dc = pd.DataFrame(dc_data)
        
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
# TAB 5: HISTORY
# ============================================================================
with tab5:
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
# TAB 6: LEARN
# ============================================================================
with tab6:
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
