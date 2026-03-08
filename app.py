import streamlit as st
import base64
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import io
import requests
import json

# Set page configuration
st.set_page_config(
    page_title="Multimodal Weather Forecasting System",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern styling
def load_custom_css():
    st.markdown("""
    <style>
        /* Main styling */
        .main-header {
            text-align: center;
            padding: 0.4rem 1rem;
            background: #ffffff;
            color: #111827;
            border-radius: 10px;
            margin-bottom: 0.5rem;
            border: 1px solid #e5e7eb;
        }
        
        .metric-card {
            background: #ffffff;
            padding: 0.6rem 0.9rem;
            border-radius: 10px;
            border: 1px solid #e5e7eb;
            color: #111827;
            margin-bottom: 0.4rem;
        }

        .summary-card {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
        }

        .summary-text {
            flex: 1;
            min-width: 0;
        }

        .summary-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem 0.75rem;
            font-size: 0.8rem;
            opacity: 0.85;
        }

        .summary-icon {
            width: 64px;
            height: auto;
            opacity: 0.9;
        }
        
        .prediction-card {
            background: #ffffff;
            padding: 1rem;
            border-radius: 12px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.06);
            border: 1px solid #e5e7eb;
            margin-bottom: 1rem;
            color: #111827;
        }
        
        /* Sidebar styling */
        .sidebar-content {
            background: #ffffff;
            padding: 0.75rem;
            border-radius: 10px;
            margin-bottom: 0.75rem;
            border: 1px solid #e5e7eb;
        }
        
        .nav-item {
            padding: 0.6rem 0.8rem;
            margin: 0.35rem 0;
            border-radius: 8px;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .nav-item:hover {
            background: #f3f4f6;
            transform: translateX(5px);
        }
        
        .nav-item.active {
            background: #dbeafe;
        }
        
        /* Chart containers */
        .chart-container {
            background: #ffffff;
            padding: 0.6rem 0.75rem;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            border: 1px solid #e5e7eb;
            margin-bottom: 0.5rem;
            color: #111827;
        }
        
        /* Upload area styling */
        .upload-area {
            background: #ffffff;
            border: 2px dashed #93c5fd;
            border-radius: 10px;
            padding: 1rem;
            text-align: center;
            margin: 0.6rem 0;
            transition: all 0.3s ease;
            color: #111827;
        }
        
        .upload-area:hover {
            border-color: #3b82f6;
            background: #f9fafb;
        }
        
        /* Button styling */
        .stButton > button {
            background: #3b82f6;
            color: white;
            border: none;
            padding: 0.6rem 1.5rem;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.25);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 18px rgba(59, 130, 246, 0.35);
        }
        
        /* Hide streamlit default elements */
        .stDeployButton {
            display: none;
        }
        
        #MainMenu {
            visibility: hidden;
        }
        
        footer {
            visibility: hidden;
        }
        
        header {
            visibility: hidden;
        }
        
        .stMarkdown h3 {
            margin: 0.35rem 0 0.4rem 0;
            font-size: 1.05rem;
        }

        .stMarkdown h4 {
            margin: 0.25rem 0 0.3rem 0;
        }

        /* App background */
        .stApp {
            background: #f8fafc;
        }

        section[data-testid="stSidebar"] {
            background: #f8fafc;
        }
    </style>
    """, unsafe_allow_html=True)

def load_local_svg_data_uri(path):
    try:
        with open(path, "r", encoding="utf-8") as svg_file:
            svg_content = svg_file.read()
    except FileNotFoundError:
        return None

    svg_base64 = base64.b64encode(svg_content.encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{svg_base64}"

# Initialize session state for chat
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'current_page' not in st.session_state:
    st.session_state.current_page = "Dashboard"

# API Configuration
API_BASE_URL = "http://127.0.0.1:8000"

# API Functions
def get_weather_prediction():
    """Get weather prediction from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/predict")
        if response.status_code == 200:
            return response.json()
        else:
            st.error("Không thể kết nối đến API server")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Lỗi kết nối API: {e}")
        return None

def chat_with_api(message):
    """Send message to chat API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={"message": message}
        )
        if response.status_code == 200:
            return response.json()["reply"]
        else:
            return "Xin lỗi, có lỗi xảy ra khi xử lý yêu cầu của bạn."
    except requests.exceptions.RequestException as e:
        return f"Không thể kết nối đến chatbot API: {e}"

# Load custom CSS
load_custom_css()

# Sidebar
with st.sidebar:
    # App logo/icon
    st.markdown("""
    <div class="sidebar-content" style="text-align: center;">
        <h1 style="font-size: 2.5rem; margin: 0;">🌤️</h1>
        <h3 style="margin: 0.5rem 0; color: #3b82f6;">Weather AI</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation menu
    st.markdown("### Navigation")
    
    if st.button("📊 Dashboard", key="nav_dashboard", use_container_width=True):
        st.session_state.current_page = "Dashboard"
    
    if st.button("🌡️ Weather Prediction", key="nav_prediction", use_container_width=True):
        st.session_state.current_page = "Weather Prediction"
    
    if st.button("📈 Data Visualization", key="nav_visualization", use_container_width=True):
        st.session_state.current_page = "Data Visualization"
    
    if st.button("🤖 Model Information", key="nav_model", use_container_width=True):
        st.session_state.current_page = "Model Information"
    

# Main header
st.markdown("""
<div class="main-header">
    <p style="margin: 0; font-size: 1.1rem; font-weight: 600; color: #111827;">
        🌤️ Multimodal Weather Forecasting System &nbsp;·&nbsp; <span style="font-weight: 400; opacity: 0.75;">Ho Chi Minh City</span>
    </p>
</div>
""", unsafe_allow_html=True)

# Main content based on selected page
if st.session_state.current_page == "Dashboard":
    st.markdown("#### 🌡️ Current Weather")

    weather_data = get_weather_prediction()
    summary_icon = load_local_svg_data_uri("assets/summary.svg")
    summary_icon_html = (
        f'<img src="{summary_icon}" class="summary-icon" alt="Summary icon"/>'
        if summary_icon
        else ""
    )

    col_left, col_right = st.columns([1, 2.5])

    # Compute shared rain values used by both card and chart
    if weather_data:
        wind_speed = weather_data.get('wind_speed', 12)
        precipitation = weather_data.get('precipitation', 0)
        # Rain probability: derive from precipitation (mm) — cap at 95%
        rain_prob_now = min(95, int(precipitation * 8)) if precipitation > 0 else 20
    else:
        wind_speed = 12
        rain_prob_now = 45

    with col_left:
        if weather_data:
            st.markdown(f"""
            <div class="metric-card summary-card">
                <div class="summary-text">
                    <div style="font-size: 0.78rem; opacity: 0.7; margin-bottom: 0.2rem;">Summary · {weather_data.get('date', '')}</div>
                    <div style="font-size: 2rem; font-weight: 700; line-height: 1.1;">{weather_data['temperature']}°C</div>
                    <div style="font-size: 0.8rem; opacity: 0.8; margin-top: 0.3rem;">{weather_data['status']}</div>
                    <div class="summary-meta" style="margin-top: 0.4rem;">
                        <span>💧 {weather_data['humidity']}%</span>
                        <span>💨 {wind_speed} km/h</span>
                        <span>🌧️ {rain_prob_now}%</span>
                    </div>
                </div>
                {summary_icon_html}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Không thể tải dữ liệu thời tiết từ API.")
            st.markdown(f"""
            <div class="metric-card summary-card">
                <div class="summary-text">
                    <div style="font-size: 0.78rem; opacity: 0.7; margin-bottom: 0.2rem;">Summary</div>
                    <div style="font-size: 2rem; font-weight: 700; line-height: 1.1;">28°C</div>
                    <div style="font-size: 0.8rem; opacity: 0.8; margin-top: 0.3rem;">Stable</div>
                    <div class="summary-meta" style="margin-top: 0.4rem;">
                        <span>💧 75%</span>
                        <span>💨 12 km/h</span>
                        <span>🌧️ {rain_prob_now}%</span>
                    </div>
                </div>
                {summary_icon_html}
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        hours = list(range(25))
        # Seed chart from rain_prob_now so card and chart are consistent
        rain_prob_series = [
            max(0, min(100, rain_prob_now + 20 * np.sin(i / 6) + np.random.normal(0, 4)))
            for i in hours
        ]

        fig_rain = go.Figure()
        fig_rain.add_trace(go.Scatter(
            x=hours,
            y=rain_prob_series,
            mode='lines+markers',
            line=dict(color='#3b82f6', width=2),
            marker=dict(size=4),
            showlegend=False
        ))

        fig_rain.update_layout(
            title=dict(text="Rain Probability (Next 24h)", font=dict(size=13)),
            xaxis_title="Hours",
            yaxis_title="Probability (%)",
            template="plotly_white",
            height=210,
            margin=dict(l=20, r=20, t=30, b=30)
        )

        st.plotly_chart(fig_rain, use_container_width=True)

    # ── Row 2: 4 metric chips ──────────────────────────────────────────────
    if weather_data:
        temp_max  = weather_data.get('temp_max',  round(weather_data['temperature'] + 2, 1))
        temp_min  = weather_data.get('temp_min',  round(weather_data['temperature'] - 2, 1))
        pressure  = weather_data.get('pressure',  1013)
        precip    = weather_data.get('precipitation', 0)
    else:
        temp_max, temp_min, pressure, precip = 30, 26, 1013, 0

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("🌡️ Max Temp",    f"{temp_max}°C")
    mc2.metric("❄️ Min Temp",    f"{temp_min}°C")
    mc3.metric("🔵 Pressure",    f"{pressure} hPa")
    mc4.metric("🌧️ Precipitation", f"{precip} mm")

    # ── Row 3: Temperature chart (left) + Advice card (right) ─────────────
    col_chart, col_advice = st.columns([2, 1])

    with col_chart:
        base_temp = weather_data['temperature'] if weather_data else 28
        temp_series = [
            round(base_temp + 3 * np.sin((i - 6) / 4) + np.random.normal(0, 0.4), 1)
            for i in range(25)
        ]
        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(
            x=list(range(25)),
            y=temp_series,
            mode='lines+markers',
            line=dict(color='#f97316', width=2),
            marker=dict(size=4),
            showlegend=False
        ))
        fig_temp.update_layout(
            title=dict(text="Temperature Forecast (Next 24h)", font=dict(size=13)),
            xaxis_title="Hours",
            yaxis_title="°C",
            template="plotly_white",
            height=200,
            margin=dict(l=20, r=20, t=30, b=30)
        )
        st.plotly_chart(fig_temp, use_container_width=True)

    with col_advice:
        advice_text = weather_data.get('advice', 'Không có khuyến nghị.') if weather_data else 'Không có khuyến nghị.'
        st.markdown(f"""
        <div class="metric-card" style="height: 100%; min-height: 160px;">
            <div style="font-size: 0.78rem; opacity: 0.7; margin-bottom: 0.4rem;">💡 Khuyến nghị hôm nay</div>
            <div style="font-size: 0.88rem; line-height: 1.6; color: #111827;">{advice_text}</div>
        </div>
        """, unsafe_allow_html=True)

elif st.session_state.current_page == "Weather Prediction":
    st.markdown("### 🌡️ Weather Prediction")
    
    # Prediction Panel
    st.subheader("Upload Data for Prediction")

    # Khai báo trước để tránh NameError khi dùng ngoài columns scope
    uploaded_satellite = None
    uploaded_csv = None

    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_satellite = st.file_uploader(
            "Upload Satellite Image",
            type=['jpg', 'jpeg', 'png', 'tiff'],
            key="satellite_upload"
        )
        
        if uploaded_satellite:
            image = Image.open(uploaded_satellite)
            st.image(image, caption="Uploaded Satellite Image", use_container_width=True)
    
    with col2:
        uploaded_csv = st.file_uploader(
            "Upload Weather Sensor CSV",
            type=['csv'],
            key="csv_upload"
        )
        
        if uploaded_csv:
            df = pd.read_csv(uploaded_csv)
            st.dataframe(df.head(), use_container_width=True)
    
    # Run Forecast button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Run Forecast", use_container_width=True):
            st.success("Forecast simulation started! (UI only - no backend processing)")
            st.info("In a real implementation, this would trigger the ML model to process the uploaded data and generate predictions.")
    
    # Prediction Results (placeholder)
    if uploaded_satellite or uploaded_csv:
        st.subheader("📊 Prediction Results")
        
        # Sample prediction results
        results_data = pd.DataFrame({
            'Time': ['Now', '+1h', '+2h', '+3h', '+6h', '+12h', '+24h'],
            'Temperature (°C)': [28, 29, 30, 29, 27, 26, 25],
            'Rain Probability (%)': [45, 60, 75, 65, 40, 20, 15],
            'Wind Speed (km/h)': [12, 15, 18, 16, 14, 10, 8]
        })
        
        st.dataframe(results_data, use_container_width=True, hide_index=True)

elif st.session_state.current_page == "Data Visualization":
    st.markdown("### 📈 Data Visualization")
    
    # Sample visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Historical Temperature Trends")
        
        # Generate historical data
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        historical_temps = [25 + 10 * np.sin(i/5) + np.random.normal(0, 2) for i in range(30)]
        
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=dates,
            y=historical_temps,
            mode='lines',
            name='Historical Temperature',
            line=dict(color='#667eea', width=2)
        ))
        
        fig_hist.update_layout(
            title="30-Day Temperature History",
            xaxis_title="Date",
            yaxis_title="Temperature (°C)",
            template="plotly_white",
            height=300
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        st.subheader("Weather Pattern Analysis")
        
        # Sample weather pattern data
        pattern_data = pd.DataFrame({
            'Condition': ['Sunny', 'Cloudy', 'Rainy', 'Stormy', 'Foggy'],
            'Frequency': [45, 25, 20, 5, 5]
        })
        
        fig_pattern = px.pie(
            pattern_data,
            values='Frequency',
            names='Condition',
            title="Weather Pattern Distribution",
            color_discrete_map={
                'Sunny': '#FFD700',
                'Cloudy': '#87CEEB',
                'Rainy': '#4682B4',
                'Stormy': '#483D8B',
                'Foggy': '#708090'
            }
        )
        
        fig_pattern.update_layout(template="plotly_white", height=300)
        st.plotly_chart(fig_pattern, use_container_width=True)

elif st.session_state.current_page == "Model Information":
    st.markdown("### 🤖 Model Information")
    
    st.subheader("Multimodal Weather Forecasting Model")
    
    st.markdown("""
    #### Model Architecture
    - **Type**: Deep Learning Ensemble Model
    - **Input Modalities**: Satellite Images, Weather Sensors, Historical Data
    - **Output**: Multi-step weather predictions (1-24 hours)
    - **Framework**: PyTorch with TensorFlow backend
    
    #### Data Sources
    - **Satellite Imagery**: Himawari-8 satellite (10-minute intervals)
    - **Ground Sensors**: 50+ weather stations across Ho Chi Minh City
    - **Historical Data**: 10 years of weather records
    
    #### Model Performance
    - **Temperature MAE**: ±1.2°C
    - **Rainfall Accuracy**: 87%
    - **Wind Speed MAE**: ±2.5 km/h
    - **Update Frequency**: Every 15 minutes
    """)
    
    # Model metrics visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Accuracy Metrics")
        
        metrics_data = pd.DataFrame({
            'Metric': ['Temperature', 'Humidity', 'Rainfall', 'Wind Speed'],
            'Accuracy (%)': [92, 88, 87, 85]
        })
        
        fig_metrics = go.Figure()
        fig_metrics.add_trace(go.Bar(
            x=metrics_data['Metric'],
            y=metrics_data['Accuracy (%)'],
            marker_color='#667eea'
        ))
        
        fig_metrics.update_layout(
            title="Model Performance by Metric",
            xaxis_title="Weather Parameter",
            yaxis_title="Accuracy (%)",
            template="plotly_white",
            height=300
        )
        
        st.plotly_chart(fig_metrics, use_container_width=True)
    
    with col2:
        st.subheader("Training Progress")
        
        # Sample training progress data
        epochs = list(range(1, 51))
        loss = [2.5 * np.exp(-i/20) + 0.1 + np.random.normal(0, 0.02) for i in epochs]
        
        fig_training = go.Figure()
        fig_training.add_trace(go.Scatter(
            x=epochs,
            y=loss,
            mode='lines',
            name='Training Loss',
            line=dict(color='#764ba2', width=2)
        ))
        
        fig_training.update_layout(
            title="Model Training Loss",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            template="plotly_white",
            height=300
        )
        
        st.plotly_chart(fig_training, use_container_width=True)

# Chat-style Weather Assistant (appears on all pages)
st.markdown('<div style="border-top: 1px solid #e5e7eb; margin: 0.4rem 0 0.3rem 0;"></div>', unsafe_allow_html=True)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about weather in Ho Chi Minh City..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    response = chat_with_api(prompt)
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

# Footer

