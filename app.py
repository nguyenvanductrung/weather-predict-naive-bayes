import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="Multimodal Weather Forecasting System",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
def load_custom_css():
    st.markdown("""
    <style>
        /* Main styling */
        .main-header {
            text-align: center;
            padding: 2rem 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 15px;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            color: white;
            margin-bottom: 1rem;
            transition: transform 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px rgba(0,0,0,0.2);
        }
        
        .prediction-card {
            background: #1e1e1e;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            border: 1px solid #333;
            margin-bottom: 2rem;
        }
        
        .chat-container {
            background: #2a2a2a;
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            border: 1px solid #333;
        }
        
        /* Sidebar styling */
        .sidebar-content {
            background: #1a1a1a;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        
        .nav-item {
            padding: 0.8rem 1rem;
            margin: 0.5rem 0;
            border-radius: 8px;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .nav-item:hover {
            background: #333;
            transform: translateX(5px);
        }
        
        .nav-item.active {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        /* Chart containers */
        .chart-container {
            background: #1e1e1e;
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            border: 1px solid #333;
            margin-bottom: 2rem;
        }
        
        /* Upload area styling */
        .upload-area {
            background: #2a2a2a;
            border: 2px dashed #667eea;
            border-radius: 10px;
            padding: 2rem;
            text-align: center;
            margin: 1rem 0;
            transition: all 0.3s ease;
        }
        
        .upload-area:hover {
            border-color: #764ba2;
            background: #333;
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 0.8rem 2rem;
            border-radius: 8px;
            font-weight: bold;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
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
        
        /* Chat message styling */
        .user-message {
            background: #667eea;
            color: white;
            padding: 1rem;
            border-radius: 15px;
            margin: 0.5rem 0;
            max-width: 80%;
            margin-left: auto;
        }
        
        .assistant-message {
            background: #333;
            color: white;
            padding: 1rem;
            border-radius: 15px;
            margin: 0.5rem 0;
            max-width: 80%;
        }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state for chat
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'current_page' not in st.session_state:
    st.session_state.current_page = "Dashboard"

# Load custom CSS
load_custom_css()

# Sidebar
with st.sidebar:
    # App logo/icon
    st.markdown("""
    <div class="sidebar-content" style="text-align: center;">
        <h1 style="font-size: 2.5rem; margin: 0;">🌤️</h1>
        <h3 style="margin: 0.5rem 0; color: #667eea;">Weather AI</h3>
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
    
    # Settings section
    st.markdown("### ⚙️ Settings")
    
    forecast_horizon = st.selectbox(
        "Forecast Horizon",
        ["1 hour", "6 hours", "24 hours"],
        index=1
    )
    
    data_modality = st.selectbox(
        "Data Modality",
        ["Satellite Images", "Weather Sensors", "Historical Data", "All Modalities"],
        index=3
    )

# Main header
st.markdown("""
<div class="main-header">
    <h1 style="margin: 0; font-size: 2.5rem;">🌤️ Multimodal Weather Forecasting System</h1>
    <h2 style="margin: 0.5rem 0; font-weight: 300;">Ho Chi Minh City</h2>
    <p style="margin: 0; opacity: 0.9;">Advanced AI-powered weather prediction platform</p>
</div>
""", unsafe_allow_html=True)

# Main content based on selected page
if st.session_state.current_page == "Dashboard":
    # Current Weather Section
    st.markdown("### 🌡️ Current Weather")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4 style="margin: 0; font-size: 0.9rem; opacity: 0.8;">Temperature</h4>
            <h2 style="margin: 0.5rem 0;">28°C</h2>
            <p style="margin: 0; font-size: 0.8rem; opacity: 0.7;">Feels like 32°C</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4 style="margin: 0; font-size: 0.9rem; opacity: 0.8;">Humidity</h4>
            <h2 style="margin: 0.5rem 0;">75%</h2>
            <p style="margin: 0; font-size: 0.8rem; opacity: 0.7;">Moderate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4 style="margin: 0; font-size: 0.9rem; opacity: 0.8;">Wind Speed</h4>
            <h2 style="margin: 0.5rem 0;">12 km/h</h2>
            <p style="margin: 0; font-size: 0.8rem; opacity: 0.7;">Northeast</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h4 style="margin: 0; font-size: 0.9rem; opacity: 0.8;">Rain Probability</h4>
            <h2 style="margin: 0.5rem 0;">45%</h2>
            <p style="margin: 0; font-size: 0.8rem; opacity: 0.7;">Moderate chance</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Forecast Visualization
    st.markdown("### 📈 Forecast Visualization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Temperature Forecast")
        
        # Generate sample temperature data
        hours = list(range(25))
        temps = [28 + np.sin(i/4) * 3 + np.random.normal(0, 0.5) for i in hours]
        
        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(
            x=hours,
            y=temps,
            mode='lines+markers',
            name='Temperature',
            line=dict(color='#667eea', width=3),
            marker=dict(size=6)
        ))
        
        fig_temp.update_layout(
            title="24-Hour Temperature Forecast",
            xaxis_title="Hours from now",
            yaxis_title="Temperature (°C)",
            template="plotly_dark",
            height=300,
            showlegend=False
        )
        
        st.plotly_chart(fig_temp, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Rainfall Probability")
        
        # Generate sample rainfall probability data
        rain_prob = [20 + 30 * np.sin(i/6 + 1) + np.random.normal(0, 5) for i in hours]
        rain_prob = [max(0, min(100, p)) for p in rain_prob]
        
        fig_rain = go.Figure()
        fig_rain.add_trace(go.Scatter(
            x=hours,
            y=rain_prob,
            mode='lines+markers',
            name='Rain Probability',
            line=dict(color='#764ba2', width=3),
            marker=dict(size=6),
            fill='tonexty'
        ))
        
        fig_rain.update_layout(
            title="24-Hour Rainfall Probability",
            xaxis_title="Hours from now",
            yaxis_title="Probability (%)",
            template="plotly_dark",
            height=300,
            showlegend=False
        )
        
        st.plotly_chart(fig_rain, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Multimodal Data Section
    st.markdown("### 🛰️ Multimodal Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Satellite Image")
        # Placeholder for satellite image
        st.image("https://picsum.photos/seed/satellite/400/300.jpg", caption="Latest Satellite Image")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Radar Visualization")
        # Placeholder for radar
        st.image("https://picsum.photos/seed/radar/400/300.jpg", caption="Weather Radar")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Sensor Data")
        
        # Sample sensor data table
        sensor_data = pd.DataFrame({
            'Sensor': ['Temp-01', 'Hum-01', 'Wind-01', 'Press-01'],
            'Value': ['28.5°C', '75%', '12 km/h', '1013 hPa'],
            'Status': ['Normal', 'Normal', 'Normal', 'Normal']
        })
        
        st.dataframe(sensor_data, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.current_page == "Weather Prediction":
    st.markdown("### 🌡️ Weather Prediction")
    
    # Prediction Panel
    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
    st.subheader("Upload Data for Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="upload-area">', unsafe_allow_html=True)
        uploaded_satellite = st.file_uploader(
            "Upload Satellite Image",
            type=['jpg', 'jpeg', 'png', 'tiff'],
            key="satellite_upload"
        )
        
        if uploaded_satellite:
            image = Image.open(uploaded_satellite)
            st.image(image, caption="Uploaded Satellite Image", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="upload-area">', unsafe_allow_html=True)
        uploaded_csv = st.file_uploader(
            "Upload Weather Sensor CSV",
            type=['csv'],
            key="csv_upload"
        )
        
        if uploaded_csv:
            df = pd.read_csv(uploaded_csv)
            st.dataframe(df.head(), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Run Forecast button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Run Forecast", use_container_width=True):
            st.success("Forecast simulation started! (UI only - no backend processing)")
            st.info("In a real implementation, this would trigger the ML model to process the uploaded data and generate predictions.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction Results (placeholder)
    if uploaded_satellite or uploaded_csv:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("📊 Prediction Results")
        
        # Sample prediction results
        results_data = pd.DataFrame({
            'Time': ['Now', '+1h', '+2h', '+3h', '+6h', '+12h', '+24h'],
            'Temperature (°C)': [28, 29, 30, 29, 27, 26, 25],
            'Rain Probability (%)': [45, 60, 75, 65, 40, 20, 15],
            'Wind Speed (km/h)': [12, 15, 18, 16, 14, 10, 8]
        })
        
        st.dataframe(results_data, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.current_page == "Data Visualization":
    st.markdown("### 📈 Data Visualization")
    
    # Sample visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
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
            template="plotly_dark",
            height=300
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
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
        
        fig_pattern.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig_pattern, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.current_page == "Model Information":
    st.markdown("### 🤖 Model Information")
    
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
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
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Model metrics visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
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
            template="plotly_dark",
            height=300
        )
        
        st.plotly_chart(fig_metrics, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
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
            template="plotly_dark",
            height=300
        )
        
        st.plotly_chart(fig_training, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Chat-style Weather Assistant (appears on all pages)
st.markdown("---")
st.markdown("### 💬 Weather Assistant")

st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about weather in Ho Chi Minh City..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate assistant response (simulated)
    if "tomorrow" in prompt.lower():
        response = "Based on current weather patterns, tomorrow in Ho Chi Minh City is expected to be partly cloudy with a high of 31°C and a 30% chance of rain in the afternoon."
    elif "temperature" in prompt.lower():
        response = "The current temperature in Ho Chi Minh City is 28°C, with a feels-like temperature of 32°C due to humidity."
    elif "rain" in prompt.lower():
        response = "There's a 45% chance of rain in the next 6 hours. The highest probability is between 2-4 PM."
    elif "weekend" in prompt.lower():
        response = "This weekend looks mostly sunny with temperatures around 30-32°C. Perfect for outdoor activities!"
    else:
        response = "I'm here to help with weather information for Ho Chi Minh City. You can ask about temperature, rain chances, forecasts, and weather patterns."
    
    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; padding: 2rem 0; color: #666;">
    <p>Multimodal Weather Forecasting System © 2024 | Powered by AI</p>
    <p style="font-size: 0.8rem;">Real-time weather predictions for Ho Chi Minh City</p>
</div>
""", unsafe_allow_html=True)
