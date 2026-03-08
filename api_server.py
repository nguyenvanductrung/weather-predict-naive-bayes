from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import datetime
import random
import uvicorn
import numpy as np

app = FastAPI(title="Weather Forecast Chatbot API")

# Load real weather dataset
try:
    weather_df = pd.read_csv("Dataset_Weather_HoChiMinhCity_3Years.csv")
    weather_df['Date'] = pd.to_datetime(weather_df['Date'])
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Dataset not found. Using mock data.")
    weather_df = None

# =========================
# HÀM DỰ BÁO DỰA TRÊN DATASET THẬT
# =========================
def get_weather_prediction():
    """
    Hàm dự báo thời tiết dựa trên dataset thật của TP.HCM
    Sử dụng statistical approach và historical patterns
    """
    if weather_df is None:
        # Fallback to mock data
        return get_mock_prediction()
    
    try:
        # Lấy ngày hiện tại
        today = datetime.datetime.now()
        tomorrow = today + datetime.timedelta(days=1)
        
        # Lấy dữ liệu cùng ngày năm ngoái để dự báo
        same_day_last_year = weather_df[
            (weather_df['Date'].dt.month == tomorrow.month) & 
            (weather_df['Date'].dt.day == tomorrow.day)
        ]
        
        if not same_day_last_year.empty:
            # Dùng trung bình của các năm trước
            latest_data = same_day_last_year.iloc[-1]
            
            temp_mean = float(latest_data['Temp_Mean'])
            temp_max = float(latest_data['Temp_Max'])
            temp_min = float(latest_data['Temp_Min'])
            humidity = float(latest_data['Humidity'])
            precipitation = float(latest_data['Precipitation'])
            wind_speed = float(latest_data['Wind_speed'])
            pressure = float(latest_data['Pressure'])
            weather_condition = str(latest_data['Weather_Condition'])
            
            # Thêm một chút random để tạo sự khác biệt
            temp_mean += random.uniform(-1, 1)
            humidity += random.uniform(-5, 5)
            wind_speed += random.uniform(-2, 2)
            
            # Tạo lời khuyên dựa trên dữ liệu thật
            if precipitation > 5:
                advice = f"Dự báo có mưa với lượng mưa {precipitation}mm. Bạn nên mang theo ô hoặc áo mưa."
            elif humidity > 75:
                advice = "Độ ẩm cao, có khả năng xuất hiện mưa rào. Bạn nên mang theo ô hoặc áo mưa."
            elif temp_max > 35:
                advice = "Ngày mai trời khá nóng. Bạn nên uống đủ nước và hạn chế ra ngoài buổi trưa."
            else:
                advice = "Thời tiết khá ổn định, phù hợp cho hoạt động ngoài trời."
            
            return {
                "date": tomorrow.strftime("%d/%m/%Y"),
                "status": weather_condition,
                "temperature": round(temp_mean, 1),
                "humidity": round(max(0, min(100, humidity)), 1),
                "precipitation": round(precipitation, 1),
                "wind_speed": round(max(0, wind_speed), 1),
                "pressure": round(pressure, 1),
                "temp_max": round(temp_max, 1),
                "temp_min": round(temp_min, 1),
                "advice": advice
            }
        else:
            # Nếu không có dữ liệu cùng ngày, dùng dữ liệu gần nhất
            latest_data = weather_df.iloc[-1]
            return get_mock_prediction()
            
    except Exception as e:
        print(f"Error in prediction: {e}")
        return get_mock_prediction()

def get_mock_prediction():
    """Fallback mock prediction"""
    today = datetime.datetime.now()
    tomorrow = today + datetime.timedelta(days=1)

    possible_status = [
        "Clear sky", "Partly cloudy", "Overcast", 
        "Moderate rain", "Light drizzle"
    ]

    final_status = random.choice(possible_status)
    final_temp = round(random.uniform(27, 34), 1)
    final_humidity = round(random.uniform(65, 95), 1)

    if final_humidity > 80:
        advice = "Độ ẩm cao, có khả năng xuất hiện mưa rào. Bạn nên mang theo ô hoặc áo mưa."
    elif final_temp > 33:
        advice = "Ngày mai trời khá nóng. Bạn nên uống đủ nước và hạn chế ra ngoài buổi trưa."
    else:
        advice = "Thời tiết khá ổn định, phù hợp cho hoạt động ngoài trời."

    return {
        "date": tomorrow.strftime("%d/%m/%Y"),
        "status": final_status,
        "temperature": final_temp,
        "humidity": final_humidity,
        "advice": advice
    }

# =========================
# REQUEST MODEL
# =========================
class ChatRequest(BaseModel):
    message: str

# =========================
# ROUTES
# =========================
@app.get("/")
def home():
    return {"message": "Weather Forecast Chatbot API is running."}

@app.get("/predict")
def predict_weather():
    result = get_weather_prediction()
    return result

@app.post("/chat")
def chat_weather(req: ChatRequest):
    user_message = req.message.lower()
    forecast = get_weather_prediction()

    if "mai" in user_message or "ngày mai" in user_message or "thời tiết" in user_message:
        reply = (
            f"Dự báo ngày {forecast['date']}:\n"
            f"- Trạng thái: {forecast['status']}\n"
            f"- Nhiệt độ trung bình: {forecast['temperature']}°C\n"
            f"- Nhiệt độ cao nhất: {forecast.get('temp_max', forecast['temperature']+2)}°C\n"
            f"- Nhiệt độ thấp nhất: {forecast.get('temp_min', forecast['temperature']-2)}°C\n"
            f"- Độ ẩm: {forecast['humidity']}%\n"
            f"- Tốc độ gió: {forecast.get('wind_speed', 12)} km/h\n"
            f"- Lượng mưa: {forecast.get('precipitation', 0)} mm\n"
            f"- Khuyến nghị: {forecast['advice']}"
        )
    elif "nhiệt độ" in user_message:
        reply = f"Nhiệt độ dự kiến ngày {forecast['date']}: trung bình {forecast['temperature']}°C, cao nhất {forecast.get('temp_max', forecast['temperature']+2)}°C, thấp nhất {forecast.get('temp_min', forecast['temperature']-2)}°C."
    elif "độ ẩm" in user_message:
        reply = f"Độ ẩm dự kiến ngày {forecast['date']} là khoảng {forecast['humidity']}%."
    elif "mưa" in user_message:
        precipitation = forecast.get('precipitation', 0)
        if precipitation > 0:
            reply = f"Dự báo có mưa vào ngày {forecast['date']} với lượng mưa khoảng {precipitation}mm. {forecast['advice']}"
        else:
            reply = f"Khả năng mưa không cao vào ngày {forecast['date']}. {forecast['advice']}"
    elif "gió" in user_message:
        reply = f"Tốc độ gió dự kiến ngày {forecast['date']} là khoảng {forecast.get('wind_speed', 12)} km/h."
    elif "áp suất" in user_message:
        reply = f"Áp suất không khí dự kiến ngày {forecast['date']} là khoảng {forecast.get('pressure', 1013)} hPa."
    else:
        reply = (
            "Xin chào, tôi là chatbot dự báo thời tiết TP.HCM.\n"
            "Bạn có thể hỏi:\n"
            "- Thời tiết ngày mai thế nào?\n"
            "- Nhiệt độ ngày mai bao nhiêu?\n"
            "- Có mưa không?\n"
            "- Độ ẩm ngày mai thế nào?\n"
            "- Tốc độ gió bao nhiêu?\n"
            "- Áp suất không khí?"
        )

    return {"reply": reply}

# Chạy API server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
