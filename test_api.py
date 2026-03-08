import requests
import json

# Test GET /predict
response = requests.get("http://127.0.0.1:8000/predict")
print("=== GET /predict ===")
print("Status:", response.status_code)
print("Response:", json.dumps(response.json(), indent=2, ensure_ascii=False))

# Test POST /chat
print("\n=== POST /chat ===")
chat_data = {"message": "thời tiết ngày mai thế nào"}
response = requests.post("http://127.0.0.1:8000/chat", json=chat_data)
print("Status:", response.status_code)
print("Response:", json.dumps(response.json(), indent=2, ensure_ascii=False))
