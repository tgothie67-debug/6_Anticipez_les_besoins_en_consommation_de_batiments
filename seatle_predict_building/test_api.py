import requests

with open("seatle_predict_building/payload.json") as f:
    payload = f.read()

response = requests.post(
    "http://localhost:3000/predict",
    data=payload,
    headers={"Content-Type": "application/json"}
)

print(response.status_code)
print(response.text)