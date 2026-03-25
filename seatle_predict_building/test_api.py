import requests

# Fichier JSON avec les nouveaux bâtiments à prédire
with open("payload.json") as f:
    payload = f.read()

response = requests.post(
    "http://35.180.125.72:3000/predict", # lien cloud AWS
    data=payload,
    headers={"Content-Type": "application/json"}
)

print(response.status_code)
print(response.text)