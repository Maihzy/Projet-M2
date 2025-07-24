import requests

with open("DATA/raw/Classeur1.csv", "rb") as f:
    files = {"file": f}
    response = requests.post(
        "http://localhost:8000/predict-batch",
        files=files
    )
print(response.json())