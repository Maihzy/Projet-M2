import requests

payload = {
    "dti": 18.5,
    "annual_inc": 50000,
    "loan_amnt": 15000,
    "int_rate": 12.5,
    "revol_util": 45.0
}
response = requests.post(
    "http://localhost:8000/predict-simple",
    json=payload
)
print(response.json())