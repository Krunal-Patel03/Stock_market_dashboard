import requests
from datetime import datetime
ticker = "AAPL"
url = f'https://eodhd.com/api/sentiments?s={ticker}&from=2024-01-01&to=2024-05-03&api_token=662a3477190930.15607695&fmt=json'
s_data = requests.get(url).json()
s_key = list(s_data.keys())
comp_state = s_key[0]

# Extracting date and normalized values
dates = [entry['date'] for entry in s_data[comp_state]]
sentiments = [entry['normalized'] for entry in s_data[comp_state]]
print(dates)
print(sentiments)