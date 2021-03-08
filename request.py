import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Recency (months)':2, 'Frequency (times)':9, 'Monetary (c.c. blood)':2000},'Time (months)':20)

print(r.json())