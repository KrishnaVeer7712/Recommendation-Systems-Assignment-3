import requests

url = 'http://localhost:5000/recommend'
r = requests.post(url,json={'key':2})

print(r.json())