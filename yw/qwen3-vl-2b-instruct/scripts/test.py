import requests

url = "http://192.168.127.5:4543/api/generate"

payload = {
    "prompt": "介绍湖南大学",
    "max_new_tokens": 100,
    "temperature": 0.7,
    "top_p": 0.9
}

resp = requests.post(url, json=payload)

print(resp.json())

resp.close()