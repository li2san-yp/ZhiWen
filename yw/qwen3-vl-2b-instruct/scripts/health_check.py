import requests

url = "http://192.168.116.5:4543/api/health"

resp = requests.get(url)

print(resp.text)

resp.close()