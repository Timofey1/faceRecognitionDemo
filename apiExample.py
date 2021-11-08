import requests

BASE = "https://faceapi.nanosemantics.ai/"
files = {"file": open("image1.png", "rb")}
data = {"name":"brad pitt"}

# Find simular faces
response = requests.post(BASE+"find", files=files)

# Photo upl
response = requests.post(BASE+"api",data=data, files=files)

# Get data from db
response = requests.get(BASE+"db/2")

# Delete data
response = requests.delete(BASE+"db/1")

# Face detector, returns dict with facial areas and landmarks
response = requests.post(BASE+"detect", files=files)

# print(response.json())
