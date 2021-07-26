import requests

BASE = "http://127.0.0.1:5000/"
files = {"file": open("image.jpg", "rb")}
data = {"name":"Fname Lname"}

# # Find simular faces
# response = requests.get(BASE+"photoUpl", files=files)

# # Photo upl 
# response = requests.post(BASE+"photoUpl",data=data, files=files)

# # Get data from db
# response = requests.get(BASE+"dbApi/<faceId>")

# # Delete data
# response = requests.delete(BASE+"dbApi/<faceId>")

# # Face detector, returns dict with facial areas and landmarks
# response = requests.post(BASE+"mtcnnFaceApi", files=files)

# print(response.json())
