import requests

english_text = (
    "It was the best of times, it was the worst of times, it was the age "
    "of wisdom, it was the age of foolishness, it was the epoch of belief"
)
response = requests.post("http://172.166.194.89:8000/", json=english_text)
# response = requests.post("http://127.0.0.1:8000/", json=english_text)
# response = requests.post("http://0.0.0.0:8000/", json=english_text)
# response = requests.post("http://localhost:8000/", json=english_text)

french_text = response.text

print(french_text)
# 'c'était le meilleur des temps, c'était le pire des temps .'

# import ray
# import requests
# # import numpy as np

# @ray.remote
# def send_query(text):
#     # resp = requests.get("http://localhost:8000/?text={}".format(text))
#     resp = requests.post("http://127.0.0.1:8000/", json=text)
#     return resp.text

# # Let's use Ray to send all queries in parallel
# texts = [
#     'Once upon a time,',
#     'Hi my name is Lewis and I like to',
#     'My name is Mary, and my favorite',
#     'My name is Clara and I am',
#     'My name is Julien and I like to',
#     'Today I accidentally',
#     'My greatest wish is to',
#     'In a galaxy far far away',
#     'My best talent is',
# ]
# results = ray.get([send_query.remote(text) for text in texts])
# print("Result returned:", results)