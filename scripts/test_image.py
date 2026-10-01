import requests
import time

start = time.time()
r = requests.post("http://127.0.0.1:8888/chat", json={"text": "Generate an image of a cyberpunk humanoid robot in neon Tokyo."})
print(r.json())
print(f"Time: {time.time() - start}")
