import requests
import time

start = time.time()
r = requests.post("http://127.0.0.1:8888/chat", json={"text": "Hello robot, give me a very short 1 sentence response."})
print(r.json())
print(f"Time: {time.time() - start}")
