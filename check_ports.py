import urllib.request
import time

def check_url(url):
    try:
        with urllib.request.urlopen(url) as response:
            return response.status == 200
    except Exception as e:
        return False

print("Backend (8000):", check_url("http://localhost:8000/docs"))
print("Streamlit (8501):", check_url("http://localhost:8501"))
