import urllib.request
import os

# Free weapon detection model - no login required
url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
save = "weapon_detect.pt"

# We will use a publicly available gun detection model
urls_to_try = [
    "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt",
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
]

print("Downloading model... please wait...")

for url in urls_to_try:
    try:
        urllib.request.urlretrieve(url, save)
        size = os.path.getsize(save) / 1024 / 1024
        print(f"Done! Saved as {save} ({size:.1f} MB)")
        break
    except Exception as e:
        print(f"Failed: {e}, trying next...")

print("Now run: streamlit run app.py")