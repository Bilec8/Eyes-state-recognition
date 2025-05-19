import os
import urllib.request

CLASSIFIERS = {
    "haarcascade_frontalface_alt2.xml": "https://github.com/opencv/opencv/raw/4.x/data/haarcascades/haarcascade_frontalface_alt2.xml",
    "haarcascade_frontalface_alt.xml": "https://github.com/opencv/opencv/raw/4.x/data/haarcascades/haarcascade_frontalface_alt.xml",
    "haarcascade_profileface.xml": "https://github.com/opencv/opencv/raw/4.x/data/haarcascades/haarcascade_profileface.xml"
}

SAVE_DIR = "haarcascades"

os.makedirs(SAVE_DIR, exist_ok=True)

for filename, url in CLASSIFIERS.items():
    save_path = os.path.join(SAVE_DIR, filename)
    if not os.path.exists(save_path):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, save_path)
        print(f"{filename} downloaded!")
    else:
        print(f"{filename} exists.")

print("✅ Classifiers downloaded successfully!")
