# 👁️ Eye State Detection – Python + OpenCV Tutorial 🚀

This project performs face and eye detection using pre-trained Haar Cascade classifiers and determines whether the eye is open or closed. The system processes each frame of a video, applies facial and eye detection, and computes the accuracy of predictions based on the ground truth data.

## 🧰 Requirements

- Python 3.9+
- OpenCV (with support for `detectMultiScale3`)
- Haar cascade classifiers (.xml files)
- A video file (`video.avi`)
- Ground truth file (`eye_state.txt`)

---

### 1️⃣ Set up a virtual environment

To avoid conflicts with other projects, it's best to use a virtual environment. You can set one up by following these steps:

- For Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

- For macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2️⃣ Install dependencies

```bash
pip install requirements.txt
```

├── haarcascades/               # Folder containing Haar Cascade files
│   ├── haarcascade_frontalface_alt2.xml
│   ├── haarcascade_frontalface_alt.xml
│   ├── haarcascade_profileface.xml
│   ├── haarcascade_smile.xml
│   └── haarcascade_mcs_mouth.xml
├── eye_classifier/             # Folder containing the eye detection classifier
│   └── eye_cascade_fusek.xml
├── source/                     # Source directory for video and ground truth file
│   ├── video.avi               # Input video file
│   └── eye_state.txt           # Ground truth file
├── main.py                     # Main Python script
├── download_classifiers.py     # Download script
└── README.md                   # Project documentation

## 📥 Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Bilec8/Eyes-state-recognition.git
cd Eyes-state-recognition
```
### 2️⃣ Install required dependencies
You will need the pre-trained Haar Cascade classifiers for face and eye detection. You can download them using the download_classifiers.py script:

```bash
python download_classifiers.py
```
This will download the necessary XML files into the haarcascades/ directory.

# Running the Project

To run the project, execute the main.py script:

```bash
python main.py
```

This will start the detection process, where the video will be processed frame by frame. The system will display the video and print the ground truth versus the predicted eye state (open/closed).

# 🤖 Tips
- Press q to quit the video early.
- The eye state prediction is based on the presence of detected iris-like circles via Hough Circle Transform.


# ✅ Result

Final accuracy: 91.74%

