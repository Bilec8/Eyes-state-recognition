import cv2 as cv
import time
import numpy as np

frontal_face = cv.CascadeClassifier("haarcascades/haarcascade_frontalface_alt2.xml")
profil_face = cv.CascadeClassifier("haarcascades/haarcascade_profileface.xml")

eye_face = cv.CascadeClassifier("eye_classifier/eye_cascade_fusek.xml")
mouth_face = cv.CascadeClassifier("haarcascades/haarcascade_smile.xml")

video = cv.VideoCapture("source/video.avi")

if not video.isOpened():    
    print("Error opening video file")


ground_truth = []
with open("source/eye_state.txt", "r") as f:
    ground_truth = f.read().splitlines()

correct = 0
total = 0

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    start_time = time.time()
    faces_frontal = frontal_face.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(200,200))
    faces_profile = profil_face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2, minSize=(200,200), maxSize=(500, 500))
    end_time = time.time()
    print(f"Profile face detection time: {end_time - start_time:.4f} seconds")

    all_faces = []

    for (x, y, w, h) in faces_frontal:
        all_faces.append((x, y, w, h, 'frontal'))  
    for (x, y, w, h) in faces_profile:
        all_faces.append((x, y, w, h, 'profile'))  


    if all_faces:
        x_min = min(face[0] for face in all_faces)
        y_min = min(face[1] for face in all_faces)
        x_max = max(face[0] + face[2] for face in all_faces)
        y_max = max(face[1] + face[3] for face in all_faces)

        cv.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        face_region_gray = gray[y_min:y_max, x_min:x_max]
        face_region_color = frame[y_min:y_max, x_min:x_max]

        eyes = eye_face.detectMultiScale(face_region_gray, scaleFactor=1.1, minNeighbors=5, minSize=(120, 120))
        detected_eye_state = "closed"

        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(face_region_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

            eye_region_gray = face_region_gray[ey:ey+eh, ex:ex+ew]
            eye_region_color = face_region_color[ey:ey+eh, ex:ex+ew]

            circles = cv.HoughCircles(
                    eye_region_gray,
                    cv.HOUGH_GRADIENT,  
                    dp=1,
                    minDist=30,
                    param1=60,
                    param2=15,
                    minRadius=13,
                    maxRadius=20
                                        )    
            
            if circles is not None:
                detected_eye_state = "open"
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    center = (i[0], i[1])
                    radius = i[2]
                    cv.circle(eye_region_color, center, radius, (0, 255, 0), 2)
                    cv.circle(eye_region_color, center, 2, (0, 0, 255), 3)

            if total < len(ground_truth):
                ground_truth_state = ground_truth[total]
                if detected_eye_state == ground_truth_state:
                    correct += 1

        mouth = mouth_face.detectMultiScale(face_region_gray, scaleFactor=1.5, minNeighbors=6, minSize=(120, 120))
        for (mx, my, mw, mh) in mouth:
            cv.rectangle(face_region_color, (mx, my), (mx + mw, my + mh), (0, 0, 255), 2)

    cv.imshow("Video", frame)
    total += 1

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

accuracy = correct / total if total > 0 else 0
print(f"Accuracy: {accuracy * 100:.2f}%")



video.release()
cv.destroyAllWindows()