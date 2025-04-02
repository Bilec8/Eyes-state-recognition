import cv2 as cv
import time

frontal_face = cv.CascadeClassifier("haarcascades/haarcascade_frontalface_alt2.xml")
frontal_face2 = cv.CascadeClassifier("haarcascades/haarcascades-kipr/haarcascade_frontalface_alt.xml")

profile_face = cv.CascadeClassifier("haarcascades/haarcascades-kipr/haarcascade_profileface.xml")

mouth_face = cv.CascadeClassifier("haarcascades/haarcascade_smile.xml")
mouth_face2 = cv.CascadeClassifier("haarcascades/haarcascades-kipr/haarcascade_mcs_mouth.xml")

eye_face = cv.CascadeClassifier("eye_classifier/eye_cascade_fusek.xml")

video = cv.VideoCapture("source/video.avi")

def load_ground_truth(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file.readlines()]
    
def calculate_accuracy(predictions, ground_truth):
    correct = 0
    total = len(ground_truth)

    for i in range(total):
        if predictions[i] == ground_truth[i]:
            correct += 1
    
    return correct / total if total > 0 else 0

def detect_eye_state(circles):
    return "open" if circles is not None else "close"
    
if not video.isOpened():    
    print("Error opening video file")
    
ground_truth = load_ground_truth("source/eye_state.txt")

frame_counter = 0
predictions = []

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)
    gray = cv.GaussianBlur(gray, (5, 5), 0)

    start_time = time.time()

    faces_frontal1, rejectLevelsFrontal1, levelWeightsFrontal1 = frontal_face.detectMultiScale3(gray, scaleFactor=1.3, minNeighbors=5, minSize=(200,200), outputRejectLevels=True)
    faces_frontal2, rejectLevelsFrontal2, levelWeightsFrontal2 = frontal_face2.detectMultiScale3(gray, scaleFactor=1.3, minNeighbors=5, minSize=(200,200), outputRejectLevels=True)
    faces_profile, rejectLevelsProfile, levelWeightsProfile = profile_face.detectMultiScale3(gray, scaleFactor=1.1, minNeighbors=2, minSize=(200,200), maxSize=(500, 500), outputRejectLevels=True)

    end_time = time.time()
    print(f"Detection Time: {end_time - start_time:.2f} seconds")

    best_face = None
    best_weight = -1

    for i in range(len(faces_frontal1)):
        if levelWeightsFrontal1[i] > best_weight:
            best_weight = levelWeightsFrontal1[i]
            best_face = faces_frontal1[i]

    for i in range(len(faces_frontal2)):
        if levelWeightsFrontal2[i] > best_weight:
            best_weight = levelWeightsFrontal2[i]
            best_face = faces_frontal2[i]

    for i in range(len(faces_profile)):
        if levelWeightsProfile[i] > best_weight:
            best_weight = levelWeightsProfile[i]
            best_face = faces_profile[i]

    if best_face is not None:
        x, y, w, h = best_face
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  

        face_roi = gray[y:y + h, x:x + w]

        best_face_m = None
        best_weight_m = -1

        faces_mouth, rejectLevelsMouth, levelWeightsMouth = mouth_face.detectMultiScale3(face_roi, scaleFactor=1.5, minNeighbors=6, minSize=(120, 120), outputRejectLevels=True)
        faces_mouth2, rejectLevelsMouth2, levelWeightsMouth2 = mouth_face2.detectMultiScale3(face_roi, scaleFactor=1.5, minNeighbors=6, minSize=(120, 120), outputRejectLevels=True)

        for i in range(len(faces_mouth)):
            if levelWeightsMouth[i] > best_weight_m:
                best_weight_m = levelWeightsMouth[i]
                best_face_m = faces_mouth[i]

        for i in range(len(faces_mouth2)):
            if levelWeightsMouth2[i] > best_weight_m:
                best_weight_m = levelWeightsMouth2[i]
                best_face_m = faces_mouth2[i]

        if best_face_m is not None:
            x_m, y_m, w_m, h_m = best_face_m
            cv.rectangle(frame, (x + x_m, y + y_m), (x + x_m + w_m, y + y_m + h_m), (255, 0, 0), 2) 

        eye_y_max = y + h // 2

        faces_eye, rejectionLevelsEye, weightLevelsEye = eye_face.detectMultiScale3(face_roi, scaleFactor=1.1, minNeighbors=6, minSize=(30, 30), maxSize=(120, 120),flags=cv.CASCADE_SCALE_IMAGE, outputRejectLevels=True)

        filtered_eyes = []
        for (ex, ey, ew, eh) in faces_eye:
            if y + ey < eye_y_max: 
                filtered_eyes.append((ex, ey, ew, eh))
        
        for (ex, ey, ew, eh) in filtered_eyes:
            cv.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 255), 2)

            eye_roi = face_roi[ey:ey + eh, ex:ex + ew]
            blur_eye = cv.GaussianBlur(eye_roi, (5, 5), 0)
            circles = cv.HoughCircles(
                blur_eye,
                cv.HOUGH_GRADIENT,  
                dp=1,
                minDist=25,
                param1=60,
                param2=15,
                minRadius=11,
                maxRadius=20
            )

            if circles is not None:
                circles = circles[0, :]
                for (cx, cy, r) in circles:
                    cv.circle(frame, (x + ex + int(cx), y + ey + int(cy)), int(r), (0, 0, 255), 2)

            if circles is not None:
                eye_state = "open"
            else:
                eye_state = "close"

        #print(f"Eye State: {eye_state}")

        predictions.append(eye_state)
        frame_counter += 1
        

    else: 
        #predictions.append("close")
        predictions.append("unknown")
        frame_counter += 1

    end_time2 = time.time()
    #print(f"Processing Time: {end_time2 - start_time:.2f} seconds")

    cv.imshow("Video", frame)

    print(f"Ground Truth: {ground_truth[frame_counter - 1]}, Prediction: {predictions[frame_counter - 1]}")
    #time.sleep(0.5)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

accuracy = calculate_accuracy(predictions, ground_truth)
print(f"Accuracy: {accuracy * 100:.2f}%")

video.release()
cv.destroyAllWindows()