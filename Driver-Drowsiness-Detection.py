import cv2
import dlib
import time
import numpy as np
from scipy.spatial import distance as dist
import playsound  # Cross-platform sound alert

# Load pre-trained face and landmark detector
face_detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download file

# Define eye aspect ratio (EAR) function
def calculate_ear(eye_points):
    A = dist.euclidean(eye_points[1], eye_points[5])  # Vertical distance
    B = dist.euclidean(eye_points[2], eye_points[4])  # Vertical distance
    C = dist.euclidean(eye_points[0], eye_points[3])  # Horizontal distance
    ear = (A + B) / (2.0 * C)
    return ear

# Beep sound function
def beep_alert():
    playsound.playsound("beep.mp3")  # Ensure beep.mp3 is in the same folder

# Capture video from the laptop camera
cap = cv2.VideoCapture(0)

# EAR threshold and detection duration
EAR_THRESHOLD = 0.2  # Adjust if necessary
SLEEP_TIME_THRESHOLD = 3  # Time (in seconds) to detect sleep

# Initialize timers
face_lost_time = None
sleep_start_time = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector(gray)

    if len(faces) > 0:
        face_lost_time = None  # Reset face lost timer

        for face in faces:
            landmarks = landmark_detector(gray, face)

            # Get eye landmarks (Left: 36-41, Right: 42-47)
            left_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
            right_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

            # Compute EAR for both eyes
            left_ear = calculate_ear(left_eye_points)
            right_ear = calculate_ear(right_eye_points)
            avg_ear = (left_ear + right_ear) / 2.0

            # If EAR is below threshold (eyes closed)
            if avg_ear < EAR_THRESHOLD:
                if sleep_start_time is None:
                    sleep_start_time = time.time()

                if time.time() - sleep_start_time >= SLEEP_TIME_THRESHOLD:
                    print("🚨 Driver is asleep for 3 seconds! Beep Alert!")
                    beep_alert()  # Play sound
                    sleep_start_time = None  # Reset after alert
            else:
                sleep_start_time = None  # Reset if eyes are open

    else:
        if face_lost_time is None:
            face_lost_time = time.time()

        if time.time() - face_lost_time >= 3:
            print("🚨 No Face Detected for 3 seconds! Beep Alert!")
            beep_alert()  # Play sound
            face_lost_time = None  # Reset after alert

    # Display the camera feed
    cv2.imshow("Driver Monitoring", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
