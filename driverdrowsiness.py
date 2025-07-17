import cv2
import dlib
import numpy as np
import pygame
import threading
from scipy.spatial import distance
from scipy.io.wavfile import write
import os

# Initialize pygame mixer for alarm
pygame.mixer.init()

# Generate alarm sound if not exists
def generate_alarm():
    sample_rate = 44100
    duration = 3
    frequency = 1000
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = 0.5 * np.sin(2 * np.pi * frequency * t)
    wave = np.int16(wave * 32767)
    write("alarm.wav", sample_rate, wave)

if not os.path.exists("alarm.wav"):
    generate_alarm()

def play_alarm():
    pygame.mixer.music.load("alarm.wav")
    pygame.mixer.music.play(-1)

def stop_alarm():
    pygame.mixer.music.stop()

# Load Dlib models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# EAR & MAR Calculation Functions
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])
    B = distance.euclidean(mouth[4], mouth[8])
    C = distance.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)

# Thresholds
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.5
CLOSED_EYE_FRAMES = 20
YAWN_FRAMES = 20

# Initialize alarm control variables
eye_counter, yawn_counter = 0, 0
alarm_on = False
lock = threading.Lock()

# Start video capture
cap = cv2.VideoCapture(0)

def process_video():
    global eye_counter, yawn_counter, alarm_on

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        alert_text = ""

        for face in faces:
            landmarks = predictor(gray, face)

            # Extract facial landmarks
            left_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)])
            right_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)])
            mouth = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(48, 68)])

            # Compute EAR and MAR
            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
            mar = mouth_aspect_ratio(mouth)

            # Check drowsiness
            if ear < EAR_THRESHOLD:
                eye_counter += 1
                if eye_counter >= CLOSED_EYE_FRAMES:
                    alert_text = "DROWSINESS ALERT!"
                    if not alarm_on:
                        with lock:
                            alarm_on = True
                            threading.Thread(target=play_alarm, daemon=True).start()
            else:
                eye_counter = max(0, eye_counter - 1)

            # Check yawning
            if mar > MAR_THRESHOLD:
                yawn_counter += 1
                if yawn_counter >= YAWN_FRAMES:
                    alert_text = "YAWNING ALERT!"
                    if not alarm_on:
                        with lock:
                            alarm_on = True
                            threading.Thread(target=play_alarm, daemon=True).start()
            else:
                yawn_counter = max(0, yawn_counter - 1)

            # If no alerts, stop alarm
            if eye_counter == 0 and yawn_counter == 0 and alarm_on:
                with lock:
                    stop_alarm()
                    alarm_on = False

            # Draw bounding box (No Dots)
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display alert text on screen
        if alert_text:
            cv2.putText(frame, alert_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Show video frame
        cv2.imshow("Driver Drowsiness Detection", frame)

        # Press 'Q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Run video processing in the main thread
process_video()

# Release resources
cap.release()
cv2.destroyAllWindows()
