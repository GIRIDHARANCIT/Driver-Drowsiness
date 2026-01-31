import cv2
import dlib
import numpy as np
from tensorflow.keras.models import load_model
import pygame
import os

# -----------------------------------------------------------
# Load CNN model and shape predictor
model = load_model('python_app/model/drowsiness_model.h5')
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
predictor_path = os.path.join(BASE_DIR, 'shape_predictor', 'shape_predictor_68_face_landmarks.dat')

predictor = dlib.shape_predictor(predictor_path)
detector = dlib.get_frontal_face_detector()
pygame.mixer.init()
alert_sound = pygame.mixer.Sound('alert.wav') 
def get_eye_regions(shape):
    left_eye = shape[36:42]
    right_eye = shape[42:48]
    return left_eye, right_eye
def crop_eye(frame, eye_points):
    x = np.min(eye_points[:,0])
    y = np.min(eye_points[:,1])
    w = np.max(eye_points[:,0]) - x
    h = np.max(eye_points[:,1]) - y
    eye_img = frame[y:y+h, x:x+w]
    eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    eye_img = cv2.resize(eye_img, (64, 64))
    eye_img = eye_img.reshape(1, 64, 64, 1) / 255.0
    return eye_img
cap = cv2.VideoCapture(0)
print("🚀 Starting webcam... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        shape_np = np.zeros((68, 2), dtype=int)

        for i in range(68):
            shape_np[i] = (shape.part(i).x, shape.part(i).y)
        left_eye, right_eye = get_eye_regions(shape_np)
        for (x, y) in np.concatenate((left_eye, right_eye)):
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        left_eye_img = crop_eye(frame, left_eye)
        right_eye_img = crop_eye(frame, right_eye)
        pred_left = model.predict(left_eye_img)[0][0]
        pred_right = model.predict(right_eye_img)[0][0]
        avg_pred = (pred_left + pred_right) / 2

        status = "Alert"
        color = (0, 255, 0)

        if avg_pred > 0.5:
            status = "Drowsy"
            color = (0, 0, 255)
            pygame.mixer.Sound.play(alert_sound)

        cv2.putText(frame, f"Status: {status}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Webcam stopped.")
