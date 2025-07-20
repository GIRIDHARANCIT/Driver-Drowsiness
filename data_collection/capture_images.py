import cv2
import dlib
import os

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../python_app/shape_predictor/shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)
count_alert = 0
count_drowsy = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Left eye landmarks: points 36–41
        x1 = min([landmarks.part(n).x for n in range(36, 42)])
        y1 = min([landmarks.part(n).y for n in range(36, 42)])
        x2 = max([landmarks.part(n).x for n in range(36, 42)])
        y2 = max([landmarks.part(n).y for n in range(36, 42)])

        eye = frame[y1:y2, x1:x2]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 1)

        cv2.putText(frame, "Press 'a' = alert | 'd' = drowsy | 'q' = quit", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imshow('Data Collection', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('a') and faces:
        filename = f"dataset/alert/{count_alert}.jpg"
        cv2.imwrite(filename, eye)
        print(f"Saved {filename}")
        count_alert += 1

    elif key == ord('d') and faces:
        filename = f"dataset/drowsy/{count_drowsy}.jpg"
        cv2.imwrite(filename, eye)
        print(f"Saved {filename}")
        count_drowsy += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
