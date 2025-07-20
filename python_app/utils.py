import numpy as np
import pygame
import time

def get_eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

def sound_alarm():
    pygame.mixer.init()
    pygame.mixer.music.load("alarm.wav")
    pygame.mixer.music.play()
    time.sleep(1)
    pygame.mixer.music.stop()
