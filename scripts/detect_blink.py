import cv2
import dlib

# モデル読み込み
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 目のランドマークインデックス
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

# EAR（Eye Aspect Ratio）計算関数
from scipy.spatial import distance

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# 写真読み込みと検出処理
image = cv2.imread('photo.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = detector(gray)

for face in faces:
    landmarks = predictor(gray, face)
    left_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in LEFT_EYE]
    right_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in RIGHT_EYE]
    
    left_EAR = eye_aspect_ratio(left_eye)
    right_EAR = eye_aspect_ratio(right_eye)
    ear = (left_EAR + right_EAR) / 2.0
    
    if ear < 0.21:  # 閾値（調整可）
        print("目を閉じている")
    else:
        print("目を開けている")
