import cv2
import dlib

# モデル読み込み
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')

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
image = cv2.imread('data/face3.png')
if image is None:
    print("画像が読み込めません。パスを確認してください。")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = detector(gray)

print(f"検出された顔の数: {len(faces)}")

for face in faces:
    landmarks = predictor(gray, face)
    left_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in LEFT_EYE]
    right_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in RIGHT_EYE]

    left_EAR = eye_aspect_ratio(left_eye)
    right_EAR = eye_aspect_ratio(right_eye)
    ear = (left_EAR + right_EAR) / 2.0
    print(f"EAR: {ear:.3f}")

    if ear < 0.21:
        print("目を閉じている")
    else:
        print("目を開けている")
