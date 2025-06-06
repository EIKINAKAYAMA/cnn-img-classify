import cv2
import dlib
import os
from scipy.spatial import distance
from mtcnn import MTCNN

# EARしきい値
EAR_THRESHOLD = 0.21

# MTCNNで顔検出
detector_mtcnn = MTCNN()

# dlibのランドマークモデル
predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')

# 目のランドマークインデックス
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

# EAR計算関数
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# フォルダ内の画像を処理
def process_folder(folder_path):
    print(f"\nEARしきい値: {EAR_THRESHOLD} 以下で「目を閉じている」と判定\n")

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            file_path = os.path.join(folder_path, filename)
            image = cv2.imread(file_path)

            if image is None:
                print(f"{filename}: 画像を読み込めません")
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = detector_mtcnn.detect_faces(rgb)

            print(f"{filename}")
            print(f"  検出された顔の数: {len(results)}")

            if len(results) == 0:
                print(f"  → 顔が見つかりません\n")
                continue

            for i, result in enumerate(results):
                x, y, width, height = result['box']
                # dlibに渡すためのrectを作成
                rect = dlib.rectangle(left=x, top=y, right=x + width, bottom=y + height)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                landmarks = predictor(gray, rect)

                left_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in LEFT_EYE]
                right_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in RIGHT_EYE]

                left_EAR = eye_aspect_ratio(left_eye)
                right_EAR = eye_aspect_ratio(right_eye)
                ear = (left_EAR + right_EAR) / 2.0

                print(f"  顔{i+1}のEAR: {ear:.3f}")
                if ear < EAR_THRESHOLD:
                    print("  → 目を閉じている")
                else:
                    print("  → 目を開けている")
            print()  # 改行

# 使用例
if __name__ == "__main__":
    folder = "data/blink"  # 処理するフォルダパス
    process_folder(folder)
