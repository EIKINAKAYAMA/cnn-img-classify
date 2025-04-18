import cv2
import numpy as np
import os

# パラメータ（調整可能）
BRIGHTNESS_LOW_THRESHOLD = 70       # 低露出と判定する平均輝度の下限
BRIGHTNESS_HIGH_THRESHOLD = 190     # 過露出と判定する平均輝度の上限
CONTRAST_THRESHOLD = 20             # 低コントラストと判定する標準偏差の下限

def evaluate_exposure(image_path):
    # 画像読み込み（グレースケール）
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # 平均輝度と標準偏差（コントラスト）を計算
    mean_brightness = np.mean(img)
    std_contrast = np.std(img)

    # 判定ロジック
    if mean_brightness < BRIGHTNESS_LOW_THRESHOLD:
        result = "低露出"
    elif mean_brightness > BRIGHTNESS_HIGH_THRESHOLD:
        result = "過露出"
    else:
        result = "適正露出"

    if std_contrast < CONTRAST_THRESHOLD:
        result += "（低コントラスト）"

    return {
        "result": result,
        "mean": mean_brightness,
        "std": std_contrast
    }

def process_folder(folder_path):
    print(f"\nしきい値:")
    print(f"  ・平均輝度 < {BRIGHTNESS_LOW_THRESHOLD} → 低露出")
    print(f"  ・平均輝度 > {BRIGHTNESS_HIGH_THRESHOLD} → 過露出")
    print(f"  ・標準偏差 < {CONTRAST_THRESHOLD} → 低コントラスト\n")

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
            file_path = os.path.join(folder_path, filename)
            result_data = evaluate_exposure(file_path)

            if result_data:
                print(f"{filename}")
                print(f"  評価結果: {result_data['result']}")
                print(f"  平均輝度: {result_data['mean']:.2f}")
                print(f"  標準偏差: {result_data['std']:.2f}\n")
            else:
                print(f"{filename} の読み込みに失敗しました。\n")

if __name__ == "__main__":
    folder = "data/brightness"  # 評価対象のフォルダパス
    process_folder(folder)
