import os
from collections import defaultdict
from datetime import datetime

from PIL import Image
from PIL.ExifTags import TAGS

import torch
from transformers import CLIPProcessor, CLIPModel
from facenet_pytorch import MTCNN

# --- モデル準備 ---
# デバイス設定
device = "cuda" if torch.cuda.is_available() else "cpu"

# CLIPモデル (use_fast=True で高速プロセッサを利用)
clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
).to(device)
clip_processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32", use_fast=False
)
# CLIP用ラベル
clip_labels = [
    "a group photo with many people",
    "a portrait of a bride and groom couple",
    "an empty wedding venue scenery"
]

# MTCNNによる顔検出
mtcnn = MTCNN(keep_all=True, device=device)

# --- 補助関数 ---

def get_exif_datetime(img_path):
    """EXIFの撮影日時を取得"""
    try:
        img = Image.open(img_path)
        exif = img._getexif() or {}
        for tag, val in exif.items():
            decoded = TAGS.get(tag, tag)
            if decoded == 'DateTimeOriginal':
                return datetime.strptime(val, "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    return None


def count_faces(img_path):
    """MTCNNを使って画像内の顔をカウント"""
    try:
        img = Image.open(img_path).convert('RGB')
        boxes, _ = mtcnn.detect(img)
        return len(boxes) if boxes is not None else 0
    except:
        return 0


def clip_zero_shot(img_path, labels):
    """CLIPでラベルとの類似度スコアを返す"""
    img = Image.open(img_path).convert("RGB")
    inputs = clip_processor(text=labels, images=img, return_tensors="pt", padding=True).to(device)
    outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)[0].cpu().tolist()
    return dict(zip(labels, probs))


def classify_image(img_path, face_thresh=3):
    """顔検出＋CLIPによるゼロショット分類"""
    n_faces = count_faces(img_path)
    if n_faces == 0:
        return "会場写真"
    if n_faces >= face_thresh:
        return "集合写真"
    if n_faces == 2:
        scores = clip_zero_shot(img_path, clip_labels)
        top = max(scores, key=scores.get)
        return "新郎新婦の写真" if top == clip_labels[1] else "集合写真"
    if n_faces == 1:
        return "顔写真"
    return "その他"


def group_by_classification(image_paths):
    """分類ルールに沿ってカテゴリごとに画像パスをグルーピング"""
    groups = defaultdict(list)
    for path in image_paths:
        cat = classify_image(path)
        groups[cat].append(path)
    return groups


def group_by_datetime(image_paths, threshold_seconds=60):
    """撮影日時が近いものをまとめるグルーピング"""
    time_list = []
    for path in image_paths:
        dt = get_exif_datetime(path)
        if dt:
            time_list.append((path, dt))
    time_list.sort(key=lambda x: x[1])

    groups = []
    current = []
    for i, (path, dt) in enumerate(time_list):
        if i == 0:
            current = [path]
        else:
            prev_dt = time_list[i-1][1]
            if (dt - prev_dt).total_seconds() <= threshold_seconds:
                current.append(path)
            else:
                groups.append(current)
                current = [path]
    if current:
        groups.append(current)

    return {i: grp for i, grp in enumerate(groups)}


def print_groups(groups, title):
    print(f"=== {title} ===")
    for key, items in groups.items():
        print(f"\nGroup {key}:")
        for p in items:
            print(f"  - {os.path.basename(p)}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="結婚式画像の分類と撮影日時グルーピング"
    )
    # positional argument with default
    parser.add_argument(
        'input_folder',
        nargs='?', default='.',
        help="対象フォルダパス (省略時はカレントディレクトリ)"
    )
    args = parser.parse_args()

    # 対象画像リスト取得
    image_paths = [
        os.path.join(args.input_folder, f)
        for f in os.listdir(args.input_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    if not image_paths:
        print("画像が見つかりません。")
        exit()

    # 1. 分類結果
    classification_groups = group_by_classification(image_paths)
    print_groups(classification_groups, "分類結果 (顔検出＋CLIP)")

    # 2. 撮影日時グルーピング
    datetime_groups = group_by_datetime(image_paths)
    print_groups(datetime_groups, "グルーピング結果 (撮影日時)")
