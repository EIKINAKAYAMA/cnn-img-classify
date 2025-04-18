import os
from PIL import Image
import torch
from torchvision import models, transforms
from facenet_pytorch import MTCNN
import cv2

# --- GPU or CPU 設定 ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- MTCNN 初期化（顔検出用） ---
mtcnn = MTCNN(keep_all=True, device=device)

# --- 画像フォルダ指定 ---
img_dir = "data/group"
paths = [os.path.join(img_dir, fname) for fname in os.listdir(img_dir) if fname.lower().endswith((".png", ".jpg", ".jpeg"))]

# --- VGG16 特徴抽出モデル設定 ---
vgg_model = models.vgg16(pretrained=True).features.eval().to(device)
vgg_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- 画像特徴抽出 ---
def get_vgg_embedding(img_path):
    img = Image.open(img_path).convert("RGB")
    tensor = vgg_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = vgg_model(tensor)
        emb = torch.flatten(features, start_dim=1)
    return emb.squeeze().cpu()

# --- 顔数カウント（MTCNN） ---
def count_faces(img_path):
    img = Image.open(img_path).convert("RGB")
    boxes, _ = mtcnn.detect(img)
    return 0 if boxes is None else len(boxes)

# --- 顔数による分類 ---
def classify_image(img_path, face_thresh=3):
    n = count_faces(img_path)
    if n == 0:
        return "会場写真"
    elif n >= face_thresh:
        return "集合写真"
    elif n in [1, 2]:
        return "顔写真"
    else:
        return "その他"

# --- 大分類：顔数ごとのグループ ---
def group_by_classification(paths):
    groups = {}
    for p in paths:
        label = classify_image(p)
        groups.setdefault(label, []).append(p)
    return groups

# --- 目線判定（OpenCV Haar Cascade） ---
def is_eye_contact(img_path):
    image = cv2.imread(img_path)
    if image is None:
        return False
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            return True
    return False

# --- 類似構図によるグルーピング（VGGベース） ---
def group_by_vgg_similarity(paths, sim_thresh=0.9):
    embs = [(p, get_vgg_embedding(p)) for p in paths]
    used = set()
    groups = []
    for i, (p1, e1) in enumerate(embs):
        if p1 in used:
            continue
        group = [p1]
        used.add(p1)
        for p2, e2 in embs[i+1:]:
            if p2 in used:
                continue
            sim = torch.nn.functional.cosine_similarity(e1, e2, dim=0).item()
            if sim >= sim_thresh:
                group.append(p2)
                used.add(p2)
        groups.append(group)
    return {i: g for i, g in enumerate(groups)}

# --- グループ出力表示 ---
def print_groups(groups, title):
    print(f"\n--- {title} ---")
    for key, paths in groups.items():
        label = f"Group {key}" if isinstance(key, int) else key
        print(f"[{label}]")
        for p in paths:
            print(f"  - {os.path.basename(p)}")

# --- メイン処理 ---
if __name__ == "__main__":
    # 大分類（顔数）
    main_groups = group_by_classification(paths)
    print_groups(main_groups, "【大分類】顔・集合・会場など")

    # 顔写真：目線あり/なし分類
    if '顔写真' in main_groups:
        face_paths = main_groups['顔写真']
        eye_groups = {'目線あり': [], '目線なし': []}
        for p in face_paths:
            key = '目線あり' if is_eye_contact(p) else '目線なし'
            eye_groups[key].append(p)
        print_groups(eye_groups, "【顔写真の細分類】目線あり/なし")

    # 構図類似度（VGG特徴量）でのグルーピング
    for label in ['集合写真', '会場写真']:
        if label in main_groups:
            sub = group_by_vgg_similarity(main_groups[label])
            print_groups(sub, f"【{label}の細分類】構図類似度")
