import os
from PIL import Image
import torch
from torchvision import models, transforms

# --- デバイス設定 ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 入力画像フォルダ ---
img_dir = "data/group"
paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

# --- VGG16 特徴抽出モデルの用意 ---
vgg_model = models.vgg16(pretrained=True).features.eval().to(device)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- 特徴ベクトルを得る関数 ---
def get_vgg_embedding(img_path):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = vgg_model(x)
        emb = torch.flatten(feat, start_dim=1)
    return emb.squeeze().cpu()

# --- 類似性に基づくグルーピング ---
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

# --- 結果表示 ---
def print_groups(groups):
    print(f"\n--- VGG特徴量に基づく構図グルーピング ---")
    for gid, paths in groups.items():
        print(f"[Group {gid}]")
        for p in paths:
            print(f"  - {os.path.basename(p)}")

# --- 実行 ---
if __name__ == "__main__":
    grouped = group_by_vgg_similarity(paths, sim_thresh=0.9)
    print_groups(grouped)
