# 画像認識用の検証アプリ
* スクリプト・モデル・テストデータは内包。

・目つぶり検出
dlibの68点顔ランドマークモデルを用いて、静止画像中の人物が目を閉じているかどうかを判定します。  
主に「目のアスペクト比（EAR）」を使って、目の開閉状態を計算しています。

・写真群のグルーピング

・明るさ判定

---

## 💻 動作環境

- Python 3.9 以上
- OpenCV
- dlib
- scipy

---

### 1. 目瞑り
```
python detect_blink.py
```

サンプルデータ一覧

 - face1.png  / 目が開いているサンプル (default)
 - face2.png  / 目が閉じているサンプル
 - face3.png  / 顔検出に失敗するサンプル