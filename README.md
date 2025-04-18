# 画像認識用の検証アプリ
* スクリプト・モデル・テストデータは内包。

・目つぶり検出
dlibの68点顔ランドマークモデルを用いて、静止画像中の人物が目を閉じているかどうかを判定します。  
主に「目のアスペクト比（EAR）」を使って、目の開閉状態を計算しています。

・写真群のグルーピング

・明るさ判定
OpenCVを使用して画像の露出を評価し、「過露出」「低露出」「適正露出」のカテゴリに分類するPythonスクリプトを示します。ヒストグラム、平均輝度、コントラスト（標準偏差）を用いて評価します。しきい値は調整可能なパラメータとして定義しています。

---

## 💻 動作環境

- Python 3.9 以上
- OpenCV
- dlib
- scipy
- tensorflow
- scikit-learn
- pillow 
- facenet_pytorch(分類用)
- transformers(分類用)

---

### 1. 目瞑り
```
python detect_blink.py
```

サンプルデータ一覧

 - face1.png  / 目が開いているサンプル (default)
 - face2.png  / 目が閉じているサンプル
 - face3.png  / 顔検出に失敗するサンプル

### 2. 写真のグルーピング
```
python group_similar.py.py
```

サンプルデータ一覧

 - face1.png  / 目が開いているサンプル (default)
 - face2.png  / 目が閉じているサンプル
 - face3.png  / 顔検出に失敗するサンプル

### 3. 明るさ判定
```
python check_brightness.py
```

サンプルデータ一覧

 - darkness.png  / 暗い画像のサンプル
 - middle.png  / 普通の画像のサンプル