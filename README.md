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
- mtcnn (dlibと相性が良く使いやすいので、こっちを採用)
- tensorflow
- scikit-learn
- pillow 
- facenet_pytorch(顔検知としては、mtcnnより精度が高いのでgroupで使用)
- transformers

---

### 1. 目瞑り
```
python detect_blink.py
```

サンプルデータ一覧

 - face1.png  / 目が開いているサンプル (default)
 - face2.png  / 目が閉じているサンプル
 - face3.png  / 目が閉じているサンプル（CNNで顔判定が可能。OpenCVのCNNなしでは不可だった）

### 2. 写真のグルーピング
```
python group_similar.py
```

サンプルデータ一覧（量が多いので割愛、色々なデータ）

### 3. 明るさ判定
```
python check_brightness.py
```

サンプルデータ一覧

 - darkness.png  / 暗い画像のサンプル
 - middle.png  / 普通の画像のサンプル