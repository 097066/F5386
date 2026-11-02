import cv2
import numpy as np

model_path = "models/DenseNet_121.caffemodel"
config_path = "models/DenseNet_121.prototxt.txt"
class_path = "models/classification_classes_ILSVRC2012.txt"

# 讀取類別名稱
with open(class_path, "r", encoding="utf-8") as f:
    class_names = [
        line.split(",")[0].strip()
        for line in f
    ]

# 載入模型
model = cv2.dnn.readNet(
    model=model_path,
    config=config_path,
    framework="Caffe"
)

# 讀取圖片
img = cv2.imread("images/ba_pi.png")

if img is None:
    raise FileNotFoundError("無法讀取圖片，請確認圖片路徑。")

# 保留原本的前處理設定
blob = cv2.dnn.blobFromImage(
    image=img,
    scalefactor=0.01,
    size=(224, 224),
    mean=(104, 117, 123),
    swapRB=False,
    crop=False
)

model.setInput(blob)
outputs = model.forward()

# 攤平成一維陣列
scores = outputs.flatten()

def softmax(x):
    # 減去最大值，避免 exp 溢位
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

# 若模型最後一層已是 Softmax，直接使用輸出；
# 否則才進行 Softmax，避免重複轉換。
last_layer = model.getLayer(model.getUnconnectedOutLayers()[0])
if last_layer.type.lower() == "softmax":
    probs = scores
else:
    probs = softmax(scores)

# 依分數由大到小排序，取前五名
top5_ids = np.argsort(probs)[::-1][:5]

# 在圖片上方增加黑色區域，顯示分類結果
display = cv2.copyMakeBorder(
    img, 180, 0, 0, 0,
    cv2.BORDER_CONSTANT,
    value=(0, 0, 0)
)

for rank, label_id in enumerate(top5_ids, start=1):
    name = class_names[label_id]
    probability = float(probs[label_id]) * 100

    text = f"{rank}. {name}: {probability:.2f}%"
    print(text)

    cv2.putText(
        display,
        text,
        (10, 30 * rank),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        1,
        cv2.LINE_AA
    )

cv2.imshow("Top 5 Classification", display)
cv2.waitKey(0)
cv2.destroyAllWindows()