import cv2
import numpy as np

# --- 參數配置 ---
MIN_CONFIDENCE = 0.5  # 偵測到的文字區域的最小信心分數
NMS_THRESHOLD = 0.05  # 非最大值抑制 (NMS) 的閾值
WIDTH = 320  # 輸入影像的寬度 (必須是32的倍數，常見為320, 640, 1280)
HEIGHT = 320  # 輸入影像的高度 (必須是32的倍數)
MODEL_PATH = "models/frozen_east_text_detection.pb"
IMAGE_PATH = "images/car.jpg"
OUTPUT_LAYERS = [
    "feature_fusion/Conv_7/Sigmoid",  # 信心分數圖 (Scores)
    "feature_fusion/concat_3"  # 幾何資訊 (Geometry: 4距離 + 1角度)
]


def decode_predictions(scores, geometry, min_confidence):
    """
    對 EAST 模型的輸出進行後處理，從分數圖和幾何資訊中提取邊界框和信心分數。
    """
    # 獲取分數圖和幾何輸出的維度
    (numRows, numCols) = scores.shape[2:4]

    # 初始化列表來儲存邊界框的矩形和分數
    rects = []
    confidences = []

    # 迴圈遍歷所有的行和列 (即特徵圖上的每個像素點)
    for y in range(0, numRows):
        # 提取當前行的數據
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # 迴圈遍歷當前行中的每一列
        for x in range(0, numCols):
            # 1. 檢查信心分數
            if scoresData[x] < min_confidence:
                continue

            # 2. 獲取幾何資訊和縮放因子
            (offsetX, offsetY) = (x * 4.0, y * 4.0)  # 縮放比例是 4
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # 3. 獲取四個距離 (d1, d2, d3, d4)
            h = xData0[x] + xData2[x]  # h = d1 (上) + d3 (下)
            w = xData1[x] + xData3[x]  # w = d2 (右) + d4 (左)

            # 4. 計算邊界框的中心點 (Cx, Cy)
            # EAST 模型輸出點是中心點的偏移量，這裡計算的是實際中心
            center_x = offsetX + (xData1[x] - xData3[x]) * cos / 2.0
            center_y = offsetY + (xData1[x] - xData3[x]) * sin / 2.0

            # 5. 計算四個頂點座標 (這是旋轉矩形的頂點)
            # 公式來源於旋轉變換：[x'] = [cos -sin][x] + [Cx]
            #                    [y']   [sin cos][y]   [Cy]

            # 頂點 1 (左上)
            endX = center_x - w / 2.0 * cos + h / 2.0 * sin
            endY = center_y - w / 2.0 * sin - h / 2.0 * cos

            # 頂點 2 (右上)
            endX_2 = center_x + w / 2.0 * cos + h / 2.0 * sin
            endY_2 = center_y + w / 2.0 * sin - h / 2.0 * cos

            # 頂點 3 (右下)
            endX_3 = center_x + w / 2.0 * cos - h / 2.0 * sin
            endY_3 = center_y + w / 2.0 * sin + h / 2.0 * cos

            # 頂點 4 (左下)
            endX_4 = center_x - w / 2.0 * cos - h / 2.0 * sin
            endY_4 = center_y - w / 2.0 * sin + h / 2.0 * cos

            # 將四個角點作為一個旋轉矩形儲存
            box = np.array([
                [endX, endY],
                [endX_2, endY_2],
                [endX_3, endY_3],
                [endX_4, endY_4]
            ], dtype="float32")

            # 這裡我們將旋轉矩形資訊存入 rects 列表，並將分數存入 confidences
            # 為了方便後續使用 OpenCV 的 NMS，我們將旋轉矩形轉換為最小邊界矩形 (Bounding Box)
            (x_min, y_min) = np.min(box, axis=0)
            (x_max, y_max) = np.max(box, axis=0)

            # NMS 需要標準矩形 (x, y, w, h)
            rects.append((int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)))
            confidences.append(float(scoresData[x]))

    return (rects, confidences)


# --- 主要執行區塊 ---
try:
    # 1. 載入模型和影像
    net = cv2.dnn.readNet(MODEL_PATH)
    image = cv2.imread(IMAGE_PATH)

    # 檢查影像是否成功載入
    if image is None:
        print(f"[錯誤] 無法載入影像：{IMAGE_PATH}")
        exit()

    # 獲取原始影像的尺寸
    (H, W) = image.shape[:2]
    HEIGHT = (H // 32 + 1) * 32
    WIDTH = (W // 32 + 1) * 32

    # 2. 影像預處理 (建立 Blob)
    # 將影像縮放並轉換為 DNN 模型的輸入格式 (Blob)
    # 影像會被縮放到 (WIDTH, HEIGHT)，縮放因子用於後續將座標轉換回原圖
    blob = cv2.dnn.blobFromImage(image, 1.0, (WIDTH, HEIGHT),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    # 3. 前向傳播 (Inference)
    net.setInput(blob)
    layerOutputs = net.forward(OUTPUT_LAYERS)

    # 4. 後處理：解碼輸出
    # layerOutputs[0] 是分數圖，layerOutputs[1] 是幾何資訊
    (rects, confidences) = decode_predictions(layerOutputs[0], layerOutputs[1], MIN_CONFIDENCE)

    # 5. 非最大值抑制 (NMS)
    # 應用 NMS 來抑制重疊的、較低的信心邊界框
    indices = cv2.dnn.NMSBoxes(rects, confidences, MIN_CONFIDENCE, NMS_THRESHOLD)

    # 6. 繪製結果
    # 處理 NMSBox 輸出的結果索引，並在原圖上繪製最終的邊界框
    if len(indices) > 0:
        for i in indices.flatten():
            # 獲取標準邊界框 (x, y, w, h)
            (x, y, w, h) = rects[i]

            # 將座標從輸入尺寸 (WIDTH, HEIGHT) 縮放回原始影像尺寸 (W, H)
            rW = W / float(WIDTH)
            rH = H / float(HEIGHT)

            # 最終的邊界框座標 (轉換回原圖尺寸)
            startX = int(x * rW)
            startY = int(y * rH)
            endX = int((x + w) * rW)
            endY = int((y + h) * rH)

            # 在影像上繪製綠色矩形框
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

            # 繪製信心分數
            text = f"{confidences[i]:.2f}"
            cv2.putText(image, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 7. 顯示結果
    cv2.imshow("Text Detection Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except Exception as e:
    print(f"執行過程中發生錯誤: {e}")
    print("請檢查 'frozen_east_text_detection.pb' 和 'test_image.jpg' 文件路徑是否正確。")
