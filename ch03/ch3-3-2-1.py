import cv2

# 載入人臉偵測模型（只需載入一次）
model_path = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
config_path = "models/deploy.prototxt"

model = cv2.dnn.readNet(
    model=model_path,
    config=config_path,
    framework="Caffe"
)

# 開啟攝影機：0 通常是預設攝影機
cap = cv2.VideoCapture(0)

try:
    if not cap.isOpened():
        raise RuntimeError("無法開啟攝影機，請確認連接狀態或攝影機編號。")

    while True:
        # 讀取一幀影像
        success, image = cap.read()

        if not success:
            print("無法取得攝影機影像。")
            break

        image_height, image_width = image.shape[:2]

        # 將影像轉換成模型需要的輸入格式
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 117.0, 123.0),
            swapRB=False,
            crop=False
        )

        # 執行人臉偵測
        model.setInput(blob)
        results = model.forward()

        face_count = 0

        for face in results[0, 0]:
            confidence = float(face[2])

            if confidence <= 0.5:
                continue

            # 將比例座標換算成原始影像的像素座標
            x1 = int(face[3] * image_width)
            y1 = int(face[4] * image_height)
            x2 = int(face[5] * image_width)
            y2 = int(face[6] * image_height)

            # 限制座標，避免超出影像邊界
            x1 = max(0, min(x1, image_width - 1))
            y1 = max(0, min(y1, image_height - 1))
            x2 = max(0, min(x2, image_width - 1))
            y2 = max(0, min(y2, image_height - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            face_count += 1

            # 畫出人臉框（BGR：藍色）
            cv2.rectangle(
                image, (x1, y1), (x2, y2),
                (255, 0, 0), 2
            )

            # 顯示信心分數
            cv2.putText(
                image,
                f"{confidence:.1%}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

        # 顯示偵測到的人臉數量
        cv2.putText(
            image,
            f"Faces: {face_count} | Q / Esc: Exit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow("Detected Faces", image)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break

finally:
    # 結束或發生錯誤時，釋放攝影機與視窗
    cap.release()
    cv2.destroyAllWindows()