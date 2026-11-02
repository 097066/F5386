import cv2

faceCascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalcatface.xml")

image = cv2.imread("images/feature-aespa.webp")
image = cv2.imread("images/cat.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 從圖片偵測人臉
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.02,
    minNeighbors=3,
    minSize=(10, 10),
    maxSize=(100, 100)
)

print("人臉數:", len(faces))

# 在偵測出的人臉繪出長方形外框
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("preview", image)
cv2.waitKey(0)

cv2.destroyAllWindows()
