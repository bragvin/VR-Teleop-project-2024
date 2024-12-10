import cv2
from ultralytics import YOLO

# Last inn SAM-modellen
model = YOLO("sam2.1_b.pt")

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)
    annotated_frame = results[0].plot()

    cv2.imshow("SAM", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Skriv ut modellinfo
model.info()

# Kjør inferens og lagre resultatet
# result = model.predict(source=0, save=True, save_dir="runs/segment")
