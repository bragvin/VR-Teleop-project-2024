import cv2
from ultralytics import YOLO

#velg hvilken yolomodell du ønsker å bruke
model = YOLO('yolo11n.pt')

cap = cv2.VideoCapture(0)

# setter størrelsen for terminalvinduet
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)
    annotated_frame = results[0].plot()

    # åpner en skjerm i terminalvindu så man kan se modellen.
    cv2.imshow('YOLO11m', annotated_frame)

    # for å avslutte programmet, trykk på q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()