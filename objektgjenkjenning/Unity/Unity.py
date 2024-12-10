from ultralytics import YOLO
import cv2
import socket
import json
import base64

# Initialiser YOLO-modellen
model = YOLO('yolo11n.pt')

# Sett opp kamera og socket
cap = cv2.VideoCapture(0)
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('127.0.0.1', 8080))
server_socket.listen(1)

print("Venter på Unity...")
client_socket, addr = server_socket.accept()
print(f"Unity tilkoblet fra {addr}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO-analyse
    results = model(frame)
    detections = []
    for result in results:
        for box in result.boxes:
            detection = {
                'label': int(box.cls),  # Klassenummer
                'box': box.xyxy.tolist(),  # Bounding box som liste
                'confidence': float(box.conf)  # Konfidensverdi
            }
            detections.append(detection)
            
    # Konverter bildet til base64
    _, buffer = cv2.imencode('.jpg', frame)
    image_as_text = base64.b64encode(buffer).decode('utf-8')

    # Pakk dataene sammen
    data = {
        'image': image_as_text,
        'detections': detections
    }

    # Send data til Unity
    client_socket.send(json.dumps(data).encode('utf-8'))

    # Vis bildet med OpenCV for debugging (valgfritt)
    #cv2.imshow("YOLO Detection", frame)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

cap.release()
cv2.destroyAllWindows()
