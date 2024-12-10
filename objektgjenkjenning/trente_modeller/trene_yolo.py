from ultralytics import YOLO

# last inn en ferdigtrent modell du ønsker å trene på et spesifikt datasett
model = YOLO("yolo11n.pt")

# Train the model
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)