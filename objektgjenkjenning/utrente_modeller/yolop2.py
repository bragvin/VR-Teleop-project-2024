from ultralytics import YOLO

model = YOLO("../trente_modeller/yolo11m.pt")

results = model("../media/amsterdam.jpeg", save=True, show=True)