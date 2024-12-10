import cv2
import torch
import numpy as np
from effdet import create_model, DetBenchPredict
from effdet.data import resolve_input_config
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from PIL import Image
import time

# Last inn EfficientDet-modell
model_name = 'tf_efficientdet_d0'
model = create_model(model_name, pretrained=True)
config = resolve_data_config(None, model=model)
transform = create_transform(**config)

bench = DetBenchPredict(model)
bench = bench.eval().cpu()  # Bruk .cuda() hvis du har CUDA-kompatibel GPU

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    # Preprosess bilde
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    img_pil = img_pil.resize((512, 512))  # Endre størrelse til 512x512
    img_tensor = transform(img_pil).unsqueeze(0).cpu()  # Bruk .cuda() hvis du har CUDA-kompatibel GPU

    # Kjør inferens
    with torch.no_grad():
        output = bench(img_tensor)

    # Hent deteksjoner
    boxes = output[0].detach().cpu().numpy()
    scores = output[1].detach().cpu().numpy()
    classes = output[2].detach().cpu().numpy()

    # Annoter ramme og tell deteksjoner
    detections = {}
    height, width = frame.shape[:2]
    for box, score, class_id in zip(boxes, scores, classes):
        if score > 0.5:  # Threshold for deteksjon
            x1, y1, x2, y2 = box.astype(int)
            # Skaler bounding box tilbake til original bildestørrelse
            x1, x2 = int(x1 * width / 512), int(x2 * width / 512)
            y1, y2 = int(y1 * height / 512), int(y2 * height / 512)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            class_name = f"Class {int(class_id)}"
            label = f"{class_name}: {score:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if class_name in detections:
                detections[class_name] += 1
            else:
                detections[class_name] = 1

    end_time = time.time()
    total_time = (end_time - start_time) * 1000  # Konverter til millisekunder

    detection_string = ", ".join([f"{count} {name}" for name, count in detections.items()])
    print(f"0: 512x512 {detection_string}, {total_time:.1f}ms")
    print(f"Speed: {total_time:.1f}ms per image at shape {img_tensor.shape}")
    print()

    cv2.imshow("EfficientDet Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()