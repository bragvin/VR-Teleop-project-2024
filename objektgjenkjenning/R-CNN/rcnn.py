import cv2
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
model = model.eval().to(device)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()

    # Preprocess
    preprocess_start = time.time()
    image = torchvision.transforms.functional.to_tensor(frame).to(device)
    preprocess_end = time.time()
    preprocess_time = (preprocess_end - preprocess_start) * 1000

    # Inference
    inference_start = time.time()
    with torch.no_grad():
        prediction = model([image])[0]
    inference_end = time.time()
    inference_time = (inference_end - inference_start) * 1000

    # Postprocess
    postprocess_start = time.time()

    detected_objects = {}
    for label, score in zip(prediction['labels'], prediction['scores']):
        if score > 0.5:
            class_name = COCO_INSTANCE_CATEGORY_NAMES[label]
            if class_name in detected_objects:
                detected_objects[class_name] += 1
            else:
                detected_objects[class_name] = 1

    postprocess_end = time.time()
    postprocess_time = (postprocess_end - postprocess_start) * 1000

    total_time = time.time() - start_time

    # Skriv ut deteksjonsresultater og hastighet
    detection_string = ", ".join([f"{count} {name}s" for name, count in detected_objects.items()])
    print(f"0: {image.shape[2]}x{image.shape[1]} {detection_string}, {total_time * 1000:.1f}ms")
    print(
        f"Speed: {preprocess_time:.1f}ms preprocess, {inference_time:.1f}ms inference, {postprocess_time:.1f}ms postprocess per image at shape (1, {image.shape[0]}, {image.shape[1]}, {image.shape[2]})")
    print()

    # Tegn bounding boxes (valgfritt)
    for box, score, label in zip(prediction['boxes'], prediction['scores'], prediction['labels']):
        if score > 0.5:
            x1, y1, x2, y2 = box.cpu().int()
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            class_name = COCO_INSTANCE_CATEGORY_NAMES[label]
            cv2.putText(frame, f"{class_name}: {score:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2)

    cv2.imshow("Faster R-CNN Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()