import cv2
from ultralytics import YOLO

model = YOLO('yolo11m.pt')

def process_frame(frame):
	results = model(frame)
	annotated_frame = results[0].plot()
	return annotated_frame