from ultralytics import YOLO
import cv2

model = YOLO("./utils/fruit_model.pt")
model.predict(source="./utils/fruits.jpg", show=True, save=True, conf=0.5) # use source=0 for webcam