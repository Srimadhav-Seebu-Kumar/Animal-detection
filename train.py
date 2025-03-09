from ultralytics import YOLO

model = YOLO('yolov8m.pt') 

model.train(data='Human-and-Animal-1\data.yaml', epochs=30, batch=16, imgsz=640)
