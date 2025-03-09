from ultralytics import YOLO

model = YOLO(r"C:\Users\ssk22\OneDrive\Documents\Chekrin project\Wild-domestic\runs\detect\train6\weights\last.pt") 

model.train(data='passport-details-1\data.yaml', epochs=30, batch=16, imgsz=640)
