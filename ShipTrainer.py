from ultralytics import YOLO

model= YOLO(r'D:\ShipDetection\yolov8m-obb.pt')

model.train(data='ship.yaml',
            epochs=100, 
            imgsz=1024, 
            batch=16,  
            )