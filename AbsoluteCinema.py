from ultralytics import YOLO
model= YOLO(r"D:\Ship_Detection_Data_Sentinel1\train\weights\best.pt")

results= model.predict(r"D:\Ship_Detection_Data_Sentinel1\Dataset_v8_split\val\images", imgsz=1024, save= True, conf= 0.1 )
