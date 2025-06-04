from ultralytics import YOLO
model= YOLO(r"D:\Ship_Detection_Data_Sentinel2\train3\weights\best.pt")

results= model.predict(r"D:\Ship_Detection_Data_Sentinel2\Dataset_v8_split_aug\val\images", imgsz=1280, save= True, conf= 0.1, project="InferenceResults", name="Sentinel2Results" )
