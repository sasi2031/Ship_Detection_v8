from ultralytics import YOLO
model= YOLO(r"D:\Ship_Detection_Data\train9\weights\best.pt")
results= model.predict(data=r"D:\Ship_Detection_Data\train9\args.yaml",source=r"D:\karach_trail2_v8\images",project=r"D:\Ship_Detection_Data\InferencingResults", name="Prediction_Results", save=True, show_labels=False)