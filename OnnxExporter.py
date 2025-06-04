from ultralytics import YOLO
model= YOLO(r"D:\Ship_Detection_Data_12Classes\train\weights\best.pt")

model.export(format="onnx", opset=10)