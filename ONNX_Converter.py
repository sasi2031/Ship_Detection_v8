from ultralytics import YOLO
import os

# Load the YOLOv8 model
model = YOLO(r"C:\Users\Sasi Kanth\Desktop\April Purpose\analytics_backend\utils\all_models\model_obb_detection\ship\OD_ship_best.pt")


# Export to ONNX with opset 11
model.export(
    format="onnx",  # Export format
    opset=11,       # ONNX opset version
    
    save=True       # Save the output
)
