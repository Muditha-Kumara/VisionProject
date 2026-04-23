from ultralytics import YOLO

model = YOLO("yolov8n.pt")
# Export to ONNX with specific constraints for NPU compatibility
model.export(format="onnx", opset=12, simplify=True)
