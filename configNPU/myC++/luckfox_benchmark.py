import os
import cv2
import numpy as np
import csv
import time
from rknnlite.api import RKNNLite

# --- CONFIGURATION ---
MODEL_PATH = 'yolov8n.rknn'
POLYGON_CSV = 'lane_polygons.csv'
IMAGE_DIR = 'test_images'
OUTPUT_CSV = 'luckfox_results.csv'
OBJ_THRESH = 0.25
NMS_THRESH = 0.45
IMG_SIZE = (640, 640)

# YOLOv8 Classes (Standard COCO)
CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck'] #... truncated

def load_polygons(csv_path):
    lanes = {}
    with open(csv_path, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['lane_name']
            if name not in lanes: lanes[name] = []
            lanes[name].append([int(row['x']), int(row['y'])])
    return {k: np.array(v, dtype=np.int32) for k, v in lanes.items()}

def post_process(outputs):
    # This is a simplified post-process for YOLOv8 output shapes
    # Note: Depending on your export, you may need to adjust indices
    data = outputs[0][0]
    data = data.transpose()
    
    boxes, confs, class_ids = [], [], []
    for row in data:
        score = row[4:].max()
        if score > OBJ_THRESH:
            class_id = row[4:].argmax()
            if class_id in [2, 3, 5, 7]: # Car, Motorcycle, Bus, Truck
                x, y, w, h = row[0:4]
                boxes.append([int(x - w/2), int(y - h/2), int(w), int(h)])
                confs.append(float(score))
                class_ids.append(class_id)
                
    indices = cv2.dnn.NMSBoxes(boxes, confs, OBJ_THRESH, NMS_THRESH)
    return [(boxes[i], confs[i], class_ids[i]) for i in indices]

# --- MAIN EXECUTION ---
def main():
    # 1. Initialize NPU
    rknn = RKNNLite()
    print("--> Loading RKNN model")
    if rknn.load_rknn(MODEL_PATH) != 0:
        print("Load model failed!"); return
    if rknn.init_runtime() != 0:
        print("Init runtime failed!"); return

    # 2. Load Metadata
    lanes = load_polygons(POLYGON_CSV)
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png'))]
    
    results_log = []

    print(f"--> Starting benchmark on {len(image_files)} images")
    for img_name in image_files:
        img_path = os.path.join(IMAGE_DIR, img_name)
        frame = cv2.imread(img_path)
        if frame is None: continue
        
        orig_h, orig_w = frame.shape[:2]
        
        # Pre-process
        img = cv2.resize(frame, IMG_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Inference
        start_time = time.time()
        outputs = rknn.inference(inputs=[img])
        infer_time = (time.time() - start_time) * 1000
        
        # Post-process
        detections = post_process(outputs)
        
        # Lane Counting Logic
        lane_counts = {lane: 0 for lane in lanes}
        for (box, conf, cls) in detections:
            # Scale box back to original image size
            x, y, w, h = box
            cx = int((x + w/2) * (orig_w / 640))
            cy = int((y + h/2) * (orig_h / 640))
            
            for lane_name, poly in lanes.items():
                if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                    lane_counts[lane_name] += 1
        
        # Log Results
        row = {"image": img_name, "latency_ms": f"{infer_time:.2f}"}
        row.update(lane_counts)
        results_log.append(row)
        print(f"Processed {img_name} - {infer_time:.1f}ms")

    # 3. Save Metrics
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results_log[0].keys())
        writer.writeheader()
        writer.writerows(results_log)
    
    print(f"--> Done. Metrics saved to {OUTPUT_CSV}")
    rknn.release()

if __name__ == '__main__':
    main()
