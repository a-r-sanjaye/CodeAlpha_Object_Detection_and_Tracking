import cv2
import torch
import torchvision
from torchvision.transforms import functional as F
import numpy as np
from sort import Sort

CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    print(f"Using device: {DEVICE}")

    print("Loading Faster R-CNN (MobileNetV3) model...")
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights='DEFAULT')
    model.to(DEVICE)
    model.eval()
    
    tracker = Sort(max_age=10, min_hits=3, iou_threshold=IOU_THRESHOLD)
    
    cap = cv2.VideoCapture(0) 
    
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    COCO_INSTANCE_CATEGORY_NAMES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    print("Starting detection loop. Press 'q' to exit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        image_tensor = F.to_tensor(frame).to(DEVICE)
        image_tensor = image_tensor.unsqueeze(0)
        
        with torch.no_grad():
            predictions = model(image_tensor)
            
        pred = predictions[0]
        boxes = pred['boxes'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        
        mask = scores > CONFIDENCE_THRESHOLD
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]
        
        detections_for_sort = []
        for box, score in zip(boxes, scores):
            detections_for_sort.append(np.append(box, score))
        
        detections_for_sort = np.array(detections_for_sort)
        
        track_bbs_ids = tracker.update(detections_for_sort)
        
        for track in track_bbs_ids:
            x1, y1, x2, y2, track_id = track
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Find the label for this track by matching with original detections
            # We look for the detection with highest IoU with this track box
            current_track_box = [x1, y1, x2, y2]
            best_iou = 0
            best_label_index = 0 # Default to background if no match
            
            for i, det_box in enumerate(boxes):
                # Calculate IoU (Intersection over Union)
                dx1 = max(x1, det_box[0])
                dy1 = max(y1, det_box[1])
                dx2 = min(x2, det_box[2])
                dy2 = min(y2, det_box[3])
                
                if dx2 >= dx1 and dy2 >= dy1:
                    intersection = (dx2 - dx1) * (dy2 - dy1)
                    area_track = (x2 - x1) * (y2 - y1)
                    area_det = (det_box[2] - det_box[0]) * (det_box[3] - det_box[1])
                    union = area_track + area_det - intersection
                    
                    iou = intersection / union if union > 0 else 0
                    if iou > best_iou:
                        best_iou = iou
                        best_label_index = labels[i]
            
            class_name = COCO_INSTANCE_CATEGORY_NAMES[best_label_index] if best_label_index < len(COCO_INSTANCE_CATEGORY_NAMES) else "Unknown"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            label_text = f"{class_name} ID: {int(track_id)}"
            
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Object Detection & Tracking', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
