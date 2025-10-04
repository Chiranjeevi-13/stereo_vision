from ultralytics import YOLO
import cv2
import numpy as np

class ObjectDetector:
    """
    YOLOv8 object detector for stereo vision pipeline.
    """
    
    def __init__(self, model_name='yolov8n.pt', confidence=0.5, iou_threshold=0.45):
        """
        Initialize YOLO detector.
        
        Args:
            model_name: YOLO model to use (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
                       n = nano (fastest, least accurate)
                       s = small
                       m = medium
                       l = large
                       x = extra large (slowest, most accurate)
            confidence: Detection confidence threshold (0.0-1.0)
            iou_threshold: IoU threshold for Non-Maximum Suppression
        """
        print(f"Loading YOLO model: {model_name}...")
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        print("YOLO model loaded successfully")
    
    def detect(self, image, conf=None, iou=None):
        """
        Detect objects in an image.
        
        Args:
            image: Input image (H x W x 3) BGR
            conf: Override confidence threshold
            iou: Override IoU threshold
            
        Returns:
            detections: List of detection dictionaries with:
                       - bbox: [x1, y1, x2, y2] bounding box coordinates
                       - class_id: Integer class ID
                       - class_name: String class name
                       - confidence: Detection confidence (0.0-1.0)
        """
        # Use provided thresholds or defaults
        conf_thresh = conf if conf is not None else self.confidence
        iou_thresh = iou if iou is not None else self.iou_threshold
        
        # Run inference
        results = self.model(image, conf=conf_thresh, iou=iou_thresh, verbose=False)
        
        # Parse results
        detections = []
        
        # results is a list (batch of images), we only have one image
        result = results[0]
        
        # Extract boxes, classes, and confidences
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes [x1, y1, x2, y2]
        classes = result.boxes.cls.cpu().numpy()  # Class IDs
        confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
        
        for box, cls_id, conf in zip(boxes, classes, confidences):
            detection = {
                'bbox': box.tolist(),  # [x1, y1, x2, y2]
                'class_id': int(cls_id),
                'class_name': self.model.names[int(cls_id)],
                'confidence': float(conf)
            }
            detections.append(detection)
        
        return detections
    
    def draw_detections(self, image, detections):
        """
        Draw bounding boxes and labels on image.
        
        Args:
            image: Input image (H x W x 3) BGR
            detections: List of detection dictionaries
            
        Returns:
            annotated_image: Image with drawn boxes (H x W x 3) BGR
        """
        annotated = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            class_name = det['class_name']
            confidence = det['confidence']
            
            # Draw bounding box
            color = (0, 255, 0)  # Green
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label = f"{class_name} {confidence:.2f}"
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - label_h - baseline), (x1 + label_w, y1), color, -1)
            
            # Draw label text
            cv2.putText(annotated, label, (x1, y1 - baseline), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return annotated
