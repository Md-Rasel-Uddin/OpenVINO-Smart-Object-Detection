# detector.py
import cv2
import numpy as np
import time
from openvino.runtime import Core


class ObjectDetector:
    def __init__(self, model_path: str, device: str = "CPU", num_streams: int = 2):
        """
        Initialize OpenVINO model for async inference
        """
        self.ie = Core()
        self.model_path = model_path
        self.device = device
        self.num_streams = num_streams

        # Load model
        model = self.ie.read_model(model=f"{model_path}")
        self.compiled_model = self.ie.compile_model(model=model, device_name=device)
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)

        self.last_inference_time = 0

        # COCO-like label map (person, vehicle, bike)
        self.labels = {1: "person", 2: "vehicle", 3: "bike"}

        print(f"[INFO] OpenVINO model loaded: {model_path} on {device}")

    async def detect_async(self, frame):
        """
        Perform asynchronous detection inference
        """
        start_time = time.time()

        # Preprocess
        input_image = self._preprocess(frame)

        # Run inference
        results = self.compiled_model([input_image])[self.output_layer]

        # Postprocess detections
        detections = self._postprocess(results, frame.shape)

        self.last_inference_time = (time.time() - start_time) * 1000  # ms
        return detections

    def _preprocess(self, frame):
        """
        Resize and format image to model input shape
        """
        n, c, h, w = self.input_layer.shape
        resized = cv2.resize(frame, (w, h))
        input_image = resized.transpose((2, 0, 1))  # HWC → CHW
        input_image = np.expand_dims(input_image, axis=0)
        return input_image

    def _postprocess(self, results, original_shape):
        """
        Filter and format detections
        """
        detections = []
        ih, iw, _ = original_shape

        for det in results[0][0]:
            image_id, label_id, conf, x_min, y_min, x_max, y_max = det
            if conf > 0.5:
                x_min = int(x_min * iw)
                y_min = int(y_min * ih)
                x_max = int(x_max * iw)
                y_max = int(y_max * ih)
                detections.append({
                    "label": self.labels.get(int(label_id), str(label_id)),
                    "confidence": float(conf),
                    "bbox": [x_min, y_min, x_max, y_max]
                })
        return detections

    def draw_detections(self, frame, detections):
        """
        Draw bounding boxes and labels on image
        """
        for det in detections:
            x_min, y_min, x_max, y_max = det["bbox"]
            label = det["label"]
            conf = det["confidence"]
            color = (0, 255, 0) if label == "person" else (255, 0, 0)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

    def draw_tracked_objects(self, frame, tracked_objects):
        """
        Draw tracked object IDs on frame
        """
        for obj_id, det in tracked_objects.items():
            x_min, y_min, x_max, y_max = det["bbox"]
            label = det["label"]
            color = (0, 255, 255)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(frame, f"{label} #{obj_id}", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame
