# tracker.py
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np


class ObjectTracker:
    def __init__(self, method: str = "centroid", max_disappeared: int = 15):
        """
        Initialize centroid-based tracker
        """
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared

    def register(self, centroid, bbox, label):
        self.objects[self.next_object_id] = {
            "centroid": centroid,
            "bbox": bbox,
            "label": label
        }
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, detections):
        """
        Update tracker with new detections
        """
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        # Compute centroids
        input_centroids = np.zeros((len(detections), 2), dtype="int")
        for (i, det) in enumerate(detections):
            x_min, y_min, x_max, y_max = det["bbox"]
            cX = int((x_min + x_max) / 2.0)
            cY = int((y_min + y_max) / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i, det in enumerate(detections):
                self.register(input_centroids[i], det["bbox"], det["label"])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = [obj["centroid"] for obj in self.objects.values()]

            # Compute distance matrix between old and new centroids
            D = dist.cdist(np.array(object_centroids), input_centroids)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            assigned_cols = set()
            for (row, col) in zip(rows, cols):
                if col in assigned_cols:
                    continue
                object_id = object_ids[row]
                self.objects[object_id]["centroid"] = input_centroids[col]
                self.objects[object_id]["bbox"] = detections[col]["bbox"]
                self.objects[object_id]["label"] = detections[col]["label"]
                self.disappeared[object_id] = 0
                assigned_cols.add(col)

            unassigned_rows = set(range(0, D.shape[0])) - set(rows)
            unassigned_cols = set(range(0, D.shape[1])) - assigned_cols

            # Handle disappeared objects
            for row in unassigned_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # Register new detections
            for col in unassigned_cols:
                self.register(input_centroids[col], detections[col]["bbox"], detections[col]["label"])

        return self.objects
