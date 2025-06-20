import os
import cv2
import torch
from ultralytics import YOLO
import json
import numpy as np
import time

class DebrisDetector:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = "C:/Users/adolf/OneDrive/Desktop/Axón_Drift_App/models/yolo/debris_detector2/weights/best.pt"
        self.model = YOLO(model_path)
        self.output_dir = "C:/Users/adolf/OneDrive/Desktop/Axón_Drift_App/output"
        os.makedirs(os.path.join(self.output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "predictions"), exist_ok=True)

    def _process_results(self, results, frame):
        predictions = {
            "boxes": [],
            "classes": [],
            "scores": [],
            "timestamp": datetime.datetime.now().isoformat()
        }
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            for box, score, cls in zip(boxes, scores, classes):
                predictions["boxes"].append(box.tolist())
                predictions["classes"].append(int(cls))
                predictions["scores"].append(float(score))
        img = result.plot()  # Dibuja las detecciones en la imagen
        return {"image": img, "predictions": predictions}

    def _save_predictions(self, predictions):
        timestamp = predictions["timestamp"].replace(":", "-")
        output_path = os.path.join(self.output_dir, "predictions", f"{timestamp}.json")
        with open(output_path, "w") as f:
            json.dump(predictions, f, indent=4)
        print(f"Predicciones guardadas localmente en: {output_path}")

    def predict_image(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen en {img_path}")
        results = self.model.predict(img, conf=0.1)  # Umbral de confianza
        pred = self._process_results(results, img)
        output_path = os.path.join(self.output_dir, "images", f"pred_{os.path.basename(img_path)}")
        cv2.imwrite(output_path, pred["image"])
        self._save_predictions(pred["predictions"])
        return pred, output_path

    def predict_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video en {video_path}")
        output_path = os.path.join(self.output_dir, "videos", f"pred_{os.path.basename(video_path)}")
        os.makedirs(os.path.join(self.output_dir, "videos"), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model.predict(frame, conf=0.1)
            pred = self._process_results(results, frame)
            out.write(pred["image"])
            self._save_predictions(pred["predictions"])
        cap.release()
        out.release()
        return pred, output_path

    def predict_camera(self):
        cap = cv2.VideoCapture("http://192.168.1.68:8000/video")
        max_attempts = 5
        attempt = 0
        while attempt < max_attempts and not cap.isOpened():
            print(f"Intento {attempt + 1} de conectar a la cámara...")
            time.sleep(2)  # Esperar 2 segundos antes de reintentar
            cap = cv2.VideoCapture("http://192.168.1.68:8000/video")
            attempt += 1
        if not cap.isOpened():
            raise ValueError("No se pudo abrir la cámara después de varios intentos")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model.predict(frame, conf=0.1)
            pred = self._process_results(results, frame)
            self._save_predictions(pred["predictions"])
            yield pred
        cap.release()

# Añadir importación de datetime si no está presente
import datetime