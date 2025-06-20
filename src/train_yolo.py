from ultralytics import YOLO
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Definir rutas absolutas
data_path = "C:/Users/adolf/OneDrive/Desktop/Axón_Drift_App/data/dataset.yaml"
project_path = "C:/Users/adolf/OneDrive/Desktop/Axón_Drift_App/models/yolo"

# Verificar existencia de dataset.yaml y archivos
if not os.path.exists(data_path):
    logger.error(f"Error: El archivo dataset.yaml no se encuentra en {data_path}")
    raise FileNotFoundError(f"Dataset configuration file not found at {data_path}")

model = YOLO('yolov5su.pt')  # Usar modelo recomendado

try:
    model.train(
        data=data_path,
        epochs=100,  # Aumentar épocas para mejor convergencia
        imgsz=640,
        batch=4,  # Ajustado para CPU
        name='debris_detector',
        project=project_path,
        device='cpu',
        patience=20,
        save_period=10,
        pretrained=True,
        single_cls=False,  # Múltiples clases (4 categorías)
        cos_lr=True,       # Usar aprendizaje con coseno
        lr0=0.001,        # Tasa de aprendizaje inicial
        lrf=0.1           # Factor de reducción de aprendizaje
    )
    logger.info("Entrenamiento completado")
except Exception as e:
    logger.error(f"Error durante el entrenamiento: {e}")
    raise