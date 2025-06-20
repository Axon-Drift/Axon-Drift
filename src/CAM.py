import cv2
cap = cv2.VideoCapture("http://192.168.1.68:8000/video")  # URL proporcionada por DroidCam
ret, frame = cap.read()
print("Cámara conectada" if ret else "Error en cámara")
cap.release()