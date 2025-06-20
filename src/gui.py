import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import threading
import os
from inference import DebrisDetector
import webbrowser

class DebrisDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Detección de Escombros Espaciales")
        self.detector = DebrisDetector(model_path="C:/Users/adolf/OneDrive/Desktop/Axón_Drift_App/models/yolo/debris_detector2/weights/best.pt")
        self.camera_active = False
        self.camera_window = None
        self.video_window = None
        self.image_window = None
        self.powerbi_opened = False

        self.label = tk.Label(root, text="Detección de Escombros Espaciales")
        self.label.pack(pady=10)

        self.btn_image = tk.Button(root, text="Procesar Imagen", command=self.process_image)
        self.btn_image.pack(pady=5)

        self.btn_video = tk.Button(root, text="Procesar Video", command=self.process_video)
        self.btn_video.pack(pady=5)

        self.btn_camera = tk.Button(root, text="Usar Cámara", command=self.toggle_camera)
        self.btn_camera.pack(pady=5)

        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.pack(pady=10)

        self.status_label = tk.Label(root, text="Estado: Listo")
        self.status_label.pack(pady=5)

    def open_powerbi(self):
        if not self.powerbi_opened:
            powerbi_url = "https://app.powerbi.com/reportEmbed?reportId=61ea4911-7b2a-4d4c-88e0-7304aba71159&autoAuth=true&ctid=d73d9a37-04fd-4229-8f42-b1f8657af496&actionBarEnabled=true&reportCopilotInEmbed=true"
            webbrowser.open(powerbi_url)
            self.powerbi_opened = True

    def reset_powerbi(self):
        self.powerbi_opened = False

    def process_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
        if file_path:
            self.status_label.config(text="Procesando imagen...")
            self.open_powerbi()  # Abrir Power BI al iniciar detección
            try:
                predictions, output_path = self.detector.predict_image(file_path)
                img = Image.open(output_path)
                img = img.resize((640, 480))
                self.photo = ImageTk.PhotoImage(img)
                if self.image_window and self.image_window.winfo_exists():
                    self.image_window.destroy()
                self.image_window = tk.Toplevel(self.root)
                self.image_window.title("Resultado Imagen")
                self.image_canvas = tk.Canvas(self.image_window, width=640, height=480)
                self.image_canvas.pack(pady=10)
                self.image_canvas.create_image(0, 0, anchor="nw", image=self.photo)
                self.image_window.update()
                self.status_label.config(text="Imagen procesada, predicciones guardadas")
               # Permitir abrir Power BI en la próxima detección
            except Exception as e:
                self.status_label.config(text=f"Error procesando imagen: {str(e)}")

    def process_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
        if file_path:
            self.status_label.config(text="Procesando video...")
            self.open_powerbi()  # Abrir Power BI al iniciar detección
            try:
                predictions, output_path = self.detector.predict_video(file_path)
                if self.video_window and self.video_window.winfo_exists():
                    self.video_window.destroy()
                self.video_window = tk.Toplevel(self.root)
                self.video_window.title("Resultado Video")
                self.video_canvas = tk.Canvas(self.video_window, width=640, height=480)
                self.video_canvas.pack(pady=10)
                cap = cv2.VideoCapture(output_path)
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(img)
                    img = img.resize((640, 480))
                    self.video_photo = ImageTk.PhotoImage(img)
                    self.video_canvas.create_image(0, 0, anchor="nw", image=self.video_photo)
                    self.video_window.update()
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                cap.release()
                messagebox.showinfo("Éxito", f"Video procesado y guardado en {output_path}")
                self.status_label.config(text="Video procesado, predicciones guardadas")
                # Permitir abrir Power BI en la próxima detección
            except Exception as e:
                self.status_label.config(text=f"Error procesando video: {str(e)}")
            finally:
                if self.video_window and self.video_window.winfo_exists():
                    self.video_window.destroy()

    def toggle_camera(self):
        if not self.camera_active:
            self.camera_active = True
            self.btn_camera.config(text="Detener Cámara")
            self.status_label.config(text="Cámara activa...")
            threading.Thread(target=self.stream_camera, daemon=True).start()
        else:
            self.camera_active = False
            self.btn_camera.config(text="Usar Cámara")
            self.status_label.config(text="Cámara detenida")
              # Permitir abrir Power BI en la próxima detección
            if self.camera_window and self.camera_window.winfo_exists():
                self.camera_window.destroy()
                self.camera_window = None

    def stream_camera(self):
        try:
            cap = cv2.VideoCapture("http://192.168.1.68:8000/video")
            if not cap.isOpened():
                self.status_label.config(text="Error: No se pudo abrir la cámara")
                self.camera_active = False
                self.btn_camera.config(text="Usar Cámara")
                return

            self.camera_window = tk.Toplevel(self.root)
            self.camera_window.title("Cámara en Vivo")
            self.camera_canvas = tk.Canvas(self.camera_window, width=640, height=480)
            self.camera_canvas.pack(pady=10)

            self.open_powerbi()  # Abrir Power BI al iniciar detección

            for pred in self.detector.predict_camera():
                if not self.camera_active:
                    break
                try:
                    img = cv2.cvtColor(pred["image"], cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(img)
                    img = img.resize((640, 480))
                    self.camera_photo = ImageTk.PhotoImage(img)
                    self.camera_canvas.create_image(0, 0, anchor="nw", image=self.camera_photo)
                    self.camera_window.update()
                except Exception as e:
                    self.status_label.config(text=f"Error en predicción: {str(e)}")
                    break
        except Exception as e:
            self.status_label.config(text=f"Error en stream_camera: {str(e)}")
            self.camera_active = False
            self.btn_camera.config(text="Usar Cámara")
            if self.camera_window and self.camera_window.winfo_exists():
                self.camera_window.destroy()
                self.camera_window = None
        finally:
            if cap.isOpened():
                cap.release()
            self.camera_active = False
            self.btn_camera.config(text="Usar Cámara")
            self.status_label.config(text="Cámara detenida")
     # Permitir abrir Power BI en la próxima detección
            if self.camera_window and self.camera_window.winfo_exists():
                self.camera_window.destroy()
                self.camera_window = None

if __name__ == "__main__":
    root = tk.Tk()
    app = DebrisDetectionGUI(root)
    root.mainloop()