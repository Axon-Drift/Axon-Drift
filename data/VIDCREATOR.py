import numpy as np
import cv2
import random

# Crear fondo de estrellas
def create_starfield(width, height, num_stars):
    starfield = np.zeros((height, width, 3), dtype=np.uint8)
    for _ in range(num_stars):
        x, y = random.randint(0, width - 1), random.randint(0, height - 1)
        brightness = random.randint(100, 255)
        size = random.choice([1, 2])
        if size == 1:
            starfield[y, x] = (brightness, brightness, brightness)
        else:
            cv2.circle(starfield, (x, y), size, (brightness, brightness, brightness), -1)
    return starfield

# Generar sprite de desecho espacial
def generate_debris_sprite(category_name, size_px=300):
    sprite = np.zeros((size_px, size_px, 4), dtype=np.uint8)
    rgb = np.random.randint(150, 255, 3)
    color = (int(rgb[0]), int(rgb[1]), int(rgb[2]), 255)
    center_x, center_y = size_px // 2, size_px // 2
    angle = random.uniform(0, 360)

    if category_name == "fragmento_pequeno":
        w, h = int(size_px * 0.6), int(size_px * 0.6)
        rect = ((center_x, center_y), (w, h), angle)
        box_points = np.int32(cv2.boxPoints(rect))
        cv2.drawContours(sprite, [box_points], 0, color, -1)
    elif category_name == "panel_solar":
        w, h = int(size_px * 0.9), int(size_px * 0.3)
        rect = ((center_x, center_y), (w, h), angle)
        box_points = np.int32(cv2.boxPoints(rect))
        cv2.drawContours(sprite, [box_points], 0, color, -1)
    elif category_name == "sensor":
        axis_1, axis_2 = int(size_px * 0.7), int(size_px * 0.7)
        cv2.ellipse(sprite, (center_x, center_y), (axis_1 // 2, axis_2 // 2), angle, 0, 360, color, -1)
    elif category_name == "resto_electronico":
        size = int(size_px * 0.8)
        points = np.array([
            [center_x, center_y - size // 2],
            [center_x + size // 2, center_y],
            [center_x, center_y + size // 2],
            [center_x - size // 3, center_y + size // 3],
            [center_x - size // 2, center_y - size // 3]
        ], dtype=np.int32)
        M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1)
        box_points = cv2.transform(np.array([points]), M)[0].astype(np.int32)
        cv2.drawContours(sprite, [box_points], 0, color, -1)
    return sprite

# Superponer imagen RGBA sobre fondo RGB (sin difuminado)
def overlay_image_alpha(background, overlay, x, y):
    h, w = overlay.shape[:2]
    b_h, b_w = background.shape[:2]
    x1, x2 = max(0, x), min(b_w, x + w)
    y1, y2 = max(0, y), min(b_h, y + h)
    overlay_x1, overlay_x2 = max(0, -x), min(w, b_w - x)
    overlay_y1, overlay_y2 = max(0, -y), min(h, b_h - y)
    if x1 >= x2 or y1 >= y2:
        return
    overlay_crop = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
    alpha = overlay_crop[:, :, 3] / 255.0
    alpha_inv = 1.0 - alpha
    background_crop = background[y1:y2, x1:x2]
    for c in range(3):
        background_crop[:, :, c] = (
            alpha * overlay_crop[:, :, c] +
            alpha_inv * background_crop[:, :, c]
        )

# Parámetros del video
width, height = 640, 480
num_debris = 10
categories = ["fragmento_pequeno", "panel_solar", "sensor", "resto_electronico"]
background = create_starfield(width, height, 200)
video_path = "simulacion_espacial.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 20.0
out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

# Inicializar desechos espaciales
debris_list = []
for i in range(num_debris):
    sprite = generate_debris_sprite(random.choice(categories))
    start_z = random.uniform(80, 150)

    # Posición inicial X a la izquierda o derecha (borde)
    side = random.choice(['left', 'right'])
    if side == 'left':
        x_start = random.uniform(20, 80)
        x_end = random.uniform(20, 80)
    else:
        x_start = random.uniform(width - 80, width - 20)
        x_end = random.uniform(width - 80, width - 20)

    debris_list.append({
        "sprite": sprite,
        "x_start": x_start,
        "x_end": x_end,
        "y": random.uniform(100, height - 100),
        "z": start_z,
        "dz": random.uniform(-1.0, -0.5),  # Movimiento lento hacia "adelante"
        "start_frame": i * 15 + random.randint(0, 10)
    })

total_frames = 300  # Duración total del video en frames
max_sprite_size = 300  # Tamaño máximo de sprite para evitar errores

for frame_idx in range(total_frames):
    frame = background.copy()
    for d in debris_list:
        if frame_idx < d["start_frame"]:
            continue

        # Actualizar profundidad z, limitar mínimo para evitar escala gigante
        d["z"] += d["dz"]
        if d["z"] <= 0:
            continue

        z_for_scale = max(d["z"], 5)  # Evitar escala muy grande
        scale = 100 / z_for_scale
        size = max(10, min(int(scale * 80), max_sprite_size))

        # Redimensionar sprite
        resized = cv2.resize(d["sprite"], (size, size), interpolation=cv2.INTER_AREA)

        # Interpolación lineal para movimiento lateral suave de un lado a otro
        total_move_frames = total_frames - d["start_frame"]
        progress_x = min(1.0, max(0.0, (frame_idx - d["start_frame"]) / total_move_frames))
        x_pos = d["x_start"] + progress_x * (d["x_end"] - d["x_start"])

        # Posición vertical fija (puedes agregar oscilaciones si quieres)
        y_pos = d["y"]

        draw_x = int(x_pos - size / 2)
        draw_y = int(y_pos - size / 2)

        overlay_image_alpha(frame, resized, draw_x, draw_y)

    out.write(frame)

out.release()

print(f"Video guardado en: {video_path}")
