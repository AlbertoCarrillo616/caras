from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)

# === Ruta absoluta al clasificador Haar ===
CASCADE_PATH = os.path.abspath("haarcascade_frontalface_default.xml")
print(f"Cargando clasificador Haar desde: {CASCADE_PATH}")

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    raise IOError(f"No se pudo cargar el clasificador Haar: {CASCADE_PATH}")

# === Carpeta donde se guardan los rostros ===
FACES_DIR = "faces"
os.makedirs(FACES_DIR, exist_ok=True)

# === Función para detectar rostros ===
def detect_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces

# === Página HTML principal ===
@app.route('/')
def index():
    return render_template("captura_rostros.html")

# === Registrar rostros desde navegador ===
@app.route('/register', methods=['POST'])
def register():
    name = request.form.get("name")
    file = request.files.get("image")

    if not name or not file:
        return jsonify({"error": "Falta el nombre o archivo de imagen"}), 400

    arr = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "No se pudo decodificar la imagen"}), 400

    faces = detect_faces(img)
    if len(faces) == 0:
        return jsonify({"error": "No se detectaron caras"}), 400

    folder = os.path.join(FACES_DIR, name)
    os.makedirs(folder, exist_ok=True)
    saved = []

    for i, (x, y, w, h) in enumerate(faces):
        face = img[y:y+h, x:x+w]
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.jpg"
        path = os.path.join(folder, filename)
        cv2.imwrite(path, face)
        saved.append(filename)

    return jsonify({"saved": saved, "person": name})

# === Detectar coincidencias desde navegador o ESP32 ===
@app.route('/detect', methods=['POST'])
def detect():
    if 'image' in request.files:
        file = request.files['image']
        arr = np.frombuffer(file.read(), np.uint8)
    else:
        arr = np.frombuffer(request.data, np.uint8)  # ← ESP32 manda así

    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "No se pudo decodificar la imagen"}), 400

    faces = detect_faces(img)
    if len(faces) == 0:
        return jsonify({"match": False, "reason": "No se detectaron rostros"})

    x, y, w, h = faces[0]
    new_face = cv2.resize(img[y:y+h, x:x+w], (100, 100))

    for person in os.listdir(FACES_DIR):
        person_path = os.path.join(FACES_DIR, person)
        for file_name in os.listdir(person_path):
            path = os.path.join(person_path, file_name)
            ref_img = cv2.imread(path)
            if ref_img is None:
                continue
            try:
                ref_img = cv2.resize(ref_img, (100, 100))
                diff = np.mean((ref_img - new_face) ** 2)
                if diff < 2000:
                    return jsonify({"match": True, "person": person})
            except:
                continue

    return jsonify({"match": False})

# === Iniciar el servidor Flask ===
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)