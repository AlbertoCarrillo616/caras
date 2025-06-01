from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import os
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore
import subprocess

# === Flask app ===
app = Flask(__name__)

# === Firebase Firestore ===
cred = credentials.Certificate("reconomiciento-facial-firebase-adminsdk-fbsvc-4f88b81dd4.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# === Clasificador Haar ===
CASCADE_PATH = os.path.abspath("haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    raise IOError(f"No se pudo cargar el clasificador Haar: {CASCADE_PATH}")

# === Directorio de rostros ===
FACES_DIR = "faces"
os.makedirs(FACES_DIR, exist_ok=True)

# === Función para detectar rostros ===
def detect_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces

# === Guardar registro en Firestore ===
def registrar_en_firestore(nombre, archivo):
    doc = {
        "nombre": nombre,
        "archivo": archivo,
        "fecha": datetime.now().isoformat()
    }
    db.collection("rostros_registrados").add(doc)
    print(f"✅ Registrado en Firestore: {doc}")

# === Subir cambios a GitHub ===
def subir_a_github(mensaje="Auto: nueva cara registrada"):
    try:
        subprocess.call(["git", "add", "."])
        subprocess.call(["git", "commit", "-m", mensaje])
        subprocess.call(["git", "push"])
        print("✅ Cambios subidos a GitHub")
    except Exception as e:
        print(f"❌ Error al subir a GitHub: {e}")

# === Página principal HTML ===
@app.route('/')
def index():
    return render_template("captura_rostros.html")

# === Registrar rostro ===
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
        registrar_en_firestore(name, f"{name}/{filename}")

    # 🔁 Subir cambios a GitHub
    subir_a_github(f"Auto: nuevo rostro registrado para {name}")

    return jsonify({"saved": saved, "person": name})

# === Detección de coincidencias ===
@app.route('/detect', methods=['POST'])
def detect():
    if 'image' in request.files:
        file = request.files['image']
        arr = np.frombuffer(file.read(), np.uint8)
    else:
        arr = np.frombuffer(request.data, np.uint8)

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

@app.route('/subir_manual', methods=['GET', 'POST'])
def subir_manual():
    if request.method == 'GET':
        return '''
        <h2>Subir imagen local para prueba</h2>
        <form method="POST" enctype="multipart/form-data">
            Nombre: <input type="text" name="name"><br><br>
            Imagen: <input type="file" name="image"><br><br>
            <input type="submit" value="Subir imagen">
        </form>
        '''

    # POST
    name = request.form.get("name")
    file = request.files.get("image")

    if not name or not file:
        return "❌ Falta nombre o archivo"

    arr = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        return "❌ No se pudo leer la imagen"

    faces = detect_faces(img)
    if len(faces) == 0:
        return "⚠️ No se detectaron rostros"

    folder = os.path.join(FACES_DIR, name)
    os.makedirs(folder, exist_ok=True)
    saved = []

    for i, (x, y, w, h) in enumerate(faces):
        face = img[y:y+h, x:x+w]
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.jpg"
        path = os.path.join(folder, filename)
        cv2.imwrite(path, face)
        saved.append(filename)
        registrar_en_firestore(name, f"{name}/{filename}")

    subir_a_github(f"Auto: rostro manual registrado para {name}")

    return f"✅ Rostro guardado como {saved}"


# === Iniciar servidor ===
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
