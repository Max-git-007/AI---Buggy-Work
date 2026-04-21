import requests
from flask import Flask, render_template, request, jsonify
import joblib
import os
import math
from werkzeug.utils import secure_filename

# Optional TensorFlow — may not be available on Python 3.14
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image
    TF_AVAILABLE = True
except Exception:
    tf = None
    image = None
    TF_AVAILABLE = False

# Serve `frontend/` as static files so `index.html` and assets are available
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))
app = Flask(__name__, static_folder=frontend_dir, static_url_path='')
model = None
if TF_AVAILABLE:
    try:
        model = tf.keras.models.load_model("model.h5")
    except Exception:
        model = None

# ================= LOAD MODELS =================
# If model artifacts are missing, try to run the training scripts to create them.
import subprocess
import sys

def _ensure_model(path, script=None):
    if os.path.exists(path):
        return True
    if script and os.path.exists(script):
        print(f"Model {path} not found — attempting to create it by running {script}")
        try:
            subprocess.run([sys.executable, script], check=True)
        except Exception as e:
            print("Training script failed:", e)
    return os.path.exists(path)

# Crop model
crop_model = None
crop_model_path = "model/crop_model.pkl"
if _ensure_model(crop_model_path, "model/crop_model_pkl.py"):
    try:
        crop_model = joblib.load(crop_model_path)
    except Exception as e:
        print("Failed to load crop model:", e)

# Fertilizer model placeholders; loaded later after weather endpoint
fertilizer_model = None
crop_labels = None
fertilizer_model_path = "model/fertilizer_model.pkl"
crop_labels_path = "model/crop_labels.pkl"
if _ensure_model(fertilizer_model_path, "model/train_fertilizer_model.py"):
    try:
        fertilizer_model = joblib.load(fertilizer_model_path)
        crop_labels = joblib.load(crop_labels_path)
    except Exception as e:
        print("Failed to load fertilizer model/labels:", e)

# If scikit-learn models aren't available (no binary wheels on this machine),
# provide lightweight pure-Python fallbacks so the app remains functional.
if crop_model is None:
    print("⚠️ Crop model not found — using simple fallback predictor (no scikit-learn)")
    class SimpleCropPredictor:
        def __init__(self):
            # Small prototype dataset (N, P, K, temperature, humidity, ph, rainfall)
            self.X = [
                [90, 40, 40, 30, 70, 6.5, 200],  # Rice-like
                [80, 35, 30, 20, 60, 7.0, 100],  # Wheat-like
                [70, 30, 25, 25, 65, 6.5, 150],  # Maize-like
                [50, 20, 20, 28, 55, 6.8, 80],   # Cotton-like
                [40, 20, 20, 22, 80, 5.5, 30],   # Potato-like
                [30, 15, 15, 27, 70, 6.0, 50],   # Tomato-like
            ]
            self.y = ["Rice", "Wheat", "Maize", "Cotton", "Potato", "Tomato"]
        def predict(self, X_input):
            preds = []
            for xi in X_input:
                best = None
                best_idx = 0
                for i, xref in enumerate(self.X):
                    dist = math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(xref, xi)))
                    if best is None or dist < best:
                        best = dist
                        best_idx = i
                preds.append(self.y[best_idx])
            return preds
    crop_model = SimpleCropPredictor()

if fertilizer_model is None or crop_labels is None:
    print("⚠️ Fertilizer model/labels not found — using simple fallback predictor (no scikit-learn)")
    # fallback crop labels and small dataset mirroring train_fertilizer_model.py
    crop_labels = ["Rice", "Wheat", "Maize"]
    class SimpleFertilizerPredictor:
        def __init__(self):
            # Rows: [crop_code, N, P, K]
            self.X = [
                [0, 90, 40, 40],
                [0, 40, 20, 20],
                [1, 80, 35, 30],
                [1, 30, 15, 15],
                [2, 70, 30, 25],
                [2, 20, 10, 10]
            ]
            self.y = ["Urea", "Compost", "DAP", "Compost", "NPK", "Compost"]
        def predict(self, X_input):
            preds = []
            for xi in X_input:
                best = None
                best_idx = 0
                for i, xref in enumerate(self.X):
                    dist = math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(xref, xi)))
                    if best is None or dist < best:
                        best = dist
                        best_idx = i
                preds.append(self.y[best_idx])
            return preds
    fertilizer_model = SimpleFertilizerPredictor()
# Disease model (TensorFlow) — only if TF available
disease_model = None
if TF_AVAILABLE:
    try:
        disease_model = tf.keras.models.load_model("model/disease_model.h5")
    except Exception:
        disease_model = None

CLASS_NAMES = [
    "Apple Scab", "Apple Black Rot", "Corn Leaf Blight",
    "Potato Early Blight", "Potato Late Blight", "Tomato Healthy"
]

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ================= HOME =================
from jinja2 import TemplateNotFound


@app.route("/")
def home():
    try:
        return render_template("index.html")
    except TemplateNotFound:
        # If there's no template folder, serve the static frontend/index.html
        return app.send_static_file('index.html')


# ================= CROP PREDICTION =================
@app.route("/predict-crop", methods=["POST"])
def predict_crop():
    if crop_model is None:
        return jsonify({"error": "Crop model not available. Run 'python model/crop_model_pkl.py' to create it."}), 501

    data = request.json

    features = [[
        data["N"], data["P"], data["K"],
        data["temperature"], data["humidity"],
        data["ph"], data["rainfall"]
    ]]

    try:
        prediction_raw = crop_model.predict(features)
        # Accept both list-like and numpy-like outputs
        if isinstance(prediction_raw, (list, tuple)):
            prediction = prediction_raw[0]
        else:
            try:
                prediction = prediction_raw[0]
            except Exception:
                prediction = prediction_raw
    except Exception as e:
        return jsonify({"error": f"Crop model prediction failed: {e}"}), 500

    return jsonify({"crop": prediction})

API_KEY = "YOUR_OPENWEATHER_API_KEY"

@app.route("/weather", methods=["POST"])
def get_weather():
    city = request.json.get("city")

    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    res = requests.get(url)
    data = res.json()

    if data.get("cod") != 200:
        return jsonify({"error": "City not found"})

    weather_info = {
        "temperature": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "description": data["weather"][0]["description"]
    }

    return jsonify(weather_info)
# fertilizer_model and crop_labels are created/loaded at startup if possible
@app.route("/predict-fertilizer", methods=["POST"])
def predict_fertilizer():
    if fertilizer_model is None or crop_labels is None:
        return jsonify({"error": "Fertilizer model not available. Run 'python model/train_fertilizer_model.py' to create it."}), 501

    data = request.json

    crop_name = data["crop"]
    N, P, K = data["N"], data["P"], data["K"]

    try:
        crop_code = list(crop_labels).index(crop_name)
    except ValueError:
        return jsonify({"error": "Unknown crop name"}), 400

    try:
        prediction_raw = fertilizer_model.predict([[crop_code, N, P, K]])
        if isinstance(prediction_raw, (list, tuple)):
            prediction = prediction_raw[0]
        else:
            try:
                prediction = prediction_raw[0]
            except Exception:
                prediction = prediction_raw
    except Exception as e:
        return jsonify({"error": f"Fertilizer model prediction failed: {e}"}), 500

    return jsonify({"fertilizer": prediction})

# ================= DISEASE PREDICTION =================
@app.route("/predict-disease", methods=["POST"])
def predict_disease():
    # If TensorFlow/disease model isn't available, return a helpful error
    if disease_model is None or image is None:
        return jsonify({
            "error": "Disease model not available on this Python environment (no TensorFlow). Run with Python 3.11/3.10, use a Docker image, or convert the model to ONNX and use onnxruntime."
        }), 501

    img_file = request.files.get("image")
    if img_file is None:
        return jsonify({"error": "No image uploaded"}), 400

    filename = secure_filename(img_file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    img_file.save(filepath)

    img = image.load_img(filepath, target_size=(224,224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = disease_model.predict(img_array)
    class_index = int(np.argmax(preds))
    confidence = float(np.max(preds))

    return jsonify({
        "disease": CLASS_NAMES[class_index],
        "confidence": f"{confidence*100:.2f}%"
    })


if __name__ == "__main__":
    app.run(debug=True)
 