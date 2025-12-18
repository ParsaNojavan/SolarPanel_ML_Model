from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_swagger_ui import get_swaggerui_blueprint
from flask_cors import CORS
import os
from ml.solar_model import SolarMLSystem

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"csv", "xlsx"}

app = Flask(__name__)
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    supports_credentials=True
)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ml_system = SolarMLSystem()

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------- SWAGGER ----------------
SWAGGER_URL = "/swagger"
API_URL = "/static/swagger.yaml"
swaggerui_blueprint = get_swaggerui_blueprint(SWAGGER_URL, API_URL, config={"app_name": "Solar Panel ML API"})
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# ---------------- HOME ----------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Solar Panel ML API is running 🚀"})

# ---------------- TRAIN ----------------
@app.route("/train", methods=["POST"])
def train():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Only CSV or XLSX allowed"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    try:
        results = ml_system.train(file_path)
        return jsonify({"status": "training completed", "results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- PREDICT ----------------
@app.route("/predict", methods=["POST"])
def predict():
    if not ml_system.is_trained:
        return jsonify({"error": "Model not trained yet"}), 400
    try:
        return jsonify(ml_system.predict(request.json))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- ANALYZE ----------------
@app.route("/analyze", methods=["POST"])
def analyze():
    if not ml_system.is_trained:
        return jsonify({"error": "Model not trained yet"}), 400
    try:
        return jsonify(ml_system.analyze_point(request.json))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- FULL ANALYSIS ----------------
@app.route("/full_analysis", methods=["POST"])
def full_analysis():
    if not ml_system.is_trained:
        return jsonify({"error": "Model not trained yet"}), 400
    try:
        return jsonify(ml_system.full_analysis(request.json))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
