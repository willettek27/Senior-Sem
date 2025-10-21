# app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
from model import predict_model


app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return {"message": "API is running successfully."}

@app.route("/predict", methods=["POST"])
def predict():
    query = request.get_json()
    url = query.get("url") 

    if not query or not url:
        return jsonify({"error": "No URL provided"}), 400

    result = predict_model(url)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


