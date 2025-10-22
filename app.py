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
def prediction():
    try:
        query = request.get_json()
        url = query.get("url") 

        if not url:
            return jsonify({"error": "No URL provided"}), 400

        result = predict_model(url)
    # Make sure confidence is always present and float
        response = {
            "prediction": result.get("prediction", "Unknown"),
            "confidence": float(result.get("confidence", 0.0))
        }

        return jsonify(response)  

    except Exception as e:
        print(f"Error in /predict: {e}")
        return jsonify({"error": "Error scanning URL", "confidence": 0}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


