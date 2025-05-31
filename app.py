from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)

# Load model and configs
cls_model = joblib.load("trained_data/model_cls.pkl")
features = joblib.load("trained_data/model_features.pkl")
label_encoder = joblib.load("trained_data/grade_encoder.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Prepare input DataFrame
    input_df = pd.DataFrame([data])
    input_df = input_df.reindex(columns=features, fill_value=0)

    # Predict
    pred_encoded = cls_model.predict(input_df)[0]
    result = label_encoder.inverse_transform([pred_encoded])[0]

    return jsonify({"Result": result})

if __name__ == "__main__":
    app.run(debug=True)
