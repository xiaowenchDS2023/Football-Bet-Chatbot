from flask import Flask, request, jsonify
from flask_cors import CORS  # 导入 CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # 启用 CORS 支持

# 加载模型
random_forest_model = joblib.load("final_random_forest_model.joblib")
voting_regressor_model = joblib.load("voting_regressor_model.joblib")

@app.route('/predict_classification', methods=['POST'])
def predict_classification():
    data = request.get_json()
    if 'features' not in data:
        return jsonify({"error": "Missing 'features' in request"}), 400
    
    features = np.array(data['features']).reshape(1, -1)
    prediction = random_forest_model.predict(features)
    return jsonify({'classification_prediction': int(prediction[0])})

@app.route('/predict_regression', methods=['POST'])
def predict_regression():
    data = request.get_json()
    if 'features' not in data:
        return jsonify({"error": "Missing 'features' in request"}), 400
    
    features = np.array(data['features']).reshape(1, -1)
    prediction = voting_regressor_model.predict(features)
    return jsonify({'regression_prediction': float(prediction[0])})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
