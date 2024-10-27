from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# 加载模型
random_forest_model = joblib.load("final_random_forest_model.joblib")
voting_regressor_model = joblib.load("voting_regressor_model.joblib")

# 根路径，检查服务是否运行
@app.route('/')
def home():
    return "Football Bet Chatbot API is running"

# 分类预测端点
@app.route('/predict_classification', methods=['POST'])
def predict_classification():
    data = request.get_json()
    # 检查 'features' 是否存在
    if 'features' not in data:
        return jsonify({"error": "Missing 'features' in request"}), 400
    
    features = np.array(data['features']).reshape(1, -1)
    prediction = random_forest_model.predict(features)
    return jsonify({'classification_prediction': int(prediction[0])})

# 回归预测端点
@app.route('/predict_regression', methods=['POST'])
def predict_regression():
    data = request.get_json()
    # 检查 'features' 是否存在
    if 'features' not in data:
        return jsonify({"error": "Missing 'features' in request"}), 400
    
    features = np.array(data['features']).reshape(1, -1)
    prediction = voting_regressor_model.predict(features)
    return jsonify({'regression_prediction': float(prediction[0])})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
