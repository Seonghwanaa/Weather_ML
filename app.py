"""
지하차도 침수 예측 Flask API 서버

실행 방법:
    1. python train_model.py  (모델 생성)
    2. python app.py          (서버 실행)
    3. 브라우저에서 http://localhost:5000 접속
"""

import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# 모델 로드
with open("model/best_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("model/features.pkl", "rb") as f:
    meta = pickle.load(f)

FEATURES = meta["features"]
MODEL_NAME = meta["best_model_name"]
print(f"모델 로드 완료: {MODEL_NAME}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    values = [
        float(data.get("rainfall_1h", 0)),
        float(data.get("rainfall_3h", 0)),
        float(data.get("rainfall_6h", 0)),
        float(data.get("rainfall_24h", 0)),
        float(data.get("depth_m", 3.5)),
        float(data.get("length_m", 200)),
        int(data.get("built_year", 2000)),
        int(data.get("lanes", 4)),
        int(data.get("has_drainage_pump", 0)),
        int(data.get("nearby_river", 0)),
    ]

    prob = float(model.predict_proba([values])[0][1])

    if prob >= 0.7:
        risk = "높음"
        color = "#e74c3c"
    elif prob >= 0.4:
        risk = "중간"
        color = "#f39c12"
    else:
        risk = "낮음"
        color = "#27ae60"

    return jsonify({
        "probability": round(prob, 4),
        "percent": round(prob * 100, 1),
        "risk_level": risk,
        "color": color,
        "model": MODEL_NAME,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
