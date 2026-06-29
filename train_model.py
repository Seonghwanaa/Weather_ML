"""
지하차도 침수 예측 모델 학습 스크립트

실행 방법:
    python train_model.py

실행 후 model/ 폴더에 best_model.pkl, scaler.pkl, features.pkl 생성됨
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[경고] XGBoost 없음. pip install xgboost 로 설치하세요.")

OUTPUT_DIR = "model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. 데이터 로드 ──
df = pd.read_csv("data/지하차도_ML_최종데이터.csv", encoding="utf-8-sig")
print(f"데이터 로드: {len(df)}행  (침수: {(df['flooded']==1).sum()}건 / 미침수: {(df['flooded']==0).sum()}건)")

# ── 2. 피처 정의 ──
FEATURES = [
    "rainfall_1h_mm",    # 시간당 강수량 (mm)
    "rainfall_3h_mm",    # 3시간 누적 강수량 (mm)
    "rainfall_6h_mm",    # 6시간 누적 강수량 (mm)
    "rainfall_24h_mm",   # 24시간 누적 강수량 (mm)
    "depth_m",           # 지하차도 깊이 (m)
    "length_m",          # 지하차도 길이 (m)
    "built_year",        # 준공연도
    "lanes",             # 차로수
    "has_drainage_pump", # 배수펌프 유무 (0/1)
    "nearby_river",      # 하천 인접 여부 (0/1)
]

X = df[FEATURES].values
y = df["flooded"].values

# ── 3. 학습 / 테스트 분리 ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"학습: {len(X_train)}건  /  테스트: {len(X_test)}건")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── 4. 모델 학습 및 평가 ──
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=42, class_weight="balanced"
    ),
}
if HAS_XGB:
    models["XGBoost"] = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
        random_state=42, eval_metric="logloss", verbosity=0,
    )

results = {}
print("\n" + "=" * 55)
for name, model in models.items():
    Xtr = X_train_sc if name == "Logistic Regression" else X_train
    Xte = X_test_sc  if name == "Logistic Regression" else X_test

    cv_auc = cross_val_score(model, Xtr, y_train, cv=cv, scoring="roc_auc")
    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    y_prob = model.predict_proba(Xte)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n[{name}]")
    print(f"  CV AUC  : {cv_auc.mean():.4f} (+/-{cv_auc.std():.4f})")
    print(f"  Test AUC: {auc:.4f}  |  Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["미침수", "침수"]))

    results[name] = {"model": model, "auc": auc}

# ── 5. 피처 중요도 (Random Forest) ──
rf = results["Random Forest"]["model"]
importances = sorted(zip(FEATURES, rf.feature_importances_), key=lambda x: -x[1])
print("\n[Random Forest 피처 중요도]")
for feat, imp in importances:
    print(f"  {feat:25s}  {imp:.4f}  {'#' * int(imp * 50)}")

# ── 6. 최고 모델 저장 ──
best_name = max(results, key=lambda k: results[k]["auc"])
best_model = results[best_name]["model"]
print(f"\n최고 모델: {best_name}  (AUC={results[best_name]['auc']:.4f})")

with open(f"{OUTPUT_DIR}/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open(f"{OUTPUT_DIR}/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open(f"{OUTPUT_DIR}/features.pkl", "wb") as f:
    pickle.dump({"features": FEATURES, "best_model_name": best_name}, f)

print(f"모델 저장 완료: {OUTPUT_DIR}/")
