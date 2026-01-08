import json
import numpy as np
import joblib
from collections import OrderedDict

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, ElasticNet, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from features.feature_engineering import build_features_train

import os




os.makedirs("models", exist_ok=True)


data = []
with open("data/problems_data.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))

scores_all = np.array(
    [row["problem_score"] for row in data],
    dtype=np.float64
)

print(f"Loaded {len(data)} problems")



print("➡️ Extracting features")
X, word_tfidf, char_tfidf, scaler = build_features_train(data)
print("✅ Feature extraction done")


scores = scores_all[:X.shape[0]]

print(f"After noise removal: {X.shape[0]} problems")



X_train, X_test, y_train, y_test = train_test_split(
    X,
    scores,
    test_size=0.2,
    random_state=42
)



baseline_pred = np.full_like(y_test, y_train.mean())
baseline_mae = mean_absolute_error(y_test, baseline_pred)

print(f"\n📉 Baseline MAE (mean predictor): {baseline_mae:.4f}")



models = OrderedDict({
    "LinearRegression": LinearRegression(),

    "ElasticNet": ElasticNet(
        alpha=0.1,
        l1_ratio=0.5,
        random_state=42
    ),

    "HuberRegressor": HuberRegressor(
        epsilon=1.35,
        max_iter=500
    ),

    "RandomForestRegressor": RandomForestRegressor(
        n_estimators=400,
        max_depth=30,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42
    ),

    "GradientBoostingRegressor": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
})



results = {}
trained_models = {}

print("\n🚀 Training & Evaluating Regression Models\n")

for name, model in models.items():
    print(f"➡️ {name}")

    # Dense only for linear models
    if name in ["LinearRegression", "ElasticNet", "HuberRegressor"]:
        X_tr = X_train.toarray()
        X_te = X_test.toarray()
    else:
        X_tr = X_train
        X_te = X_test

    model.fit(X_tr, y_train)
    preds = model.predict(X_te)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    results[name] = mae
    trained_models[name] = model

    # Save each model
    joblib.dump(model, f"models/{name}.pkl")

    print(f"   MAE  = {mae:.4f}")
    print(f"   RMSE = {rmse:.4f}")
    print(f"   R²   = {r2:.4f}")
    print(f"   💾 Saved as models/{name}.pkl\n")



# SELECT & SAVE BEST REGRESSOR

best_model_name = min(results, key=results.get)
best_model = trained_models[best_model_name]

print(f"🏆 Best Regressor: {best_model_name}")
print(f"📉 Best MAE: {results[best_model_name]:.4f}")

joblib.dump(best_model, "models/best_regressor.pkl")
print("✅ Best regressor ALSO saved as models/best_regressor.pkl")




joblib.dump(word_tfidf, "models/word_tfidf.pkl")
joblib.dump(char_tfidf, "models/char_tfidf.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Preprocessing objects saved")


print("\n🏁 REGRESSION MODEL COMPARISON SUMMARY\n")

for name, mae in results.items():
    print(f"{name:<26} | MAE = {mae:.4f}")
