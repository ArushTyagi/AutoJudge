import json
import numpy as np
import joblib
import os
from collections import OrderedDict

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from features.feature_engineering import build_features_train


# ======================================================
# SETUP
# ======================================================
os.makedirs("models", exist_ok=True)


# ======================================================
# LOAD DATA
# ======================================================
data = []
with open("data/problems_data.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))

labels_all = np.array(
    [row["problem_class"].capitalize() for row in data]
)

print(f"Loaded {len(data)} problems")


# ======================================================
# FEATURE EXTRACTION (noise handled inside)
# ======================================================
print("➡️ Extracting features")
X, word_tfidf, char_tfidf, scaler = build_features_train(data)
print("✅ Feature extraction done")

labels = labels_all[:X.shape[0]]
print(f"After noise removal: {X.shape[0]} problems")


# ======================================================
# TRAIN / TEST SPLIT
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)


# ======================================================
# LABEL ENCODING
# ======================================================
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)


# ======================================================
# MODELS TO COMPARE
# ======================================================
models = OrderedDict({
    "LogisticRegression": LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        n_jobs=-1
    ),

    "LinearSVM": LinearSVC(
        class_weight="balanced",
        max_iter=5000
    ),

    "RBFSVM": SVC(
        kernel="rbf",
        class_weight="balanced",
        probability=True
    ),

    "RandomForestClassifier": RandomForestClassifier(
        n_estimators=300,
        max_depth=25,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )
})


# ======================================================
# TRAIN, EVALUATE & SAVE ALL STANDARD CLASSIFIERS
# ======================================================
results = {}
trained_models = {}

print("\n🚀 Training & Evaluating Standard Classifiers\n")

for name, model in models.items():
    print(f"➡️ {name}")

    model.fit(X_train, y_train_enc)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test_enc, preds)
    bal_acc = balanced_accuracy_score(y_test_enc, preds)

    results[name] = bal_acc
    trained_models[name] = model

    # 💾 Save each model
    joblib.dump(model, f"models/{name}.pkl")

    print(f"   Accuracy          = {acc:.3f}")
    print(f"   Balanced Accuracy = {bal_acc:.3f}")
    print(f"   💾 Saved as models/{name}.pkl\n")


# ======================================================
# SELECT & SAVE BEST STANDARD CLASSIFIER
# ======================================================
best_model_name = max(results, key=results.get)
best_model = trained_models[best_model_name]

print(f"🏆 Best Standard Classifier: {best_model_name}")

joblib.dump(best_model, "models/best_classifier.pkl")
joblib.dump(le, "models/label_encoder.pkl")

print("✅ Best standard classifier saved as models/best_classifier.pkl")


# ======================================================
# ORDINAL CLASSIFIER (STANDOUT)
# ======================================================
print("\n⭐ Training Ordinal Classifier")

y_train_easy = np.array([1 if y == "Easy" else 0 for y in y_train])
y_train_hard = np.array([1 if y == "Hard" else 0 for y in y_train])

clf_easy = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    n_jobs=-1
)

clf_hard = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    n_jobs=-1
)

clf_easy.fit(X_train, y_train_easy)
clf_hard.fit(X_train, y_train_hard)

easy_prob = clf_easy.predict_proba(X_test)[:, 1]
hard_prob = clf_hard.predict_proba(X_test)[:, 1]

ordinal_preds = []
for pe, ph in zip(easy_prob, hard_prob):
    if pe >= 0.6:
        ordinal_preds.append("Easy")
    elif ph >= 0.4:
        ordinal_preds.append("Hard")
    else:
        ordinal_preds.append("Medium")

print("\n📊 Ordinal Classification Report:\n")
print(classification_report(y_test, ordinal_preds))


# ======================================================
# SAVE ORDINAL CLASSIFIER COMPONENTS
# ======================================================
joblib.dump(clf_easy, "models/ordinal_easy.pkl")
joblib.dump(clf_hard, "models/ordinal_hard.pkl")

print("✅ Ordinal classifier components saved")


# ======================================================
# SAVE PREPROCESSING OBJECTS (CRITICAL FOR UI)
# ======================================================
joblib.dump(word_tfidf, "models/word_tfidf.pkl")
joblib.dump(char_tfidf, "models/char_tfidf.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Preprocessing objects saved")


# ======================================================
# FINAL SUMMARY
# ======================================================
print("\n🏁 CLASSIFIER TRAINING COMPLETE\n")

for name, bal in results.items():
    print(f"{name:<26} | Balanced Accuracy = {bal:.3f}")
