"""
Train ML models (LR, SVM, RF, ANN, AdaBoost, XGBoost, LightGBM, DNN, GPR) for fluoride classification.
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Optional imports (handle missing libraries gracefully)
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    import numpy as np
except ImportError:
    Sequential = None

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def get_models():
    """Return a dictionary of ML models to train."""
    models = {
        "LR": LogisticRegression(max_iter=10000, solver="saga", multi_class="multinomial"),
        "SVM": SVC(kernel="rbf", probability=True),
        "RF": RandomForestClassifier(n_estimators=100, random_state=42),
        "ANN": MLPClassifier(hidden_layer_sizes=(2000,), max_iter=20000, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
    }

    if XGBClassifier:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5, subsample=0.8, random_state=42
        )
    if LGBMClassifier:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=-1, random_state=42
        )

    # Gaussian Process (can be slow, so use only for smaller datasets)
    # models["GPR"] = GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42)

    return models


def build_dnn(input_dim, output_dim):
    """Build a simple deep neural network using Keras."""
    model = Sequential([
        Dense(256, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(output_dim, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    models = get_models()
    results = {}

    for name, model in models.items():
        print(f"\n🔹 Training {name}...")
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            cm = confusion_matrix(y_test, preds)
            results[name] = {
                "accuracy": acc,
                "report": classification_report(y_test, preds, output_dict=True),
                "confusion_matrix": cm.tolist(),
            }
            joblib.dump(model, os.path.join(MODEL_DIR, f"{name}_model.pkl"))
            print(f"{name} Accuracy: {acc:.4f}")
        except Exception as e:
            print(f"⚠️ Skipping {name}: {e}")

    # DNN (Keras)
    if Sequential:
        print("\n🔹 Training Deep Neural Network (DNN)...")
        num_classes = len(set(y))
        dnn = build_dnn(X_train.shape[1], num_classes)

        X_train_np, X_test_np = np.array(X_train), np.array(X_test)
        y_train_np, y_test_np = np.array(y_train), np.array(y_test)

        dnn.fit(X_train_np, y_train_np, epochs=30, batch_size=32, verbose=0)
        dnn_preds = dnn.predict(X_test_np).argmax(axis=1)

        acc = accuracy_score(y_test_np, dnn_preds)
        cm = confusion_matrix(y_test_np, dnn_preds)
        results["DNN"] = {
            "accuracy": acc,
            "report": classification_report(y_test_np, dnn_preds, output_dict=True),
            "confusion_matrix": cm.tolist(),
        }
        dnn.save(os.path.join(MODEL_DIR, "DNN_model.h5"))
        print(f"DNN Accuracy: {acc:.4f}")

    # Save summary results
    pd.DataFrame(
        [{"Model": k, "Accuracy": v["accuracy"]} for k, v in results.items()]
    ).to_csv(os.path.join(MODEL_DIR, "model_accuracies.csv"), index=False)

    return results, (X_train, X_test, y_train, y_test)
