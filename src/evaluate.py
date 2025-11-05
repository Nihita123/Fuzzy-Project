"""
Evaluate saved models and visualize confusion matrices.
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_models():
    models = {}
    for f in os.listdir(MODEL_DIR):
        if f.endswith(".pkl"):
            name = f.replace("_model.pkl", "")
            models[name] = joblib.load(os.path.join(MODEL_DIR, f))
    return models

def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        cm = confusion_matrix(y_test, preds)
        results[name] = acc

        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{name} Confusion Matrix (Acc={acc:.2f})")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"{name}_cm.png"))
        plt.close()

        print(f"{name}: Accuracy={acc:.4f}")
        print(classification_report(y_test, preds))
    pd.DataFrame({"Model": list(results.keys()), "Accuracy": list(results.values())}).to_csv(
        os.path.join(RESULTS_DIR, "evaluation_summary.csv"), index=False
    )
