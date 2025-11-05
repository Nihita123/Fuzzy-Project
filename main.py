import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Optional imports
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
except ImportError:
    Sequential = None


# ---------- Utility: Metrics per class ----------
def compute_metrics(y_true, y_pred):
    """Compute sensitivity, specificity, and error rate per class."""
    cm = confusion_matrix(y_true, y_pred)
    num_classes = cm.shape[0]
    sensitivity, specificity, error_rate = [], [], []

    for i in range(num_classes):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - (TP + FP + FN)

        sensitivity.append(TP / (TP + FN) if (TP + FN) > 0 else 0)
        specificity.append(TN / (TN + FP) if (TN + FP) > 0 else 0)
        error_rate.append((FP + FN) / cm.sum())

    return sensitivity, specificity, error_rate


# ---------- Models ----------
def get_models():
    models = {
        "LR": LogisticRegression(max_iter=10000, solver="saga", multi_class="multinomial"),
        "SVM": SVC(kernel="rbf", probability=True),
        "RF": RandomForestClassifier(n_estimators=100, random_state=42),
        "ANN": MLPClassifier(hidden_layer_sizes=(200,), max_iter=2000, random_state=42),
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
    return models


def build_dnn(input_dim, output_dim):
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


# ---------- Training ----------
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    models = get_models()
    results, preds_dict = {}, {}

    for name, model in models.items():
        print(f"\n🔹 Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        cm = confusion_matrix(y_test, preds)
        results[name] = {
            "accuracy": acc,
            "report": classification_report(y_test, preds, output_dict=True),
            "confusion_matrix": cm.tolist(),
        }
        preds_dict[name] = preds
        print(f"{name} Accuracy: {acc:.4f}")

    # DNN if TensorFlow available
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
        preds_dict["DNN"] = dnn_preds
        print(f"DNN Accuracy: {acc:.4f}")

    return results, preds_dict, (X_train, X_test, y_train, y_test)


# ---------- Tabular Summary ----------
def summarize_results(results, preds_dict, y_test):
    summary = []
    for name in results.keys():
        sens, spec, err = compute_metrics(y_test, preds_dict[name])
        summary.append({
            "Model": name,
            "Accuracy (%)": round(results[name]["accuracy"] * 100, 2),
            "Sensitivity (%)": round(np.mean(sens) * 100, 2),
            "Specificity (%)": round(np.mean(spec) * 100, 2),
            "Error rate (%)": round(np.mean(err) * 100, 2)
        })
    df = pd.DataFrame(summary)
    print("\n🔹 Model Performance Summary:\n")
    print(df.to_string(index=False))

    # ---- Find best model ----
    best_model = df.loc[df['Accuracy (%)'].idxmax()]
    print("\n🏆 Best Suited Model:")
    print(f"➡ {best_model['Model']} achieves the highest accuracy "
          f"({best_model['Accuracy (%)']}%), with sensitivity "
          f"{best_model['Sensitivity (%)']}%, specificity "
          f"{best_model['Specificity (%)']}%, and lowest error rate "
          f"{best_model['Error rate (%)']}%.")
    return df


# ---------- Visualization ----------
def plot_performance_dashboard(results, y_test, predictions_dict):
    models = list(results.keys())
    num_classes = len(np.unique(y_test))
    colors = ["green", "gold", "red"]
    width = 0.25
    x = np.arange(len(models))

    metrics_all = {}
    for m in models:
        sens, spec, err = compute_metrics(y_test, predictions_dict[m])
        metrics_all[m] = {
            "Accuracy": results[m]["accuracy"],
            "Sensitivity": sens,
            "Specificity": spec,
            "ErrorRate": err
        }

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()

    # (a) Accuracy
    ax = axes[0]
    ax.bar(models, [metrics_all[m]["Accuracy"] for m in models], color="skyblue")
    ax.set_title("(a) Accuracy of ML models", color='red', fontsize=12)
    ax.set_ylabel("Accuracy")

    # (b) Sensitivity
    ax = axes[1]
    for i, c in enumerate(colors[:num_classes]):
        ax.bar(x + i*width, [metrics_all[m]["Sensitivity"][i] for m in models],
               width=width, color=c, label=f"Class {i}")
    ax.set_title("(b) Sensitivity of ML models", color='red', fontsize=12)
    ax.set_ylabel("Sensitivity")
    ax.set_xticks(x + width)
    ax.set_xticklabels(models)

    # (c) Specificity
    ax = axes[2]
    for i, c in enumerate(colors[:num_classes]):
        ax.bar(x + i*width, [metrics_all[m]["Specificity"][i] for m in models],
               width=width, color=c, label=f"Class {i}")
    ax.set_title("(c) Specificity of ML models", color='red', fontsize=12)
    ax.set_ylabel("Specificity")
    ax.set_xticks(x + width)
    ax.set_xticklabels(models)

    # (d) Error rate
    ax = axes[3]
    for i, c in enumerate(colors[:num_classes]):
        ax.bar(x + i*width, [metrics_all[m]["ErrorRate"][i] for m in models],
               width=width, color=c, label=f"Class {i}")
    ax.set_title("(d) Error rate of ML models", color='red', fontsize=12)
    ax.set_ylabel("Error rate")
    ax.set_xticks(x + width)
    ax.set_xticklabels(models)

    # (e) Overall performance
    ax = axes[4]
    metrics_labels = ["Accuracy", "Sensitivity", "Specificity", "ErrorRate"]
    for i, metric in enumerate(metrics_labels):
        vals = []
        for m in models:
            if metric in ["Sensitivity", "Specificity", "ErrorRate"]:
                vals.append(np.mean(metrics_all[m][metric]))
            else:
                vals.append(metrics_all[m][metric])
        ax.bar(x + i*width, vals, width=width, label=metric)
    ax.set_title("(e) Overall performance of ML models", color='red', fontsize=12)
    ax.set_ylabel("Performance")
    ax.set_xticks(x + width)
    ax.set_xticklabels(models)
    ax.legend()

    plt.tight_layout()
    plt.savefig("results/ML_model_performance.png", dpi=300)
    plt.show()


# ---------- Example Main ----------
def main():
    print("🔹 Loading sample data (replace with your dataset)...")
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=500, n_features=10, n_classes=3,
                               n_informative=6, random_state=42)

    results, preds_dict, splits = train_models(X, y)
    _, X_test, _, y_test = splits

    df_summary = summarize_results(results, preds_dict, y_test)
    df_summary.to_csv("results/model_performance_summary.csv", index=False)

    print("\n🔹 Generating performance plots...")
    plot_performance_dashboard(results, y_test, preds_dict)
    print("\n✅ Done! Figure saved in 'results/ML_model_performance.png'")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    main()
