import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from imblearn.over_sampling import SMOTE

from src.preprocess import prepare_dataset
from src.train_models import train_and_save_models
from src.regression_models import train_regression_models

def main():
    print("🔹 Preprocessing & loading dataset...")

    data_path = "data/fluoride_data.csv"
    if not os.path.exists(data_path):
        print(f"❌ Dataset not found at {data_path}")
        return

    # ✅ Preprocess dataset
    X, y, F_raw, df, features, target_col, scaler, encoder = prepare_dataset(data_path)
    print(f"✅ Data loaded: X shape = {X.shape}, y shape = {y.shape}")

    # ✅ Apply SMOTE to balance classes
    print("\n🔹 Applying SMOTE oversampling...")
    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X, y)

    print("Class distribution after SMOTE:")
    print(pd.Series(y_bal).value_counts(normalize=True))

    # ✅ Split balanced data into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.3, random_state=42, stratify=y_bal
    )
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

    # ✅ Train classification models (on balanced data)
    print("\n🚀 Sending data to training script...")
    results, splits = train_and_save_models(X_bal, y_bal, scaler=scaler, encoder=encoder)
    X_train, X_test, y_train, y_test = splits

    print("✅ Models trained and saved in /models folder")

    # ✅ Regression for actual fluoride prediction
    print("\n🔹 Training Regression Models to predict Fluoride concentration...")
    reg_results = train_regression_models(X, df[target_col])
    print("\n✅ All Done! Models saved in `models/` and results saved.")

    # --- Extra analysis section ---
    print("\n📊 Class distribution (original):")
    print(pd.Series(y).value_counts(normalize=True))

    # 🏆 Find best model by accuracy
    best_model_name = max(results, key=results.get)
    best_accuracy = results[best_model_name]
    print(f"\n🏆 Best Model: {best_model_name} (Accuracy = {best_accuracy:.4f})")

    # Re-train the best model quickly to generate preds (since models aren't returned)
    from src.model_utils import get_model
    model = get_model(best_model_name)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("\n📈 Classification Report:")
    print(classification_report(y_test, preds))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))

    # 🔹 Additional metrics
    macro_f1 = f1_score(y_test, preds, average='macro')
    macro_recall = recall_score(y_test, preds, average='macro')
    print(f"\nMacro F1 Score: {macro_f1:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")

    # Check for target leakage
    print("\n🧩 Target column:", target_col)
    print("Is target in features?", target_col in features)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    main()
