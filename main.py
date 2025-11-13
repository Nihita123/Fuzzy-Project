import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from imblearn.over_sampling import SMOTE

from src.preprocess import prepare_dataset
from src.train_models import train_and_save_models
from src.regression_models import train_regression_models
from src.model_utils import get_model
from src.fuzzy_module import fuzzy_classify_fluoride

def main():
    print("🔹 Preprocessing & loading dataset...")

    data_path = "data/fluoride_data.csv"
    if not os.path.exists(data_path):
        print(f"❌ Dataset not found at {data_path}")
        return

    # ✅ Preprocess dataset
    X, y, F_raw, df, features, target_col, scaler, encoder = prepare_dataset(data_path)
    print(f"✅ Data loaded: X shape = {X.shape}, y shape = {y.shape}")

    # ✅ Apply SMOTE
    print("\n🔹 Applying SMOTE oversampling...")
    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X, y)
    print("Class distribution after SMOTE:")
    print(pd.Series(y_bal).value_counts(normalize=True))

    # ✅ Split balanced data
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.3, random_state=42, stratify=y_bal
    )

    # ✅ Train classification models
    print("\n🚀 Training classification models...")
    results, _ = train_and_save_models(X_bal, y_bal, scaler=scaler, encoder=encoder)
    print("✅ Models trained and saved in /models folder")

   # ✅ Train regression models
    print("\n🔹 Training Regression Models to predict actual Fluoride concentration...")
    reg_results = train_regression_models(X, df[target_col])

    # Fix: pick best regressor by r² score
    best_regressor_name = max(reg_results, key=lambda k: reg_results[k]['R2'])
    best_regressor_score = reg_results[best_regressor_name]['R2']

    print(f"🏆 Best Regressor: {best_regressor_name} (R² = {best_regressor_score:.4f})")

    # Re-train the best regressor to predict fluoride
    reg_model = get_model(best_regressor_name)
    reg_model.fit(X, df[target_col])
    fluoride_preds = reg_model.predict(X)

    df["Predicted_Fluoride"] = fluoride_preds


    # ✅ Apply fuzzy logic classification based on predictions
    print("\n🌍 Regional Fluoride Fuzzy Safety Summary:")
    fuzzy_results = []
    for i in range(len(df)):
        location = df.get("Location", [f"Site_{i}"])[i]
        fluoride_value = df["Predicted_Fluoride"].iloc[i]
        fuzzy_label, fuzzy_score = fuzzy_classify_fluoride(fluoride_value)
        fuzzy_results.append((location, fluoride_value, fuzzy_label, fuzzy_score))

    fuzzy_df = pd.DataFrame(fuzzy_results, columns=["Location", "Fluoride (mg/L)", "Risk_Label", "Risk_Score"])
    print(fuzzy_df.head(10))

    fuzzy_df.to_csv("results/fuzzy_fluoride_summary.csv", index=False)
    print("\n✅ Fuzzy classification results saved to results/fuzzy_fluoride_summary.csv")

    # --- Model performance section ---
    print("\n📊 Class distribution (original):")
    print(pd.Series(y).value_counts(normalize=True))

    best_model_name = max(results, key=results.get)
    print(f"\n🏆 Best Classifier: {best_model_name} (Accuracy = {results[best_model_name]:.4f})")

    model = get_model(best_model_name)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("\n📈 Classification Report:")
    print(classification_report(y_test, preds))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))

    # Additional metrics
    macro_f1 = f1_score(y_test, preds, average='macro')
    macro_recall = recall_score(y_test, preds, average='macro')
    print(f"\nMacro F1 Score: {macro_f1:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    main()
