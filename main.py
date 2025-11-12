import os
from src.preprocess import prepare_dataset
from src.train_models import train_and_save_models
from src.regression_models import train_regression_models

def main():
    print("🔹 Preprocessing & loading dataset...")

    data_path = "data/fluoride_data.csv"
    if not os.path.exists(data_path):
        print(f"❌ Dataset not found at {data_path}")
        return

    # ✅ Preprocess only
    X, y, F_raw, df, features, target_col, scaler, encoder = prepare_dataset(data_path)
    print(f"✅ Data loaded: X shape = {X.shape}, y shape = {y.shape}")

    # ✅ Call training file
    print("\n🚀 Sending data to training script...")
    results, splits = train_and_save_models(X, y, scaler=scaler, encoder=encoder)

    print("✅ Models trained and saved in /models folder")

    # ✅ Regression model for fluoride value
    print("\n🔹 Training Regression Models to predict Fluoride concentration...")
    reg_results = train_regression_models(X, df[target_col])

    print("\n✅ All Done! Models saved in `models/` and results saved.")

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    main()
